"""Sequential (multi-time) LMTP via recursive outcome regression.

Implements a practical two-or-more time-point sequential regression estimator
for modified treatment policies (Díaz–Williams LMTP spirit) without requiring
application-specific column names.

Numeric `A_t` uses a common additive [`ShiftPolicy`](@ref). Categorical `A_t`
uses per-time [`DiscreteTreatmentPolicy`](@ref) values in `policies` (dummy-coded
Q, Díaz–Williams classification ratio at ``t = 1``). Mixed continuous/discrete
treatments are rejected.

# References

- Díaz, Williams, Hoffman & Schenck (2023), *JASA* — sequential ID / EIF for LMTP
- Díaz & van der Laan (2012), *Biometrics* — stochastic / population interventions
- Pair ID with CausalDynamics `TemporalEffectQuery` + `unroll_temporal_dag`
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    SequentialPolicy(treatments, outcome, baseline; time_vary, shift, policies)

Time-ordered treatments `A_1,…,A_T`, baseline covariates `L_0`, and optional
time-varying covariates `time_vary[t]` observed before `A_t`.

Empty `policies` selects the numeric additive-shift path. A nonempty vector of
[`DiscreteTreatmentPolicy`](@ref) (length `T`, or length 1 broadcast to every
time) selects dummy-coded factor recodes. Mixed continuous and categorical
`A_t` is not supported.
"""
struct SequentialPolicy <: Estimand
    treatments::Vector{Symbol}
    outcome::Symbol
    baseline::Vector{Symbol}
    time_vary::Vector{Vector{Symbol}}
    shift::ShiftPolicy
    policies::Vector{DiscreteTreatmentPolicy}
end

function SequentialPolicy(
    treatments::Vector{Symbol},
    outcome::Symbol,
    baseline::Vector{Symbol};
    time_vary::Vector{Vector{Symbol}} = [Symbol[] for _ in treatments],
    shift::ShiftPolicy = shift_policy_from_settings(),
    policies = DiscreteTreatmentPolicy[],
)
    length(time_vary) == length(treatments) || throw(ArgumentError(
        "time_vary length ($(length(time_vary))) must match treatments ($(length(treatments)))",
    ))
    pols = _normalise_sequential_policies(policies, length(treatments); allow_empty = true)
    return SequentialPolicy(treatments, outcome, baseline, time_vary, shift, pols)
end

estimand_engine(::SequentialPolicy) = :sequential_lmtp

"""
    run_sequential_lmtp(data, treatments, outcome; baseline, time_vary, delta, folds, learners, rng) -> NamedTuple

Recursive g-computation + TMLE-style one-step correction at the first time
point. Numeric exposures share an additive MTP shift `delta`. Categorical
exposures require `policies` ([`DiscreteTreatmentPolicy`](@ref) per time, or one
policy broadcast to all times).

Returns `(estimate, se, n, times, delta)` and, on the factor path,
`density_ratio`, `positivity`, and `n_changed`.
"""
function run_sequential_lmtp(
    data::DataFrame,
    treatments::Vector{Symbol},
    outcome::Symbol;
    baseline::Vector{Symbol},
    time_vary::Vector{Vector{Symbol}} = [Symbol[] for _ in treatments],
    delta::Float64 = 1.0,
    folds::Int = recommend_folds(nrow(data)),
    learners = recommend_learners(nrow(data)),
    learners_trt = (:logistic, :mean),
    lower_q = mtp_settings().lower_q,
    upper_q = mtp_settings().upper_q,
    shift::ShiftPolicy = additive_shift_policy(; lower_q = lower_q, upper_q = upper_q),
    policies = DiscreteTreatmentPolicy[],
    rng::AbstractRNG = StableRNG(42),
    handle_missing::Symbol = :drop,
    trunc::Real = 10.0,
)
    validate_contrast_learners(learners; context = "run_sequential_lmtp")
    T = length(treatments)
    T >= 1 || throw(ArgumentError("need at least one treatment time"))
    length(time_vary) == T || throw(ArgumentError("time_vary length must equal treatments"))
    all_cols = unique(vcat(
        baseline,
        reduce(vcat, time_vary; init = Symbol[]),
        treatments,
    ))
    miss = handle_missing_data(
        data, outcome, all_cols, handle_missing;
        rng = rng, rung = :L2, time_indexed = true,
    )
    data_clean, ipcw_w, extra_cols = miss
    baseline = unique(vcat(baseline, extra_cols))
    pols = _normalise_sequential_policies(policies, T; allow_empty = true)
    kind = _validate_sequential_treatment_kinds(data_clean, treatments, pols)
    if kind === :discrete
        return with_missingness(_run_sequential_discrete_lmtp(
            data_clean, treatments, outcome, baseline, time_vary, pols, ipcw_w;
            folds = folds, learners = learners, learners_trt = learners_trt,
            rng = rng, trunc = Float64(trunc),
        ), miss.meta)
    end
    return with_missingness(_run_sequential_numeric_lmtp(
        data_clean, treatments, outcome, baseline, time_vary, ipcw_w;
        delta = delta, folds = folds, learners = learners, shift = shift, rng = rng,
    ), miss.meta)
end

function run_sequential_lmtp(data::DataFrame, estimand::SequentialPolicy; kwargs...)
    kw = (; (p.first => p.second for p in pairs(kwargs) if p.first !== :policies)...)
    policies = haskey(kwargs, :policies) ? kwargs[:policies] : estimand.policies
    return run_sequential_lmtp(
        data, estimand.treatments, estimand.outcome;
        baseline = estimand.baseline,
        time_vary = estimand.time_vary,
        shift = estimand.shift,
        policies = policies,
        kw...,
    )
end

"""
    sequential_identification_certificate(unrolling, query; nuisance_source) -> IdentificationCertificate

Run CausalDynamics `identify` on a `TemporalUnrolling` with a
`TemporalEffectQuery`, then wrap the result for `plan_mtp` /
`execute_estimand`. Application code supplies column names separately
via [`SequentialPolicy`](@ref); this never invents sheep-specific symbols.
"""
function sequential_identification_certificate(
    unrolling::CausalDynamics.TemporalUnrolling,
    query::TemporalEffectQuery;
    nuisance_source::Symbol = :temporal_graph,
)
    id = identify(unrolling, query)
    return identification_certificate(
        id, Symbol(query.treatment), Symbol(query.outcome);
        adjustment = Symbol.(id.adjustment),
        mediators = Symbol[],
        nuisance_source = nuisance_source,
        temporal_lags = (treat_lag = query.t_treat, outcome_lag = query.t_outcome),
    )
end

"""
    _normalise_sequential_policies(policies, T; allow_empty) -> Vector{DiscreteTreatmentPolicy}

Accept a single policy (broadcast to every time), a length-`T` vector, or
(when `allow_empty`) an empty vector for the numeric path.
"""
function _normalise_sequential_policies(policies, T::Int; allow_empty::Bool = false)
    pols = if policies isa DiscreteTreatmentPolicy
        DiscreteTreatmentPolicy[policies]
    else
        collect(DiscreteTreatmentPolicy, policies)
    end
    if isempty(pols)
        allow_empty || throw(ArgumentError("sequential factor path requires at least one DiscreteTreatmentPolicy"))
        return DiscreteTreatmentPolicy[]
    end
    if length(pols) == 1
        return DiscreteTreatmentPolicy[pols[1] for _ in 1:T]
    end
    length(pols) == T || throw(ArgumentError(
        "policies length ($(length(pols))) must be 1 (broadcast to all times) or $T",
    ))
    return pols
end

"""Classify a treatment column as `:factor`, `:continuous`, `:integer`, or `:other`."""
function _sequential_treatment_kind(col)
    T = Base.nonmissingtype(eltype(col))
    (T <: AbstractString || T <: AbstractChar) && return :factor
    T <: AbstractFloat && return :continuous
    (T <: Integer || T <: Bool) && return :integer
    return :other
end

"""
    _validate_sequential_treatment_kinds(df, treatments, policies) -> Symbol

Return `:discrete` when `policies` is nonempty, otherwise `:numeric`.
Reject mixed continuous/categorical columns and categorical columns without policies.
"""
function _validate_sequential_treatment_kinds(
    df::DataFrame,
    treatments::Vector{Symbol},
    policies::Vector{DiscreteTreatmentPolicy},
)
    kinds = [_sequential_treatment_kind(df[!, trt]) for trt in treatments]
    has_factor = any(==(:factor), kinds)
    has_continuous = any(==(:continuous), kinds)
    if has_factor && has_continuous
        throw(ArgumentError(
            "sequential LMTP does not mix continuous and categorical treatments; " *
            "got $(collect(zip(treatments, kinds)))",
        ))
    end
    if !isempty(policies)
        has_continuous && throw(ArgumentError(
            "sequential factor policies require a categorical treatment at every time; " *
            "continuous A_t is not supported in this version " *
            "(got $(collect(zip(treatments, kinds))))",
        ))
        return :discrete
    end
    if has_factor
        first_factor = treatments[findfirst(==(:factor), kinds)]
        throw(ArgumentError(
            "sequential LMTP treatment :$first_factor is categorical; " *
            "pass policies as a Vector{DiscreteTreatmentPolicy} for multi-time factor recodes, " *
            "or use run_discrete_lmtp for a single time point (T=1)",
        ))
    end
    return :numeric
end

"""
    _numeric_sequential_treatment(col, name) -> Vector{Float64}

Require a numeric exposure. Categorical columns must pass `policies` on
[`run_sequential_lmtp`](@ref) or use [`run_discrete_lmtp`](@ref) at T=1.
"""
function _numeric_sequential_treatment(col, name::Symbol)
    T = Base.nonmissingtype(eltype(col))
    if T <: AbstractString || T <: AbstractChar
        throw(ArgumentError(
            "sequential LMTP treatment :$name is categorical; " *
            "pass policies as a Vector{DiscreteTreatmentPolicy} for multi-time factor recodes, " *
            "or use run_discrete_lmtp for a single time point (T=1)",
        ))
    end
    try
        return Float64.(col)
    catch err
        throw(ArgumentError(
            "sequential LMTP treatment :$name must be numeric; " *
            "use run_discrete_lmtp for a single time point (T=1), " *
            "or policies for multi-time factor recodes. $(sprint(showerror, err))",
        ))
    end
end

function _run_sequential_numeric_lmtp(
    data_clean::DataFrame,
    treatments::Vector{Symbol},
    outcome::Symbol,
    baseline::Vector{Symbol},
    time_vary::Vector{Vector{Symbol}},
    ipcw_w;
    delta::Float64,
    folds::Int,
    learners,
    shift::ShiftPolicy,
    rng::AbstractRNG,
)
    T = length(treatments)
    n = nrow(data_clean)
    y = Float64.(data_clean[!, outcome])

    a_nat = Vector{Vector{Float64}}(undef, T)
    a_pol = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        a = _numeric_sequential_treatment(data_clean[!, treatments[t]], treatments[t])
        a_nat[t] = apply_policy_values(a, 0.0, shift)
        a_pol[t] = apply_policy_values(a, delta, shift)
    end

    Q = copy(y)
    fold_sets = crossfit_indices(n, folds, rng)
    ic = zeros(n)
    baseline_schema = fit_covariate_schema(data_clean, baseline)

    for t in T:-1:1
        hist = unique(vcat(baseline, reduce(vcat, time_vary[1:t]; init = Symbol[]), treatments[1:t]))
        history_schema = fit_covariate_schema(data_clean, hist)
        Q_next = copy(Q)
        Q = zeros(n)
        for (fi, test_idx) in enumerate(fold_sets)
            train_idx = setdiff(1:n, test_idx)
            train = data_clean[train_idx, :]
            sl = _fit_sl_outcome(
                train, hist, Q_next[train_idx];
                treatment = treatments[t], learners = learners, rng = rng,
                schema = history_schema,
            )
            block = data_clean[test_idx, :]
            Q[test_idx] .= _predict_sl(
                sl, block, hist;
                treatment = treatments[t], treatment_values = a_pol[t][test_idx],
            )
            if t == 1
                a = _numeric_sequential_treatment(data_clean[!, treatments[1]], treatments[1])
                L, U = exposure_bounds(a, shift.lower_q, shift.upper_q)
                sl_a = fit_super_learner(
                    design_matrix(baseline_schema, train), a[train_idx];
                    learners = learners, rng = rng,
                )
                μ_tr = predict_super_learner(sl_a, design_matrix(baseline_schema, train))
                μ_te = predict_super_learner(sl_a, design_matrix(baseline_schema, block))
                σ_a = robust_residual_sd(a[train_idx] .- μ_tr)
                req = mean(a_pol[1][test_idx] .- a_nat[1][test_idx])
                H = _mtp_clever_covariate_clamp_aware(a[test_idx], μ_te, σ_a, req, L, U)
                H = truncate_weights(H; trunc = 10.0)
                Q_obs = _predict_sl(sl, block, hist; treatment = treatments[1])
                ic[test_idx] .= Q[test_idx] .+ H .* (Q_next[test_idx] .- Q_obs)
            end
        end
    end

    return _summarise_sequential_influence(Q, ic, ipcw_w, n, T, delta)
end

"""
    _run_sequential_discrete_lmtp(...) -> NamedTuple

Sequential regression with dummy-coded factor treatments and a classification
density ratio at ``t = 1``. Mixed continuous/discrete `A_t` is rejected upstream.
"""
function _run_sequential_discrete_lmtp(
    data_clean::DataFrame,
    treatments::Vector{Symbol},
    outcome::Symbol,
    baseline::Vector{Symbol},
    time_vary::Vector{Vector{Symbol}},
    policies::Vector{DiscreteTreatmentPolicy},
    ipcw_w;
    folds::Int,
    learners,
    learners_trt,
    rng::AbstractRNG,
    trunc::Float64,
)
    T = length(treatments)
    analysis = copy(data_clean)
    for trt in treatments
        analysis[!, trt] = _factorise_treatment(analysis[!, trt])
    end
    n = nrow(analysis)
    y = Float64.(analysis[!, outcome])
    W0 = isempty(baseline) ? zeros(n, 0) :
        Matrix{Float64}(_covariate_matrix(fit_covariate_schema(analysis, baseline), analysis))

    a_nat = Vector{Vector}(undef, T)
    a_pol = Vector{Vector}(undef, T)
    for t in 1:T
        a_nat[t] = collect(analysis[!, treatments[t]])
        a_pol[t] = _factorise_treatment(apply_discrete_policy(a_nat[t], W0, policies[t]))
        analysis[!, treatments[t]] = a_nat[t]
    end

    schema_df = _pad_sequential_factor_levels(analysis, treatments, a_pol)
    Q = copy(y)
    fold_sets = crossfit_indices(n, folds, rng)
    ic = zeros(n)
    baseline_schema = fit_covariate_schema(analysis, baseline)

    for t in T:-1:1
        hist = unique(vcat(baseline, reduce(vcat, time_vary[1:t]; init = Symbol[]), treatments[1:t]))
        history_schema = fit_covariate_schema(schema_df, hist)
        Q_next = copy(Q)
        Q = zeros(n)
        for test_idx in fold_sets
            train_idx = setdiff(1:n, test_idx)
            train = analysis[train_idx, :]
            sl = _fit_sl_outcome(
                train, hist, Q_next[train_idx];
                treatment = nothing, learners = learners, rng = rng,
                schema = history_schema,
            )
            block = analysis[test_idx, :]
            cf = _counterfactual_frame(block, treatments[t], a_pol[t][test_idx])
            Q[test_idx] .= _predict_sl(sl, cf, hist; treatment = nothing)
            if t == 1
                levels = _canonical_treatment_levels(a_nat[1], policies[1])
                clf = _fit_discrete_density_ratio(
                    a_nat[1][train_idx], a_pol[1][train_idx],
                    Matrix{Float64}(W0[train_idx, :]), levels;
                    learners = learners_trt, rng = rng,
                )
                H = _ratio_from_discrete_classifier(
                    clf, a_nat[1][test_idx], Matrix{Float64}(W0[test_idx, :]), levels;
                    trunc = trunc, mtp = policies[1].mtp, a_policy = a_pol[1][test_idx],
                )
                Q_obs = _predict_sl(sl, block, hist; treatment = nothing)
                ic[test_idx] .= Q[test_idx] .+ H .* (Q_next[test_idx] .- Q_obs)
            end
        end
    end

    pos = _sequential_discrete_positivity(a_nat, a_pol)
    n_changed = sum(t -> count(string.(a_nat[t]) .!= string.(a_pol[t])), 1:T)
    summary = _summarise_sequential_influence(Q, ic, ipcw_w, n, T, NaN)
    return merge(summary, (;
        density_ratio = :classification,
        positivity = pos,
        n_changed = n_changed,
    ))
end

"""Pad dummy rows so the outcome schema includes every policy level of each `A_t`."""
function _pad_sequential_factor_levels(
    df::DataFrame,
    treatments::Vector{Symbol},
    a_pol,
)
    extras = DataFrame[]
    template = df[1:1, :]
    for (t, trt) in enumerate(treatments)
        for lev in unique(a_pol[t])
            row = copy(template)
            row[1, trt] = lev
            push!(extras, row)
        end
    end
    return isempty(extras) ? df : vcat(df, extras...)
end

"""Combine per-time [`discrete_positivity`](@ref) reports."""
function _sequential_discrete_positivity(a_nat, a_pol)
    parts = [discrete_positivity(a_nat[t], a_pol[t]) for t in eachindex(a_nat)]
    empty_support = reduce(vcat, (p.empty_support for p in parts); init = String[])
    return (
        ok = all(p.ok for p in parts),
        empty_support = empty_support,
        n_observed_levels = map(p -> p.n_observed_levels, parts),
        n_policy_levels = map(p -> p.n_policy_levels, parts),
        times = parts,
    )
end

function _summarise_sequential_influence(Q, ic, ipcw_w, n::Int, T::Int, delta)
    if any(!=(0.0), ic)
        if _uses_ipcw_weights(ipcw_w)
            s = weighted_influence_summary(ic, ipcw_w)
            est = s.estimate
            se = s.se
        else
            est = mean(ic)
            se = std(ic) / sqrt(n)
        end
    else
        est = _uses_ipcw_weights(ipcw_w) ? transport_weighted_mean(Q, ipcw_w) : mean(Q)
        se = std(Q) / sqrt(n)
    end
    return (estimate = est, se = se, n = n, times = T, delta = delta)
end

export SequentialPolicy, run_sequential_lmtp, sequential_identification_certificate
