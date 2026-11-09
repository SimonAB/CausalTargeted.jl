"""Point-treatment repeated-outcome MSM with joint influence-function covariance.

Estimates an unstructured marginal or baseline-stratified profile

```math
\\tau(t) = E[Y_t \\mid do(A=1)] - E[Y_t \\mid do(A=0)]
```

for binary point treatment and several outcomes measured on the same units
(wide layout). Cross-fitted propensity is shared across times; outcome
regressions are fit per ``Y_t``. Unit-level AIPW influence curves are stacked
into an ``n \\times T`` matrix whose empirical covariance yields
``\\widehat{\\Sigma}`` for joint Wald contrasts (Rosenblum & van der Laan 2010
spirit; Julia-native, not an R `tmleMSM` port). Stratified estimation pools
nuisance fitting across the full sample, then applies stratum-specific targeting
and conditional Hajek influence-curve normalisation.

# References

- Rosenblum, M. & van der Laan, M. J. (2010). Targeted Maximum Likelihood
  Estimation of the Parameter of a Marginal Structural Model.
  *The International Journal of Biostatistics*, 6(2).
- Bang & Robins (2005); Chernozhukov et al. (2018) — doubly robust / cross-fit AIPW
"""

using DataFrames
using LinearAlgebra
using Random
using StableRNGs
using Statistics
using Graphs: AbstractGraph
using CausalDynamics: IdentificationResult, TotalEffectQuery, identify

"""
    RepeatedOutcomeMSM(trt, outcomes, adjustment)

Point-treatment profile estimand: binary `trt` on several outcomes measured on
the same units. Engine [`estimand_engine`](@ref) is `:repeated_msm`.
"""
struct RepeatedOutcomeMSM <: Estimand
    trt::Symbol
    outcomes::Vector{Symbol}
    adjustment::Vector{Symbol}
end

RepeatedOutcomeMSM(trt::Symbol, outcomes::AbstractVector{Symbol}, adjustment::AbstractVector{Symbol}) =
    RepeatedOutcomeMSM(trt, collect(Symbol, outcomes), collect(Symbol, adjustment))

estimand_engine(::RepeatedOutcomeMSM) = :repeated_msm

"""
    identify_repeated_outcomes(g, treatment, outcomes; node_names, missingness)
        -> Vector{IdentificationResult}

Identify `TotalEffectQuery(treatment, y)` for each outcome on the same graph.
Useful when all ``Y_t`` share a backdoor set (static ``A``, baseline ``W``).
"""
function identify_repeated_outcomes(
    g::AbstractGraph,
    treatment,
    outcomes::AbstractVector;
    node_names = nothing,
    missingness = nothing,
)
    isempty(outcomes) && throw(ArgumentError("outcomes must be nonempty"))
    return map(outcomes) do y
        q = TotalEffectQuery(treatment, y)
        if missingness === nothing
            identify(g, q; node_names = node_names)
        else
            identify(g, q; node_names = node_names, missingness = missingness)
        end
    end
end

"""
    _crossfit_propensity_folds(df, treatment, covariates, fold_sets; learners, rng)
        -> Vector{Float64}

Cross-fitted propensity on a **shared** fold partition (same folds as outcome fits).
"""
function _crossfit_propensity_folds(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    fold_sets;
    learners = (:logistic, :mean),
    rng = StableRNG(1),
)
    return crossfit_propensity(
        df, treatment, covariates, length(fold_sets), rng;
        learners = learners, fold_sets = fold_sets,
    )
end

"""
    _aipw_influence_matrix(...) -> (psi, estimates, psi_μ1, μ1, psi_μ0, μ0)

Cross-fitted AIPW influence curves for each outcome (columns), sharing folds and
propensity `g_hat`. Returns uncentred plug-in+weighting curves for ``τ``,
``μ(1)``, and ``μ(0)``, with Hajek means.
"""
function _aipw_influence_matrix(
    df::DataFrame,
    treatment::Symbol,
    outcomes::Vector{Symbol},
    adj_covars::Vector{Symbol},
    fold_sets,
    g_hat::AbstractVector{<:Real},
    ipcw_w::AbstractVector{<:Real};
    learners = DEFAULT_SL_LEARNERS,
    rng = StableRNG(1),
    estimator::Symbol = :tmle,
)
    estimator in (:tmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "estimator must be :tmle or :eif (aliases :aipw, :sdr); got :$estimator",
    ))
    n = nrow(df)
    T = length(outcomes)
    length(g_hat) == n || throw(ArgumentError("g_hat length must match nrow(df)"))
    length(ipcw_w) == n || throw(ArgumentError("ipcw_w length must match nrow(df)"))
    a = Float64.(df[!, treatment])
    all(x -> x == 0.0 || x == 1.0, a) || throw(ArgumentError(
        "run_repeated_outcome_msm requires a binary 0/1 treatment column",
    ))

    g = Float64.(g_hat)
    H = @. a / g - (1 - a) / (1 - g)
    H1 = 1.0 ./ g
    H0 = -1.0 ./ (1.0 .- g)
    covariate_schema = fit_covariate_schema(df, adj_covars)
    psi = zeros(n, T)
    psi_μ1 = zeros(n, T)
    psi_μ0 = zeros(n, T)

    for (j, outcome) in enumerate(outcomes)
        y = Float64.(df[!, outcome])
        for test_idx in fold_sets
            train_idx = setdiff(1:n, test_idx)
            train = df[train_idx, :]
            test = df[test_idx, :]
            sl = _fit_sl_outcome(
                train, adj_covars, y[train_idx];
                treatment = treatment,
                learners = learners,
                rng = rng,
                schema = covariate_schema,
            )
            Q_obs = _predict_sl(
                sl, test, adj_covars;
                treatment = treatment,
                treatment_values = a[test_idx],
            )
            Q1 = _predict_sl(
                sl, test, adj_covars;
                treatment = treatment,
                treatment_values = ones(length(test_idx)),
            )
            Q0 = _predict_sl(
                sl, test, adj_covars;
                treatment = treatment,
                treatment_values = zeros(length(test_idx)),
            )
            Ht = H[test_idx]
            if estimator === :tmle
                den = sum(abs2, Ht)
                ε = den > 1e-12 ? sum(Ht .* (y[test_idx] .- Q_obs)) / den : 0.0
                Q_obs = Q_obs .+ ε .* Ht
                Q1 = Q1 .+ ε .* H1[test_idx]
                Q0 = Q0 .+ ε .* H0[test_idx]
            end
            resid = y[test_idx] .- Q_obs
            psi[test_idx, j] = Ht .* resid .+ (Q1 .- Q0)
            # μ(1): A/g * resid + Q1; μ(0): (1-A)/(1-g) * resid + Q0
            # H1 = 1/g, H0 = -1/(1-g) ⇒ (1-A)/(1-g) = -(1-A)*H0
            psi_μ1[test_idx, j] = (a[test_idx] .* H1[test_idx]) .* resid .+ Q1
            psi_μ0[test_idx, j] = ((1 .- a[test_idx]) .* (-H0[test_idx])) .* resid .+ Q0
        end
    end

    estimates = Vector{Float64}(undef, T)
    μ1 = Vector{Float64}(undef, T)
    μ0 = Vector{Float64}(undef, T)
    for j in 1:T
        estimates[j] = transport_weighted_mean(psi[:, j], ipcw_w)
        μ1[j] = transport_weighted_mean(psi_μ1[:, j], ipcw_w)
        μ0[j] = transport_weighted_mean(psi_μ0[:, j], ipcw_w)
    end
    return psi, estimates, psi_μ1, μ1, psi_μ0, μ0
end

"""
    _msm_covariance(psi, estimates, ipcw_w; cluster=nothing) -> (Σ, se, ic)

Hajek-centred influence matrix and sampling covariance of the mean.

When `cluster` is `nothing`, ``\\widehat{\\Sigma}=n^{-2}D^\\top D`` (unit-level).
When `cluster` is a length-`n` vector of cluster ids, use the cluster sandwich
``\\widehat{\\Sigma}=n^{-2}\\sum_g s_g s_g^\\top`` with ``s_g=\\sum_{i\\in g} D_i``
(sampling hierarchy; not an LMM / BLUP estimand).
"""
function _msm_covariance(
    psi::AbstractMatrix{<:Real},
    estimates::AbstractVector{<:Real},
    ipcw_w::AbstractVector{<:Real};
    cluster::Union{Nothing, AbstractVector} = nothing,
)
    n, T = size(psi)
    length(estimates) == T || throw(ArgumentError("estimates length must equal ncol(psi)"))
    length(ipcw_w) == n || throw(ArgumentError("ipcw_w length must match nrow(psi)"))
    w = Float64.(ipcw_w)
    sw = sum(w)
    sw ≈ 0 && throw(ArgumentError("weights sum to zero"))
    w_bar = sw / n
    ic = Matrix{Float64}(undef, n, T)
    for j in 1:T
        ic[:, j] = (w ./ w_bar) .* (Float64.(psi[:, j]) .- estimates[j])
    end
    if cluster === nothing
        Σ = (ic' * ic) ./ n^2
    else
        length(cluster) == n || throw(ArgumentError("cluster length must match nrow(psi)"))
        groups = Dict{Any, Vector{Int}}()
        for i in 1:n
            push!(get!(groups, cluster[i], Int[]), i)
        end
        Σ = zeros(T, T)
        for idxs in values(groups)
            s = vec(sum(ic[idxs, :]; dims = 1))
            Σ .+= s * s'
        end
        Σ ./= n^2
    end
    Σ = Symmetric(0.5 .* (Σ .+ Σ'))
    se = sqrt.(diag(Σ))
    return Matrix{Float64}(Σ), se, ic
end

"""
    _complete_profile_R(df, outcomes) -> Vector{Float64}

Response indicator for a complete outcome profile (all ``Y_t`` observed).
"""
function _complete_profile_R(df::DataFrame, outcomes::Vector{Symbol})
    n = nrow(df)
    R = ones(n)
    for y in outcomes
        R .*= Float64.(.!ismissing.(df[!, y]))
    end
    return R
end

"""
    _ipcw_weights_from_R(df, R, covariates; learners, rng) -> Vector{Float64}

Stabilised IPCW from a 0/1 response vector `R` (units with `R=0` get weight 0).
"""
function _ipcw_weights_from_R(
    df::DataFrame,
    R::AbstractVector{<:Real},
    covariates::Vector{Symbol};
    learners = (:logistic, :mean),
    rng::AbstractRNG = StableRNG(1),
)
    n = nrow(df)
    length(R) == n || throw(ArgumentError("R length must match nrow(df)"))
    Rv = Float64.(R)
    marginal_p = mean(Rv)
    if marginal_p >= 1.0 - 1e-10 || isempty(covariates)
        w = ones(n)
        w[Rv .== 0.0] .= 0.0
        return w
    end
    X = design_matrix(df, covariates)
    sl = fit_super_learner(
        X, Rv;
        learners = learners, family = :binomial, metalearner = :invmse, rng = rng,
    )
    p_obs = clamp.(predict_super_learner(sl, X), 1e-3, 1.0 - 1e-3)
    w = marginal_p ./ p_obs
    w[Rv .== 0.0] .= 0.0
    return w
end

"""
    _handle_missing_repeated_outcomes(df, treatment, outcomes, baseline, strategy; rng)
        -> MissingDataResult

Observable policy for a **shared** analysis sample across all outcomes.

- `:drop` — complete cases on treatment, baseline, and every ``Y_t``
- `:ipcw` — drop incomplete treatment/baseline; IPCW on the complete-profile
  indicator; keep complete-profile rows
- `:impute` — mean-impute baseline only; drop missing treatment or any ``Y_t``
- `:ipcw_impute` — impute baseline, then profile IPCW

Outcomes are never imputed (fills would enter the IF).
"""
function _handle_missing_repeated_outcomes(
    df::DataFrame,
    treatment::Symbol,
    outcomes::Vector{Symbol},
    baseline::Vector{Symbol},
    strategy::Symbol;
    rng::AbstractRNG = StableRNG(1),
    required_complete::Vector{Symbol} = Symbol[],
)
    n_in = nrow(df)
    extra_cols = Symbol[]
    baseline = unique(baseline)
    required = unique(vcat([treatment], baseline, outcomes, required_complete))
    rates = _column_miss_rates(df, required)

    if strategy == :drop
        df_clean = dropmissing(df, required)
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, ones(nrow(df_clean)), extra_cols, meta)
    elseif strategy == :ipcw
        df_cc = dropmissing(df, unique(vcat([treatment], baseline, required_complete)))
        R = _complete_profile_R(df_cc, outcomes)
        w = _ipcw_weights_from_R(df_cc, R, baseline; rng = rng)
        keep = R .== 1.0
        df_clean = df_cc[keep, :]
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, w[keep], extra_cols, meta)
    elseif strategy == :impute
        df_copy = copy(df)
        extra_cols = impute_covariates_mean!(df_copy, baseline)
        df_clean = dropmissing(df_copy, unique(vcat([treatment], outcomes, required_complete)))
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, ones(nrow(df_clean)), extra_cols, meta)
    elseif strategy == :ipcw_impute
        df_copy = copy(df)
        extra_cols = impute_covariates_mean!(df_copy, baseline)
        df_cc = dropmissing(df_copy, unique(vcat([treatment], required_complete)))
        R = _complete_profile_R(df_cc, outcomes)
        covars = unique(vcat(baseline, extra_cols))
        w = _ipcw_weights_from_R(df_cc, R, covars; rng = rng)
        keep = R .== 1.0
        df_clean = df_cc[keep, :]
        meta = _missing_data_meta(
            strategy, n_in, nrow(df_clean), rates;
            rung = :L2, time_indexed = true,
        )
        return MissingDataResult(df_clean, w[keep], extra_cols, meta)
    else
        error("Unknown missing-data strategy: $strategy. Use :drop, :ipcw, :impute, or :ipcw_impute.")
    end
end

"""
    unstack_repeated_outcomes(long; id, time, outcome, treatment, covariates, prefix)
        -> DataFrame

Pivot a long table (`id`, `time`, `Y`) to wide `Y1…YT` with unit-constant
`treatment` and `covariates`. Times are sorted; columns are `prefix` * `1:T`.
"""
function unstack_repeated_outcomes(
    long::DataFrame;
    id::Symbol,
    time::Symbol,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol} = Symbol[],
    prefix::AbstractString = "Y",
)
    required = unique(vcat([id, time, outcome, treatment], covariates))
    for c in required
        hasproperty(long, c) || throw(ArgumentError("long table missing column :$c"))
    end
    times = sort(unique(long[!, time]))
    T = length(times)
    T >= 1 || throw(ArgumentError("no time levels in :$time"))
    time_index = Dict(t => i for (i, t) in enumerate(times))
    ynames = [Symbol(prefix, i) for i in 1:T]
    records = Dict{Symbol, Any}[]
    for sub in groupby(long, id)
        nrow(sub) == T || throw(ArgumentError(
            "unit $(sub[1, id]) has $(nrow(sub)) rows; expected $T distinct times",
        ))
        a0 = sub[1, treatment]
        all(==(a0), sub[!, treatment]) || throw(ArgumentError(
            "treatment :$treatment is not constant within unit $(sub[1, id])",
        ))
        rec = Dict{Symbol, Any}(treatment => a0)
        for c in covariates
            v0 = sub[1, c]
            all(==(v0), sub[!, c]) || throw(ArgumentError(
                "covariate :$c is not constant within unit $(sub[1, id])",
            ))
            rec[c] = v0
        end
        y = Vector{Any}(undef, T)
        fill!(y, missing)
        for r in eachrow(sub)
            y[time_index[r[time]]] = r[outcome]
        end
        for i in 1:T
            rec[ynames[i]] = y[i]
        end
        push!(records, rec)
    end
    wide = DataFrame(records)
    order = unique(vcat([treatment], covariates, ynames))
    return select(wide, order)
end

"""Normalise `strata` and reject missing columns."""
function _normalise_msm_strata(df::DataFrame, strata)
    cols = if strata === nothing
        Symbol[]
    elseif strata isa Symbol
        [strata]
    elseif strata isa AbstractVector{Symbol}
        collect(Symbol, strata)
    else
        throw(ArgumentError("strata must be nothing, a Symbol, or a Vector{Symbol}"))
    end
    allunique(cols) || throw(ArgumentError("strata columns must be unique"))
    for col in cols
        hasproperty(df, col) || throw(ArgumentError("strata column :$col not found"))
    end
    return cols
end

"""
Return observed joint-stratum keys in deterministic order. Explicit categorical
levels determine their component order; otherwise first occurrence does.
"""
function _ordered_msm_strata(df::DataFrame, cols::Vector{Symbol})
    isempty(cols) && return NamedTuple[]
    names_tuple = Tuple(cols)
    keys = [NamedTuple{names_tuple}(Tuple(df[i, c] for c in cols)) for i in 1:nrow(df)]
    observed = unique(keys)
    ranks = Vector{Dict{Any, Int}}(undef, length(cols))
    for (j, col) in enumerate(cols)
        values = collect(df[!, col])
        mod = parentmodule(typeof(df[!, col]))
        ordered_values = if isdefined(mod, :levels)
            explicit = collect(getfield(mod, :levels)(df[!, col]))
            empty_levels = setdiff(explicit, unique(values))
            isempty(empty_levels) || throw(ArgumentError(
                "strata column :$col has empty categorical levels $(empty_levels)",
            ))
            explicit
        else
            unique(values)
        end
        ranks[j] = Dict(v => i for (i, v) in enumerate(ordered_values))
    end
    sort!(observed; by = key -> Tuple(get(ranks[j], key[cols[j]], typemax(Int)) for j in eachindex(cols)))
    return observed
end

"""Cross-fitting folds balanced within every joint stratum × treatment cell."""
function _stratified_msm_folds(keys, a::AbstractVector{<:Real}, folds::Int, rng::AbstractRNG)
    folds = max(2, folds)
    fold_sets = [Int[] for _ in 1:folds]
    for key in unique(keys), arm in (0.0, 1.0)
        idx = findall(i -> keys[i] == key && a[i] == arm, eachindex(keys))
        length(idx) >= folds || throw(ArgumentError(
            "stratum $key treatment arm $(Int(arm)) has $(length(idx)) observations; " *
            "need at least folds=$folds so every training/test fold is supported",
        ))
        shuffled = idx[randperm(rng, length(idx))]
        for (q, i) in enumerate(shuffled)
            push!(fold_sets[mod1(q, folds)], i)
        end
    end
    foreach(sort!, fold_sets)
    return fold_sets
end

function _validate_known_propensity(values, label::AbstractString)
    g = Float64.(values)
    all(isfinite, g) || throw(ArgumentError("$label propensity values must be finite"))
    all(x -> 0.0 < x < 1.0, g) || throw(ArgumentError(
        "$label propensity values must be strictly between zero and one",
    ))
    return g
end

"""Resolve estimated/scalar/column/vector propensity after row filtering."""
function _resolve_msm_propensity(
    df_original::DataFrame,
    df_clean::DataFrame,
    row_id::Symbol,
    propensity,
    treatment::Symbol,
    adj_covars::Vector{Symbol},
    fold_sets;
    learners_trt,
    rng,
)
    n = nrow(df_clean)
    if propensity === nothing
        g = _crossfit_propensity_folds(
            df_clean, treatment, adj_covars, fold_sets;
            learners = learners_trt, rng = rng,
        )
        return g, (estimated = true, source = :estimated, column = nothing)
    elseif propensity isa Real
        g = _validate_known_propensity(fill(propensity, n), "scalar")
        return g, (estimated = false, source = :scalar, column = nothing)
    elseif propensity isa Symbol
        hasproperty(df_clean, propensity) || throw(ArgumentError(
            "propensity column :$propensity not found in analysis frame",
        ))
        g = _validate_known_propensity(df_clean[!, propensity], "column :$propensity")
        return g, (estimated = false, source = :column, column = propensity)
    elseif propensity isa AbstractVector
        length(propensity) == nrow(df_original) || throw(ArgumentError(
            "propensity vector length $(length(propensity)) must match input n=$(nrow(df_original))",
        ))
        idx = Int.(df_clean[!, row_id])
        g = _validate_known_propensity(propensity[idx], "vector")
        return g, (estimated = false, source = :vector, column = nothing)
    else
        throw(ArgumentError(
            "propensity must be nothing, a scalar probability, a column Symbol, or an input-row-aligned vector",
        ))
    end
end

"""Covariance from already normalised conditional influence curves."""
function _msm_covariance_from_ic(ic::AbstractMatrix{<:Real}; cluster = nothing)
    n, p = size(ic)
    if cluster === nothing
        covariance = (ic' * ic) ./ n^2
    else
        length(cluster) == n || throw(ArgumentError("cluster length must match nrow(ic)"))
        groups = Dict{Any, Vector{Int}}()
        for i in 1:n
            push!(get!(groups, cluster[i], Int[]), i)
        end
        covariance = zeros(p, p)
        for idx in values(groups)
            score = vec(sum(ic[idx, :]; dims = 1))
            covariance .+= score * score'
        end
        covariance ./= n^2
    end
    covariance = Matrix{Float64}(Symmetric(0.5 .* (covariance .+ covariance')))
    return covariance, sqrt.(max.(diag(covariance), 0.0))
end

"""
Shared full-sample outcome fits with stratum-specific fold fluctuations.

For stratum `s` and fold `v`, the linear fluctuation solves
`sum_{i in v} w_i I(S_i=s) H_i {Y_i-Q_i-epsilon*H_i}=0`. The conditional
Hajek influence curve is
`w_i I(S_i=s)/mean(w I_s) * {D_i(s,t)-theta_s(t)}`; this explicit denominator
normalisation is what permits all stratum × time columns to share one joint
covariance without treating conditional parameters as unconditional means.
"""
function _stratified_aipw_influence_matrix(
    df::DataFrame,
    treatment::Symbol,
    outcomes::Vector{Symbol},
    adj_covars::Vector{Symbol},
    fold_sets,
    g_hat::AbstractVector{<:Real},
    ipcw_w::AbstractVector{<:Real},
    strata_cols::Vector{Symbol},
    stratum_levels;
    learners = DEFAULT_SL_LEARNERS,
    rng = StableRNG(1),
    estimator::Symbol = :tmle,
)
    estimator in (:tmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "estimator must be :tmle or :eif (aliases :aipw, :sdr); got :$estimator",
    ))
    n = nrow(df)
    T = length(outcomes)
    K = length(stratum_levels)
    P = K * T
    a = Float64.(df[!, treatment])
    g = Float64.(g_hat)
    w = Float64.(ipcw_w)
    H = @. a / g - (1 - a) / (1 - g)
    H1 = 1.0 ./ g
    H0 = -1.0 ./ (1.0 .- g)
    key_names = Tuple(strata_cols)
    keys = [NamedTuple{key_names}(Tuple(df[i, c] for c in strata_cols)) for i in 1:n]
    masks = [keys .== Ref(level) for level in stratum_levels]
    raw_tau = zeros(n, P)
    raw_mu1 = zeros(n, P)
    raw_mu0 = zeros(n, P)
    fold_scores = zeros(length(fold_sets), P)
    schema = fit_covariate_schema(df, adj_covars)

    for (t, outcome) in enumerate(outcomes)
        y = Float64.(df[!, outcome])
        for (v, test_idx) in enumerate(fold_sets)
            train_idx = setdiff(1:n, test_idx)
            train = df[train_idx, :]
            test = df[test_idx, :]
            sl = _fit_sl_outcome(
                train, adj_covars, y[train_idx]; treatment = treatment,
                learners = learners, rng = rng, schema = schema,
            )
            Qobs_all = _predict_sl(sl, test, adj_covars;
                treatment = treatment, treatment_values = a[test_idx])
            Q1_all = _predict_sl(sl, test, adj_covars;
                treatment = treatment, treatment_values = ones(length(test_idx)))
            Q0_all = _predict_sl(sl, test, adj_covars;
                treatment = treatment, treatment_values = zeros(length(test_idx)))
            for (s, mask) in enumerate(masks)
                local_idx = findall(mask[test_idx])
                isempty(local_idx) && throw(ArgumentError(
                    "stratum $(stratum_levels[s]) is absent from test fold $v",
                ))
                idx = test_idx[local_idx]
                Qobs = Qobs_all[local_idx]
                Q1 = Q1_all[local_idx]
                Q0 = Q0_all[local_idx]
                Hs = H[idx]
                if estimator === :tmle
                    wi = w[idx]
                    den = sum(wi .* abs2.(Hs))
                    den > 1e-12 || throw(ArgumentError(
                        "stratum $(stratum_levels[s]) fold $v has a degenerate TMLE clever covariate",
                    ))
                    epsilon = sum(wi .* Hs .* (y[idx] .- Qobs)) / den
                    Qobs = Qobs .+ epsilon .* Hs
                    Q1 = Q1 .+ epsilon .* H1[idx]
                    Q0 = Q0 .+ epsilon .* H0[idx]
                end
                resid = y[idx] .- Qobs
                p = (s - 1) * T + t
                raw_tau[idx, p] = Hs .* resid .+ Q1 .- Q0
                raw_mu1[idx, p] = (a[idx] .* H1[idx]) .* resid .+ Q1
                raw_mu0[idx, p] = ((1 .- a[idx]) .* (-H0[idx])) .* resid .+ Q0
                fold_scores[v, p] = sum(w[idx] .* Hs .* resid)
            end
        end
    end

    estimates = zeros(P)
    mu1 = zeros(P)
    mu0 = zeros(P)
    ic = zeros(n, P)
    ic_mu1 = zeros(n, P)
    ic_mu0 = zeros(n, P)
    for s in 1:K, t in 1:T
        p = (s - 1) * T + t
        ws = w .* masks[s]
        sws = sum(ws)
        sws > 0 || throw(ArgumentError("stratum $(stratum_levels[s]) has zero total analysis weight"))
        estimates[p] = dot(ws, raw_tau[:, p]) / sws
        mu1[p] = dot(ws, raw_mu1[:, p]) / sws
        mu0[p] = dot(ws, raw_mu0[:, p]) / sws
        normaliser = sws / n
        ic[:, p] = (ws ./ normaliser) .* (raw_tau[:, p] .- estimates[p])
        ic_mu1[:, p] = (ws ./ normaliser) .* (raw_mu1[:, p] .- mu1[p])
        ic_mu0[:, p] = (ws ./ normaliser) .* (raw_mu0[:, p] .- mu0[p])
    end
    return (
        estimates = estimates, mu1 = mu1, mu0 = mu0,
        ic = ic, ic_mu1 = ic_mu1, ic_mu0 = ic_mu0,
        uncentered_tau = raw_tau,
        uncentered_mu1 = raw_mu1,
        uncentered_mu0 = raw_mu0,
        targeting_scores = vec(sum(fold_scores; dims = 1)),
        targeting_scores_by_fold = fold_scores,
    )
end

"""
    run_repeated_outcome_msm(df, treatment, outcomes; baseline, strata, propensity,
                             folds, learners, rng, handle_missing, estimator, cluster)
        -> NamedTuple

Estimate ``\\tau(t)`` for binary point treatment and several outcomes with a
**joint** influence-function covariance.

Returns `(estimates, se, covariance, ic, outcomes, n, positivity, missingness, …)`.
Use [`msm_contrast`](@ref) for linear forms ``c^\\top\\tau`` (e.g. ``\\tau(t_3)-\\tau(t_2)``).

Missingness uses a **shared** sample across times (complete-profile ``R``).
`estimator=:tmle` (default) applies a per-fold clever-covariate fluctuation;
`:eif` is the untargeted AIPW one-step.

`strata` may be one `Symbol` or a vector of symbols. Nuisance models are still
fit on the full eligible sample, with these columns added to `baseline`, while
targeting and Hajek standardisation are conditional on each joint stratum.
`propensity` may be a known scalar, column symbol, or input-row-aligned vector;
known values bypass treatment fitting but retain positivity diagnostics.

`cluster` may be a column `Symbol` in `df` or a length-`nrow(df)` id vector.
When set, ``\\widehat{\\Sigma}`` is cluster-robust (sampling hierarchy). This does
**not** change the estimand to BLUP / partial pooling.
"""
function run_repeated_outcome_msm(
    df::DataFrame,
    treatment::Symbol,
    outcomes::AbstractVector{Symbol};
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    learners = DEFAULT_SL_LEARNERS,
    learners_trt = (:logistic, :mean),
    rng::AbstractRNG = StableRNG(1),
    handle_missing::Symbol = :drop,
    estimator::Symbol = :tmle,
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
    strata = nothing,
    propensity = nothing,
)
    validate_contrast_learners(learners; context = "run_repeated_outcome_msm")
    outs = collect(Symbol, outcomes)
    isempty(outs) && throw(ArgumentError("outcomes must be nonempty"))
    allunique(outs) || throw(ArgumentError("outcome symbols must be unique"))

    strata_cols = _normalise_msm_strata(df, strata)
    baseline_full = unique(vcat(baseline, strata_cols))
    propensity_required = propensity isa Symbol ? [propensity] : Symbol[]
    needs_row_id = propensity isa AbstractVector || cluster isa AbstractVector
    row_id = :__causal_targeted_input_row__
    while hasproperty(df, row_id)
        row_id = Symbol(row_id, "_")
    end
    df_work = needs_row_id ? copy(df) : df
    needs_row_id && (df_work[!, row_id] = collect(1:nrow(df)))

    miss = _handle_missing_repeated_outcomes(
        df_work, treatment, outs, baseline_full, handle_missing; rng = rng,
        required_complete = propensity_required,
    )
    df_clean, ipcw_w, extra_cols = miss
    adj_covars = unique(vcat(baseline_full, extra_cols))

    a = Float64.(df_clean[!, treatment])
    all(x -> x == 0.0 || x == 1.0, a) || throw(ArgumentError(
        "run_repeated_outcome_msm requires a binary 0/1 treatment column",
    ))

    n = nrow(df_clean)
    n >= folds || throw(ArgumentError(
        "need at least $folds complete-profile rows after handle_missing=:$handle_missing; got $n",
    ))
    input_rows = needs_row_id ? Int.(df_clean[!, row_id]) : nothing
    cluster_ids = _resolve_cluster_ids(
        df_clean, cluster, n; input_n = nrow(df), input_rows = input_rows,
    )
    stratum_levels = _ordered_msm_strata(df_clean, strata_cols)
    key_names = Tuple(strata_cols)
    row_keys = isempty(strata_cols) ? NamedTuple[] : [
        NamedTuple{key_names}(Tuple(df_clean[i, c] for c in strata_cols)) for i in 1:n
    ]
    fold_sets = isempty(strata_cols) ? crossfit_indices(n, folds, rng) :
        _stratified_msm_folds(row_keys, a, folds, rng)
    g_hat, propensity_meta = _resolve_msm_propensity(
        df, df_clean, row_id, propensity, treatment, adj_covars, fold_sets;
        learners_trt = learners_trt, rng = rng,
    )

    if isempty(strata_cols)
        psi, estimates, psi_μ1, μ1, psi_μ0, μ0 = _aipw_influence_matrix(
            df_clean, treatment, outs, adj_covars, fold_sets, g_hat, ipcw_w;
            learners = learners, rng = rng, estimator = estimator,
        )
        Σ, se, ic = _msm_covariance(psi, estimates, ipcw_w; cluster = cluster_ids)
        Σ_μ1, _, ic_μ1 = _msm_covariance(psi_μ1, μ1, ipcw_w; cluster = cluster_ids)
        Σ_μ0, _, ic_μ0 = _msm_covariance(psi_μ0, μ0, ipcw_w; cluster = cluster_ids)
        parameter_index = [(
            position = t, stratum = nothing, strata_columns = Symbol[],
            outcome = outs[t], time_position = t, estimand = :tau,
        ) for t in eachindex(outs)]
        targeting_scores = nothing
        targeting_scores_by_fold = nothing
        uncentered_tau = nothing
        uncentered_mu1 = nothing
        uncentered_mu0 = nothing
    else
        strat = _stratified_aipw_influence_matrix(
            df_clean, treatment, outs, adj_covars, fold_sets, g_hat, ipcw_w,
            strata_cols, stratum_levels;
            learners = learners, rng = rng, estimator = estimator,
        )
        estimates, μ1, μ0 = strat.estimates, strat.mu1, strat.mu0
        ic, ic_μ1, ic_μ0 = strat.ic, strat.ic_mu1, strat.ic_mu0
        Σ, se = _msm_covariance_from_ic(ic; cluster = cluster_ids)
        Σ_μ1, _ = _msm_covariance_from_ic(ic_μ1; cluster = cluster_ids)
        Σ_μ0, _ = _msm_covariance_from_ic(ic_μ0; cluster = cluster_ids)
        parameter_index = NamedTuple[]
        for (s, level) in enumerate(stratum_levels), t in eachindex(outs)
            push!(parameter_index, (
                position = (s - 1) * length(outs) + t,
                stratum = level,
                strata_columns = copy(strata_cols),
                outcome = outs[t],
                time_position = t,
                estimand = :tau,
            ))
        end
        targeting_scores = strat.targeting_scores
        targeting_scores_by_fold = strat.targeting_scores_by_fold
        uncentered_tau = strat.uncentered_tau
        uncentered_mu1 = strat.uncentered_mu1
        uncentered_mu0 = strat.uncentered_mu0
    end

    min_g = minimum(g_hat)
    max_g = maximum(g_hat)
    positivity = (
        ok = min_g > 1e-3 && max_g < 1 - 1e-3,
        min_propensity = min_g,
        max_propensity = max_g,
        mean_propensity = mean(g_hat),
    )
    mu1_parameter_index = [merge(x, (estimand = :mu1,)) for x in parameter_index]
    mu0_parameter_index = [merge(x, (estimand = :mu0,)) for x in parameter_index]

    return with_missingness((
        estimates = estimates,
        se = se,
        covariance = Σ,
        ic = ic,
        mu1 = μ1,
        mu0 = μ0,
        covariance_mu1 = Σ_μ1,
        covariance_mu0 = Σ_μ0,
        ic_mu1 = ic_μ1,
        ic_mu0 = ic_μ0,
        outcomes = outs,
        n = n,
        positivity = positivity,
        propensity = g_hat,
        propensity_metadata = propensity_meta,
        parameter_index = parameter_index,
        mu1_parameter_index = mu1_parameter_index,
        mu0_parameter_index = mu0_parameter_index,
        strata = isempty(strata_cols) ? nothing : (
            columns = strata_cols,
            levels = stratum_levels,
            ordering = :categorical_levels_then_first_occurrence,
            parameter_order = :stratum_major_time_minor,
            nuisance_adjustment = adj_covars,
        ),
        targeting_scores = targeting_scores,
        targeting_scores_by_fold = targeting_scores_by_fold,
        uncentered_tau = uncentered_tau,
        uncentered_mu1 = uncentered_mu1,
        uncentered_mu0 = uncentered_mu0,
        cluster = cluster_ids,
        covariance_kind = cluster_ids === nothing ? :unit : :cluster,
    ), miss.meta)
end

"""
    _resolve_cluster_ids(df, cluster, n) -> Union{Nothing, Vector}

Normalise `cluster` kwarg to a length-`n` vector of ids (or `nothing`).
"""
function _resolve_cluster_ids(
    df::DataFrame,
    cluster,
    n::Int;
    input_n::Int = n,
    input_rows = nothing,
)
    return resolve_cluster_ids(df, cluster, n; input_n = input_n, input_rows = input_rows)
end

"""
    msm_contrast(result, i, j) -> NamedTuple

Wald inference for ``\\tau_i - \\tau_j`` using the joint covariance in `result`.
"""
function msm_contrast(result::NamedTuple, i::Integer, j::Integer)
    T = length(result.estimates)
    (1 <= i <= T && 1 <= j <= T) || throw(BoundsError(result.estimates, (i, j)))
    c = zeros(T)
    c[i] = 1.0
    c[j] = -1.0
    return msm_contrast(result, c)
end

"""
    msm_contrast(result, c) -> NamedTuple

Wald inference for the linear form ``c^\\top\\tau`` with
``\\mathrm{Var}=c^\\top\\Sigma c``.
"""
function msm_contrast(result::NamedTuple, c::AbstractVector{<:Real})
    θ = haskey(result, :coefficients) ? result.coefficients : result.estimates
    Σ = result.covariance
    length(c) == length(θ) || throw(ArgumentError(
        "contrast length $(length(c)) must match parameter length $(length(θ))",
    ))
    c64 = Float64.(c)
    est = dot(c64, θ)
    var = dot(c64, Σ * c64)
    var < 0 && var > -1e-12 && (var = 0.0)
    var >= 0 || throw(ArgumentError("contrast variance is negative ($(var)); check Σ"))
    se = sqrt(var)
    z = 1.96
    return (
        estimate = est,
        se = se,
        ci_lower = est - z * se,
        ci_upper = est + z * se,
        contrast = c64,
    )
end

function _match_msm_stratum(result::NamedTuple, requested)
    result.strata === nothing && throw(ArgumentError(
        "msm_stratum_contrast requires a result fitted with strata=",
    ))
    cols = result.strata.columns
    key = if requested isa NamedTuple
        requested
    elseif length(cols) == 1
        NamedTuple{Tuple(cols)}((requested,))
    elseif requested isa Tuple && length(requested) == length(cols)
        NamedTuple{Tuple(cols)}(requested)
    else
        throw(ArgumentError(
            "joint stratum labels must be a NamedTuple or a $(length(cols))-tuple in $(cols) order",
        ))
    end
    idx = findfirst(==(key), result.strata.levels)
    idx === nothing && throw(ArgumentError(
        "stratum $key not found; available strata are $(result.strata.levels)",
    ))
    return idx, key
end

"""
    msm_stratum_contrast(result, high, low) -> NamedTuple

Return the complete effect-modification profile
``Gamma(t)=tau_high(t)-tau_low(t)`` with joint Wald inference. For one stratum
column, `high` and `low` may be bare values; joint strata accept named tuples or
tuples in `result.strata.columns` order.
"""
function msm_stratum_contrast(result::NamedTuple, high, low)
    high_idx, high_key = _match_msm_stratum(result, high)
    low_idx, low_key = _match_msm_stratum(result, low)
    high_idx == low_idx && throw(ArgumentError("high and low must identify different strata"))
    T = length(result.outcomes)
    P = length(result.estimates)
    C = zeros(T, P)
    for t in 1:T
        C[t, (high_idx - 1) * T + t] = 1.0
        C[t, (low_idx - 1) * T + t] = -1.0
    end
    estimates = C * result.estimates
    covariance = C * result.covariance * C'
    covariance = Matrix{Float64}(Symmetric(0.5 .* (covariance .+ covariance')))
    se = sqrt.(max.(diag(covariance), 0.0))
    z = 1.96
    return (
        estimates = estimates,
        se = se,
        covariance = covariance,
        ci_lower = estimates .- z .* se,
        ci_upper = estimates .+ z .* se,
        contrast_matrix = C,
        outcomes = result.outcomes,
        high = high_key,
        low = low_key,
    )
end

export RepeatedOutcomeMSM, run_repeated_outcome_msm, msm_contrast, msm_stratum_contrast
export identify_repeated_outcomes, unstack_repeated_outcomes