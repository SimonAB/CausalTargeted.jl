"""Sequential (multi-time) LMTP via recursive outcome regression.

Implements a practical two-or-more time-point sequential regression estimator
for modified treatment policies (Díaz–Williams LMTP spirit) without requiring
application-specific column names.

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
    SequentialPolicy(treatments, outcome, baseline, time_vary, shift)

Time-ordered treatments `A_1,…,A_T`, baseline covariates `L_0`, and optional
time-varying covariates `time_vary[t]` observed before `A_t`.
"""
struct SequentialPolicy <: Estimand
    treatments::Vector{Symbol}
    outcome::Symbol
    baseline::Vector{Symbol}
    time_vary::Vector{Vector{Symbol}}
    shift::ShiftPolicy
end

function SequentialPolicy(
    treatments::Vector{Symbol},
    outcome::Symbol,
    baseline::Vector{Symbol};
    time_vary::Vector{Vector{Symbol}} = [Symbol[] for _ in treatments],
    shift::ShiftPolicy = shift_policy_from_settings(),
)
    length(time_vary) == length(treatments) || throw(ArgumentError(
        "time_vary length ($(length(time_vary))) must match treatments ($(length(treatments)))",
    ))
    return SequentialPolicy(treatments, outcome, baseline, time_vary, shift)
end

estimand_engine(::SequentialPolicy) = :sequential_lmtp

"""
    run_sequential_lmtp(data, treatments, outcome; baseline, time_vary, delta, folds, learners, rng) -> NamedTuple

Recursive g-computation + TMLE-style one-step correction at the last time point
for a common additive MTP shift `delta` applied to every `A_t`.

Returns `(estimate, se, n, times)`.
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
    lower_q = mtp_settings().lower_q,
    upper_q = mtp_settings().upper_q,
    shift::ShiftPolicy = additive_shift_policy(; lower_q = lower_q, upper_q = upper_q),
    rng::AbstractRNG = StableRNG(42),
)
    T = length(treatments)
    T >= 1 || throw(ArgumentError("need at least one treatment time"))
    length(time_vary) == T || throw(ArgumentError("time_vary length must equal treatments"))
    n = nrow(data)
    y = Float64.(data[!, outcome])

    # Build intervened treatments under the same policy at each t
    a_nat = Vector{Vector{Float64}}(undef, T)
    a_pol = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        a = Float64.(data[!, treatments[t]])
        a_nat[t] = apply_policy_values(a, 0.0, shift)
        a_pol[t] = apply_policy_values(a, delta, shift)
    end

    # Sequential regression: Q_T = Y; for t=T…1 regress Q_t on history, predict under a_pol[t]
    Q = copy(y)
    fold_sets = crossfit_indices(n, folds, rng)
    ic = zeros(n)

    for t in T:-1:1
        hist = unique(vcat(baseline, reduce(vcat, time_vary[1:t]; init = Symbol[]), treatments[1:t]))
        hist = [c for c in hist if hasproperty(data, c)]
        Q_next = copy(Q)
        Q = zeros(n)
        for (fi, test_idx) in enumerate(fold_sets)
            train_idx = setdiff(1:n, test_idx)
            train = data[train_idx, :]
            # Fit Q on natural history at time t
            sl = _fit_sl_outcome(
                train, hist, Q_next[train_idx];
                treatment = treatments[t], learners = learners, rng = rng,
            )
            block = data[test_idx, :]
            # Predict under policy at t (other times remain observed in hist design)
            Q[test_idx] .= _predict_sl(
                sl, block, hist;
                treatment = treatments[t], treatment_values = a_pol[t][test_idx],
            )
            if t == 1
                a = Float64.(data[!, treatments[1]])
                L, U = exposure_bounds(a, shift.lower_q, shift.upper_q)
                sl_a = fit_super_learner(
                    design_matrix(train, baseline), a[train_idx];
                    learners = learners, rng = rng,
                )
                μ_tr = predict_super_learner(sl_a, design_matrix(train, baseline))
                μ_te = predict_super_learner(sl_a, design_matrix(block, baseline))
                σ_a = robust_residual_sd(a[train_idx] .- μ_tr)
                req = mean(a_pol[1][test_idx] .- a_nat[1][test_idx])
                H = _mtp_clever_covariate_clamp_aware(a[test_idx], μ_te, σ_a, req, L, U)
                H = truncate_weights(H; trunc = 10.0)
                Q_obs = _predict_sl(sl, block, hist; treatment = treatments[1])
                ic[test_idx] .= Q[test_idx] .+ H .* (Q_next[test_idx] .- Q_obs)
            end
        end
    end

    # Prefer IC mean when available
    if any(!=(0.0), ic)
        est = mean(ic)
        se = std(ic) / sqrt(n)
    else
        est = mean(Q)
        se = std(Q) / sqrt(n)
    end
    return (estimate = est, se = se, n = n, times = T, delta = delta)
end

function run_sequential_lmtp(data::DataFrame, estimand::SequentialPolicy; kwargs...)
    return run_sequential_lmtp(
        data, estimand.treatments, estimand.outcome;
        baseline = estimand.baseline,
        time_vary = estimand.time_vary,
        shift = estimand.shift,
        kwargs...,
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

export SequentialPolicy, run_sequential_lmtp, sequential_identification_certificate
