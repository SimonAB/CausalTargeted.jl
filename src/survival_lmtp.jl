"""Thin discrete-time event-time / survival LMTP.

Estimates the probability of remaining event-free through a horizon under a
common additive MTP shift on time-ordered treatments (Díaz–Hoffman–Hejazi
survival LMTP spirit). Competing risks are deferred.

Wide data layout (one row per unit):

- treatments `A1,…,AT`
- event-free indicators `S1,…,ST` (`St = 1` if still event-free at occasion `t`;
  once 0, stays 0)
- optional time-varying covariates before each `A_t`
- optional censoring indicators `C1,…,CT` (`Ct = 1` if censored at `t`; thin IPCW)

# References

- Díaz, Hoffman & Hejazi (2024), *Lifetime Data Analysis* — survival / competing-risks LMTP
- Díaz, Williams, Hoffman & Schenck (2023), *JASA* — sequential LMTP regression
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    SurvivalPolicy(treatments, surv, baseline; time_vary, censor, shift, horizon)

Discrete-time survival estimand: ``E[S_{\\mathrm{horizon}}^{d}]`` under an MTP
`d` applied to every treatment time. `surv[t]` is the event-free indicator at
occasion `t`.
"""
struct SurvivalPolicy <: Estimand
    treatments::Vector{Symbol}
    surv::Vector{Symbol}
    baseline::Vector{Symbol}
    time_vary::Vector{Vector{Symbol}}
    censor::Vector{Symbol}
    shift::ShiftPolicy
    horizon::Int
end

function SurvivalPolicy(
    treatments::Vector{Symbol},
    surv::Vector{Symbol},
    baseline::Vector{Symbol};
    time_vary::Vector{Vector{Symbol}} = [Symbol[] for _ in treatments],
    censor::Vector{Symbol} = Symbol[],
    shift::ShiftPolicy = shift_policy_from_settings(),
    horizon::Union{Nothing, Int} = nothing,
)
    T = length(treatments)
    T >= 1 || throw(ArgumentError("need at least one treatment time"))
    length(surv) == T || throw(ArgumentError(
        "surv length ($(length(surv))) must match treatments ($T)",
    ))
    length(time_vary) == T || throw(ArgumentError(
        "time_vary length must equal treatments",
    ))
    isempty(censor) || length(censor) == T || throw(ArgumentError(
        "censor length must be 0 or equal treatments ($T)",
    ))
    h = horizon === nothing ? T : Int(horizon)
    (1 <= h <= T) || throw(ArgumentError("horizon must be in 1:$T, got $h"))
    return SurvivalPolicy(treatments, surv, baseline, time_vary, censor, shift, h)
end

estimand_engine(::SurvivalPolicy) = :survival_lmtp

"""
    _at_risk_mask(data, surv, t) -> BitVector

Units still event-free entering occasion `t` (`t == 1` ⇒ all units).
"""
function _at_risk_mask(data::DataFrame, surv::Vector{Symbol}, t::Int)
    n = nrow(data)
    t == 1 && return trues(n)
    prev = Float64.(data[!, surv[t - 1]])
    return prev .> 0.5
end

"""
    _censor_ipcw(data, censor, baseline, horizon; learners, rng) -> Vector{Float64}

Thin inverse-probability-of-remaining-uncensored weights through `horizon`.
Units censored by the horizon get weight 0; others get ``1 / \\hat P(\\mathrm{uncensored} \\mid W)``.
"""
function _censor_ipcw(
    data::DataFrame,
    censor::Vector{Symbol},
    baseline::Vector{Symbol},
    horizon::Int;
    learners = DEFAULT_SL_LEARNERS,
    rng::AbstractRNG = StableRNG(0),
)
    n = nrow(data)
    isempty(censor) && return ones(n)
    cens = falses(n)
    for t in 1:horizon
        cens .|= (Float64.(data[!, censor[t]]) .> 0.5)
    end
    y_unc = Float64.(.!cens)
    X = design_matrix(data, baseline)
    sl = fit_super_learner(X, y_unc; learners = learners, rng = rng)
    p = clamp.(predict_super_learner(sl, X), 0.05, 0.95)
    w = ifelse.(cens, 0.0, 1.0 ./ p)
    pos = w .> 0
    if any(pos)
        w[pos] ./= mean(w[pos])
    end
    return w
end

"""
    run_survival_lmtp(data, treatments, surv; baseline, time_vary, censor, delta, horizon, ...) -> NamedTuple

Sequential regression estimator of event-free probability at `horizon` under a
common MTP shift `delta` on every treatment time through the horizon.

Returns `(estimate, se, n, times, horizon, delta, estimand=:survival)`.
"""
function run_survival_lmtp(
    data::DataFrame,
    treatments::Vector{Symbol},
    surv::Vector{Symbol};
    baseline::Vector{Symbol},
    time_vary::Vector{Vector{Symbol}} = [Symbol[] for _ in treatments],
    censor::Vector{Symbol} = Symbol[],
    delta::Float64 = 1.0,
    horizon::Union{Nothing, Int} = nothing,
    folds::Int = recommend_folds(nrow(data)),
    learners = recommend_learners(nrow(data)),
    lower_q = mtp_settings().lower_q,
    upper_q = mtp_settings().upper_q,
    shift::ShiftPolicy = additive_shift_policy(; lower_q = lower_q, upper_q = upper_q),
    rng::AbstractRNG = StableRNG(42),
)
    T = length(treatments)
    T >= 1 || throw(ArgumentError("need at least one treatment time"))
    length(surv) == T || throw(ArgumentError("surv length must equal treatments"))
    length(time_vary) == T || throw(ArgumentError("time_vary length must equal treatments"))
    h = horizon === nothing ? T : Int(horizon)
    (1 <= h <= T) || throw(ArgumentError("horizon must be in 1:$T"))

    n = nrow(data)
    for t in 1:h
        hasproperty(data, treatments[t]) || throw(ArgumentError("missing $(treatments[t])"))
        hasproperty(data, surv[t]) || throw(ArgumentError("missing $(surv[t])"))
    end

    a_nat = Vector{Vector{Float64}}(undef, h)
    a_pol = Vector{Vector{Float64}}(undef, h)
    for t in 1:h
        a = Float64.(data[!, treatments[t]])
        a_nat[t] = apply_policy_values(a, 0.0, shift)
        a_pol[t] = apply_policy_values(a, delta, shift)
    end

    ipcw = _censor_ipcw(data, censor, baseline, h; learners = learners, rng = rng)

    # Terminal event-free indicator at horizon (IPCW-weighted when censoring is present)
    Q = Float64.(data[!, surv[h]]) .* ipcw
    fold_sets = crossfit_indices(n, folds, rng)
    ic = zeros(n)

    for t in h:-1:1
        at_risk = _at_risk_mask(data, surv, t)
        hist = unique(vcat(
            baseline,
            reduce(vcat, time_vary[1:t]; init = Symbol[]),
            treatments[1:t],
        ))
        hist = [c for c in hist if hasproperty(data, c)]
        Q_next = copy(Q)
        Q = zeros(n)
        for test_idx in fold_sets
            train_idx = setdiff(1:n, test_idx)
            train_risk = train_idx[at_risk[train_idx]]
            test_risk = test_idx[at_risk[test_idx]]
            Q[test_idx[.!at_risk[test_idx]]] .= 0.0
            isempty(train_risk) && continue

            train = data[train_risk, :]
            sl = _fit_sl_outcome(
                train, hist, Q_next[train_risk];
                treatment = treatments[t], learners = learners, rng = rng,
            )
            if !isempty(test_risk)
                block = data[test_risk, :]
                Q[test_risk] .= _predict_sl(
                    sl, block, hist;
                    treatment = treatments[t],
                    treatment_values = a_pol[t][test_risk],
                )
            end

            if t == 1 && !isempty(test_risk)
                a = Float64.(data[!, treatments[1]])
                L, U = exposure_bounds(a, shift.lower_q, shift.upper_q)
                sl_a = fit_super_learner(
                    design_matrix(data[train_risk, :], baseline), a[train_risk];
                    learners = learners, rng = rng,
                )
                μ_te = predict_super_learner(sl_a, design_matrix(data[test_risk, :], baseline))
                μ_tr = predict_super_learner(sl_a, design_matrix(data[train_risk, :], baseline))
                σ_a = robust_residual_sd(a[train_risk] .- μ_tr)
                req = mean(a_pol[1][test_risk] .- a_nat[1][test_risk])
                H = _mtp_clever_covariate_clamp_aware(a[test_risk], μ_te, σ_a, req, L, U)
                H = truncate_weights(H; trunc = 10.0)
                Q_obs = _predict_sl(
                    sl, data[test_risk, :], hist; treatment = treatments[1],
                )
                ic[test_risk] .= Q[test_risk] .+ H .* (Q_next[test_risk] .- Q_obs)
            end
        end
    end

    if any(!=(0.0), ic)
        est = mean(ic)
        se = std(ic) / sqrt(n)
    else
        est = mean(Q)
        se = std(Q) / sqrt(n)
    end
    return (
        estimate = est,
        se = se,
        n = n,
        times = h,
        horizon = h,
        delta = delta,
        estimand = :survival,
    )
end

function run_survival_lmtp(data::DataFrame, estimand::SurvivalPolicy; kwargs...)
    return run_survival_lmtp(
        data, estimand.treatments, estimand.surv;
        baseline = estimand.baseline,
        time_vary = estimand.time_vary,
        censor = estimand.censor,
        shift = estimand.shift,
        horizon = estimand.horizon,
        kwargs...,
    )
end

"""
    survival_identification_certificate(unrolling, query; nuisance_source) -> IdentificationCertificate

Wrap temporal backdoor ID for a survival / event-time horizon query
(`TemporalEffectQuery` with `t_outcome = horizon`). Application code supplies
column names via [`SurvivalPolicy`](@ref).
"""
function survival_identification_certificate(
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

export SurvivalPolicy, run_survival_lmtp, survival_identification_certificate
