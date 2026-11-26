"""Cross-fitted LMTP TMLE / EIF / SDR for modified treatment policies.

Uses a **single shared fold partition** for outcome and density-ratio fits
via [`build_lmtp_fold_cache`](@ref) and [`lmtp_components_from_cache`](@ref).
Density-ratio kernels live in `lmtp_density_ratios.jl`; canonical-link
targeting lives in `lmtp_targeting.jl`.

Estimators (`estimator`):
- `:tmle` — score-solving separate-policy TMLE (default)
- `:eif` / `:aipw` / `:sdr` — EIF one-step (sequential DR at T = 1)
- `:itmle` — compatibility label for the same canonical weighted-intercept targeting step

# References

- Díaz et al. (2023), *JASA* — LMTP EIF, TMLE, and sequential double robustness
- van der Laan & Rubin (2006); van der Laan & Rose (2011) — TMLE
- Zheng & van der Laan (2011); Chernozhukov et al. (2018) — cross-fitting
"""

"""
    _shared_fold_lmtp_components(...) -> NamedTuple

Build cross-fitted LMTP nuisance components using a single shared fold
partition by delegating to [`build_lmtp_fold_cache`](@ref) and
[`lmtp_components_from_cache`](@ref).
"""
function _shared_fold_lmtp_components(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector{<:Real},
    a_reference::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    density_ratio::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_amount::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
    policy::Union{Nothing, AbstractContinuousPolicy} = nothing,
    policy_delta::Union{Nothing, Real} = nothing,
    smooth_policy::Union{Nothing, SmoothBoundedLogitShiftPolicy} = nothing,
    smooth_shift::Union{Nothing, Real} = nothing,
    feasible_policy::Union{Nothing, FeasibleAdditiveShiftPolicy} = nothing,
    feasible_delta::Union{Nothing, Real} = nothing,
    headroom_policy::Union{Nothing, ProportionalHeadroomPolicy} = nothing,
    headroom_delta::Union{Nothing, Real} = nothing,
    policy_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    weight_truncation::Bool = true,
)
    n = nrow(df)
    a = collect(Float64, df[!, trt])::Vector{Float64}
    a1 = collect(Float64, a_policy)::Vector{Float64}
    a0 = collect(Float64, a_reference)::Vector{Float64}
    length(a1) == n == length(a0) || throw(ArgumentError("policy vectors must match the data length"))

    resolved_policy, resolved_delta = _resolve_policy_kwargs(;
        policy, policy_delta,
        smooth_policy, smooth_shift,
        feasible_policy, feasible_delta,
        headroom_policy, headroom_delta,
    )
    clamp_config = L !== nothing || U !== nothing || shift_amount !== nothing
    _policy_validate_components(
        resolved_policy, resolved_delta, a, a1, a0;
        density_ratio, policy_mask, weight_truncation, clamp_config,
    )
    _validate_uncapped_options(
        resolved_policy, a, a1, a0;
        density_ratio, cv_trunc, clamp_config, weight_truncation,
    )
    exposure_values = _policy_density_values(resolved_policy, a)
    cache = build_lmtp_fold_cache(
        df, trt, outcome, covariates, folds, rng;
        learners_outcome, learners_trt, family_outcome, weight_truncation,
        exposure_values,
        fit_exposure = density_ratio !== :classification,
    )
    return lmtp_components_from_cache(
        cache, a1, a0;
        density_ratio, trunc, cv_trunc, trunc_candidates,
        L, U, shift_amount, shift_reference,
        policy = resolved_policy, policy_delta = resolved_delta,
        policy_mask, weight_truncation,
    )
end

"""
    lmtp_tmle_contrast(...) -> NamedTuple

`estimator`:
- `:tmle` (default) — score-solving separate-policy TMLE
- `:eif` / `:aipw` / `:sdr` — full EIF one-step `P_n[Q(d)+H(Y−Q)]` (SDR at T = 1)
- `:itmle` — compatibility label for the same canonical weighted-intercept targeting step

`density_ratio`: `:gaussian` (default), `:classification`, or `:hybrid`.

`cv_trunc=true` selects the hard truncation among `trunc_candidates` by
clever-covariate variance. `targeting_weight` scales fluctuations toward
g-computation. `epochs` remains available for compatibility; each arm's
canonical score is solved in one step.
"""
function lmtp_tmle_contrast(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector{<:Real},
    a_reference::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
    weight_truncation::Bool = true,
    targeting_weight::Real = 1.0,
    epochs::Int = 3,
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_amount::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
    policy::Union{Nothing, AbstractContinuousPolicy} = nothing,
    policy_delta::Union{Nothing, Real} = nothing,
    smooth_policy::Union{Nothing, SmoothBoundedLogitShiftPolicy} = nothing,
    smooth_shift::Union{Nothing, Real} = nothing,
    feasible_policy::Union{Nothing, FeasibleAdditiveShiftPolicy} = nothing,
    feasible_delta::Union{Nothing, Real} = nothing,
    headroom_policy::Union{Nothing, ProportionalHeadroomPolicy} = nothing,
    headroom_delta::Union{Nothing, Real} = nothing,
    policy_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
)
    c = _shared_fold_lmtp_components(
        df, trt, outcome, covariates, a_policy, a_reference, folds, rng;
        learners_outcome, family_outcome, density_ratio, learners_trt,
        trunc, cv_trunc, trunc_candidates,
        L, U, shift_amount, shift_reference,
        policy, policy_delta,
        smooth_policy, smooth_shift,
        feasible_policy, feasible_delta,
        headroom_policy, headroom_delta,
        policy_mask, weight_truncation,
    )
    return _lmtp_targeting_step(
        c; estimator = estimator, targeting_weight = targeting_weight,
        epochs = epochs, family_outcome = family_outcome,
    )
end

"""
    lmtp_tmle_policy_mean(...) -> (psi, ic)

Single-policy interventional mean under the supplied treatment policy. `:tmle`
and `:itmle` use the canonical weighted-intercept update; `:eif`, `:aipw`, and
`:sdr` use the initial one-step score. The centred influence curve is all `NaN`
when fixed weight truncation or another unsupported route gates inference. An
identity policy returns the natural sample mean and its centred outcome values.
"""
function lmtp_tmle_policy_mean(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    weight_truncation::Bool = true,
)
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "unsupported LMTP estimator=$estimator; expected :tmle, :itmle, :eif, :aipw, or :sdr",
    ))
    y = collect(Float64, df[!, outcome])::Vector{Float64}
    a = collect(Float64, df[!, trt])::Vector{Float64}
    policy = collect(Float64, a_policy)::Vector{Float64}
    length(policy) == length(a) || throw(DimensionMismatch(
        "policy length $(length(policy)) does not match sample size $(length(a))",
    ))
    if all(isequal.(policy, a))
        ψ = mean(y)
        return ψ, y .- ψ
    end
    c = _shared_fold_lmtp_components(
        df, trt, outcome, covariates, policy, a, folds, rng;
        learners_outcome, family_outcome, density_ratio, learners_trt,
        trunc, cv_trunc, weight_truncation,
    )
    arm = _single_arm_lmtp_target(
        c.y, c.Q_obs, c.Q1, c.H1, family_outcome, 1.0, estimator,
    )
    ψ = arm.psi
    all(isfinite, arm.point_score) && isfinite(ψ) || throw(DomainError(
        (arm.point_score, ψ), "single-policy LMTP mean produced non-finite point scores",
    ))
    ic = arm.point_score .- mean(arm.point_score)
    all(isfinite, ic) || throw(DomainError(
        ic, "single-policy LMTP mean EIF contains non-finite values",
    ))
    status = c.inference_status === :available && arm.boundary ? :boundary_outcome : c.inference_status
    return ψ, status === :available ? ic : fill(NaN, length(ic))
end
