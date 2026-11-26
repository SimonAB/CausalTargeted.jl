"""Continuous LMTP density-ratio kernels and polymorphic policy-component traits."""

"""
    apply_shift_policy(a, requested_shift, L, U; stratum_mask=nothing) -> Vector{Float64}

Match R `.make_shift_LU`: `clamp(a + δ, L, U)` (shift raw exposure, then clamp).
Natural policy (`δ ≈ 0`) is `clamp(a, L, U)`.
"""
function apply_shift_policy(
    a::AbstractVector{<:Real},
    requested_shift::Real,
    L::Real,
    U::Real;
    stratum_mask::Union{Nothing, BitVector} = nothing,
)
    a = Float64.(a)
    if !isfinite(requested_shift) || isapprox(requested_shift, 0; atol = 1e-12)
        return clamp.(a, L, U)
    end
    if stratum_mask === nothing
        return clamp.(a .+ requested_shift, L, U)
    end
    out = clamp.(a, L, U)
    out[stratum_mask] .= clamp.(a[stratum_mask] .+ requested_shift, L, U)
    return out
end

"""
    _gaussian_density(x, μ, σ) -> Float64

Univariate Gaussian density at `x` with mean `μ`; `σ` is lower-bounded for numerical stability in
clever-covariate ratios.
"""
function _gaussian_density(x::Real, μ::Real, σ::Real)
    σ = max(σ, 1e-6)
    return pdf(Normal(μ, σ), x)
end

@inline function _log_gaussian_ratio_checked(
    a::Float64,
    a_num::Float64,
    μ::Float64,
    σ::Float64,
    log_jac::Float64,
    lo_log::Float64,
    hi_log::Float64,
    domain_msg::String,
    range_msg::String,
)
    all(isfinite, (a, a_num, μ)) || throw(DomainError((a, a_num, μ), domain_msg))
    Δz = (a_num - a) / σ
    if Δz == 0.0 && log_jac == 0.0
        return 1.0
    end
    za = (a - μ) / σ
    znum = (a_num - μ) / σ
    log_ratio = log_jac - Δz * (za / 2 + znum / 2)
    (isfinite(log_ratio) && lo_log <= log_ratio <= hi_log) ||
        throw(DomainError(log_ratio, range_msg))
    return exp(log_ratio)
end

"""
    _mtp_clever_covariate_gaussian(a_obs, a_policy, mu_a, sigma) -> Vector

Interior MTP clever covariate using `a_policy` to recover the shift
(`a − δ ≈ 2a − a_policy`). Prefer `_mtp_clever_covariate_clamp_aware` when
bounds `(L,U)` and the requested shift are known.
"""
function _mtp_clever_covariate_gaussian(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma::Real,
)
    a_minus_δ = 2 .* Float64.(a_obs) .- Float64.(a_policy)
    num = _gaussian_density.(a_minus_δ, mu_a, sigma)
    den = max.(_gaussian_density.(a_obs, mu_a, sigma), 1e-12)
    return num ./ den
end

"""Uncapped Gaussian MTP ratio computed on the log scale."""
function _mtp_clever_covariate_gaussian_uncapped(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma::Real,
)
    length(a_obs) == length(a_policy) == length(mu_a) ||
        throw(ArgumentError("treatment and density prediction lengths must match"))
    σ = max(Float64(sigma), 1e-6)
    out = Vector{Float64}(undef, length(a_obs))
    lo, hi = log(nextfloat(0.0)), log(floatmax(Float64))
    @inbounds for i in eachindex(out)
        a = Float64(a_obs[i])
        ap = Float64(a_policy[i])
        μ = Float64(mu_a[i])
        all(isfinite, (a, ap, μ)) || throw(DomainError(
            (a, ap, μ), "uncapped Gaussian MTP ratios require finite treatments and means",
        ))
        δi = (a - ap) / σ
        if δi == 0.0
            out[i] = 1.0
            continue
        end
        za = (a - μ) / σ
        zap = (2a - ap - μ) / σ
        log_ratio = -δi * (za / 2 + zap / 2)
        (isfinite(log_ratio) && lo <= log_ratio <= hi) || throw(DomainError(
            log_ratio, "uncapped Gaussian MTP ratio is not representable in Float64",
        ))
        out[i] = exp(log_ratio)
    end
    return out
end

function _constant_additive_shift(a::AbstractVector{<:Real}, policy::AbstractVector{<:Real})
    length(a) == length(policy) || return false
    isempty(a) && return false
    shifts = Float64.(policy) .- Float64.(a)
    all(isfinite, shifts) || return false
    δ = first(shifts)
    return all(x -> isapprox(x, δ; atol = 1e-10, rtol = 1e-10), shifts)
end

function _feasible_gaussian_ratio(
    a,
    μ,
    σ,
    policy::FeasibleAdditiveShiftPolicy,
    step,
    selected;
    weight_truncation::Bool = false,
)
    n = length(a)
    length(μ) == n || throw(ArgumentError("treatment and density prediction lengths must match"))
    σ_val = max(Float64(σ), 1e-6)
    step_f = Float64(step)
    lo_log, hi_log = log(nextfloat(0.0)), log(floatmax(Float64))
    ratios = ones(Float64, n)
    @inbounds for i in 1:n
        selected !== nothing && !selected[i] && continue
        ai = Float64(a[i])
        _within_feasible_bounds(ai, policy) || continue
        # The identity branch has unit Jacobian and contributes exactly one.
        r = _within_feasible_bounds(ai + step_f, policy) ? 0.0 : 1.0
        if _within_feasible_bounds(ai - step_f, policy)
            # The shifted preimage is a-step. Evaluate g(a-step|W)/g(a|W).
            μi = Float64(μ[i])
            if !weight_truncation
                ap = ai + step_f
                all(isfinite, (ai, ap, μi)) || throw(DomainError(
                    (ai, ap, μi), "uncapped Gaussian MTP ratios require finite treatments and means",
                ))
                δi = (ai - ap) / σ_val
                if δi == 0.0
                    r += 1.0
                else
                    za = (ai - μi) / σ_val
                    zap = (2ai - ap - μi) / σ_val
                    log_ratio = -δi * (za / 2 + zap / 2)
                    (isfinite(log_ratio) && lo_log <= log_ratio <= hi_log) || throw(DomainError(
                        log_ratio, "uncapped Gaussian MTP ratio is not representable in Float64",
                    ))
                    r += exp(log_ratio)
                end
            else
                num = _gaussian_density(ai - step_f, μi, σ_val)
                den = max(_gaussian_density(ai, μi, σ_val), 1e-12)
                r += num / den
            end
        end
        ratios[i] = r
    end
    return ratios
end

function _headroom_gaussian_ratio(
    a,
    μ,
    σ,
    policy::ProportionalHeadroomPolicy,
    delta,
    selected;
    weight_truncation::Bool = false,
)
    d = _validate_headroom_delta(delta)
    length(a) == length(μ) || throw(ArgumentError(
        "treatment and density prediction lengths must match",
    ))
    selected === nothing || length(selected) == length(a) || throw(DimensionMismatch(
        "stratum mask must match the exposures",
    ))
    σ_raw = Float64(σ)
    isfinite(σ_raw) || throw(DomainError(
        σ_raw, "proportional headroom Gaussian ratios require a finite SD",
    ))
    σ_val = max(σ_raw, 1e-6)
    ratios = ones(Float64, length(a))
    d == 0.0 && return ratios

    lo_b, hi_b = policy.lower, policy.upper
    width = hi_b - lo_b
    p = abs(d)
    one_minus_p = 1.0 - p
    log_jac = -log1p(-p)
    lo_log, hi_log = log(nextfloat(0.0)), log(floatmax(Float64))
    threshold = d > 0.0 ? lo_b + p * width : hi_b - p * width

    @inbounds for i in eachindex(a)
        selected !== nothing && !selected[i] && continue
        ai = Float64(a[i])
        μi = Float64(μ[i])
        isfinite(ai) || throw(DomainError(
            ai, "proportional headroom Gaussian ratios require finite exposures",
        ))
        _within_headroom_bounds(ai, policy) || continue
        if (d > 0.0 && ai < threshold) || (d < 0.0 && ai > threshold)
            ratios[i] = 0.0
            continue
        end
        u = d > 0.0 ? (ai - p * hi_b) / one_minus_p : (ai - p * lo_b) / one_minus_p
        if !weight_truncation
            ratios[i] = _log_gaussian_ratio_checked(
                ai, u, μi, σ_val, log_jac, lo_log, hi_log,
                "uncapped proportional headroom Gaussian ratios require finite treatments and means",
                "uncapped proportional headroom Gaussian ratio is not representable in Float64",
            )
        else
            isfinite(μi) || throw(DomainError(
                μi, "proportional headroom Gaussian ratios require finite density means",
            ))
            num = _gaussian_density(u, μi, σ_val)
            den = max(_gaussian_density(ai, μi, σ_val), 1e-12)
            ratios[i] = (1.0 / one_minus_p) * (num / den)
        end
    end
    return ratios
end

function _lmtp_weight_inference_status(
    a::AbstractVector{<:Real},
    a1::AbstractVector{<:Real},
    a0::AbstractVector{<:Real};
    weight_truncation::Bool,
    clamp_config::Bool,
    smooth_policy::Bool = false,
)
    all(isequal.(a1, a0)) && return :trivial
    clamp_config && return :unsupported_clamped_policy
    smooth_policy && return weight_truncation ? :truncated_weights : :available
    (_constant_additive_shift(a, a1) && _constant_additive_shift(a, a0)) ||
        return :unsupported_policy
    return weight_truncation ? :truncated_weights : :available
end

"""
    _mtp_clever_covariate_clamp_aware(a_obs, mu_a, sigma, δ, L, U) -> Vector

Clamp-aware additive MTP density ratio for `d(a) = clamp(a+δ, L, U)`.

On the absolutely continuous interior
`{a : L < a < U and L < a−δ < U}`, returns `g(a−δ|w)/g(a|w)`.
Elsewhere returns `0` (boundary atoms / non-overlapping support do not enter
the continuous Radon–Nikodym fluctuation; g-computation still uses `Q(d(A),W)`).
"""
function _mtp_clever_covariate_clamp_aware(
    a_obs::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma::Real,
    δ::Real,
    L::Real,
    U::Real;
    eps::Real = 1e-8,
)
    n = length(a_obs)
    H = zeros(n)
    if !isfinite(δ) || isapprox(δ, 0; atol = 1e-12)
        fill!(H, 1.0)
        return H
    end
    σ = max(Float64(sigma), 1e-6)
    Lo = Float64(L) + eps
    Up = Float64(U) - eps
    @inbounds for i in 1:n
        a = Float64(a_obs[i])
        a_back = a - Float64(δ)
        if a > Lo && a < Up && a_back > Lo && a_back < Up
            den = max(_gaussian_density(a, mu_a[i], σ), 1e-12)
            H[i] = _gaussian_density(a_back, mu_a[i], σ) / den
        end
    end
    return H
end

"""
    _covariate_matrix(schema, df)

Matrix of encoded covariates from a fitted `CovariateSchema` (same contract as `transform_covariates`).
Used for stacking `W` beside treatment in density-ratio designs.
"""
function _covariate_matrix(schema::CovariateSchema, df::AbstractDataFrame)
    return transform_covariates(schema, df)
end

"""
    _covariate_matrix(df, covariates::Vector{Symbol})

Fit `fit_covariate_schema(df, covariates)` and return the design matrix via `_covariate_matrix(schema, df)`.
"""
function _covariate_matrix(df::AbstractDataFrame, covariates::Vector{Symbol})
    schema = fit_covariate_schema(df, covariates)
    return _covariate_matrix(schema, df)
end

"""
    _ratio_from_classifier(sl, a, W; trunc=10) -> Vector{Float64}

Predict Pr(label=`observed` | A, W) with fitted binomial Super Learner `sl` on `[1, A, W]`. Training uses
label 1 for natural `A` and 0 for policy `A`; Bayes odds give ``g_{\\text{policy}} / g_{\\text{obs}} = (1-p)/p``,
then clamp to `[1/trunc, trunc]`.
"""
function _ratio_from_classifier(sl, a::AbstractVector{<:Real}, W::Matrix{Float64}; trunc::Real = 10.0)
    X = hcat(ones(length(a)), Float64.(a), W)
    # The classifier labels observed samples 1 and policy samples 0, so the
    # shared inverse-odds transform applies unchanged.
    return _discrete_policy_ratio(predict_super_learner(sl, X); trunc = trunc)
end

"""
    _fit_density_ratio_classifier(a_obs, a_policy, W; learners, rng)

Continuous-treatment 2n stack: rows are `([a_obs; a_policy], [W; W])` with binary label `S=1` on the first
half (observed law) and `S=0` on the second (policy law). Returns the fitted Super Learner for use with
`_ratio_from_classifier`.
"""
function _fit_density_ratio_classifier(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    W::Matrix{Float64};
    learners = (:logistic, :mean),
    rng = StableRNG(1),
)
    n = length(a_obs)
    A_tr = vcat(Float64.(a_obs), Float64.(a_policy))
    W_tr = vcat(W, W)
    S_tr = vcat(ones(n), zeros(n))
    X_tr = hcat(ones(2n), A_tr, W_tr)
    return fit_super_learner(
        X_tr, S_tr;
        learners = learners,
        family = :binomial,
        metalearner = :invmse,
        rng = rng,
    )
end

# ---------------------------------------------------------------------------
# Polymorphic policy-component validation, triviality, status, and ratios
# ---------------------------------------------------------------------------

"""
    _resolve_policy_kwargs(; policy, policy_delta, smooth_policy, smooth_shift,
                           feasible_policy, feasible_delta, headroom_policy, headroom_delta)
        -> (resolved_policy, resolved_delta)

Unify modern `policy`/`policy_delta` kwargs with backwards-compatible per-policy
kwargs, enforcing mutual exclusion.
"""
function _resolve_policy_kwargs(;
    policy::Union{Nothing, AbstractContinuousPolicy} = nothing,
    policy_delta::Union{Nothing, Real} = nothing,
    smooth_policy::Union{Nothing, SmoothBoundedLogitShiftPolicy} = nothing,
    smooth_shift::Union{Nothing, Real} = nothing,
    feasible_policy::Union{Nothing, FeasibleAdditiveShiftPolicy} = nothing,
    feasible_delta::Union{Nothing, Real} = nothing,
    headroom_policy::Union{Nothing, ProportionalHeadroomPolicy} = nothing,
    headroom_delta::Union{Nothing, Real} = nothing,
)
    n_supplied = Int(policy !== nothing) + Int(smooth_policy !== nothing) +
        Int(feasible_policy !== nothing) + Int(headroom_policy !== nothing)
    n_supplied <= 1 || throw(ArgumentError(
        "supply at most one explicit continuous policy for this contrast",
    ))
    policy !== nothing && return (policy, policy_delta)
    smooth_policy !== nothing && return (smooth_policy, smooth_shift)
    feasible_policy !== nothing && return (feasible_policy, feasible_delta)
    headroom_policy !== nothing && return (headroom_policy, headroom_delta)
    return (nothing, nothing)
end

function _validate_uncapped_options(
    policy::Union{Nothing, AbstractContinuousPolicy},
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    density_ratio::Symbol,
    cv_trunc::Bool,
    clamp_config::Bool,
    weight_truncation::Bool,
)
    weight_truncation && return nothing
    density_ratio === :gaussian || throw(ArgumentError(
        "weight_truncation=false supports only Gaussian LMTP density ratios",
    ))
    cv_trunc && throw(ArgumentError(
        "weight_truncation=false cannot be combined with cv_trunc=true",
    ))
    clamp_config && policy === nothing && throw(ArgumentError(
        "weight_truncation=false supports only unclamped LMTP policies",
    ))
    (policy !== nothing ||
        (_constant_additive_shift(a, a1) && _constant_additive_shift(a, a0))) ||
        throw(ArgumentError(
            "weight_truncation=false requires constant additive shifts or an explicit bounded policy",
        ))
    return nothing
end

function _policy_validate_components(
    ::Nothing,
    delta,
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    density_ratio::Symbol,
    policy_mask,
    weight_truncation::Bool,
    clamp_config::Bool,
)
    return 0.0
end

function _policy_validate_components(
    p::SmoothBoundedLogitShiftPolicy,
    delta,
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    density_ratio::Symbol,
    policy_mask,
    weight_truncation::Bool,
    clamp_config::Bool,
)
    !clamp_config || throw(ArgumentError(
        "supply only the smooth bounded policy for this contrast",
    ))
    (delta !== nothing && isfinite(delta)) ||
        throw(ArgumentError("smooth policy requires a finite smooth_shift"))
    expected = apply_policy_values(a, delta, p; stratum_mask = policy_mask)
    all(isapprox.(a1, expected; atol = 1e-10, rtol = 1e-10)) || throw(ArgumentError(
        "a_policy does not match the supplied smooth bounded policy and shift",
    ))
    all(isapprox.(a0, a; atol = 1e-10, rtol = 1e-10)) || throw(ArgumentError(
        "smooth bounded policy EIF inference requires the natural policy as reference",
    ))
    return Float64(delta)
end

function _policy_validate_components(
    p::FeasibleAdditiveShiftPolicy,
    delta,
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    density_ratio::Symbol,
    policy_mask,
    weight_truncation::Bool,
    clamp_config::Bool,
)
    !clamp_config || throw(ArgumentError(
        "supply only the feasible additive policy for this contrast",
    ))
    density_ratio === :gaussian || throw(ArgumentError(
        "feasible additive LMTP currently uses density_ratio=:gaussian",
    ))
    delta !== nothing || throw(ArgumentError("feasible additive policy requires feasible_delta"))
    step = _feasible_step(p, delta)
    expected = apply_policy_values(a, delta, p; stratum_mask = policy_mask)
    all(isapprox.(a1, expected; atol = 1e-10, rtol = 1e-10)) ||
        throw(ArgumentError("a_policy must match the feasible additive policy"))
    all(isapprox.(a0, a; atol = 1e-10, rtol = 1e-10)) ||
        throw(ArgumentError("feasible additive policy requires the natural reference"))
    return step
end

function _policy_validate_components(
    p::ProportionalHeadroomPolicy,
    delta,
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    density_ratio::Symbol,
    policy_mask,
    weight_truncation::Bool,
    clamp_config::Bool,
)
    !clamp_config || throw(ArgumentError(
        "supply only the proportional headroom policy for this contrast",
    ))
    density_ratio === :gaussian || throw(ArgumentError(
        "proportional headroom LMTP currently uses density_ratio=:gaussian",
    ))
    delta !== nothing || throw(ArgumentError(
        "proportional headroom policy requires headroom_delta",
    ))
    d = _validate_headroom_delta(delta)
    expected = apply_policy_values(a, d, p; stratum_mask = policy_mask)
    all(isapprox.(a1, expected; atol = 1e-10, rtol = 1e-10)) || throw(ArgumentError(
        "a_policy must match the proportional headroom policy",
    ))
    all(isapprox.(a0, a; atol = 1e-10, rtol = 1e-10)) ||
        throw(ArgumentError("proportional headroom policy requires the natural reference"))
    return d
end

_policy_is_trivial(::Nothing, param, a, a1, a0; policy_mask = nothing) = all(isequal.(a1, a0))
_policy_is_trivial(::SmoothBoundedLogitShiftPolicy, param, a, a1, a0; policy_mask = nothing) =
    all(isequal.(a1, a0))
_policy_is_trivial(p::FeasibleAdditiveShiftPolicy, param, a, a1, a0; policy_mask = nothing) =
    iszero(param) || abs(param) > p.upper - p.lower || (policy_mask !== nothing && !any(policy_mask))
_policy_is_trivial(::ProportionalHeadroomPolicy, param, a, a1, a0; policy_mask = nothing) =
    iszero(param) || (policy_mask !== nothing && !any(policy_mask))

function _policy_inference_status(
    ::Nothing,
    param,
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    weight_truncation::Bool,
    clamp_config::Bool,
    policy_mask = nothing,
)
    return _lmtp_weight_inference_status(
        a, a1, a0; weight_truncation, clamp_config, smooth_policy = false,
    )
end

function _policy_inference_status(
    p::AbstractFixedBoundContinuousPolicy,
    param,
    a::Vector{Float64},
    a1::Vector{Float64},
    a0::Vector{Float64};
    weight_truncation::Bool,
    clamp_config::Bool,
    policy_mask = nothing,
)
    _policy_is_trivial(p, param, a, a1, a0; policy_mask) && return :trivial
    return weight_truncation ? :truncated_weights : :available
end

function _policy_density_ratio(
    ::Nothing,
    param,
    a_test::Vector{Float64},
    density_a_test::Vector{Float64},
    a1_test::Vector{Float64},
    a0_test::Vector{Float64},
    μ_te::Vector{Float64},
    σ_fold::Float64,
    selected;
    weight_truncation::Bool,
    clamp_aware::Bool,
    shift_amount,
    shift_reference,
    L,
    U,
)
    if clamp_aware
        Hg1 = _mtp_clever_covariate_clamp_aware(a_test, μ_te, σ_fold, shift_amount, L, U)
        δ0 = shift_reference === nothing ? 0.0 : Float64(shift_reference)
        Hg0 = _mtp_clever_covariate_clamp_aware(a_test, μ_te, σ_fold, δ0, L, U)
        return Hg1, Hg0
    end
    ratio = weight_truncation ? _mtp_clever_covariate_gaussian :
        _mtp_clever_covariate_gaussian_uncapped
    return ratio(a_test, a1_test, μ_te, σ_fold), ratio(a_test, a0_test, μ_te, σ_fold)
end

function _policy_density_ratio(
    ::SmoothBoundedLogitShiftPolicy,
    param::Float64,
    a_test::Vector{Float64},
    density_a_test::Vector{Float64},
    a1_test::Vector{Float64},
    a0_test::Vector{Float64},
    μ_te::Vector{Float64},
    σ_fold::Float64,
    selected;
    weight_truncation::Bool,
    clamp_aware::Bool,
    shift_amount,
    shift_reference,
    L,
    U,
)
    mask_test = selected === nothing ? ones(length(a_test)) : Float64.(selected)
    z1 = density_a_test .+ param .* mask_test
    ratio = weight_truncation ? _mtp_clever_covariate_gaussian :
        _mtp_clever_covariate_gaussian_uncapped
    return ratio(density_a_test, z1, μ_te, σ_fold), ones(length(a_test))
end

function _policy_density_ratio(
    p::FeasibleAdditiveShiftPolicy,
    param::Float64,
    a_test::Vector{Float64},
    density_a_test::Vector{Float64},
    a1_test::Vector{Float64},
    a0_test::Vector{Float64},
    μ_te::Vector{Float64},
    σ_fold::Float64,
    selected;
    weight_truncation::Bool,
    clamp_aware::Bool,
    shift_amount,
    shift_reference,
    L,
    U,
)
    Hg1 = _feasible_gaussian_ratio(a_test, μ_te, σ_fold, p, param, selected; weight_truncation)
    return Hg1, ones(length(a_test))
end

function _policy_density_ratio(
    p::ProportionalHeadroomPolicy,
    param::Float64,
    a_test::Vector{Float64},
    density_a_test::Vector{Float64},
    a1_test::Vector{Float64},
    a0_test::Vector{Float64},
    μ_te::Vector{Float64},
    σ_fold::Float64,
    selected;
    weight_truncation::Bool,
    clamp_aware::Bool,
    shift_amount,
    shift_reference,
    L,
    U,
)
    Hg1 = _headroom_gaussian_ratio(a_test, μ_te, σ_fold, p, param, selected; weight_truncation)
    return Hg1, ones(length(a_test))
end
