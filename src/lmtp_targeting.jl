"""Canonical-link LMTP TMLE / EIF targeting steps."""

using Statistics

"""
    _clamp_count_predictions!(Q)

Ensure fitted means stay on the nonnegative count scale.
"""
function _clamp_count_predictions!(Q::AbstractVector{<:Real})
    Q .= max.(Q, 0.0)
    return Q
end

function _clamped_logits(q::Vector{Float64})::Vector{Float64}
    qfloor = eps(Float64)
    qceil = 1.0 - qfloor
    out = Vector{Float64}(undef, length(q))
    @inbounds for i in eachindex(q)
        p = clamp(q[i], qfloor, qceil)
        out[i] = log(p / (1.0 - p))
    end
    return out
end

@inline function _binomial_tilt_score_and_deriv(
    H::Vector{Float64},
    y::Vector{Float64},
    offset_obs::Vector{Float64},
    ε::Float64,
)
    f = 0.0
    df = 0.0
    @inbounds for i in eachindex(H, y, offset_obs)
        p = _expit_stable(offset_obs[i] + ε)
        hi = H[i]
        f += hi * (y[i] - p)
        df -= hi * p * (1.0 - p)
    end
    return f, df
end

function _solve_binomial_tilt(
    H::Vector{Float64},
    y::Vector{Float64},
    offset_obs::Vector{Float64},
    sum_h::Float64,
)::Float64
    left, right = -40.0, 40.0
    f_left = _binomial_tilt_score_and_deriv(H, y, offset_obs, left)[1]
    step = 40.0
    while f_left < 0 && abs(left) < 1e6
        left -= step
        step *= 2
        f_left = _binomial_tilt_score_and_deriv(H, y, offset_obs, left)[1]
    end
    f_right = _binomial_tilt_score_and_deriv(H, y, offset_obs, right)[1]
    step = 40.0
    while f_right > 0 && abs(right) < 1e6
        right += step
        step *= 2
        f_right = _binomial_tilt_score_and_deriv(H, y, offset_obs, right)[1]
    end
    f_left >= 0 && f_right <= 0 || throw(DomainError(
        (f_left, f_right), "binomial weighted-intercept score has no finite root",
    ))
    f_left == 0.0 && return left
    f_right == 0.0 && return right
    scale = max(1.0, sum_h)
    tol_f = 1e-13 * scale
    root = clamp(0.0, left, right)
    f_root = 0.0
    prev_abs_f = Inf
    for iter in 1:120
        f_root, df = _binomial_tilt_score_and_deriv(H, y, offset_obs, root)
        if f_root > 0.0
            left = root
        elseif f_root < 0.0
            right = root
        else
            break
        end
        abs_f = abs(f_root)
        (abs_f <= tol_f || right - left <= 1e-13 * (1.0 + abs(root))) && break
        use_newton = df < -1e-15 && !(iter > 12 && abs_f > 0.5 * prev_abs_f)
        cand = use_newton ? root - f_root / df : NaN
        root = (isfinite(cand) && left < cand < right) ? cand : 0.5 * (left + right)
        prev_abs_f = abs_f
    end
    f_final = _binomial_tilt_score_and_deriv(H, y, offset_obs, root)[1]
    abs(f_final) <= 1e-8 * scale || throw(DomainError(
        f_final, "binomial weighted-intercept score root did not converge",
    ))
    return root
end

function _weighted_intercept_tilt(
    y::Vector{Float64},
    q_obs::Vector{Float64},
    q_policy::Vector{Float64},
    H::Vector{Float64},
    family::Symbol,
    λ::Float64,
)
    all(isfinite, y) && all(isfinite, q_obs) && all(isfinite, q_policy) && all(isfinite, H) ||
        throw(DomainError((y, q_obs, q_policy, H), "TMLE tilting requires finite inputs"))
    all(H .>= 0) || throw(DomainError(H, "TMLE clever covariates must be nonnegative"))
    sum_h = sum(H)
    isfinite(sum_h) || throw(DomainError(sum_h, "TMLE clever-covariate sum must be finite"))
    sum_h > 0 || return copy(q_obs), copy(q_policy), 0.0, false

    if family in COUNT_OUTCOME_FAMILIES
        all(y .>= 0) || throw(DomainError(y, "count TMLE outcomes must be nonnegative"))
        all(q_obs .>= 0) && all(q_policy .>= 0) || throw(DomainError(
            (q_obs, q_policy), "count TMLE outcome predictions must be nonnegative",
        ))
        denominator = sum(H .* q_obs)
        numerator = sum(H .* y)
        isfinite(denominator) && isfinite(numerator) || throw(DomainError(
            (denominator, numerator), "count TMLE weighted sums must be finite",
        ))
        if numerator == 0
            if λ == 1.0
                return zeros(length(q_obs)), zeros(length(q_policy)), -Inf, true
            end
            updated_scale = exp(λ * log(floatmin(Float64)))
            return updated_scale .* q_obs, updated_scale .* q_policy, log(updated_scale), false
        end
        denominator > 0 || throw(DomainError(denominator, "count TMLE requires a positive weighted outcome offset"))
        scale = numerator / denominator
        (isfinite(scale) && scale > 0) || throw(DomainError(
            scale, "count TMLE weighted-intercept scale must be finite and positive",
        ))
        updated_scale = exp(λ * log(scale))
        return updated_scale .* q_obs, updated_scale .* q_policy, log(updated_scale), false
    elseif family === :binomial
        all(x -> 0 <= x <= 1, y) || throw(DomainError(y, "binomial TMLE outcomes must lie in [0, 1]"))
        all(x -> 0 <= x <= 1, q_obs) && all(x -> 0 <= x <= 1, q_policy) ||
            throw(DomainError((q_obs, q_policy), "binomial outcome predictions must lie in [0, 1]"))
        if all(iszero, y) || all(==(1.0), y)
            target = all(iszero, y) ? 0.0 : 1.0
            if λ == 1.0
                return fill(target, length(q_obs)), fill(target, length(q_policy)),
                    target == 0 ? -Inf : Inf, true
            end
            ε = λ * (target == 0 ? -40.0 : 40.0)
            offset_obs_b = _clamped_logits(q_obs)
            offset_policy_b = _clamped_logits(q_policy)
            return _expit_stable.(offset_obs_b .+ ε),
                _expit_stable.(offset_policy_b .+ ε), ε, false
        end
        offset_obs = _clamped_logits(q_obs)
        offset_policy = _clamped_logits(q_policy)
        root = _solve_binomial_tilt(H, y, offset_obs, sum_h)
        ε = λ * root
        return _expit_stable.(offset_obs .+ ε), _expit_stable.(offset_policy .+ ε), ε, false
    elseif family === :gaussian
        ε = λ * sum(H .* (y .- q_obs)) / sum_h
        return q_obs .+ ε, q_policy .+ ε, ε, false
    end
    throw(ArgumentError("weighted-intercept LMTP TMLE does not support family_outcome=$family"))
end

"""
    _single_arm_lmtp_target(y, q_obs, q_policy, H, family, λ, estimator) -> NamedTuple

Apply one-step EIF or canonical-link weighted-intercept TMLE targeting to a
single policy arm. Returns `(psi, point_score, q_obs, q_policy, boundary, epochs)`.
"""
function _single_arm_lmtp_target(
    y::Vector{Float64},
    q_obs::Vector{Float64},
    q_policy::Vector{Float64},
    H::Vector{Float64},
    family::Symbol,
    λ::Float64,
    estimator::Symbol,
)
    if estimator in (:eif, :aipw, :sdr)
        qp = copy(q_policy)
        family in COUNT_OUTCOME_FAMILIES && _clamp_count_predictions!(qp)
        raw = qp .+ λ .* H .* (y .- q_obs)
        ψ = mean(raw)
        return (
            psi = ψ, point_score = raw, q_obs = copy(q_obs),
            q_policy = qp, boundary = false, epochs = 0,
        )
    end
    q_obs_t, q_policy_t, _, boundary = _weighted_intercept_tilt(
        y, q_obs, q_policy, H, family, λ,
    )
    ψ = mean(q_policy_t)
    raw = q_policy_t .+ H .* (y .- q_obs_t)
    score = raw .- mean(raw) .+ ψ
    return (
        psi = ψ, point_score = score, q_obs = q_obs_t,
        q_policy = q_policy_t, boundary = boundary, epochs = 1,
    )
end

"""
    _lmtp_targeting_step(components; estimator, targeting_weight, epochs, family_outcome) -> NamedTuple

Apply the targeting step to cross-fitted nuisances and return the Wald interval.

`estimator` selects the one-step EIF path (`:eif`, `:aipw`, `:sdr`) or either
TMLE label (`:tmle`, `:itmle`). The TMLE paths fit separate weighted canonical-link
intercept tilts for each policy arm, applying each fitted intercept to both the
factual offset and its policy prediction. `epochs` is accepted for compatibility;
the canonical weighted-intercept score is solved in one step per arm.
"""
function _lmtp_targeting_step(
    components::NamedTuple;
    estimator::Symbol,
    targeting_weight::Real,
    epochs::Int,
    family_outcome::Union{Nothing, Symbol},
)
    λ = clamp(Float64(targeting_weight), 0.0, 1.0)
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "unsupported LMTP estimator=$estimator; expected :tmle, :itmle, :eif, :aipw, or :sdr",
    ))
    fam = family_outcome === nothing ?
        (hasproperty(components, :family_outcome) ? components.family_outcome : :gaussian) :
        family_outcome
    Q1 = collect(Float64, components.Q1)::Vector{Float64}
    Q0 = collect(Float64, components.Q0)::Vector{Float64}
    y = collect(Float64, components.y)::Vector{Float64}
    Q_obs = collect(Float64, components.Q_obs)::Vector{Float64}
    H1 = collect(Float64, components.H1)::Vector{Float64}
    H0 = collect(Float64, components.H0)::Vector{Float64}
    base_status = hasproperty(components, :inference_status) ?
        Symbol(components.inference_status) : :available

    # The component builder marks an algebraically identical policy pair as
    # trivial. Do not infer that status from coincidentally equal fitted arrays:
    # those can arise under degenerate fits while a base inference gate applies.
    if base_status === :trivial
        point_score = zeros(Float64, components.n)
        diagnostics = (
            score_policy1 = 0.0, score_policy0 = 0.0,
            score_norm_policy1 = 0.0, score_norm_policy0 = 0.0,
            score_combined_norm = 0.0, mean_abs_residual = 0.0,
            mean_h1 = mean(H1), mean_h0 = mean(H0), targeting_ok = true,
        )
        return (
            estimate = 0.0, se = 0.0, lower = 0.0, upper = 0.0,
            ic = copy(point_score), point_score = point_score, estimator = estimator,
            targeting_weight = λ, epochs = 0, psi1 = 0.0, psi0 = 0.0,
            trunc = components.trunc, inference_status = :trivial,
            targeting_diagnostics = diagnostics,
        )
    end

    if base_status === :available && (all(iszero, H1) || all(iszero, H0))
        base_status = :positivity_violation
    end
    λ < 1.0 && base_status === :available && (base_status = :attenuated_targeting)
    inference_status = base_status

    arm1 = _single_arm_lmtp_target(y, Q_obs, Q1, H1, fam, λ, estimator)
    arm0 = _single_arm_lmtp_target(y, Q_obs, Q0, H0, fam, λ, estimator)
    ψ1 = arm1.psi
    ψ0 = arm0.psi
    est = ψ1 - ψ0
    point_score = arm1.point_score .- arm0.point_score
    boundary_outcome = arm1.boundary || arm0.boundary
    epochs_used = arm1.epochs

    inference_status === :available && boundary_outcome &&
        (inference_status = :boundary_outcome)

    all(isfinite, point_score) && isfinite(est) || throw(DomainError(
        (point_score, est), "LMTP targeting produced non-finite point scores",
    ))
    abs(mean(point_score) - est) <= 1e-9 * (1 + abs(est)) || throw(DomainError(
        (mean(point_score), est), "LMTP point-score mean does not equal the point estimate",
    ))
    if inference_status === :available
        ic = point_score .- est
        all(isfinite, ic) || throw(DomainError(ic, "LMTP EIF contains non-finite values"))
        se = std(ic) / sqrt(components.n)
        isfinite(se) || throw(DomainError(se, "LMTP EIF standard error is not finite"))
        lwr, upr = wald_ci(est, se)
    else
        ic = fill(NaN, components.n)
        se = lwr = upr = NaN
    end
    diagnostics = tmle_score_diagnostics(
        components; λ = 1.0,
        residual_policy1 = y .- arm1.q_obs,
        residual_policy0 = y .- arm0.q_obs,
    )
    return (
        estimate = est, se = se, lower = lwr, upper = upr, ic = ic,
        point_score = Float64.(point_score), estimator = estimator,
        targeting_weight = λ, epochs = epochs_used,
        psi1 = ψ1, psi0 = ψ0, trunc = components.trunc,
        inference_status = inference_status, targeting_diagnostics = diagnostics,
    )
end

"""
    lmtp_tmle_from_components(components; estimator, targeting_weight, epochs) -> NamedTuple

Targeting step only (for cached fold nuisances or diagnostics).
"""
function lmtp_tmle_from_components(
    components::NamedTuple;
    estimator::Symbol = :tmle,
    targeting_weight::Real = 1.0,
    epochs::Int = 1,
    family_outcome::Union{Nothing, Symbol} = nothing,
)
    return _lmtp_targeting_step(
        components; estimator = estimator, targeting_weight = targeting_weight,
        epochs = epochs, family_outcome = family_outcome,
    )
end
