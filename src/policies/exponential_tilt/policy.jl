# ---------------------------------------------------------------------------
# Policy struct, tilt score, weights, and analytical support bounds
# ---------------------------------------------------------------------------

"""
    ExponentialTiltPolicy(; bound = nothing, center = 0.0)

Stochastic exponential-tilting intervention policy with score function:

- Unbounded (`bound = nothing`): ``\\phi(a) = a - c``
- Bounded (`bound = M > 0`): ``\\phi(a) = M \\tanh((a - c) / M)``

The tilt weight is ``w_\\delta(a) = \\exp(\\delta \\phi(a))``. For `bound = M`,
``|\\phi(a)| \\le M``, so ``\\exp(-2|\\delta|M) \\le w_\\delta(A) / m_0(W) \\le \\exp(2|\\delta|M)``
almost surely without ad-hoc truncation.
"""
struct ExponentialTiltPolicy <: AbstractContinuousPolicy
    bound::Union{Nothing, Float64}
    center::Float64

    function ExponentialTiltPolicy(bound::Union{Nothing, Real}, center::Real)
        c = Float64(center)
        isfinite(c) || throw(ArgumentError(
            "exponential tilt policy requires a finite center; got $center",
        ))
        if bound === nothing
            return new(nothing, c)
        end
        b = Float64(bound)
        (isfinite(b) && b > 0) || throw(ArgumentError(
            "exponential tilt policy bound must be nothing or finite and > 0; got $bound",
        ))
        return new(b, c)
    end
end

ExponentialTiltPolicy(; bound::Union{Nothing, Real} = nothing, center::Real = 0.0) =
    ExponentialTiltPolicy(bound, center)

"""
    exponential_tilt_policy(; bound = nothing, center = 0.0) -> ExponentialTiltPolicy

Construct an [`ExponentialTiltPolicy`](@ref). Set `bound = M > 0` for the smooth
bounded tilt ``\\phi(a) = M \\tanh((a - c) / M)`` or leave `bound = nothing` for the
linear tilt ``\\phi(a) = a - c``.
"""
exponential_tilt_policy(; bound::Union{Nothing, Real} = nothing, center::Real = 0.0) =
    ExponentialTiltPolicy(bound, center)

"""
    tilt_score(a, policy::ExponentialTiltPolicy)

Evaluate the tilt score ``\\phi(a)`` at a scalar or vector of finite exposures.
"""
function tilt_score(a::Real, policy::ExponentialTiltPolicy)
    x = Float64(a)
    isfinite(x) || throw(ArgumentError("tilt_score requires finite exposure values; got $a"))
    d = x - policy.center
    return policy.bound === nothing ? d : policy.bound * tanh(d / policy.bound)
end

function tilt_score(a::AbstractVector{<:Real}, policy::ExponentialTiltPolicy)
    out = Vector{Float64}(undef, length(a))
    @inbounds for i in eachindex(a)
        out[i] = tilt_score(a[i], policy)
    end
    return out
end

"""
    tilt_log_weight(a, delta, policy::ExponentialTiltPolicy)

Compute ``\\log w_\\delta(a) = \\delta \\phi(a)`` for a finite tilt strength `delta`.
"""
function tilt_log_weight(a::Real, delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    return δ * tilt_score(a, policy)
end

function tilt_log_weight(a::AbstractVector{<:Real}, delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    scores = tilt_score(a, policy)
    return δ .* scores
end

"""
    tilt_weight(a, delta, policy::ExponentialTiltPolicy)

Compute the unnormalised tilt weight ``w_\\delta(a) = \\exp(\\delta \\phi(a))``.
"""
tilt_weight(a::Real, delta::Real, policy::ExponentialTiltPolicy) =
    exp(tilt_log_weight(a, delta, policy))

tilt_weight(a::AbstractVector{<:Real}, delta::Real, policy::ExponentialTiltPolicy) =
    exp.(tilt_log_weight(a, delta, policy))

"""
    tilt_normalizer_log_bounds(delta, policy::ExponentialTiltPolicy) -> (lo, hi)

Exact theoretical bounds ``[-|\\delta|M, |\\delta|M]`` on ``\\log m_0(W) = \\log \\mathbb{E}[w_\\delta(A) \\mid W]``
for a bounded policy (`(-Inf, Inf)` when unbounded).
"""
function tilt_normalizer_log_bounds(delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    policy.bound === nothing && return (-Inf, Inf)
    lim = abs(δ) * policy.bound
    return (-lim, lim)
end

"""
    tilt_weight_bounds(delta, policy::ExponentialTiltPolicy) -> (lo, hi)

Exact theoretical bounds ``[\\exp(-2|\\delta|M), \\exp(2|\\delta|M)]`` on the density ratio
``H_\\delta(A, W) = w_\\delta(A) / m_0(W)`` for a bounded policy (`(0.0, Inf)` when unbounded).
"""
function tilt_weight_bounds(delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    policy.bound === nothing && return (0.0, Inf)
    lim = 2.0 * abs(δ) * policy.bound
    return (exp(-lim), exp(lim))
end
