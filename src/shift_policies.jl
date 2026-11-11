"""Policy constructors for modified treatment policies beyond additive z-shifts.

# References

- Díaz & van der Laan (2012), *Biometrics* — stochastic / population interventions
- Díaz et al. (2023), *JASA* — longitudinal MTPs (additive and more general *d*)
"""

using Statistics

"""A fixed-bound, smooth logit-scale shift for continuous treatment in `(lower, upper)`."""
struct SmoothBoundedLogitShiftPolicy
    lower::Float64
    upper::Float64

    function SmoothBoundedLogitShiftPolicy(lower::Real, upper::Real)
        lo, hi = Float64(lower), Float64(upper)
        isfinite(lo) && isfinite(hi) && lo < hi && isfinite(hi - lo) || throw(ArgumentError(
            "smooth bounded policy bounds must be finite and satisfy lower < upper",
        ))
        new(lo, hi)
    end
end

"""
    smooth_bounded_logit_shift_policy(lower, upper)

Construct a smooth, strictly increasing policy on a prespecified bounded
support. A grid value `δ` shifts the logit of the exposure's relative position:
`L + (U-L) * logistic(logit((A-L)/(U-L)) + δ)`. Bounds must be fixed before
analysing the sample; observed exposures must lie strictly between them.
"""
smooth_bounded_logit_shift_policy(lower::Real, upper::Real) =
    SmoothBoundedLogitShiftPolicy(lower, upper)

function _policy_logistic(x::Float64)
    return x >= 0 ? inv(1 + exp(-x)) : (exp(x) / (1 + exp(x)))
end

"""Map interior bounded exposures to logit coordinates for density fitting."""
function _bounded_logit_values(a::AbstractVector{<:Real}, policy::SmoothBoundedLogitShiftPolicy)
    all(x -> isfinite(x) && policy.lower < x < policy.upper, a) || throw(ArgumentError(
        "smooth bounded policy requires every analysed exposure strictly inside the fixed bounds",
    ))
    return log.(Float64.(a) .- policy.lower) .- log.(policy.upper .- Float64.(a))
end

"""Apply the smooth bounded logit-shift policy, optionally within a stratum."""
function apply_policy_values(
    a::AbstractVector{<:Real},
    delta::Real,
    policy::SmoothBoundedLogitShiftPolicy;
    stratum_mask = nothing,
)
    isfinite(delta) || throw(ArgumentError("smooth bounded policy delta must be finite"))
    values = Float64.(a)
    mask = stratum_mask === nothing ? trues(length(values)) : BitVector(stratum_mask)
    length(mask) == length(values) || throw(DimensionMismatch(
        "stratum mask length $(length(mask)) does not match exposure length $(length(values))",
    ))
    selected = findall(mask)
    all(i -> isfinite(values[i]) && policy.lower < values[i] < policy.upper, selected) ||
        throw(ArgumentError(
            "smooth bounded policy requires shifted exposures strictly inside the fixed bounds",
        ))
    output = copy(values)
    isapprox(delta, 0; atol = 1e-12) && return output
    logit_values = _bounded_logit_values(values[selected], policy)
    width = policy.upper - policy.lower
    output[selected] .= policy.lower .+ width .* _policy_logistic.(logit_values .+ Float64(delta))
    all(i -> policy.lower < output[i] < policy.upper, selected) || throw(ArgumentError(
        "smooth bounded policy shift is too large to represent strictly inside the fixed bounds",
    ))
    return output
end

"""
    additive_shift_policy(; scale="z", lower_q=0.01, upper_q=0.99) -> ShiftPolicy

Standard additive MTP (current default): `A ↦ clamp(A + δ·s, L, U)`.
"""
additive_shift_policy(; scale = "z", lower_q = 0.01, upper_q = 0.99) =
    ShiftPolicy(scale, lower_q, upper_q)

"""
    multiplicative_shift_policy(; lower_q=0.01, upper_q=0.99) -> ShiftPolicy

Multiplicative MTP encoded as `scale = "multiplicative"`: realised shift uses
`A ↦ clamp(A * (1 + δ), L, U)` via [`apply_policy_values`](@ref).
"""
multiplicative_shift_policy(; lower_q = 0.01, upper_q = 0.99) =
    ShiftPolicy("multiplicative", lower_q, upper_q)

"""
    threshold_shift_policy(; lower_q=0.01, upper_q=0.99) -> ShiftPolicy

Threshold / “raise if below” policy (`scale = "threshold"`): for positive `δ`,
observations with `A < median(A)` are shifted up by `δ · sd(A)` (clamped).
"""
threshold_shift_policy(; lower_q = 0.01, upper_q = 0.99) =
    ShiftPolicy("threshold", lower_q, upper_q)

"""
    apply_policy_values(a, delta, policy::ShiftPolicy; stratum_mask=nothing) -> Vector{Float64}

Map grid `delta` to intervened exposures under additive, multiplicative, or
threshold policies. Additive scales (`z`, `raw`, …) delegate to
[`apply_shift_policy`](@ref).
"""
function apply_policy_values(
    a::AbstractVector{<:Real},
    delta::Real,
    policy::ShiftPolicy;
    stratum_mask = nothing,
)
    x = Float64.(a)
    L, U = exposure_bounds(x, policy.lower_q, policy.upper_q)
    if policy.scale in ("z", "raw", "global_sd", "stratum_sd")
        mask = stratum_mask === nothing ? trues(length(x)) : BitVector(stratum_mask)
        # Build a temporary STRAT-less requested shift using full-sample scale
        req = if policy.scale == "raw"
            Float64(delta)
        else
            sd = std(x[mask])
            (!isfinite(sd) || sd <= 0) && return clamp.(x, L, U)
            Float64(delta) * sd
        end
        return apply_shift_policy(x, req, L, U; stratum_mask = stratum_mask)
    elseif policy.scale == "multiplicative"
        factor = 1 + Float64(delta)
        out = copy(x)
        mask = stratum_mask === nothing ? trues(length(x)) : BitVector(stratum_mask)
        out[mask] .= clamp.(x[mask] .* factor, L, U)
        return out
    elseif policy.scale == "threshold"
        out = clamp.(x, L, U)
        mask = stratum_mask === nothing ? trues(length(x)) : BitVector(stratum_mask)
        isapprox(delta, 0; atol = 1e-12) && return out
        sub = x[mask]
        med = median(sub)
        sd = std(sub)
        (!isfinite(sd) || sd <= 0) && return out
        bump = Float64(delta) * sd
        for i in eachindex(x)
            mask[i] || continue
            if (delta > 0 && x[i] < med) || (delta < 0 && x[i] > med)
                out[i] = clamp(x[i] + bump, L, U)
            end
        end
        return out
    end
    error("Unknown ShiftPolicy scale: $(policy.scale)")
end

export additive_shift_policy, multiplicative_shift_policy, threshold_shift_policy
export apply_policy_values
export SmoothBoundedLogitShiftPolicy, smooth_bounded_logit_shift_policy
