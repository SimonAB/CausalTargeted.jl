"""Policy constructors for modified treatment policies beyond additive z-shifts.

# References

- Díaz & van der Laan (2012), *Biometrics* — stochastic / population interventions
- Díaz et al. (2023), *JASA* — longitudinal MTPs (additive and more general *d*)
"""

"""A fixed-bound, smooth logit-scale shift for continuous treatment in `(lower, upper)`."""
struct SmoothBoundedLogitShiftPolicy <: AbstractFixedBoundContinuousPolicy
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

"""An additive shift with fixed eligibility bounds and a fixed exposure SD."""
struct FeasibleAdditiveShiftPolicy <: AbstractFixedBoundContinuousPolicy
    lower::Float64
    upper::Float64
    sd_scale::Float64

    function FeasibleAdditiveShiftPolicy(lower::Real, upper::Real, sd_scale::Real)
        lo, hi, scale = Float64(lower), Float64(upper), Float64(sd_scale)
        isfinite(lo) && isfinite(hi - lo) && lo < hi || throw(ArgumentError(
            "feasible additive policy requires finite bounds with lower < upper",
        ))
        isfinite(scale) && scale > 0 || throw(ArgumentError(
            "feasible additive policy requires a finite, positive sd_scale",
        ))
        new(lo, hi, scale)
    end
end

"""
    feasible_additive_shift_policy(lower, upper; sd_scale)

Shift an exposure by `delta * sd_scale` when its original and proposed values
both lie in the fixed interval `[lower, upper]`. Retain the exposure otherwise.
Choose the bounds and SD before fitting. Use `sd_scale=1` when the exposure
and bounds are already expressed on the fixed standardised scale.
"""
feasible_additive_shift_policy(lower::Real, upper::Real; sd_scale::Real) =
    FeasibleAdditiveShiftPolicy(lower, upper, sd_scale)

function _feasible_step(policy::FeasibleAdditiveShiftPolicy, delta::Real)
    step = Float64(delta) * policy.sd_scale
    isfinite(step) || throw(ArgumentError("feasible additive shift must be finite"))
    return step
end

_within_feasible_bounds(a, policy::FeasibleAdditiveShiftPolicy) = policy.lower <= a <= policy.upper

"""Apply a feasible additive shift, optionally within a stratum."""
function apply_policy_values(
    a::AbstractVector{<:Real},
    delta::Real,
    policy::FeasibleAdditiveShiftPolicy;
    stratum_mask = nothing,
)
    step = _feasible_step(policy, delta)
    out = Float64.(a)
    all(isfinite, out) || throw(ArgumentError("feasible additive policy requires finite exposures"))
    selected = stratum_mask === nothing ? trues(length(out)) : stratum_mask
    length(selected) == length(out) || throw(DimensionMismatch("stratum mask must match the exposures"))
    for i in eachindex(out, selected)
        selected[i] || continue
        proposed = out[i] + step
        if _within_feasible_bounds(out[i], policy) && _within_feasible_bounds(proposed, policy)
            out[i] = proposed
        end
    end
    return out
end

"""A deterministic, raw-scale, range-adaptive contraction policy on prespecified bounds `[lower, upper]`."""
struct ProportionalHeadroomPolicy <: AbstractFixedBoundContinuousPolicy
    lower::Float64
    upper::Float64

    function ProportionalHeadroomPolicy(lower::Real, upper::Real)
        lo, hi = Float64(lower), Float64(upper)
        isfinite(lo) && isfinite(hi) && lo < hi && isfinite(hi - lo) || throw(ArgumentError(
            "proportional headroom policy bounds must be finite and satisfy lower < upper",
        ))
        new(lo, hi)
    end
end

"""
    proportional_headroom_policy(lower, upper)

Construct a deterministic, range-adaptive contraction policy on the prespecified
interval `[lower, upper]`. For a proportion `delta` in `(-1, 1)` and an exposure
`a` in `[lower, upper]`:

- `delta >= 0`: `a + delta * (upper - a)` (moves a fraction `delta` of remaining headroom toward `upper`)
- `delta < 0`: `a + abs(delta) * (lower - a)` (moves a fraction `abs(delta)` of remaining headroom toward `lower`)

Exposures outside `[lower, upper]` or outside `stratum_mask` are retained unchanged.
Bounds must be fixed before analysing the sample.
"""
proportional_headroom_policy(lower::Real, upper::Real) =
    ProportionalHeadroomPolicy(lower, upper)

function _validate_headroom_delta(delta::Real)
    d = Float64(delta)
    (isfinite(d) && -1.0 < d < 1.0) || throw(ArgumentError(
        "proportional headroom policy requires finite delta with -1 < delta < 1",
    ))
    return d
end

_within_headroom_bounds(a, policy::ProportionalHeadroomPolicy) = policy.lower <= a <= policy.upper

"""Apply a proportional headroom shift, optionally within a stratum."""
function apply_policy_values(
    a::AbstractVector{<:Real},
    delta::Real,
    policy::ProportionalHeadroomPolicy;
    stratum_mask = nothing,
)
    d = _validate_headroom_delta(delta)
    out = Float64.(a)
    all(isfinite, out) || throw(ArgumentError("proportional headroom policy requires finite exposures"))
    selected = stratum_mask === nothing ? trues(length(out)) : stratum_mask
    length(selected) == length(out) || throw(DimensionMismatch("stratum mask must match the exposures"))
    d == 0.0 && return out
    lo, hi = policy.lower, policy.upper
    p = abs(d)
    for i in eachindex(out, selected)
        selected[i] || continue
        ai = out[i]
        if _within_headroom_bounds(ai, policy)
            out[i] = d >= 0.0 ? ai + d * (hi - ai) : ai + p * (lo - ai)
        end
    end
    return out
end

@inline _policy_logistic(x::Float64) = _expit_stable(x)

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

# ---------------------------------------------------------------------------
# Internal polymorphic policy trait interface (defaults + fixed-bound types)
# ---------------------------------------------------------------------------

_policy_support_status(::SmoothBoundedLogitShiftPolicy) = :smooth_bounded_policy
_policy_support_status(::FeasibleAdditiveShiftPolicy) = :feasible_additive_policy
_policy_support_status(::ProportionalHeadroomPolicy) = :proportional_headroom_policy

_policy_requested_shift(::SmoothBoundedLogitShiftPolicy, delta::Real) = Float64(delta)
_policy_requested_shift(p::FeasibleAdditiveShiftPolicy, delta::Real) = _feasible_step(p, delta)
_policy_requested_shift(::ProportionalHeadroomPolicy, delta::Real) = _validate_headroom_delta(delta)

_policy_density_values(::Any, a) = nothing
_policy_density_values(p::SmoothBoundedLogitShiftPolicy, a) = _bounded_logit_values(a, p)

_policy_uses_fold_cache(::Any) = true

_policy_has_structural_zeros(::Any) = false
_policy_has_structural_zeros(::FeasibleAdditiveShiftPolicy) = true
_policy_has_structural_zeros(::ProportionalHeadroomPolicy) = true

_policy_reports_retention(::Any) = false
_policy_reports_retention(::FeasibleAdditiveShiftPolicy) = true
_policy_reports_retention(::ProportionalHeadroomPolicy) = true

function _validate_explicit_policy_grid_common(; positivity::Bool, imputation)
    positivity && throw(ArgumentError(
        "positivity=true currently reports quantile-clamp diagnostics and is unavailable for explicit bounded policies",
    ))
    imputation !== nothing && throw(ArgumentError(
        "explicit bounded LMTP policies are not yet supported with multiple imputation",
    ))
    return nothing
end

function _policy_validate_grid_options(
    ::Nothing,
    deltas;
    density_ratio::Symbol,
    estimator::Symbol,
    family_outcome::Symbol,
    cv_trunc::Bool,
    weight_truncation::Bool,
    positivity::Bool,
    imputation,
)
    if !weight_truncation && any(d -> !isapprox(d, 0; atol = 1e-12), deltas)
        throw(ArgumentError(
            "weight_truncation=false is not supported for nonzero quantile-clamped LMTP grids",
        ))
    end
    return nothing
end

function _validate_fixed_bound_grid_options(;
    density_ratio::Symbol,
    cv_trunc::Bool,
    weight_truncation::Bool,
    positivity::Bool,
    imputation,
)
    _validate_explicit_policy_grid_common(; positivity, imputation)
    !weight_truncation && density_ratio !== :gaussian && throw(ArgumentError(
        "EIF intervals for explicit bounded policies require density_ratio=:gaussian",
    ))
    !weight_truncation && cv_trunc && throw(ArgumentError(
        "uncapped EIF inference does not support cv_trunc=true",
    ))
    return nothing
end

function _policy_validate_grid_options(
    ::SmoothBoundedLogitShiftPolicy,
    deltas;
    density_ratio::Symbol,
    estimator::Symbol,
    family_outcome::Symbol,
    cv_trunc::Bool,
    weight_truncation::Bool,
    positivity::Bool,
    imputation,
)
    _validate_fixed_bound_grid_options(; density_ratio, cv_trunc, weight_truncation, positivity, imputation)
    return nothing
end

function _policy_validate_grid_options(
    ::FeasibleAdditiveShiftPolicy,
    deltas;
    density_ratio::Symbol,
    estimator::Symbol,
    family_outcome::Symbol,
    cv_trunc::Bool,
    weight_truncation::Bool,
    positivity::Bool,
    imputation,
)
    _validate_fixed_bound_grid_options(; density_ratio, cv_trunc, weight_truncation, positivity, imputation)
    density_ratio !== :gaussian && throw(ArgumentError(
        "feasible additive LMTP currently uses density_ratio=:gaussian",
    ))
    return nothing
end

function _policy_validate_grid_options(
    ::ProportionalHeadroomPolicy,
    deltas;
    density_ratio::Symbol,
    estimator::Symbol,
    family_outcome::Symbol,
    cv_trunc::Bool,
    weight_truncation::Bool,
    positivity::Bool,
    imputation,
)
    _validate_fixed_bound_grid_options(; density_ratio, cv_trunc, weight_truncation, positivity, imputation)
    density_ratio !== :gaussian && throw(ArgumentError(
        "proportional headroom LMTP currently uses density_ratio=:gaussian",
    ))
    foreach(_validate_headroom_delta, deltas)
    return nothing
end
