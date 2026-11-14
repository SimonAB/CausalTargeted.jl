"""Separate probability-coordinate quadrature for a two-arm logistic posterior.

This shares QuadGK, but no posterior kernels or tail helpers, with the coefficient
reference. Nested error estimates are practical diagnostics, not rigorous bounds.
"""
module PolicyProbabilityReference

using Distributions: MvNormal, logpdf
using LinearAlgebra
using QuadGK: quadgk

export probability_reference, probability_cdf, probability_survival,
       probability_quantile, probability_tail_details

@inline logistic(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
@inline logit(p) = log(p) - log1p(-p)

struct ProbabilityReference{D<:MvNormal}
    counts::NTuple{4,Int}
    prior::D
    η_mean::Vector{Float64}
    η_covariance::Matrix{Float64}
    split::Float64
    log_peak::Float64
    mass::Float64
    mass_error::Float64
    rtol::Float64
    atol::Float64
end

function log_density(prior, counts, p₀, p₁)
    # A proper Gaussian in logits dominates every polynomial endpoint factor.
    0 < p₀ < 1 && 0 < p₁ < 1 || return -Inf
    n₀, y₀, n₁, y₁ = counts
    η₀, η₁ = logit(p₀), logit(p₁)
    # Binomial coefficients are constant in the probabilities and cancel from
    # every normalised result. Counts of zero deliberately contribute nothing.
    return logpdf(prior, [η₀, η₁ - η₀]) +
        (y₀ - 1) * log(p₀) + (n₀ - y₀ - 1) * log1p(-p₀) +
        (y₁ - 1) * log(p₁) + (n₁ - y₁ - 1) * log1p(-p₁)
end

"""Choose a finite density rescaling and a count-informed integration split.

The Gaussian-prior count likelihood in arm logits is strictly log-concave.
Its Newton solution supplies a useful reference density, not a claimed maximum
of the transformed density. This search never truncates the integration domain.
"""
function density_reference(prior, counts, μ, Σ)
    n₀, y₀, n₁, y₁ = counts
    n, y, W = [n₀, n₁], [y₀, y₁], inv(Σ)
    η = copy(μ)
    for _ in 1:100
        p = logistic.(η)
        gradient = y - n .* p - W * (η - μ)
        direction = (W + Diagonal(n .* p .* (1 .- p))) \ gradient
        norm(direction) < 1e-10 && break
        # Damping bounds each step in logits; strict concavity makes the
        # stationary point unique, including under complete separation.
        η += direction / max(1.0, norm(direction))
    end
    p = logistic.(η)
    scale = log_density(prior, counts, p...)
    isfinite(scale) || error("probability reference could not choose a finite density scale")
    return p[1], scale
end

"""Construct a full-unit-square reference from arm totals and success counts."""
function probability_reference(control_n, control_successes, treated_n,
        treated_successes, mean::AbstractVector, covariance::AbstractMatrix;
        rtol = 1e-7, atol = 1e-10)
    raw = (control_n, control_successes, treated_n, treated_successes)
    all(x -> x isa Real && isfinite(x) && x >= 0 && isinteger(x), raw) ||
        throw(ArgumentError("arm counts must be nonnegative finite integers"))
    counts = Int.(raw)
    counts[2] <= counts[1] && counts[4] <= counts[3] ||
        throw(ArgumentError("successes cannot exceed arm totals"))
    length(mean) == 2 && size(covariance) == (2, 2) ||
        throw(DimensionMismatch("two coefficients and a 2 by 2 covariance required"))
    all(isfinite, mean) && all(isfinite, covariance) && issymmetric(covariance) &&
        isposdef(Symmetric(covariance)) || throw(ArgumentError("proper finite Gaussian prior required"))
    0 < rtol < 1 && 0 < atol < 1 || throw(ArgumentError("tolerances must lie in (0,1)"))
    μ, V = Float64.(mean), Matrix{Float64}(covariance)
    prior = MvNormal(μ, Symmetric(V))
    A = [1.0 0.0; 1.0 1.0]
    η_mean, η_covariance = A * μ, A * V * A'
    split, log_peak = density_reference(prior, counts, η_mean, η_covariance)
    provisional = ProbabilityReference(counts, prior, η_mean, η_covariance,
        split, log_peak, 1.0, 0.0, rtol, atol)
    detail = probability_integral(provisional, Inf)
    isfinite(detail.value) && detail.value > 0 || error("probability reference normalisation failed")
    return ProbabilityReference(counts, prior, η_mean, η_covariance,
        split, log_peak, detail.value, detail.error, rtol, atol)
end

function conditional_split(r, p₀)
    Σ = r.η_covariance
    μ = r.η_mean[2] + Σ[2, 1] / Σ[1, 1] * (logit(p₀) - r.η_mean[1])
    variance = Σ[2, 2] - Σ[2, 1]^2 / Σ[1, 1]
    n, y = r.counts[3], r.counts[4]
    # The conditional logit score is monotone, with this exact finite bracket.
    left, right = μ + variance * (y - n), μ + variance * y
    for _ in 1:45
        middle = (left + right) / 2
        y - n * logistic(middle) - (middle - μ) / variance > 0 ?
            (left = middle) : (right = middle)
    end
    return logistic((left + right) / 2)
end

function inner_probability_integral(r::ProbabilityReference, p₀::Float64,
        lower::Float64, upper::Float64)::Tuple{Float64,Float64}
    lower == upper && return (0.0, 0.0)
    split = conditional_split(r, p₀)
    boundaries = [lower, upper]
    lower < split < upper && insert!(boundaries, 2, split)
    return quadgk(p₁ -> exp(log_density(r.prior, r.counts, p₀, p₁) - r.log_peak),
        boundaries...; rtol = r.rtol, atol = r.atol)
end

function probability_integral(r::ProbabilityReference, u::Float64; upper_tail = false)
    function outer(p₀)
        0 < p₀ < 1 || return zeros(2)
        boundary = clamp(p₀ + u, 0.0, 1.0)
        lower, upper = upper_tail ? (boundary, 1.0) : (0.0, boundary)
        value, error = inner_probability_integral(r, p₀, lower, upper)
        return [value, error]
    end
    boundaries = [0.0, r.split, 1.0]
    # The event changes from empty/full to a finite sloping boundary here.
    -1 < u < 0 && push!(boundaries, -u)
    0 < u < 1 && push!(boundaries, 1 - u)
    value, outer_error = quadgk(outer, sort!(unique!(boundaries))...;
        rtol = r.rtol, atol = r.atol, norm = x -> abs(x[1]))
    inner_error = abs(value[2])
    return (; value = value[1], error = outer_error + inner_error, outer_error, inner_error)
end

"""Integrate a tail directly, retaining inner, outer and normalisation estimates."""
function probability_tail_details(r, u::Real; upper_tail = false)
    isnan(u) && throw(ArgumentError("contrast must not be NaN"))
    if u <= -1 || u >= 1
        probability = Float64(upper_tail ? u <= -1 : u >= 1)
        return (; probability, error = 0.0, outer_error = 0.0, inner_error = 0.0, normalisation_error = 0.0)
    end
    detail = probability_integral(r, Float64(u); upper_tail)
    probability = detail.value / r.mass
    normalisation_error = abs(probability) * r.mass_error / r.mass
    return (; probability, error = detail.error / r.mass + normalisation_error,
        outer_error = detail.outer_error / r.mass, inner_error = detail.inner_error / r.mass,
        normalisation_error)
end

"""CDF of the arm contrast p₁-p₀, evaluated by direct lower-tail quadrature."""
probability_cdf(r, u::Real) = probability_tail_details(r, u).probability

"""Survival probability, integrated independently rather than subtracted from one."""
probability_survival(r, u::Real) = probability_tail_details(r, u; upper_tail = true).probability

"""Invert the CDF, checking both contrast bracket and probability residual."""
function probability_quantile(r, p::Real; probability_tolerance = max(1e-8, 10r.rtol),
        contrast_tolerance = 1e-7)
    0 < p < 1 || throw(ArgumentError("quantile probability must lie in (0,1)"))
    left, right = -1.0, 1.0
    for _ in 1:100
        middle = (left + right) / 2
        value = p <= 0.5 ? probability_cdf(r, middle) : 1 - probability_survival(r, middle)
        right - left <= contrast_tolerance && abs(value - p) <= probability_tolerance && return middle
        value < p ? (left = middle) : (right = middle)
    end
    error("probability-space quantile inversion did not converge")
end

end # module
