"""
Independent low-dimensional logistic posterior integration for tail validation.

QuadGK integrates the original Bernoulli likelihood times a Gaussian prior on
an infinite domain. Neither variational moments nor Hermite nodes are used.
Scalar coefficients and a two-coefficient static policy contrast are supported.
Normalising mass, CDFs and quantiles have their own integration/refinement checks;
agreement of posterior moments alone never certifies a tail probability.
"""
module LogisticTailReference

using QuadGK: quadgk
using LinearAlgebra
using Distributions: Normal, cdf

export adaptive_reference, posterior_cdf, posterior_quantile, posterior_tails,
       interval_content, logistic_probability, bivariate_policy_reference,
       bivariate_moments, coefficient_cdf, coefficient_quantile, coefficient_tails,
       coefficient_interval_content, refine_bivariate_reference,
       policy_cdf, policy_quantile, policy_tails

@inline logistic_probability(x::Real) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
@inline log_logistic(x::Real) = x >= 0 ? -log1p(exp(-x)) : x - log1p(exp(x))

"""Scalar Gaussian-prior logistic posterior, standardised by its specified prior."""
struct ScalarLogisticReference
    x::Vector{Float64}
    y::Vector{Float64}
    prior_mean::Float64
    prior_sd::Float64
    mode::Float64
    log_peak::Float64
    mass::Float64
    mass_error::Float64
    mean::Float64
    variance::Float64
    rtol::Float64
    atol::Float64
end

function log_kernel(z, x, y, μ, σ)
    isfinite(z) || return -Inf
    β = μ + σ * z
    result = -z^2 / 2
    for i in eachindex(x, y)
        # A zero design has constant likelihood, including at extreme β.
        x[i] == 0 && (result -= log(2); continue)
        result += log_logistic(y[i] == 1 ? x[i] * β : -x[i] * β)
    end
    return result
end

function integrate_kernel(f, density, mode, lower, upper; rtol, atol)
    lower == upper && return (0.0, 0.0)
    boundaries = lower < mode < upper ? (lower, mode, upper) : (lower, upper)
    # quadgk never evaluates infinite endpoints, but explicit zero avoids 0*Inf
    # if an implementation subsequently changes its endpoint policy.
    return quadgk(z -> isfinite(z) ? f(z) * density(z) : 0.0, boundaries...; rtol, atol)
end

"""Construct an adaptive reference with independently integrated first two moments."""
function adaptive_reference(x::AbstractVector, y::AbstractVector, μ::Real, variance::Real;
        rtol = 1e-10, atol = 1e-12)
    length(x) == length(y) || throw(DimensionMismatch("design and outcome lengths"))
    all(isfinite, x) && all(yi -> yi == 0 || yi == 1, y) ||
        throw(ArgumentError("finite design and binary observed outcomes required"))
    isfinite(μ) && isfinite(variance) && variance > 0 || throw(ArgumentError("proper finite scalar Gaussian prior required"))
    0 < rtol < 1 && 0 < atol < 1 || throw(ArgumentError("integration tolerances must lie in (0,1)"))
    x, y, μ, σ = Float64.(x), Float64.(y), Float64(μ), sqrt(Float64(variance))
    score(z) = -z + σ * sum(xi * (yi - logistic_probability(xi * (μ + σ * z))) for (xi, yi) in zip(x, y); init = 0.0)
    left, right = -1.0, 1.0
    while score(left) < 0; left *= 2; end
    while score(right) > 0; right *= 2; end
    for _ in 1:100
        middle = (left + right) / 2
        score(middle) > 0 ? (left = middle) : (right = middle)
    end
    mode = (left + right) / 2
    log_peak = log_kernel(mode, x, y, μ, σ)
    density(z) = exp(log_kernel(z, x, y, μ, σ) - log_peak)
    mass, mass_error = integrate_kernel(one, density, mode, -Inf, Inf; rtol, atol)
    mass > 0 && isfinite(mass) || error("adaptive posterior normalisation failed")
    first, _ = integrate_kernel(identity, density, mode, -Inf, Inf; rtol, atol)
    zmean = first / mass
    second, _ = integrate_kernel(z -> (z - zmean)^2, density, mode, -Inf, Inf; rtol, atol)
    return ScalarLogisticReference(x, y, μ, σ, mode, log_peak, mass, mass_error,
        μ + σ * zmean, σ^2 * second / mass, rtol, atol)
end

function tail_integral(reference, β, upper::Bool)
    r = reference
    z = (β - r.prior_mean) / r.prior_sd
    density(z) = exp(log_kernel(z, r.x, r.y, r.prior_mean, r.prior_sd) - r.log_peak)
    lower, upper_bound = upper ? (z, Inf) : (-Inf, z)
    value, error = integrate_kernel(one, density, r.mode, lower, upper_bound; rtol = r.rtol, atol = r.atol)
    probability = value / r.mass
    # Propagate QuadGK's absolute error estimates, including normalisation.
    probability_error = error / r.mass + abs(probability) * r.mass_error / r.mass
    return (; probability, error = probability_error)
end

"""Integrate either tail directly; use the complementary tail to avoid cancellation."""
function posterior_cdf(reference, β::Real)
    isnan(β) && throw(ArgumentError("CDF argument must not be NaN"))
    β == -Inf && return 0.0
    β == Inf && return 1.0
    z = (β - reference.prior_mean) / reference.prior_sd
    z <= reference.mode && return tail_integral(reference, β, false).probability
    return 1 - tail_integral(reference, β, true).probability
end

"""Invert the adaptive CDF; termination checks both bracket width and CDF residual."""
function posterior_quantile(reference, probability::Real; probability_tolerance = 1e-9,
        coefficient_tolerance = 1e-8)
    0 < probability < 1 || throw(ArgumentError("quantile probability must lie in (0,1)"))
    σ, μ = reference.prior_sd, reference.prior_mean
    left, right = μ - σ, μ + σ
    while posterior_cdf(reference, left) > probability; left = μ - 2 * (μ - left); end
    while posterior_cdf(reference, right) < probability; right = μ + 2 * (right - μ); end
    for _ in 1:200
        middle = (left + right) / 2
        value = posterior_cdf(reference, middle)
        if right - left <= coefficient_tolerance * max(1, σ) &&
                abs(value - probability) <= probability_tolerance
            return middle
        end
        value < probability ? (left = middle) : (right = middle)
    end
    error("adaptive posterior quantile inversion did not converge")
end

"""Check each endpoint directly, including upper-tail mass and normalisation error."""
function posterior_tails(reference; level = 0.95)
    0 < level < 1 || throw(ArgumentError("level must lie in (0,1)"))
    probability = (1 - level) / 2
    lower = posterior_quantile(reference, probability)
    upper = posterior_quantile(reference, 1 - probability)
    lower_tail = tail_integral(reference, lower, false)
    upper_tail = tail_integral(reference, upper, true)
    return (; lower, upper, lower_tail, upper_tail,
        maximum_tail_residual = max(abs(lower_tail.probability - probability),
            abs(upper_tail.probability - probability)),
        maximum_integration_error = max(lower_tail.error, upper_tail.error))
end

"""Exact-reference content and tail imbalance of a proposed interval."""
function interval_content(reference, lower, upper)
    lower <= upper || throw(ArgumentError("interval endpoints must be ordered"))
    left = tail_integral(reference, lower, false).probability
    right = tail_integral(reference, upper, true).probability
    return (; content = 1 - left - right, lower_tail = left, upper_tail = right)
end

"""
Two-coefficient posterior for the fixed-cohort policy contrast
`logistic(β₀ + β₁) - logistic(β₀)` (baseline A=0, policy A=1).

Integration conditions on β₀, with β₁ integrated up to the analytic policy
boundary. This is a continuous nested adaptive rule, independent of Hermite
nodes, variational fits and public posterior draws.
"""
struct BivariatePolicyReference
    X::Matrix{Float64}
    y::Vector{Float64}
    μ::Vector{Float64}
    L::Matrix{Float64}
    mode::Vector{Float64}
    log_peak::Float64
    mass::Float64
    mass_error::Float64
    rtol::Float64
    atol::Float64
end

function bivariate_log_kernel(z, X, y, μ, L)
    all(isfinite, z) || return -Inf
    β = μ + L * z
    result = -dot(z, z) / 2
    for i in eachindex(y)
        η = dot(view(X, i, :), β)
        result += log_logistic(y[i] == 1 ? η : -η)
    end
    return result
end

function bivariate_policy_reference(X::AbstractMatrix, y::AbstractVector,
        μ::AbstractVector, V::AbstractMatrix; rtol = 1e-7, atol = 1e-10)
    size(X) == (length(y), 2) && length(μ) == 2 && size(V) == (2, 2) ||
        throw(DimensionMismatch("policy reference requires two logistic coefficients"))
    all(isfinite, X) && all(isfinite, μ) && all(isfinite, V) &&
        all(yi -> yi == 0 || yi == 1, y) || throw(ArgumentError("finite design, prior and binary outcomes required"))
    issymmetric(V) && isposdef(Symmetric(V)) || throw(ArgumentError("proper Gaussian covariance required"))
    0 < rtol < 1 && 0 < atol < 1 || throw(ArgumentError("integration tolerances must lie in (0,1)"))
    X, y, μ = Matrix{Float64}(X), Float64.(y), Float64.(μ)
    L = Matrix(cholesky(Symmetric(Matrix{Float64}(V))).L)
    # A Newton mode of the original log-concave posterior supplies a numerical
    # rescaling only. Integration does not replace the density by its Hessian.
    Z = X * L
    z = zeros(2)
    for _ in 1:100
        probabilities = logistic_probability.(X * μ + Z * z)
        score = Z' * (y - probabilities) - z
        information = Matrix{Float64}(I, 2, 2) + Z' * ((probabilities .* (1 .- probabilities)) .* Z)
        direction = information \ score
        norm(direction) <= 1e-11 && break
        previous = bivariate_log_kernel(z, X, y, μ, L)
        scale = 1.0
        while bivariate_log_kernel(z + scale * direction, X, y, μ, L) < previous
            scale /= 2
            scale < 1e-10 && error("reference mode line search failed")
        end
        z += scale * direction
    end
    log_peak = bivariate_log_kernel(z, X, y, μ, L)
    unnormalised = BivariatePolicyReference(X, y, μ, L, z, log_peak, 1.0, 0.0, rtol, atol)
    mass, mass_error = policy_integral(unnormalised, Inf; upper_tail = false)
    mass > 0 && isfinite(mass) || error("bivariate posterior normalisation failed")
    return BivariatePolicyReference(X, y, μ, L, z, log_peak, mass, mass_error, rtol, atol)
end

function policy_integral(r::BivariatePolicyReference, contrast; upper_tail = false)
    # β₀=μ₀+L₀₀z₀; β₁|z₀=μ₁+L₁₀z₀+L₁₁z₁.
    # Outer integrand is mass only; nested absolute-error estimates are not smooth
    # and must not enter the outer adaptive rule.
    function outer_integrand(z₀)
        isfinite(z₀) || return 0.0
        β₀ = r.μ[1] + r.L[1, 1] * z₀
        probability_boundary = contrast + logistic_probability(β₀)
        boundary = probability_boundary <= 0 ? -Inf : probability_boundary >= 1 ? Inf :
            (log(probability_boundary) - log1p(-probability_boundary) - β₀ - r.μ[2] - r.L[2, 1] * z₀) / r.L[2, 2]
        lower, upper = upper_tail ? (boundary, Inf) : (-Inf, boundary)
        lower == upper && return 0.0
        if isempty(r.y)
            probability = upper_tail ? cdf(Normal(), -boundary) : cdf(Normal(), boundary)
            return exp(-z₀^2 / 2) / sqrt(2π) * probability
        end
        function density(z₁)
            isfinite(z₁) || return 0.0
            β₁ = r.μ[2] + r.L[2, 1] * z₀ + r.L[2, 2] * z₁
            log_density = -(z₀^2 + z₁^2) / 2 - r.log_peak
            for i in eachindex(r.y)
                η = r.X[i, 1] * β₀ + r.X[i, 2] * β₁
                log_density += log_logistic(r.y[i] == 1 ? η : -η)
            end
            return exp(log_density)
        end
        value, _ = integrate_kernel(one, density, r.mode[2], lower, upper; rtol = r.rtol, atol = r.atol)
        return value
    end
    boundaries = [-Inf, r.mode[1], Inf]
    # Split at the point where the conditional event changes from empty/full
    # to a finite logit boundary; otherwise an adaptive rule may skip the kink.
    if -1 < contrast < 0
        push!(boundaries, (log(-contrast) - log1p(contrast) - r.μ[1]) / r.L[1, 1])
    elseif 0 < contrast < 1
        push!(boundaries, (log1p(-contrast) - log(contrast) - r.μ[1]) / r.L[1, 1])
    end
    result, error = quadgk(outer_integrand, sort!(unique!(boundaries))...; rtol = r.rtol, atol = r.atol)
    # QuadGK estimates are not formal interval bounds; refine independently
    # before treating a quantile as checked.
    return result, error
end

function policy_tail_details(r, contrast; upper_tail = false)
    value, error = policy_integral(r, contrast; upper_tail)
    probability = value / r.mass
    return (; probability, error = error / r.mass + abs(probability) * r.mass_error / r.mass)
end

"""Adaptive posterior CDF of logistic(β₀+β₁)-logistic(β₀)."""
function policy_cdf(reference::BivariatePolicyReference, contrast::Real)
    contrast <= -1 && return 0.0
    contrast >= 1 && return 1.0
    isnan(contrast) && throw(ArgumentError("CDF argument must not be NaN"))
    return contrast <= 0 ? policy_tail_details(reference, contrast).probability :
        1 - policy_tail_details(reference, contrast; upper_tail = true).probability
end

function policy_quantile(reference::BivariatePolicyReference, probability::Real;
        probability_tolerance = max(1e-8, 10reference.rtol), contrast_tolerance = 1e-7)
    0 < probability < 1 || throw(ArgumentError("quantile probability must lie in (0,1)"))
    left, right = -1.0, 1.0
    for _ in 1:100
        middle = (left + right) / 2
        value = policy_cdf(reference, middle)
        if right - left <= contrast_tolerance && abs(value - probability) <= probability_tolerance
            return middle
        end
        value < probability ? (left = middle) : (right = middle)
    end
    error("adaptive policy quantile inversion did not converge")
end

function policy_tails(reference::BivariatePolicyReference; level = 0.95)
    0 < level < 1 || throw(ArgumentError("level must lie in (0,1)"))
    probability = (1 - level) / 2
    lower = policy_quantile(reference, probability)
    upper = policy_quantile(reference, 1 - probability)
    lower_tail = policy_tail_details(reference, lower)
    upper_tail = policy_tail_details(reference, upper; upper_tail = true)
    return (; lower, upper, lower_tail, upper_tail,
        maximum_tail_residual = max(abs(lower_tail.probability - probability), abs(upper_tail.probability - probability)),
        maximum_integration_error = max(lower_tail.error, upper_tail.error))
end

"""Nested adaptive moments of the two-coefficient posterior; independent of Hermite grids."""
function bivariate_moments(r::BivariatePolicyReference)
    function outer(z₀)
        isfinite(z₀) || return zeros(6)
        β₀ = r.μ[1] + r.L[1, 1] * z₀
        function moments(z₁)
            isfinite(z₁) || return zeros(6)
            w = exp(bivariate_log_kernel([z₀, z₁], r.X, r.y, r.μ, r.L) - r.log_peak)
            β₁ = r.μ[2] + r.L[2, 1] * z₀ + r.L[2, 2] * z₁
            return w .* [1.0, β₀, β₁, β₀^2, β₀ * β₁, β₁^2]
        end
        value, _ = quadgk(moments, -Inf, r.mode[2], Inf; rtol = r.rtol, atol = r.atol)
        return value
    end
    raw, outer_error = quadgk(outer, -Inf, r.mode[1], Inf; rtol = r.rtol, atol = r.atol)
    mass = raw[1]
    mass > 0 && isfinite(mass) || error("bivariate moment normalisation failed")
    μ = [raw[2], raw[3]] ./ mass
    second = [raw[4] raw[5]; raw[5] raw[6]] ./ mass
    Σ = Symmetric(second - μ * μ')
    # Absolute mass disagreement versus the policy-normalising integral is retained
    # as a diagnostic; moment claims still require independent tolerance refinement.
    return (; mean = μ, covariance = Matrix(Σ), mass, mass_disagreement = abs(mass - r.mass) / r.mass,
        integration_error = outer_error)
end

"""Integrate one coefficient's marginal tail under the bivariate posterior."""
function coefficient_tail_details(r::BivariatePolicyReference, index::Integer, threshold::Real;
        upper_tail = false)
    index in (1, 2) || throw(ArgumentError("coefficient index must be 1 or 2"))
    function inner_mass(z₀, lower, upper)
        isfinite(z₀) || return 0.0
        lower == upper && return 0.0
        function density(z₁)
            isfinite(z₁) || return 0.0
            return exp(bivariate_log_kernel([z₀, z₁], r.X, r.y, r.μ, r.L) - r.log_peak)
        end
        value, _ = integrate_kernel(one, density, r.mode[2], lower, upper; rtol = r.rtol, atol = r.atol)
        return value
    end
    if index == 1
        # β₀ = μ₀ + L₁₁ z₀, so the event is a half-line in z₀ only.
        split = (threshold - r.μ[1]) / r.L[1, 1]
        lo, hi = upper_tail ? (split, Inf) : (-Inf, split)
        splits = filter(z -> lo < z < hi, float.([r.mode[1]]))
        result, error = quadgk(z₀ -> inner_mass(z₀, -Inf, Inf), lo, splits..., hi;
            rtol = r.rtol, atol = r.atol)
    else
        function outer(z₀)
            boundary = (threshold - r.μ[2] - r.L[2, 1] * z₀) / r.L[2, 2]
            lower, upper = upper_tail ? (boundary, Inf) : (-Inf, boundary)
            return inner_mass(z₀, lower, upper)
        end
        result, error = quadgk(outer, -Inf, r.mode[1], Inf; rtol = r.rtol, atol = r.atol)
    end
    probability = result / r.mass
    return (; probability, error = error / r.mass + abs(probability) * r.mass_error / r.mass)
end

function coefficient_cdf(reference::BivariatePolicyReference, index::Integer, value::Real)
    isnan(value) && throw(ArgumentError("CDF argument must not be NaN"))
    value == -Inf && return 0.0
    value == Inf && return 1.0
    detail = coefficient_tail_details(reference, index, value)
    # Prefer the lighter tail to limit cancellation.
    detail.probability <= 0.5 && return detail.probability
    return 1 - coefficient_tail_details(reference, index, value; upper_tail = true).probability
end

function coefficient_quantile(reference::BivariatePolicyReference, index::Integer, probability::Real;
        probability_tolerance = max(1e-6, 20reference.rtol),
        coefficient_tolerance = max(1e-4, 50reference.rtol))
    0 < probability < 1 || throw(ArgumentError("quantile probability must lie in (0,1)"))
    σ = sqrt(reference.L[index, index]^2 + (index == 2 ? reference.L[2, 1]^2 : 0.0))
    μ = reference.μ[index]
    left, right = μ - σ, μ + σ
    while coefficient_cdf(reference, index, left) > probability; left = μ - 2 * (μ - left); end
    while coefficient_cdf(reference, index, right) < probability; right = μ + 2 * (right - μ); end
    for _ in 1:200
        middle = (left + right) / 2
        value = coefficient_cdf(reference, index, middle)
        if right - left <= coefficient_tolerance * max(1, σ) &&
                abs(value - probability) <= probability_tolerance
            return middle
        end
        value < probability ? (left = middle) : (right = middle)
    end
    error("adaptive coefficient quantile inversion did not converge")
end

function coefficient_tails(reference::BivariatePolicyReference, index::Integer; level = 0.95)
    0 < level < 1 || throw(ArgumentError("level must lie in (0,1)"))
    probability = (1 - level) / 2
    lower = coefficient_quantile(reference, index, probability)
    upper = coefficient_quantile(reference, index, 1 - probability)
    lower_tail = coefficient_tail_details(reference, index, lower)
    upper_tail = coefficient_tail_details(reference, index, upper; upper_tail = true)
    return (; lower, upper, lower_tail, upper_tail,
        maximum_tail_residual = max(abs(lower_tail.probability - probability),
            abs(upper_tail.probability - probability)),
        maximum_integration_error = max(lower_tail.error, upper_tail.error))
end

function coefficient_interval_content(reference::BivariatePolicyReference, index::Integer,
        lower::Real, upper::Real)
    lower <= upper || throw(ArgumentError("interval endpoints must be ordered"))
    left = coefficient_tail_details(reference, index, lower).probability
    right = coefficient_tail_details(reference, index, upper; upper_tail = true).probability
    return (; content = 1 - left - right, lower_tail = left, upper_tail = right)
end

"""
Independently refine moments, interval content and endpoints.

Moment agreement alone never marks content or endpoints as validated. Endpoints
are inverted once under the working rule, then checked by evaluating the tighter
CDF at those points (residual and integration-error gates). Content of probe
intervals is compared under both rules. Failures are listed explicitly.
"""
function refine_bivariate_reference(X, y, μ, V;
        rtol = 1e-5, atol = 1e-8, tight_rtol = 1e-7, tight_atol = 1e-10,
        moment_mean_tol = 5e-4, moment_cov_tol = 5e-4,
        content_tol = 5e-4, endpoint_cdf_tol = 5e-4,
        residual_tol = 5e-4, integration_tol = 5e-4,
        check_policy = true, probe_intervals = nothing)
    working = bivariate_policy_reference(X, y, μ, V; rtol, atol)
    tight = bivariate_policy_reference(X, y, μ, V; rtol = tight_rtol, atol = tight_atol)
    working_moments = bivariate_moments(working)
    tight_moments = bivariate_moments(tight)
    mean_change = maximum(abs.(working_moments.mean - tight_moments.mean) ./
        max.(sqrt.(diag(tight_moments.covariance)), 1e-8))
    cov_change = norm(working_moments.covariance - tight_moments.covariance) /
        max(norm(tight_moments.covariance), 1e-12)
    moments_converged = mean_change <= moment_mean_tol && cov_change <= moment_cov_tol &&
        working_moments.mass_disagreement <= 1e-4 && tight_moments.mass_disagreement <= 1e-4
    coefficient_reports = NamedTuple[]
    content_converged = true
    endpoints_converged = true
    for index in 1:2
        working_tails = coefficient_tails(working, index)
        # Validate working endpoints against the tighter CDF instead of a second inversion.
        lower_detail = coefficient_tail_details(tight, index, working_tails.lower)
        upper_detail = coefficient_tail_details(tight, index, working_tails.upper; upper_tail = true)
        lower_residual = abs(lower_detail.probability - 0.025)
        upper_residual = abs(upper_detail.probability - 0.025)
        endpoint_ok = lower_residual <= endpoint_cdf_tol && upper_residual <= endpoint_cdf_tol &&
            max(lower_detail.error, upper_detail.error) <= integration_tol &&
            working_tails.maximum_tail_residual <= residual_tol
        endpoints_converged &= endpoint_ok
        probes = probe_intervals === nothing ?
            [(working_tails.lower, working_tails.upper)] : probe_intervals[index]
        contents = NamedTuple[]
        for (lower, upper) in probes
            coarse = coefficient_interval_content(working, index, lower, upper)
            fine = coefficient_interval_content(tight, index, lower, upper)
            change = abs(coarse.content - fine.content)
            push!(contents, (; lower, upper, content = fine.content,
                lower_tail = fine.lower_tail, upper_tail = fine.upper_tail, content_change = change))
            content_converged &= change <= content_tol
        end
        push!(coefficient_reports, (; index,
            reference_interval = (working_tails.lower, working_tails.upper),
            lower_cdf_residual = lower_residual, upper_cdf_residual = upper_residual,
            maximum_integration_error = max(lower_detail.error, upper_detail.error),
            endpoints_converged = endpoint_ok, contents))
    end
    policy = nothing
    if check_policy
        policy_working = policy_tails(working)
        lower_detail = policy_tail_details(tight, policy_working.lower)
        upper_detail = policy_tail_details(tight, policy_working.upper; upper_tail = true)
        lower_residual = abs(lower_detail.probability - 0.025)
        upper_residual = abs(upper_detail.probability - 0.025)
        policy_endpoints_converged = lower_residual <= endpoint_cdf_tol &&
            upper_residual <= endpoint_cdf_tol &&
            max(lower_detail.error, upper_detail.error) <= integration_tol &&
            policy_working.maximum_tail_residual <= residual_tol
        endpoints_converged &= policy_endpoints_converged
        coarse_content = policy_cdf(working, policy_working.upper) -
            policy_cdf(working, policy_working.lower)
        fine_content = policy_cdf(tight, policy_working.upper) -
            policy_cdf(tight, policy_working.lower)
        policy_content_change = abs(coarse_content - fine_content)
        content_converged &= policy_content_change <= content_tol
        policy = (; reference_interval = (policy_working.lower, policy_working.upper),
            lower_cdf_residual = lower_residual, upper_cdf_residual = upper_residual,
            maximum_integration_error = max(lower_detail.error, upper_detail.error),
            endpoints_converged = policy_endpoints_converged,
            content = fine_content, content_change = policy_content_change)
    end
    failures = String[]
    moments_converged || push!(failures, "moments")
    content_converged || push!(failures, "interval_content")
    endpoints_converged || push!(failures, "endpoints")
    return (; working, tight, moments = tight_moments,
        moments_converged, content_converged, endpoints_converged,
        reference_converged = moments_converged && content_converged && endpoints_converged,
        reference_failures = failures,
        mean_change, covariance_change = cov_change,
        coefficients = coefficient_reports, policy)
end

end # module
