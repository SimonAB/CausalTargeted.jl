"""
Independent low-dimensional logistic posterior reference for tests and stress runs.

The integration rule is a product Gauss–Hermite rule under the *prior*, without
using a variational fit, its covariance, or its update equations. FastGaussQuadrature
is a test extra, not a runtime dependency. Only one or two coefficients are supported.
"""
module LogisticPosteriorReference

using FastGaussQuadrature: gausshermite
using LinearAlgebra
using Statistics

export prior_grid, logistic_reference, converged_reference, weighted_quantile,
       policy_reference, gaussian_policy_summary, logistic_probability

@inline logistic_probability(x::Real) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
@inline log_logistic(x::Real) = x >= 0 ? -log1p(exp(-x)) : x - log1p(exp(x))

"""Prior-transformed Hermite nodes and normalised prior log weights."""
function prior_grid(μ::AbstractVector, V::AbstractMatrix; points::Int = 160)
    p = length(μ)
    p in (1, 2) || throw(ArgumentError("the independent reference supports p=1 or p=2"))
    size(V) == (p, p) || throw(DimensionMismatch("prior covariance dimensions"))
    points >= 2 || throw(ArgumentError("at least two Hermite points are required"))
    L = cholesky(Symmetric(Matrix{Float64}(V))).L
    nodes, weights = gausshermite(points)
    standard_nodes = Matrix{Float64}(undef, p, points^p)
    if p == 1
        standard_nodes[1, :] .= nodes
    else
        column = 0
        for j in eachindex(nodes), i in eachindex(nodes)
            column += 1
            standard_nodes[1, column] = nodes[i]
            standard_nodes[2, column] = nodes[j]
        end
    end
    log_weights = p == 1 ? log.(weights) .- log(π) / 2 :
        [log(weights[i]) + log(weights[j]) - log(π) for j in eachindex(nodes) for i in eachindex(nodes)]
    return (; coefficients = μ .+ sqrt(2) .* (L * standard_nodes), log_weights, points)
end

"""Normalise log weights without exponentiating the unnormalised likelihood."""
function normalise_log_weights(log_weights)
    largest = maximum(log_weights)
    isfinite(largest) || throw(ArgumentError("all posterior quadrature weights vanished"))
    weights = exp.(log_weights .- largest)
    total = sum(weights)
    return weights ./ total, largest + log(total)
end

"""Integrate posterior moments using a Gaussian-prior Hermite product rule."""
function logistic_reference(X::AbstractMatrix, y::AbstractVector, μ::AbstractVector,
                            V::AbstractMatrix; points::Int = 160, grid = nothing)
    size(X, 1) == length(y) || throw(DimensionMismatch("design and response lengths"))
    size(X, 2) == length(μ) || throw(DimensionMismatch("design and prior dimensions"))
    all(yi -> yi == 0 || yi == 1, y) || throw(ArgumentError("responses must be binary"))
    integration = isnothing(grid) ? prior_grid(μ, V; points) : grid
    coefficients = integration.coefficients
    predictors = X * coefficients
    log_weights = copy(integration.log_weights)
    for j in axes(predictors, 2), i in axes(predictors, 1)
        log_weights[j] += log_logistic(y[i] == 1 ? predictors[i, j] : -predictors[i, j])
    end
    weights, log_evidence = normalise_log_weights(log_weights)
    posterior_mean = coefficients * weights
    centred = coefficients .- posterior_mean
    posterior_covariance = Matrix(Symmetric((centred .* transpose(weights)) * centred'))
    return (; mean = posterior_mean, covariance = posterior_covariance,
            coefficients, weights, log_evidence, points = integration.points)
end

"""
Refine a posterior rule and report moment agreement, without silently accepting
an unconverged reference. Quantiles of the weighted grid converge more slowly.
"""
function converged_reference(X, y, μ, V; orders = (80, 160, 240),
                             mean_tolerance = 2e-5, covariance_tolerance = 2e-5)
    previous = logistic_reference(X, y, μ, V; points = first(orders))
    history = NamedTuple[]
    for points in Iterators.drop(orders, 1)
        current = logistic_reference(X, y, μ, V; points)
        scale = sqrt.(max.(diag(current.covariance), eps(Float64)))
        mean_change = maximum(abs.(current.mean - previous.mean) ./ scale)
        covariance_change = norm(current.covariance - previous.covariance) /
                            max(norm(current.covariance), eps(Float64))
        push!(history, (; points, mean_change, covariance_change))
        converged = mean_change <= mean_tolerance && covariance_change <= covariance_tolerance
        converged && return merge(current, (; converged, history))
        previous = current
    end
    return merge(previous, (; converged = false, history))
end

"""Left-continuous quantile of a normalised, discrete quadrature measure."""
function weighted_quantile(values, weights, probability::Real)
    0 <= probability <= 1 || throw(ArgumentError("quantile probability outside [0,1]"))
    order = sortperm(values)
    cumulative = 0.0
    for j in order
        cumulative += weights[j]
        cumulative >= probability && return values[j]
    end
    return values[last(order)]
end

"""Posterior fixed-cohort policy contrast, evaluated on the same weighted nodes."""
function policy_reference(reference, X, X_policy; level = 0.95)
    size(X) == size(X_policy) || throw(DimensionMismatch("policy design dimensions"))
    baseline = X * reference.coefficients
    policy = X_policy * reference.coefficients
    contrasts = [mean(logistic_probability.(view(policy, :, j))) -
                 mean(logistic_probability.(view(baseline, :, j))) for j in axes(policy, 2)]
    location = sum(reference.weights .* contrasts)
    variance = sum(reference.weights .* (contrasts .- location) .^ 2)
    tail = (1 - level) / 2
    return (; mean = location, sd = sqrt(max(variance, 0.0)),
            lower = weighted_quantile(contrasts, reference.weights, tail),
            upper = weighted_quantile(contrasts, reference.weights, 1 - tail))
end

"""Integrate a Gaussian moment approximation through the same policy functional."""
function gaussian_policy_summary(μ, V, X, X_policy; points = 80, level = 0.95)
    grid = prior_grid(μ, V; points)
    weights, _ = normalise_log_weights(grid.log_weights)
    return policy_reference((; coefficients = grid.coefficients, weights), X, X_policy; level)
end

end # module
