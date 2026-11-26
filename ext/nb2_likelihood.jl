"""Subject-integrated NB2/log/random-intercept likelihood and information."""

struct _NB2FitData
    y::Vector{Float64}
    X::Matrix{Float64}
    groups::Vector{Vector{Int}}
    nodes::Vector{Float64}
    logweights::Vector{Float64}
    logfactorials::Vector{Float64}
    quadrature_method::Symbol
end

struct _FixedGHQuadrature end
struct _AdaptiveGHQuadrature end

@inline function _softplus(x::Float64)
    x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
end

@inline function _logistic(x::Float64)
    if x >= 0
        z = exp(-x)
        return 1 / (1 + z)
    end
    z = exp(x)
    return z / (1 + z)
end

function _stable_logsumexp(values::AbstractVector{Float64})
    maximum_value = maximum(values)
    isfinite(maximum_value) || return maximum_value
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function _nb2_logpmf(y::Real, eta::Real, theta::Real)
    y >= 0 || throw(DomainError(y, "NB2 counts must be nonnegative"))
    theta > 0 || throw(DomainError(theta, "NB2 theta must be positive"))
    yf = Float64(y)
    etaf = Float64(eta)
    thetaf = Float64(theta)
    logtheta = log(thetaf)
    logdenominator = logtheta + _softplus(etaf - logtheta)
    return loggamma(yf + thetaf) - loggamma(thetaf) - loggamma(yf + 1) +
        thetaf * (logtheta - logdenominator) + yf * (etaf - logdenominator)
end

function _adaptive_mode(
    fitdata::_NB2FitData,
    indices::Vector{Int},
    eta_fixed::Vector{Float64},
    sigma::Float64,
    logtheta::Float64,
    theta_value::Float64,
)
    inverse_variance = inv(sigma^2)
    score_curvature = function (b::Float64)
        score = -b * inverse_variance
        curvature = inverse_variance
        for observation in indices
            y = fitdata.y[observation]
            fraction = _logistic(eta_fixed[observation] + b - logtheta)
            score += y - (theta_value + y) * fraction
            curvature += (theta_value + y) * fraction * (1 - fraction)
        end
        isfinite(score) && isfinite(curvature) && curvature > 0 || throw(ArgumentError(
            "could not compute a finite positive conditional curvature for a subject",
        ))
        return score, curvature
    end

    lower = -max(1.0, sigma)
    upper = max(1.0, sigma)
    lower_score, _ = score_curvature(lower)
    upper_score, _ = score_curvature(upper)
    for _ in 1:60
        lower_score >= 0 && upper_score <= 0 && break
        if lower_score < 0
            lower *= 2
            lower_score, _ = score_curvature(lower)
        end
        if upper_score > 0
            upper *= 2
            upper_score, _ = score_curvature(upper)
        end
    end
    lower_score >= 0 && upper_score <= 0 || throw(ArgumentError(
        "could not bracket a subject's conditional random-intercept mode",
    ))

    mode = clamp(0.0, lower, upper)
    for _ in 1:80
        score, curvature = score_curvature(mode)
        abs(score) <= 1.0e-10 * (1 + length(indices)) && return mode, curvature
        if score > 0
            lower = mode
        else
            upper = mode
        end
        candidate = mode + score / curvature
        if !(isfinite(candidate) && lower < candidate < upper)
            candidate = (lower + upper) / 2
        end
        if abs(candidate - mode) <= 1.0e-12 * (1 + abs(mode))
            final_score, final_curvature = score_curvature(candidate)
            abs(final_score) <= 1.0e-8 * (1 + length(indices)) &&
                return candidate, final_curvature
        end
        mode = candidate
    end
    throw(ArgumentError(
        "conditional random-intercept mode did not converge within 80 safeguarded Newton steps",
    ))
end

@inline function _nb2_subject_rule(
    ::_FixedGHQuadrature,
    fitdata::_NB2FitData,
    indices::Vector{Int},
    eta_fixed::Vector{Float64},
    sigma::Float64,
    logtheta::Float64,
    theta_value::Float64,
)
    return 0.0, sqrt(2.0) * sigma, -0.5 * log(pi)
end

@inline function _nb2_subject_rule(
    ::_AdaptiveGHQuadrature,
    fitdata::_NB2FitData,
    indices::Vector{Int},
    eta_fixed::Vector{Float64},
    sigma::Float64,
    logtheta::Float64,
    theta_value::Float64,
)
    mode, curvature = _adaptive_mode(
        fitdata, indices, eta_fixed, sigma, logtheta, theta_value,
    )
    return mode, sqrt(2.0 / curvature), 0.5 * log(2.0 / curvature)
end

@inline function _nb2_node_logterm_and_sigma_score(
    ::_FixedGHQuadrature,
    conditional_loglikelihood::Float64,
    eta_score_sum::Float64,
    b::Float64,
    node::Float64,
    logweight::Float64,
    logsigma::Float64,
    inverse_variance::Float64,
)
    return logweight + conditional_loglikelihood, eta_score_sum * b
end

@inline function _nb2_node_logterm_and_sigma_score(
    ::_AdaptiveGHQuadrature,
    conditional_loglikelihood::Float64,
    eta_score_sum::Float64,
    b::Float64,
    node::Float64,
    logweight::Float64,
    logsigma::Float64,
    inverse_variance::Float64,
)
    log_integrand = conditional_loglikelihood -
        0.5 * b^2 * inverse_variance - logsigma - 0.5 * log(2pi)
    # Exact marginal score identity: includes the sigma derivative of the
    # Gaussian random-effect density without differentiating adaptive nodes.
    return logweight + log_integrand + node^2, -1.0 + b^2 * inverse_variance
end

function _nb2_likelihood_with_rule(
    rule::R,
    fitdata::_NB2FitData,
    parameters::AbstractVector{<:Real};
    fixed_logtheta::Union{Nothing, Float64} = nothing,
    gradient::Bool = true,
) where {R}
    p = size(fitdata.X, 2)
    expected_length = p + (fixed_logtheta === nothing ? 2 : 1)
    length(parameters) == expected_length || throw(DimensionMismatch(
        "parameter vector has length $(length(parameters)); expected $expected_length",
    ))
    beta = view(parameters, 1:p)
    logsigma = Float64(parameters[p + 1])
    logtheta = fixed_logtheta === nothing ? Float64(parameters[p + 2]) : fixed_logtheta
    sigma = exp(logsigma)
    theta_value = exp(logtheta)
    if !(isfinite(sigma) && sigma > 0 && isfinite(theta_value) && theta_value > 0)
        return -Inf, fill(NaN, expected_length)
    end

    eta_fixed = Vector{Float64}(fitdata.X * beta)
    all(isfinite, eta_fixed) || return -Inf, fill(NaN, expected_length)
    psi_theta = digamma(theta_value)
    gamma_terms = similar(fitdata.y)
    psi_differences = gradient ? similar(fitdata.y) : Float64[]
    for observation in eachindex(fitdata.y)
        y = fitdata.y[observation]
        gamma_terms[observation] =
            loggamma(y + theta_value) - loggamma(theta_value) -
            fitdata.logfactorials[observation]
        gradient && (psi_differences[observation] = digamma(y + theta_value) - psi_theta)
    end

    q = length(fitdata.nodes)
    log_terms = Vector{Float64}(undef, q)
    score_beta = gradient ? Matrix{Float64}(undef, q, p) : Matrix{Float64}(undef, 0, 0)
    score_logsigma = gradient ? Vector{Float64}(undef, q) : Float64[]
    score_logtheta = gradient ? Vector{Float64}(undef, q) : Float64[]
    total_loglikelihood = 0.0
    total_gradient = gradient ? zeros(Float64, expected_length) : Float64[]
    inverse_variance = inv(sigma^2)

    for indices in fitdata.groups
        mode, local_scale, subject_log_const = _nb2_subject_rule(
            rule, fitdata, indices, eta_fixed, sigma, logtheta, theta_value,
        )
        for k in 1:q
            node = fitdata.nodes[k]
            b = mode + local_scale * node
            conditional_loglikelihood = 0.0
            gradient && fill!(view(score_beta, k, :), 0.0)
            eta_score_sum = 0.0
            theta_score_sum = 0.0
            for observation in indices
                y = fitdata.y[observation]
                eta = eta_fixed[observation] + b
                difference = eta - logtheta
                mean_fraction = _logistic(difference)
                logdenominator = logtheta + _softplus(difference)
                conditional_loglikelihood += gamma_terms[observation] +
                    theta_value * (logtheta - logdenominator) +
                    y * (eta - logdenominator)
                if gradient
                    eta_score = y - (theta_value + y) * mean_fraction
                    for column in 1:p
                        score_beta[k, column] +=
                            eta_score * fitdata.X[observation, column]
                    end
                    eta_score_sum += eta_score
                    theta_score_sum += theta_value * (
                        psi_differences[observation] + logtheta - logdenominator + 1 -
                        (1 + y / theta_value) * (1 - mean_fraction)
                    )
                end
            end
            log_term, logsigma_score = _nb2_node_logterm_and_sigma_score(
                rule, conditional_loglikelihood, eta_score_sum, b, node,
                fitdata.logweights[k], logsigma, inverse_variance,
            )
            log_terms[k] = log_term
            if gradient
                score_logsigma[k] = logsigma_score
                score_logtheta[k] = theta_score_sum
            end
        end

        subject_logsum = _stable_logsumexp(log_terms)
        isfinite(subject_logsum) || return -Inf, fill(NaN, expected_length)
        total_loglikelihood += subject_logsum + subject_log_const
        if gradient
            for k in 1:q
                posterior_weight = exp(log_terms[k] - subject_logsum)
                for column in 1:p
                    total_gradient[column] += posterior_weight * score_beta[k, column]
                end
                total_gradient[p + 1] += posterior_weight * score_logsigma[k]
                fixed_logtheta === nothing &&
                    (total_gradient[p + 2] += posterior_weight * score_logtheta[k])
            end
        end
    end
    return total_loglikelihood, total_gradient
end

_nb2_likelihood_fixed(
    fitdata::_NB2FitData,
    parameters::AbstractVector{<:Real};
    fixed_logtheta::Union{Nothing, Float64} = nothing,
    gradient::Bool = true,
) = _nb2_likelihood_with_rule(
    _FixedGHQuadrature(), fitdata, parameters; fixed_logtheta, gradient,
)

_nb2_likelihood_adaptive(
    fitdata::_NB2FitData,
    parameters::AbstractVector{<:Real};
    fixed_logtheta::Union{Nothing, Float64} = nothing,
    gradient::Bool = true,
) = _nb2_likelihood_with_rule(
    _AdaptiveGHQuadrature(), fitdata, parameters; fixed_logtheta, gradient,
)

function _nb2_likelihood(
    fitdata::_NB2FitData,
    parameters::AbstractVector{<:Real};
    fixed_logtheta::Union{Nothing, Float64} = nothing,
    gradient::Bool = true,
)
    if fitdata.quadrature_method == :adaptive_gauss_hermite
        return _nb2_likelihood_adaptive(
            fitdata, parameters; fixed_logtheta, gradient,
        )
    end
    return _nb2_likelihood_fixed(fitdata, parameters; fixed_logtheta, gradient)
end

function _conditional_beta_information(
    fitdata::_NB2FitData,
    beta::Vector{Float64},
    logsigma::Float64,
    logtheta::Float64,
)
    p = length(beta)
    q = length(fitdata.nodes)
    sigma = exp(logsigma)
    theta_value = exp(logtheta)
    eta_fixed = fitdata.X * beta
    log_terms = Vector{Float64}(undef, q)
    scores = zeros(Float64, q, p)
    hessians = [zeros(Float64, p, p) for _ in 1:q]
    information = zeros(Float64, p, p)
    root_two_sigma = sqrt(2.0) * sigma
    inverse_variance = inv(sigma^2)

    for indices in fitdata.groups
        mode, curvature = if fitdata.quadrature_method == :adaptive_gauss_hermite
            _adaptive_mode(fitdata, indices, eta_fixed, sigma, logtheta, theta_value)
        else
            (0.0, NaN)
        end
        local_scale = fitdata.quadrature_method == :adaptive_gauss_hermite ?
            sqrt(2 / curvature) : root_two_sigma
        for k in 1:q
            fill!(view(scores, k, :), 0.0)
            fill!(hessians[k], 0.0)
            b = mode + local_scale * fitdata.nodes[k]
            conditional_loglikelihood = 0.0
            for observation in indices
                y = fitdata.y[observation]
                eta = eta_fixed[observation] + b
                difference = eta - logtheta
                fraction = _logistic(difference)
                logdenominator = logtheta + _softplus(difference)
                conditional_loglikelihood +=
                    loggamma(y + theta_value) - loggamma(theta_value) -
                    fitdata.logfactorials[observation] +
                    theta_value * (logtheta - logdenominator) +
                    y * (eta - logdenominator)
                eta_score = y - (theta_value + y) * fraction
                eta_second = -(theta_value + y) * fraction * (1 - fraction)
                for a in 1:p
                    xa = fitdata.X[observation, a]
                    scores[k, a] += eta_score * xa
                    for c in 1:p
                        hessians[k][a, c] +=
                            eta_second * xa * fitdata.X[observation, c]
                    end
                end
            end
            log_terms[k] = if fitdata.quadrature_method == :adaptive_gauss_hermite
                log_integrand = conditional_loglikelihood -
                    0.5b^2 * inverse_variance - logsigma - 0.5log(2pi)
                fitdata.logweights[k] + log_integrand + fitdata.nodes[k]^2
            else
                fitdata.logweights[k] + conditional_loglikelihood
            end
        end
        subject_logsum = _stable_logsumexp(log_terms)
        mean_score = zeros(Float64, p)
        expected_hessian_and_outer = zeros(Float64, p, p)
        for k in 1:q
            weight = exp(log_terms[k] - subject_logsum)
            score = view(scores, k, :)
            mean_score .+= weight .* score
            expected_hessian_and_outer .+=
                weight .* (hessians[k] .+ score * transpose(score))
        end
        subject_hessian = expected_hessian_and_outer .-
            mean_score * transpose(mean_score)
        information .-= subject_hessian
    end
    information = Matrix(Symmetric((information .+ transpose(information)) ./ 2))
    eigenvalues = eigvals(Symmetric(information))
    minimum(eigenvalues) > 0 || throw(ArgumentError(
        "fixed-effect observed information is not positive definite; minimum " *
        "eigenvalue is $(minimum(eigenvalues))",
    ))
    covariance = inv(Symmetric(information))
    all(isfinite, covariance) || throw(ArgumentError(
        "fixed-effect covariance contains non-finite values",
    ))
    return Matrix{Float64}((covariance .+ transpose(covariance)) ./ 2)
end
