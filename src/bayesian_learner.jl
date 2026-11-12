"""A Gaussian Super Learner candidate with an explicitly specified prior."""
struct BayesianGaussianLearner
    prior::ConjugateGaussianPrior
end

"""
    BayesianGaussianLearner(; prior)

Construct a Gaussian regression candidate using a conjugate Gaussian prior.
The prior dimension must equal the fitted design-matrix width, including its
intercept and any treatment column. The candidate is available only for
`family=:gaussian`.
"""
BayesianGaussianLearner(; prior::ConjugateGaussianPrior) = BayesianGaussianLearner(prior)

"""Gaussian candidate fit retaining posterior moments for posterior-mean prediction."""
struct BayesianGaussianCandidateFit
    posterior_mean::Vector{Float64}
    posterior_covariance::Matrix{Float64}
end

_candidate_key(::BayesianGaussianLearner) = :bayesian_gaussian

function _fit_any_candidate(
    candidate::BayesianGaussianLearner,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    family::Symbol,
)
    family === :gaussian || throw(ArgumentError(
        "BayesianGaussianLearner requires family=:gaussian; got family=:$family",
    ))
    posterior = _bayesian_gaussian_posterior(X, y, candidate.prior)
    return BayesianGaussianCandidateFit(
        posterior.posterior_mean,
        posterior.posterior_covariance,
    )
end

"""Predict with the posterior mean; posterior covariance is not propagated."""
function _predict_learner(fit::BayesianGaussianCandidateFit, X::Matrix{Float64})
    size(X, 2) == length(fit.posterior_mean) || throw(DimensionMismatch(
        "prediction design has $(size(X, 2)) columns but posterior mean has " *
        "$(length(fit.posterior_mean)) coefficients",
    ))
    return X * fit.posterior_mean
end
