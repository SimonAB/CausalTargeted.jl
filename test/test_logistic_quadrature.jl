using Test, LinearAlgebra, DataFrames, CausalTargeted
include("logistic_reference.jl")

@testset "Independent logistic quadrature accuracy" begin
    reference = LogisticPosteriorReference
    μ, V = [0.3, -0.2], [1.0 0.3; 0.3 2.0]
    prior_only = reference.logistic_reference(zeros(3, 2), [0, 1, 0], μ, V; points = 80)
    @test prior_only.mean ≈ μ atol = 1e-10
    @test prior_only.covariance ≈ V atol = 1e-10
    a = repeat([-0.9, -0.5, 0.1, 0.5, 0.9], inner = 4)
    X = hcat(ones(20), a)
    y = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    prior = GaussianCoefficientPrior(mean = zeros(2), covariance = 2.25 * Matrix{Float64}(I, 2, 2))
    truth = reference.converged_reference(X, y, prior.mean, prior.covariance)
    fit = fit_bayesian_logistic(DataFrame(A = a, Y = y), :Y, :A, Symbol[]; prior)
    @test truth.converged
    @test fit.status === :converged
    # Prespecified moderate-fixture bounds; adversarial cases are reported by
    # test/stress/logistic_calibration.jl without weakening these thresholds.
    @test maximum(abs.(fit.posterior_mean - truth.mean) ./ sqrt.(diag(truth.covariance))) <= 0.20
    @test norm(fit.corrected_covariance - truth.covariance) / norm(truth.covariance) <= 0.25
end
