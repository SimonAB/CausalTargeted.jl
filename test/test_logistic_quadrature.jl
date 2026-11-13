using Test, LinearAlgebra, DataFrames, CausalTargeted, Statistics
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

    @testset "Interval content against the quadrature reference" begin
        content = reference.coefficient_interval_content(truth, fit.posterior_mean, fit.posterior_covariance)
        @test length(content.contents) == 2
        @test all(0 .< content.contents .< 1)
        X_policy = copy(X)
        X_policy[:, end] .= median(a)
        exact_policy = reference.policy_reference(truth, X, X_policy)
        approx_policy = reference.gaussian_policy_summary(fit.posterior_mean, fit.posterior_covariance, X, X_policy)
        policy_mass = reference.policy_interval_content(truth, X, X_policy,
            approx_policy.lower, approx_policy.upper)
        @test 0 < policy_mass.content < 1
        @test isapprox(policy_mass.content + policy_mass.lower_tail + policy_mass.upper_tail, 1; atol = 1e-12)
        @test abs(approx_policy.mean - exact_policy.mean) < 0.05
    end

    @testset "Moderately scaled predictor remains comparable" begin
        W = 10.0 .* a
        Xw = hcat(ones(20), W)
        yw = Float64.(W .> 0)
        prior_w = GaussianCoefficientPrior(mean = zeros(2), covariance = Matrix{Float64}(I, 2, 2))
        truth_w = reference.converged_reference(Xw, yw, prior_w.mean, prior_w.covariance;
            orders = (80, 160, 240, 400, 640))
        fit_w = fit_bayesian_logistic(DataFrame(A = W, Y = yw), :Y, :A, Symbol[]; prior = prior_w)
        @test truth_w.converged
        @test fit_w.status === :converged
        mean_error = maximum(abs.(fit_w.posterior_mean - truth_w.mean) ./ sqrt.(diag(truth_w.covariance)))
        cov_error = norm(fit_w.posterior_covariance - truth_w.covariance) / norm(truth_w.covariance)
        @test isfinite(mean_error) && isfinite(cov_error)
        @test mean_error < 1.0
    end

    @testset "Gram-scale predictors break prior-centred Hermite" begin
        W = 500.0 .* a
        Xw = hcat(ones(20), W)
        yw = Float64.(W .> 0)
        prior_w = GaussianCoefficientPrior(mean = zeros(2), covariance = Matrix{Float64}(I, 2, 2))
        truth_w = reference.converged_reference(Xw, yw, prior_w.mean, prior_w.covariance;
            orders = (80, 160, 240, 400, 640, 960))
        fit_w = fit_bayesian_logistic(DataFrame(A = W, Y = yw), :Y, :A, Symbol[];
            prior = prior_w, maxiter = 2000)
        @test !truth_w.converged
        # The variational loop may still return a finite Gaussian; the point of
        # this fixture is that prior-centred Hermite cannot score it.
        @test fit_w.status in (:converged, :maxiter)
        @test all(isfinite, fit_w.posterior_mean)
    end
end
