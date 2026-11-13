using Test, LinearAlgebra, DataFrames, Distributions, Random
using CausalTargeted, GraphPPL, RxInfer

isdefined(@__MODULE__, :LogisticTailReference) || include("logistic_tail_reference.jl")
isdefined(@__MODULE__, :LogisticPosteriorReference) || include("logistic_reference.jl")

@testset "Independent logistic posterior tails" begin
    reference = LogisticTailReference
    @testset "Adaptive quantiles have independent error checks" begin
        prior = Normal(0.3, 1.7)
        truth = reference.adaptive_reference(zeros(4), [0, 1, 0, 1], mean(prior), var(prior))
        @test truth.mean ≈ mean(prior) atol = 1e-10
        @test truth.variance ≈ var(prior) atol = 1e-10
        for probability in (0.001, 0.025, 0.5, 0.975, 0.999)
            q = reference.posterior_quantile(truth, probability)
            @test q ≈ quantile(prior, probability) atol = 1e-7
            @test reference.posterior_cdf(truth, q) ≈ probability atol = 1e-9
        end
        @test reference.posterior_cdf(truth, -Inf) == 0
        @test reference.posterior_cdf(truth, Inf) == 1
        @test_throws ArgumentError reference.posterior_quantile(truth, 0)
        tails = reference.posterior_tails(truth)
        @test tails.maximum_tail_residual <= 1e-9
        @test tails.maximum_integration_error <= 1e-9
        # Exact prior moments do not make a discrete Hermite quantile exact.
        grid = LogisticPosteriorReference.logistic_reference(zeros(4, 1), [0, 1, 0, 1],
            [mean(prior)], reshape([var(prior)], 1, 1); points = 80)
        @test grid.mean[1] ≈ truth.mean atol = 1e-10
        @test grid.covariance[1, 1] ≈ truth.variance atol = 1e-10
        grid_lower = LogisticPosteriorReference.weighted_quantile(vec(grid.coefficients), grid.weights, 0.025)
        @test abs(reference.posterior_cdf(truth, grid_lower) - 0.025) > 1e-3
    end

    @testset "Bivariate policy CDF agrees with analytic Gaussian events" begin
        μ, V = [0.3, -0.4], [2.25 0.5; 0.5 1.0]
        prior_policy = reference.bivariate_policy_reference(zeros(0, 2), Float64[], μ, V;
            rtol = 1e-9, atol = 1e-12)
        # The contrast has the sign of β₁, irrespective of intercept or prior correlation.
        @test reference.policy_cdf(prior_policy, 0.0) ≈ cdf(Normal(μ[2], sqrt(V[2, 2])), 0) atol = 1e-8
        @test reference.policy_cdf(prior_policy, -1.0) == 0
        @test reference.policy_cdf(prior_policy, 1.0) == 1
        constant_likelihood = reference.bivariate_policy_reference(zeros(4, 2), [0, 1, 0, 1], μ, V;
            rtol = 1e-9, atol = 1e-12)
        for contrast in (-0.2, 0.2)
            @test reference.policy_cdf(constant_likelihood, contrast) ≈
                reference.policy_cdf(prior_policy, contrast) atol = 1e-8
        end
        symmetric = reference.bivariate_policy_reference(zeros(0, 2), Float64[], zeros(2), V;
            rtol = 1e-9, atol = 1e-12)
        @test reference.policy_cdf(symmetric, -0.3) + reference.policy_cdf(symmetric, 0.3) ≈ 1 atol = 1e-8
    end

    @testset "Bivariate refine separates moments, content and endpoints" begin
        μ, V = zeros(2), Matrix(0.25 * I(2))
        X = hcat(ones(8), repeat([-0.5, 0.5], 4))
        y = Float64.([0, 0, 1, 1, 0, 1, 0, 1])
        refined = reference.refine_bivariate_reference(X, y, μ, V; check_policy = true)
        @test refined.moments_converged
        @test refined.content_converged
        @test refined.endpoints_converged
        @test refined.reference_converged
        @test isempty(refined.reference_failures)
        @test length(refined.coefficients) == 2
        moments = reference.bivariate_moments(refined.tight)
        @test moments.mean ≈ refined.moments.mean rtol = 1e-8
        content = reference.coefficient_interval_content(refined.tight, 1,
            refined.coefficients[1].reference_interval...)
        @test content.content ≈ 0.95 atol = 5e-3
    end

    @testset "Moderate scalar inference and public policy pushforward" begin
        a = [zeros(4); ones(20)]
        y = [0, 1, 0, 1, zeros(Int, 12)..., ones(Int, 8)...]
        prior = GaussianCoefficientPrior(mean = [0.0], covariance = reshape([2.25], 1, 1))
        basis = (_, values) -> reshape(Float64.(values), :, 1)
        data = DataFrame(A = a, Y = y)
        truth = reference.adaptive_reference(a, y, 0.0, 2.25)
        tighter = reference.adaptive_reference(a, y, 0.0, 2.25; rtol = 1e-12, atol = 1e-14)
        tails = reference.posterior_tails(truth)
        tight_tails = reference.posterior_tails(tighter)
        @test max(abs(tails.lower - tight_tails.lower), abs(tails.upper - tight_tails.upper)) <= 1e-7
        @test tails.maximum_tail_residual <= 1e-9
        @test tails.maximum_integration_error <= 1e-9
        fit = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior, basis, maxiter = 1000, tol = 1e-10)
        @test fit.status === :converged
        @test abs(fit.posterior_mean[1] - truth.mean) / sqrt(truth.variance) <= 0.20
        @test abs(fit.corrected_covariance[1, 1] / truth.variance - 1) <= 0.25
        gaussian = Normal(fit.posterior_mean[1], sqrt(fit.posterior_covariance[1, 1]))
        proposed = reference.interval_content(truth, quantile(gaussian, 0.025), quantile(gaussian, 0.975))
        # Prespecified moderate-fixture endpoint gate, not a universal promise.
        @test max(abs(proposed.lower_tail - 0.025), abs(proposed.upper_tail - 0.025)) <= 0.025
        result = bayesian_policy_contrast(fit, DataFrame(A = [0.0]), discrete_static_policy(1.0);
            deltas = [1.0], n_draws = 40_000, rng = Xoshiro(2703))
        # Here the functional is strictly monotone: logistic(β) - 1/2.
        # This checks the public sampling layer against its own Gaussian law,
        # separately from the Gaussian approximation's exact-posterior error.
        expected = reference.logistic_probability.([quantile(gaussian, 0.025), quantile(gaussian, 0.975)]) .- 0.5
        @test result.lower[1] ≈ expected[1] atol = 0.007
        @test result.upper[1] ≈ expected[2] atol = 0.007
        @test result.metadata.uncertainty === :gaussian_moment_approximation
    end

    @testset "Missing outcomes contribute no likelihood" begin
        a = repeat([-1.0, -0.5, 0.5, 1.0], 4)
        y = Union{Missing,Int}[0, 0, 1, 1, 0, missing, 0, 1, 1, 0, missing, 1, 0, 1, 1, 0]
        data = DataFrame(A = a, Y = y)
        prior = GaussianCoefficientPrior(mean = [0.2], covariance = reshape([1.0], 1, 1))
        basis = (_, values) -> reshape(Float64.(values), :, 1)
        observed = findall(!ismissing, y)
        fit = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior, basis, handle_missing = :mar)
        complete = fit_bayesian_logistic(data[observed, :], :Y, :A, Symbol[]; prior, basis)
        truth = reference.adaptive_reference(a[observed], Float64.(y[observed]), 0.2, 1.0)
        @test fit.status === :converged
        @test fit.posterior_mean ≈ complete.posterior_mean atol = 1e-12
        @test fit.posterior_covariance ≈ complete.posterior_covariance atol = 1e-12
        @test fit.metadata.n_missing == 2
        @test fit.metadata.missingness_status === :user_assumed_mar
        @test abs(fit.posterior_mean[1] - truth.mean) / sqrt(truth.variance) <= 0.20
        @test abs(fit.posterior_covariance[1, 1] / truth.variance - 1) <= 0.25
        data[!, :Y] .= missing
        absent = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior, basis, handle_missing = :mar)
        prior_reference = reference.adaptive_reference(Float64[], Float64[], 0.2, 1.0)
        @test absent.posterior_mean[1] ≈ prior_reference.mean atol = 1e-10
        @test absent.posterior_covariance[1, 1] ≈ prior_reference.variance atol = 1e-10
    end
end

include("test_logistic_tail_regressions.jl")
