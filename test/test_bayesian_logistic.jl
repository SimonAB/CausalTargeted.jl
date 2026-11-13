using Test, DataFrames, LinearAlgebra, Random, Statistics
using CausalTargeted

@testset "Bayesian logistic g-computation" begin
    prior2 = GaussianCoefficientPrior(mean = zeros(2), covariance = Matrix{Float64}(I, 2, 2))
    @test_throws ArgumentError GaussianCoefficientPrior(mean = [NaN], covariance = ones(1, 1))
    @test_throws ArgumentError GaussianCoefficientPrior(mean = zeros(2), covariance = [1.0 1; 0 1])
    @test_throws ArgumentError GaussianCoefficientPrior(mean = zeros(2), covariance = ones(2, 2))
    @test_throws DimensionMismatch GaussianCoefficientPrior(mean = zeros(2), covariance = ones(1, 1))
    data = DataFrame(A = repeat([-0.7, -0.3, 0.3, 0.7], 8), Y = repeat([0, 1, 0, 1], 8))
    @test_throws ArgumentError fit_bayesian_logistic(data[1:0, :], :Y, :A, Symbol[]; prior = prior2)
    invalid = copy(data); invalid.Y[1] = 2
    @test_throws ArgumentError fit_bayesian_logistic(invalid, :Y, :A, Symbol[]; prior = prior2)
    invalid[!, :Y] = fill(missing, nrow(data))
    @test_throws ArgumentError fit_bayesian_logistic(invalid, :Y, :A, Symbol[]; prior = prior2)
    absent = fit_bayesian_logistic(invalid, :Y, :A, Symbol[]; prior = prior2, handle_missing = :mar)
    @test absent.posterior_mean == prior2.mean
    @test absent.posterior_covariance ≈ prior2.covariance
    missing_predictor = copy(data); missing_predictor[!, :A] = Vector{Union{Missing,Float64}}(data.A); missing_predictor.A[1] = missing
    @test_throws ArgumentError fit_bayesian_logistic(missing_predictor, :Y, :A, Symbol[]; prior = prior2)
    fit = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior = prior2)
    @test fit.metadata.graph_identification_status == :user_assumed
    @test fit.metadata.outcome_model_status == :user_assumed
    @test fit.metadata.identification_status == :user_assumed
    @test fit.metadata.likelihood == :bernoulli_logit
    @test fit.metadata.n_observed == nrow(data)
    @test size(fit.variational_covariance) == (2, 2)
    @test fit.status == :converged
    @test all(isfinite, fit.posterior_covariance)
    variational = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior = prior2, covariance = :variational)
    @test variational.posterior_covariance == variational.variational_covariance
    @test variational.metadata.covariance_method == :variational
    policy = SmoothBoundedLogitShiftPolicy(-1, 1)
    result = bayesian_policy_contrast(fit, data, policy; deltas = [0, 0.5, 0.5], n_draws = 500, rng = Xoshiro(42))
    repeat_result = bayesian_policy_contrast(fit, data, policy; deltas = [0, 0.5, 0.5], n_draws = 500, rng = Xoshiro(42))
    @test result.mean == repeat_result.mean
    @test result.covariance == repeat_result.covariance
    @test result.mean[1] == result.lower[1] == result.upper[1] == result.mcse[1] == 0
    @test result.mean[2] == result.mean[3]
    @test result.covariance[2, 3] ≈ result.covariance[2, 2]
    @test result.metadata.uncertainty == :gaussian_moment_approximation
    @test result.metadata.target == :fixed_cohort
    @test result.metadata.n_draws == 500
    @test_throws ArgumentError bayesian_policy_contrast(fit, data, ShiftPolicy("raw", 0.01, 0.99); deltas = [1])
    @test_throws ArgumentError bayesian_policy_contrast(fit, data, policy; deltas = [Inf])
    @test_throws ArgumentError bayesian_policy_contrast(fit, data, policy; deltas = [1], n_draws = 1)
    @test_throws ArgumentError bayesian_policy_contrast(fit, data, policy; deltas = [1], level = 1)
    @test_throws DimensionMismatch fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior = GaussianCoefficientPrior(mean = zeros(3), covariance = Matrix{Float64}(I, 3, 3)))

    mar = copy(data); mar[!, :Y] = Vector{Union{Missing,Int}}(mar.Y); mar.Y[end] = missing
    marfit = fit_bayesian_logistic(mar, :Y, :A, Symbol[]; prior = prior2, handle_missing = :mar)
    complete = fit_bayesian_logistic(mar[1:end-1, :], :Y, :A, Symbol[]; prior = prior2)
    @test marfit.posterior_mean ≈ complete.posterior_mean
    @test marfit.metadata.missingness_status == :user_assumed_mar
    @test marfit.metadata.n_missing == 1

    @testset "Frozen categories and predictor-only custom basis" begin
        categorical = DataFrame(A = repeat(["a", "b"], 16), W = repeat(["low", "high", "high", "low"], 8), Y = data.Y)
        prior3 = GaussianCoefficientPrior(mean = zeros(3), covariance = Matrix{Float64}(I, 3, 3))
        cfit = fit_bayesian_logistic(categorical, :Y, :A, [:W]; prior = prior3)
        calls = Ref(0)
        discrete = discrete_shift_policy((a, w) -> (calls[] += 1; "b"))
        cresult = bayesian_policy_contrast(cfit, categorical, discrete; deltas = [0, 1], n_draws = 100)
        @test calls[] == nrow(categorical)
        @test cresult.mean[1] == cresult.mean[2]
        unknown = copy(categorical); unknown.A[1] = "c"
        @test_throws ArgumentError bayesian_policy_contrast(cfit, unknown, discrete; deltas = [0])
        @test_throws ArgumentError bayesian_policy_contrast(cfit, categorical, discrete_static_policy("c"); deltas = [0])
        basis = (df, a) -> begin
            @test propertynames(df) == [:A]
            hcat(ones(length(a)), a, a .^ 2)
        end
        bfit = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior = prior3, basis)
        @test bfit.design_metadata.basis_source == :user_function
        @test length(bfit.posterior_mean) == 3
    end

    @testset "Nonconvergence withholds uncertainty" begin
        failed = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior = prior2, maxiter = 1, tol = 1e-16)
        @test failed.status != :converged
        @test all(isnan, failed.posterior_covariance)
        contrast = bayesian_policy_contrast(failed, data, policy; deltas = [0, 1])
        @test contrast.metadata.uncertainty == :unavailable
        @test all(isfinite, contrast.mean)
        @test all(isnan, contrast.lower)
        @test all(isnan, contrast.covariance)
    end

    @testset "Poorly scaled continuous predictors" begin
        # Gram-scale continuous W with a modest slope needs several hundred
        # Jaakkola updates under a unit Gaussian prior (Twins-like scale).
        rng = Xoshiro(17)
        n = 400
        W = 500.0 .* randn(rng, n)
        A = Float64.(rand(rng, n) .< 0.5)
        η = -0.4 .+ 0.8 .* A .+ 0.01 .* W
        Y = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-η)))
        scaled = DataFrame(A = A, W = W, Y = Y)
        prior3 = GaussianCoefficientPrior(mean = zeros(3), covariance = Matrix{Float64}(I, 3, 3))
        capped = fit_bayesian_logistic(scaled, :Y, :A, [:W]; prior = prior3, maxiter = 200)
        @test capped.status == :maxiter
        @test !capped.metadata.uncertainty_available
        default = fit_bayesian_logistic(scaled, :Y, :A, [:W]; prior = prior3)
        @test default.status == :converged
        @test default.metadata.n_iterations > 200
        @test default.metadata.uncertainty_available
        @test all(isfinite, default.posterior_covariance)
    end
end
