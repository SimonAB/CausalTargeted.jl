using Test, DataFrames, LinearAlgebra, CausalTargeted

@testset "Bayesian discrete policy support" begin
    data = DataFrame(A = repeat([0, 1], 4), Y = repeat([0, 1, 1, 0], 2))
    V = Matrix{Float64}(I, 2, 2)
    gaussian = fit_bayesian_gcomp(data, :Y, :A, Symbol[];
        prior = ConjugateGaussianPrior(mean = zeros(2), covariance = V, shape = 2, scale = 1))
    logistic = fit_bayesian_logistic(data, :Y, :A, Symbol[];
        prior = GaussianCoefficientPrior(mean = zeros(2), covariance = V))
    for fit in (gaussian, logistic)
        @test_throws ArgumentError bayesian_policy_contrast(fit, data, discrete_static_policy(2); deltas = [0])
        @test_throws ArgumentError bayesian_policy_contrast(fit, data,
            discrete_shift_policy((a, w) -> 2); deltas = [0])
        @test_throws ArgumentError bayesian_policy_contrast(fit, DataFrame(A = [2]),
            discrete_static_policy(1); deltas = [0])
        # Support comes from fitting rows, rather than this target subset.
        result = bayesian_policy_contrast(fit, data[data.A .== 0, :],
            discrete_static_policy("1"); deltas = [0])
        @test all(isfinite, result.mean)
    end
end
