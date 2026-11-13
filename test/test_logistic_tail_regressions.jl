using Test, LinearAlgebra, Distributions
isdefined(@__MODULE__, :LogisticTailReference) || include("logistic_tail_reference.jl")

@testset "Bivariate reference regressions" begin
    reference = LogisticTailReference
    μ, V = [0.3, -0.4], [1.0 0.3; 0.3 2.0]
    working = reference.bivariate_policy_reference(zeros(0, 2), Float64[], μ, V;
        rtol = 1e-5, atol = 1e-8)
    tight = reference.bivariate_policy_reference(zeros(0, 2), Float64[], μ, V;
        rtol = 1e-9, atol = 1e-12)
    @testset "Prior-only coefficient probabilities" begin
        for index in 1:2
            @test reference.coefficient_cdf(tight, index, μ[index]) ≈ 0.5 atol = 1e-7
            gaussian = Normal(μ[index], sqrt(V[index, index]))
            lower, upper = quantile(gaussian, 0.1), quantile(gaussian, 0.9)
            content = reference.coefficient_interval_content(tight, index, lower, upper)
            @test content.content ≈ 0.8 atol = 1e-7
            @test content.lower_tail ≈ 0.1 atol = 1e-7
            @test content.upper_tail ≈ 0.1 atol = 1e-7
            q = reference.coefficient_quantile(tight, index, 0.1)
            @test q ≈ lower atol = 2e-4
            @test reference.coefficient_cdf(tight, index, q) ≈ 0.1 atol = 1e-6
        end
    end
    moments = reference.bivariate_moments(tight)
    @test moments.mean ≈ μ atol = 1e-7
    @test moments.covariance ≈ V atol = 1e-7
    @test moments.mass_disagreement < 1e-7
    @testset "Actual proposed interval refinement" begin
        # Neither proposed interval is the reference's equal-tailed 95% interval.
        coefficient = reference.refine_coefficient_interval(working, tight, 2, -0.8, 0.6)
        @test coefficient.converged
        @test coefficient.content ≈ cdf(Normal(μ[2], sqrt(V[2, 2])), 0.6) -
            cdf(Normal(μ[2], sqrt(V[2, 2])), -0.8) atol = 1e-6
        policy = reference.refine_policy_interval(working, tight, -0.1, 0.3)
        @test policy.converged
        @test policy.content ≈ reference.policy_cdf(tight, 0.3) - reference.policy_cdf(tight, -0.1) atol = 1e-7
        # A deliberately misnormalised working rule must not validate these intervals.
        bad = reference.BivariatePolicyReference(working.X, working.y, working.μ,
            working.L, working.mode, working.log_peak, 1.1working.mass,
            working.mass_error, working.rtol, working.atol)
        @test !reference.refine_coefficient_interval(bad, tight, 2, -0.8, 0.6).converged
        @test !reference.refine_policy_interval(bad, tight, -0.1, 0.3).converged
        # Opposite tail errors can cancel exactly in total interval content.
        cancellation = reference.interval_refinement((probability = 0.02, error = 0.0),
            (probability = 0.03, error = 0.0), (probability = 0.025, error = 0.0),
            (probability = 0.025, error = 0.0))
        @test cancellation.content_change ≈ 0.0 atol = eps()
        @test cancellation.tail_change ≈ 0.005
        @test !cancellation.converged
        # Stable total content can mask unstable individual tails.
        @test !reference.refine_coefficient_interval(bad, tight, 2, -0.8, 0.6;
            content_tol = 1.0, tail_tol = 1e-5).converged
        uncertain = reference.BivariatePolicyReference(working.X, working.y, working.μ,
            working.L, working.mode, working.log_peak, working.mass,
            0.1working.mass, working.rtol, working.atol)
        @test !reference.refine_policy_interval(uncertain, tight, -0.1, 0.3).converged
        @test_throws ArgumentError reference.refine_policy_interval(working, tight, 0.3, -0.1)
    end
end
