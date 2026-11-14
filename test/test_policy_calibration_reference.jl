using Test, LinearAlgebra, Distributions
isdefined(@__MODULE__, :LogisticTailReference) || include("logistic_tail_reference.jl")

@testset "Frequency likelihood reference" begin
    reference = LogisticTailReference
    X = [1.0 0; 1 0; 1 1; 1 1]
    y, counts = [0.0, 1, 0, 1], [4, 2, 1, 5]
    μ, V = [0.1, -0.2], [1.0 0.2; 0.2 1.5]
    grouped = reference.bivariate_policy_reference(X, y, μ, V; weights = counts)
    expanded_X = reduce(vcat, [repeat(X[i:i, :], counts[i], 1) for i in 1:4])
    expanded_y = reduce(vcat, [fill(y[i], counts[i]) for i in 1:4])
    expanded = reference.bivariate_policy_reference(expanded_X, expanded_y, μ, V)
    for z in ([0.0, 0], [0.3, -0.8], [-1.2, 0.7])
        @test reference.bivariate_log_kernel(z, X, y, μ, grouped.L, counts) ≈
            reference.bivariate_log_kernel(z, expanded_X, expanded_y, μ, expanded.L) atol = 1e-12
    end
    for contrast in (-0.1, 0.0, 0.2)
        @test reference.policy_cdf(grouped, contrast) ≈ reference.policy_cdf(expanded, contrast) atol = 1e-7
    end
    @test grouped.mode ≈ expanded.mode atol = 1e-9
    @test grouped.mass ≈ expanded.mass rtol = 1e-7
    @test_throws ArgumentError reference.bivariate_policy_reference(X, y, μ, V; weights = [4, 2, -1, 5])
    @test_throws ArgumentError reference.bivariate_policy_reference(X, y, μ, V; weights = [4, 2, 1.5, 5])
    @test_throws DimensionMismatch reference.bivariate_policy_reference(X, y, μ, V; weights = [4, 2])
end

include("stress/policy_calibration_reference.jl")
@testset "CDF decision gates" begin
    p = (cdf = 0.5, survival = 0.5, error = 1e-10, log_mass = log(2π))
    good = assess_policy_cdf(p, p)
    @test good.available && good.covered
    @test good.pit == 0.5
    outside = merge(p, (cdf = 0.99, survival = 0.01,))
    @test assess_policy_cdf(outside, outside).available
    @test !assess_policy_cdf(outside, outside).covered
    boundary = merge(p, (cdf = 0.025, survival = 0.975,))
    @test assess_policy_cdf(boundary, boundary).status == :boundary_ambiguous
    @test ismissing(assess_policy_cdf(boundary, boundary).covered)
    @test assess_policy_cdf(merge(p, (cdf = 0.51, survival = 0.49,)), p).status == :cdf_disagreement
    @test assess_policy_cdf(merge(p, (log_mass = p.log_mass + 0.1,)), p).status == :mass_disagreement
    @test assess_policy_cdf(merge(p, (error = 1e-2,)), p).status == :integration_error
    @test assess_policy_cdf(merge(p, (cdf = NaN,)), p).status == :invalid_probability
    @test assess_policy_cdf(merge(p, (cdf = 0.6, survival = 0.5,)),
        merge(p, (cdf = 0.6, survival = 0.5,))).status == :integration_error
    @test_throws ArgumentError assess_policy_cdf(p, p; cdf_tol = -1)
end

@testset "Gaussian policy PIT has analytic zero-contrast check" begin
    μ, V = [0.3, -0.4], [1.0 0.3; 0.3 2.0]
    result = checked_policy_cdf(zeros(0, 2), Float64[], μ, V, 0.0; gaussian = true)
    @test result.available
    @test result.pit ≈ Distributions.cdf(Distributions.Normal(μ[2], sqrt(V[2, 2])), 0.0) atol = 1e-7
    probes = [checked_policy_cdf(zeros(0, 2), Float64[], μ, V, x; gaussian = true)
        for x in (-1.0, -0.2, 0.2, 1.0)]
    @test all(r.available for r in probes)
    @test issorted([r.pit for r in probes])
    @test first(probes).pit ≈ 0.0 atol = 1e-7
    @test last(probes).pit ≈ 1.0 atol = 1e-7
    @test_throws ArgumentError checked_policy_cdf(ones(2, 2), [0, 1], μ, V, 0.0; gaussian = true)
end

@testset "Nested quadrature errors are retained" begin
    reference = LogisticTailReference
    μ, V = [0.3, -0.4], [1.0 0.3; 0.3 2.0]
    r = reference.bivariate_policy_reference(zeros(4, 2), [0, 1, 0, 1], μ, V)
    detail = reference.policy_integral_details(r, 0.0)
    @test detail.inner_error > 0
    @test detail.error ≈ detail.outer_error + detail.inner_error
    tail = reference.policy_tail_details(r, 0.0)
    @test tail.error ≈ tail.outer_error + tail.inner_error + tail.normalisation_error
    @test tail.inner_error > 0
    @test tail.probability ≈ cdf(Normal(μ[2], sqrt(V[2, 2])), 0.0) atol = 1e-7
    coefficient = reference.coefficient_tail_details(r, 1, μ[1])
    @test coefficient.inner_error > 0
    @test coefficient.error ≈ coefficient.outer_error + coefficient.inner_error + coefficient.normalisation_error
    @test coefficient.probability ≈ 0.5 atol = 1e-7
end
