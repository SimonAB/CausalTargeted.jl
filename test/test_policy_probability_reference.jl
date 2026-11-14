using Test, LinearAlgebra, Distributions
BLAS.set_num_threads(1)
include("stress/policy_probability_reference.jl")
isdefined(@__MODULE__, :LogisticTailReference) || include("logistic_tail_reference.jl")

@testset "Independent probability-space policy reference" begin
    P = PolicyProbabilityReference
    T = LogisticTailReference
    fixtures = [
        (name = "prior only", counts = (0, 0, 0, 0), μ = [0.0, 0.0], V = [1.0 0.0; 0.0 1.0]),
        (name = "balanced", counts = (10, 4, 10, 7), μ = zeros(2), V = [2.25 0.0; 0.0 2.25]),
        (name = "one control", counts = (1, 0, 19, 12), μ = zeros(2), V = [2.25 0.0; 0.0 2.25]),
        (name = "complete separation", counts = (10, 0, 10, 10), μ = zeros(2), V = [2.25 0.0; 0.0 2.25]),
        (name = "quasi separation", counts = (10, 0, 10, 9), μ = zeros(2), V = [2.25 0.0; 0.0 2.25]),
        (name = "strong prior", counts = (10, 0, 10, 10), μ = [0.5, -0.5], V = [0.25 0.0; 0.0 0.25]),
        (name = "correlated prior", counts = (8, 3, 12, 9), μ = [-0.3, 0.8], V = [1.0 0.7; 0.7 2.0]),
    ]
    for f in fixtures
        println("Probability reference fixture: ", f.name)
        flush(stdout)
        @testset "$(f.name)" begin
            n₀, y₀, n₁, y₁ = f.counts
            X = vcat(repeat([1.0 0.0], n₀), repeat([1.0 1.0], n₁))
            y = vcat(ones(y₀), zeros(n₀-y₀), ones(y₁), zeros(n₁-y₁))
            coarse = P.probability_reference(f.counts..., f.μ, f.V; rtol = 1e-6, atol = 1e-9)
            fine = P.probability_reference(f.counts..., f.μ, f.V; rtol = 1e-8, atol = 1e-11)
            other_coarse = T.bivariate_policy_reference(X, y, f.μ, f.V; rtol = 1e-6, atol = 1e-9)
            other = T.bivariate_policy_reference(X, y, f.μ, f.V; rtol = 1e-8, atol = 1e-11)
            @test coarse.mass_error / coarse.mass <= 1e-5
            @test fine.mass_error / fine.mass <= 1e-5
            @test other.mass_error / other.mass <= 1e-5
            @test other_coarse.mass_error / other_coarse.mass <= 1e-5
            @test abs(coarse.log_peak + log(coarse.mass) - fine.log_peak - log(fine.mass)) <= 1e-5
            @test abs(other_coarse.log_peak + log(other_coarse.mass) - other.log_peak - log(other.mass)) <= 1e-5
            @test P.probability_cdf(fine, -1) == 0
            @test P.probability_cdf(fine, 1) == 1
            @test P.probability_survival(fine, -1) == 1
            @test P.probability_survival(fine, 1) == 0
            for u in (-0.3, 0.0, 0.3)
                detail = P.probability_tail_details(fine, u)
                upper = P.probability_tail_details(fine, u; upper_tail = true)
                @test max(detail.error, upper.error) <= 1e-5
                @test detail.probability + upper.probability ≈ 1 atol = 1e-5
                @test P.probability_cdf(coarse, u) ≈ detail.probability atol = 1e-5
                @test detail.probability ≈ T.policy_cdf(other, u) atol = 5e-5
                other_lower = T.policy_tail_details(other, u)
                other_upper = T.policy_tail_details(other, u; upper_tail = true)
                @test max(other_lower.error, other_upper.error) <= 1e-5
                @test other_lower.probability + other_upper.probability ≈ 1 atol = 1e-5
                @test T.policy_cdf(other_coarse, u) ≈ other_lower.probability atol = 1e-5
            end
            for p in (0.025, 0.975)
                q = P.probability_quantile(fine, p)
                @test q ≈ T.policy_quantile(other, p) atol = 1e-4
                @test P.probability_cdf(fine, q) ≈ p atol = 1e-5
                @test T.policy_cdf(other, q) ≈ p atol = 5e-5
                @test P.probability_cdf(coarse, q) ≈ p atol = 1e-5
                @test T.policy_cdf(other_coarse, q) ≈ p atol = 1e-5
                @test P.probability_tail_details(fine, q).error <= 1e-5
                @test T.policy_tail_details(other, q).error <= 1e-5
            end
            if n₀ + n₁ == 0
                @test P.probability_cdf(fine, 0) ≈ 0.5 atol = 1e-6
                @test fine.log_peak + log(fine.mass) ≈ 0 atol = 1e-6
            end
        end
    end
    @test_throws ArgumentError P.probability_reference(2, 3, 2, 1, zeros(2), Matrix{Float64}(I, 2, 2))
    @test_throws ArgumentError P.probability_reference(2, 1, 2, 1, zeros(2), [1.0 2.0; 2.0 1.0])
    prior = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
    for p in ((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0))
        @test P.log_density(prior, (0, 0, 0, 0), p...) == -Inf
    end
end
