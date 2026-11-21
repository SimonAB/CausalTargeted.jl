using Test, LinearAlgebra, Distributions, Random, StableRNGs, Statistics
using QuadGK: quadgk

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "stress", "logistic_psis_reference.jl"))
isdefined(@__MODULE__, :PolicyProbabilityReference) ||
    include(joinpath(@__DIR__, "stress", "policy_probability_reference.jl"))

const EXPECTED_PSIS_FIELDS = (
    :available, :failure_reason, :khat, :ess, :mcse, :mcse_to_sd,
    :beta_mean, :beta_cov, :policy_mean, :policy_sd,
    :policy_lower, :policy_upper, :policy_draws, :weights, :seconds,
)

@inline _test_logistic(x::Real) = x >= 0 ? inv(1.0 + exp(-x)) : (ex = exp(x); ex / (1.0 + ex))

function _exact_arm_mean(prob_ref)
    val, _ = quadgk(
        u -> u >= 0 ? PolicyProbabilityReference.probability_survival(prob_ref, u) :
            -PolicyProbabilityReference.probability_cdf(prob_ref, u),
        -1.0, 0.0, 1.0;
        rtol = 1e-4, atol = 1e-5,
    )
    return val
end

function _simulate_multicovariate_dataset(; n::Int, rho::Float64, interact::Bool, seed::Int)
    rng = StableRNG(seed)
    Σ_w = [1.0 rho; rho 1.0]
    W = Matrix(transpose(rand(rng, MvNormal(zeros(2), Σ_w), n)))
    w1, w2 = W[:, 1], W[:, 2]
    p_a = _test_logistic.(0.5 .* w1 .- 0.4 .* w2)
    a = Float64.(rand(rng, n) .< p_a)
    η = -0.5 .+ 1.0 .* a .+ 0.6 .* w1 .- 0.5 .* w2 .+ (interact ? 0.9 .* a .* w1 : 0.0)
    y = Float64.(rand(rng, n) .< _test_logistic.(η))
    if interact
        X = hcat(ones(n), a, w1, w2, a .* w1)
        X_alt = hcat(ones(n), ones(n), w1, w2, w1)
    else
        X = hcat(ones(n), a, w1, w2)
        X_alt = hcat(ones(n), ones(n), w1, w2)
    end
    return (; X, y, X_alt, X_obs = X)
end

@testset "Logistic PSIS Reference" begin
    @testset "1. 2D anchor agreement against PolicyProbabilityReference" begin
        anchors = [
            (name = "balanced", counts = (20, 7, 20, 14),
                μ0 = [0.0, 0.0], Σ0 = [2.25 0.0; 0.0 2.25]),
            (name = "unbalanced_rare_control", counts = (8, 2, 32, 22),
                μ0 = [0.0, 0.0], Σ0 = [2.25 0.0; 0.0 2.25]),
            (name = "correlated_prior", counts = (12, 3, 14, 10),
                μ0 = [-0.3, 0.8], Σ0 = [1.0 0.7; 0.7 2.0]),
        ]

        for (idx, a) in enumerate(anchors)
            @testset "$(a.name)" begin
                n0, y0, n1, y1 = a.counts
                n = n0 + n1
                X = vcat(repeat([1.0 0.0], n0), repeat([1.0 1.0], n1))
                y = vcat(ones(y0), zeros(n0 - y0), ones(y1), zeros(n1 - y1))
                X_alt = repeat([1.0 1.0], n)
                X_obs = X
                cohort_scale = n0 / n

                exact_ref = PolicyProbabilityReference.probability_reference(
                    n0, y0, n1, y1, a.μ0, a.Σ0; rtol = 1e-6, atol = 1e-9)
                exact_mean = cohort_scale * _exact_arm_mean(exact_ref)
                exact_lower = cohort_scale * PolicyProbabilityReference.probability_quantile(exact_ref, 0.025)
                exact_upper = cohort_scale * PolicyProbabilityReference.probability_quantile(exact_ref, 0.975)

                psis_ref = evaluate_logistic_psis_reference(
                    X, y, a.μ0, a.Σ0, X_alt, X_obs;
                    rng = StableRNG(20261121 + idx),
                )

                mean_err = abs(psis_ref.policy_mean - exact_mean)
                lower_err = abs(psis_ref.policy_lower - exact_lower)
                upper_err = abs(psis_ref.policy_upper - exact_upper)
                println("2D anchor [$(a.name)]: mean_err=$(round(mean_err, digits=6)), " *
                        "lower_err=$(round(lower_err, digits=6)), upper_err=$(round(upper_err, digits=6)), " *
                        "khat=$(round(psis_ref.khat, digits=4)), ess=$(round(psis_ref.ess, digits=1))")

                @test propertynames(psis_ref) == EXPECTED_PSIS_FIELDS
                @test psis_ref.available
                @test psis_ref.failure_reason == ""
                @test psis_ref.khat < 0.7
                @test psis_ref.ess >= 400.0
                @test psis_ref.mcse_to_sd <= 0.05
                @test mean_err <= 0.01
                @test lower_err <= 0.015
                @test upper_err <= 0.015
            end
        end
    end

    @testset "2. Multi-covariate (4D and 5D) datasets pass gates" begin
        configs = [
            (name = "4D_baseline", n = 100, rho = 0.0, interact = false, seed = 401),
            (name = "4D_correlated", n = 120, rho = 0.7, interact = false, seed = 402),
            (name = "5D_interaction", n = 120, rho = 0.3, interact = true, seed = 501),
        ]

        for cfg in configs
            @testset "$(cfg.name)" begin
                ds = _simulate_multicovariate_dataset(;
                    n = cfg.n, rho = cfg.rho, interact = cfg.interact, seed = cfg.seed)
                p = size(ds.X, 2)
                μ0 = zeros(p)
                Σ0 = 2.25 * Matrix{Float64}(I, p, p)

                res = evaluate_logistic_psis_reference(
                    ds.X, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
                    rng = StableRNG(20261121 + cfg.seed),
                )

                @test propertynames(res) == EXPECTED_PSIS_FIELDS
                @test res.available
                @test res.failure_reason == ""
                @test res.khat < 0.7
                @test res.ess >= 400.0
                @test res.mcse_to_sd <= 0.05
                @test res.seconds >= 0.0 && isfinite(res.seconds)

                @test length(res.beta_mean) == p
                @test all(isfinite, res.beta_mean)
                @test size(res.beta_cov) == (p, p)
                @test all(isfinite, res.beta_cov)
                @test issymmetric(res.beta_cov)
                @test isposdef(res.beta_cov)

                @test length(res.weights) == 20_000
                @test length(res.policy_draws) == 20_000
                @test all(isfinite, res.weights)
                @test sum(res.weights) ≈ 1.0 atol = 1e-12
                @test res.ess ≈ 1.0 / sum(abs2, res.weights)

                @test isfinite(res.policy_mean)
                @test isfinite(res.policy_sd) && res.policy_sd > 0.0
                @test res.policy_lower < res.policy_mean < res.policy_upper
                @test res.mcse ≈ res.policy_sd / sqrt(res.ess)
                @test res.mcse_to_sd ≈ res.mcse / res.policy_sd

                content = weighted_interval_content(
                    res.policy_draws, res.weights, res.policy_lower, res.policy_upper)
                @test 0.94 <= content <= 0.96
                @test isnan(weighted_interval_content(res.policy_draws, res.weights, NaN, res.policy_upper))
                @test isnan(weighted_interval_content(res.policy_draws, res.weights, res.policy_lower, Inf))
                @test weighted_interval_content(res.policy_draws, res.weights, 1.0, -1.0) == 0.0
            end
        end
    end

    @testset "3. Gate failure contract" begin
        ds = _simulate_multicovariate_dataset(n = 80, rho = 0.3, interact = false, seed = 601)
        p = size(ds.X, 2)
        μ0 = zeros(p)
        Σ0 = 2.25 * Matrix{Float64}(I, p, p)

        ess_fail = evaluate_logistic_psis_reference(
            ds.X, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
            ess_min = 1e9, rng = StableRNG(701),
        )
        @test propertynames(ess_fail) == EXPECTED_PSIS_FIELDS
        @test !ess_fail.available
        @test ess_fail.failure_reason == "ess_gate"

        khat_fail = evaluate_logistic_psis_reference(
            ds.X, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
            khat_max = -1.0, rng = StableRNG(702),
        )
        @test !khat_fail.available
        @test khat_fail.failure_reason == "khat_gate"

        mcse_fail = evaluate_logistic_psis_reference(
            ds.X, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
            mcse_sd_max = 1e-9, rng = StableRNG(703),
        )
        @test !mcse_fail.available
        @test mcse_fail.failure_reason == "mcse_gate"

        bad_cov_fail = evaluate_logistic_psis_reference(
            ds.X[:, 1:2], ds.y, zeros(2), [1.0 2.0; 2.0 1.0],
            ds.X_alt[:, 1:2], ds.X_obs[:, 1:2];
            rng = StableRNG(704),
        )
        @test !bad_cov_fail.available
        @test bad_cov_fail.failure_reason == "hessian_not_pd"

        X_nan = copy(ds.X)
        X_nan[1, 1] = NaN
        newton_fail = evaluate_logistic_psis_reference(
            X_nan, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
            rng = StableRNG(705),
        )
        @test !newton_fail.available
        @test newton_fail.failure_reason == "newton_failed"
    end

    @testset "4. Deterministic reproducibility" begin
        ds = _simulate_multicovariate_dataset(n = 80, rho = 0.3, interact = true, seed = 801)
        p = size(ds.X, 2)
        μ0 = zeros(p)
        Σ0 = 2.25 * Matrix{Float64}(I, p, p)

        r1 = evaluate_logistic_psis_reference(
            ds.X, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
            n_draws = 4_000, rng = StableRNG(9999),
        )
        r2 = evaluate_logistic_psis_reference(
            ds.X, ds.y, μ0, Σ0, ds.X_alt, ds.X_obs;
            n_draws = 4_000, rng = StableRNG(9999),
        )

        @test r1.available && r2.available
        @test r1.failure_reason == r2.failure_reason
        @test r1.khat == r2.khat
        @test r1.ess == r2.ess
        @test r1.mcse == r2.mcse
        @test r1.mcse_to_sd == r2.mcse_to_sd
        @test r1.beta_mean == r2.beta_mean
        @test r1.beta_cov == r2.beta_cov
        @test r1.policy_mean == r2.policy_mean
        @test r1.policy_sd == r2.policy_sd
        @test r1.policy_lower == r2.policy_lower
        @test r1.policy_upper == r2.policy_upper
        @test r1.policy_draws == r2.policy_draws
        @test r1.weights == r2.weights
    end
end
