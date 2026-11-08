using CausalTargeted
using DataFrames
using Distributions
using Random
using StableRNGs
using Statistics
using Test

@testset "two-part discrete LMTP" begin
    rng = StableRNG(11)
    n = 500
    arms = ["R", "SS", "SC"]
    A = rand(rng, arms, n)
    W = randn(rng, n)
    p_pres = [a == "SS" ? 0.65 : a == "SC" ? 0.45 : 0.30 for a in A]
    pres = [rand(rng, Bernoulli(p)) for p in p_pres]
    shift = [a == "SS" ? 1.2 : a == "SC" ? 0.4 : 0.0 for a in A]
    intensity = [
        p ? max(0.1, 0.5 + 0.3 * W[i] + shift[i] + 0.2 * randn(rng)) : 0.0
        for (i, p) in enumerate(pres)
    ]
    df = DataFrame(A = A, W = W, Y_pres = Float64.(pres), Y_int = intensity)

    @testset "arm contrast" begin
        res = run_two_part_discrete_lmtp_contrast(
            df, :A;
            presence = :Y_pres,
            intensity = :Y_int,
            arm_hi = "SS",
            arm_ref = "R",
            levels = arms,
            baseline = [:W],
            folds = 3,
            learners_outcome = (:glm, :mean),
            weight_truncation = false,
            rng = StableRNG(12),
        )
        @test isfinite(res.presence.estimate)
        @test isfinite(res.presence.se)
        @test res.presence.n == n
        @test isfinite(res.intensity.estimate)
        @test isfinite(res.intensity.se)
        @test res.presence.se ≈ std(res.presence.hi.ic .- res.presence.ref.ic) / sqrt(n)
        @test res.intensity.se ≈ std(res.intensity.hi.ic .- res.intensity.ref.ic) /
            sqrt(res.intensity.n)
        @test res.intensity.n == count(>(0), df.Y_pres)
        @test res.intensity.conditional_on === :Y_pres
        @test res.contrast == "SS_vs_R"
    end

    @testset "vector clusters follow positive-intensity rows" begin
        cluster_ids = repeat(1:div(n, 5); inner = 5)
        df_missing = copy(df)
        df_missing.Y_pres = Union{Missing, Float64}[value for value in df.Y_pres]
        df_missing.Y_pres[1:8] .= missing
        result = run_two_part_discrete_lmtp_contrast(
            df_missing, :A;
            presence = :Y_pres, intensity = :Y_int,
            arm_hi = "SS", arm_ref = "R", levels = arms,
            baseline = [:W], folds = 2, learners_outcome = (:glm, :mean),
            cluster = cluster_ids, handle_missing = :drop, rng = StableRNG(19),
        )
        @test result.presence.hi.covariance_kind == :cluster
        @test result.intensity.hi.covariance_kind == :cluster
        @test length(result.intensity.hi.cluster) == result.intensity.n
    end

    @testset "single policy + execute_estimand" begin
        policy = discrete_static_policy("SS"; levels = arms)
        single = run_two_part_discrete_lmtp(
            df, :A;
            presence = :Y_pres,
            intensity = :Y_int,
            policy = policy,
            baseline = [:W],
            folds = 3,
            learners_outcome = (:glm, :mean),
            rng = StableRNG(13),
        )
        @test isfinite(single.presence.estimate)
        @test isfinite(single.intensity.estimate)
        @test single.n_intensity == count(>(0), df.Y_pres)

        estimand = TwoPartInterventionalMean(:A, :Y_pres, :Y_int, [:W], policy)
        @test estimand_engine(estimand) === :two_part_discrete_lmtp
        grid = execute_estimand(estimand, df; folds = 2, rng = StableRNG(14), metadata = false)
        @test nrow(grid) == 2
        @test all(isfinite, grid.est)
    end

    @testset "synthetic recovery (#35 DGP)" begin
        rng = StableRNG(7)
        df = DataFrame(
            id = repeat(1:100; inner = 3),
            arm = rand(rng, ["R", "SS"], 300),
        )
        logit_p = @. -1.0 + (df.arm == "SS") * 0.6
        p = 1.0 ./ (1.0 .+ exp.(-logit_p))
        df.y = zeros(300)
        for i in 1:300
            if rand(rng) < p[i]
                df.y[i] = exp(randn(rng) * 0.5 + (df.arm[i] == "SS") * 0.4)
            end
        end
        df.y_bin = Float64.(df.y .> 0)
        df.y_log = [y > 0 ? log(y) : missing for y in df.y]

        res = run_two_part_discrete_lmtp_contrast(
            df, :arm;
            presence = :y_bin,
            intensity = :y_log,
            arm_hi = "SS",
            arm_ref = "R",
            levels = ["R", "SS"],
            baseline = Symbol[],
            cluster = :id,
            folds = 3,
            learners_outcome = (:glm, :mean),
            rng = StableRNG(8),
        )
        @test res.presence.estimate > 0.05
        @test res.intensity.estimate > 0.05
        @test res.presence.hi.covariance_kind == :cluster
    end
end
