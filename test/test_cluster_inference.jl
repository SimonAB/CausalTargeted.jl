using CausalTargeted
using DataFrames
using Distributions
using Random
using StableRNGs
using Statistics
using Test

@testset "cluster-aware cross-fitting" begin
    ids = repeat(1:12; inner = 3)
    folds = CausalTargeted.crossfit_indices(36, 4, StableRNG(101); cluster = ids)
    @test length(folds) == 4
    @test sort(vcat(folds...)) == collect(1:36)
    for cluster_id in unique(ids)
        @test count(fold -> any(i -> ids[i] == cluster_id, fold), folds) == 1
    end
    @test_throws ArgumentError CausalTargeted.crossfit_indices(4, 3, StableRNG(102); cluster = [1, 1, 1, 1])
end

@testset "discrete classifier returns policy-to-natural ratios" begin
    p_observed = [0.2, 0.8, 0.5]
    followed = [true, true, false]
    ratio = CausalTargeted._discrete_policy_ratio(
        p_observed; trunc = 100.0, support = followed,
    )
    @test ratio ≈ [4.0, 0.25, 0.0]
end

@testset "discrete cluster intervals use finite-cluster correction" begin
    ic = [-1.0, 0.4, 0.6, -0.2, 0.5, -0.3, 0.1, -0.1]
    cluster = [1, 1, 2, 2, 3, 3, 4, 4]
    estimate = 0.25
    interval = CausalTargeted._discrete_influence_interval(
        estimate, ic; cluster = cluster,
    )
    expected_se = sqrt(
        CausalTargeted.cluster_robust_variance(ic; cluster = cluster) * 4 / 3,
    )
    critical = quantile(Distributions.TDist(3), 0.975)
    @test interval.se ≈ expected_se
    @test interval.lower ≈ estimate - critical * expected_se
    @test interval.upper ≈ estimate + critical * expected_se
    @test interval.df == 3
    @test interval.n_clusters == 4
end

@testset "discrete cluster inference alignment" begin
    rng = StableRNG(103)
    n = 120
    clusters = repeat(1:30; inner = 4)
    arms = rand(rng, ["R", "SS"], n)
    y = Union{Missing, Float64}[
        0.4 * (arms[i] == "SS") + 0.2 * clusters[i] + randn(rng) for i in 1:n
    ]
    y[1:12] .= missing
    df = DataFrame(arm = arms, y = y, w = randn(rng, n))

    result = run_discrete_lmtp(
        df, :arm, :y;
        policy = discrete_static_policy("SS"; levels = ["R", "SS"]),
        baseline = [:w],
        folds = 3,
        learners_outcome = (:glm, :mean),
        cluster = clusters,
        rng = StableRNG(104),
    )
    @test result.covariance_kind == :cluster
    retained = .!ismissing.(df.y)
    @test result.cluster == clusters[retained]
    @test isfinite(result.se)
end

@testset "paired discrete contrast score covariance" begin
    rng = StableRNG(105)
    n = 180
    cluster = repeat(1:45; inner = 4)
    arm = rand(rng, ["R", "SS", "SC"], n)
    df = DataFrame(
        arm = arm,
        y = 0.5 .* (arm .== "SS") .- 0.2 .* (arm .== "SC") .+
            0.15 .* cluster .+ randn(rng),
        cluster = cluster,
    )
    result = run_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = "SS", arm_ref = "R", levels = ["R", "SS", "SC"],
        baseline = Symbol[], folds = 3, learners_outcome = (:glm, :mean),
        cluster = :cluster, rng = StableRNG(106),
    )
    expected = CausalTargeted._discrete_influence_interval(
        result.estimate, result.hi.ic .- result.ref.ic; cluster = cluster,
    ).se
    @test result.se ≈ expected
    @test result.hi.cluster == result.ref.cluster == cluster
    @test result.nuisance_fit_counts == (outcome_model_fits = 3, treatment_model_fits = 6)
end

@testset "IPCW cluster variance is retained" begin
    rng = StableRNG(107)
    n = 160
    cluster = repeat(1:40; inner = 4)
    arm = rand(rng, ["R", "SS"], n)
    w = randn(rng, n)
    observed = rand(rng, n) .< (0.45 .+ 0.35 .* (1 ./(1 .+ exp.(-w))))
    y = Union{Missing, Float64}[observed[i] ? 0.3 * (arm[i] == "SS") + randn(rng) : missing for i in 1:n]
    df = DataFrame(arm = arm, y = y, w = w, cluster = cluster)
    result = run_discrete_lmtp(
        df, :arm, :y;
        policy = discrete_static_policy("SS"; levels = ["R", "SS"]),
        baseline = [:w], folds = 3, learners_outcome = (:glm, :mean),
        handle_missing = :ipcw, cluster = :cluster, rng = StableRNG(108),
    )
    @test result.covariance_kind == :cluster
    @test result.se ≈ CausalTargeted._discrete_influence_interval(
        result.estimate, result.ic; cluster = result.cluster,
    ).se
end
