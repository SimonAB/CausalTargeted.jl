using CausalTargeted
using DataFrames
using Random
using StableRNGs
using Statistics
using Test

@testset "discrete LMTP targeting and inference contract" begin
    rng = StableRNG(1601)
    n = 120
    w = randn(rng, n)
    a = rand(rng, ["0", "1"], n)
    y = 0.4 .* (a .== "1") .+ 0.2 .* w .+ randn(rng, n)
    df = DataFrame(A = a, W = w, Y = y)
    policy = discrete_static_policy("1"; levels = ["0", "1"])
    common = (; policy, baseline = [:W], folds = 3,
        learners_outcome = (:glm, :mean), learners_trt = (:logistic, :mean))

    capped = run_discrete_lmtp(df, :A, :Y; common..., rng = StableRNG(1602))
    @test capped.inference_status == :truncated_weights
    @test all(isfinite, capped.point_score)
    @test mean(capped.point_score) ≈ capped.estimate
    @test all(isnan, capped.ic)
    @test isnan(capped.se) && isnan(capped.lower) && isnan(capped.upper)

    uncapped = run_discrete_lmtp(
        df, :A, :Y; common..., weight_truncation = false, rng = StableRNG(1602),
    )
    @test uncapped.inference_status == :available
    @test isfinite(uncapped.se)
    @test all(isfinite, uncapped.ic)
    @test all(isfinite, uncapped.point_score)
    @test mean(uncapped.point_score) ≈ uncapped.estimate

    contrast = run_discrete_lmtp_contrast(
        df, :A, :Y; arm_hi = "1", arm_ref = "0", levels = ["0", "1"],
        baseline = [:W], folds = 3, learners_outcome = (:glm, :mean),
        learners_trt = (:logistic, :mean), rng = StableRNG(1603),
    )
    @test contrast.inference_status == :truncated_weights
    @test mean(contrast.point_score) ≈ contrast.estimate
    @test all(isfinite, contrast.point_score)
    @test all(isnan, contrast.ic)
    @test isnan(contrast.se) && isnan(contrast.lower) && isnan(contrast.upper)

    identity_policy = discrete_recode_policy(Dict("0" => "0", "1" => "1"))
    identity = run_discrete_lmtp(
        df, :A, :Y; policy = identity_policy, baseline = [:W], folds = 3,
        learners_outcome = (:glm, :mean), rng = StableRNG(1607),
    )
    @test identity.inference_status == :trivial
    @test identity.estimate == identity.se == identity.lower == identity.upper == 0.0
    @test all(iszero, identity.ic)
    @test all(iszero, identity.point_score)

    equal_arms = run_discrete_lmtp_contrast(
        df, :A, :Y; arm_hi = "1", arm_ref = "1", levels = ["0", "1"],
        baseline = [:W], folds = 3, learners_outcome = (:glm, :mean),
        rng = StableRNG(1608),
    )
    @test equal_arms.inference_status == :trivial
    @test equal_arms.estimate == equal_arms.se == equal_arms.lower == equal_arms.upper == 0.0
    @test all(iszero, equal_arms.ic)
    @test all(iszero, equal_arms.point_score)

    one_cluster = ones(Int, n)
    identity_cluster = run_discrete_lmtp(
        df, :A, :Y; policy = identity_policy, baseline = [:W], folds = 3,
        learners_outcome = (:glm, :mean), cluster = one_cluster, rng = StableRNG(1609),
    )
    @test identity_cluster.inference_status == :trivial
    @test identity_cluster.estimate == identity_cluster.se ==
        identity_cluster.lower == identity_cluster.upper == 0.0
    @test all(iszero, identity_cluster.ic)
    @test identity_cluster.n_clusters == 1
    @test_throws ArgumentError run_discrete_lmtp(
        df, :A, :Y; policy = identity_policy, baseline = [:W], estimator = :bogus,
        cluster = one_cluster, rng = StableRNG(1614),
    )

    equal_arms_cluster = run_discrete_lmtp_contrast(
        df, :A, :Y; arm_hi = "1", arm_ref = "1", levels = ["0", "1"],
        baseline = [:W], folds = 3, learners_outcome = (:glm, :mean),
        cluster = one_cluster, rng = StableRNG(1612),
    )
    @test equal_arms_cluster.inference_status == :trivial
    @test equal_arms_cluster.estimate == equal_arms_cluster.se ==
        equal_arms_cluster.lower == equal_arms_cluster.upper == 0.0
    @test all(iszero, equal_arms_cluster.ic)
    @test equal_arms_cluster.n_clusters == 1
    @test_throws ArgumentError run_discrete_lmtp_contrast(
        df, :A, :Y; arm_hi = "1", arm_ref = "1", levels = ["0", "1"],
        baseline = [:W], estimator = :bogus, cluster = one_cluster,
        rng = StableRNG(1615),
    )

    @test_throws ArgumentError run_discrete_lmtp(
        df, :A, :Y; policy, baseline = [:W], folds = 3,
        learners_outcome = (:glm, :mean), cluster = one_cluster, rng = StableRNG(1613),
    )

    y_miss = Union{Missing, Float64}[y...]
    y_miss[1:20] .= missing
    df_miss = DataFrame(A = a, W = w, Y = y_miss)
    ipcw = run_discrete_lmtp(
        df_miss, :A, :Y; common..., handle_missing = :ipcw,
        weight_truncation = false, rng = StableRNG(1604),
    )
    @test ipcw.inference_status == :unsupported_ipcw
    @test all(isfinite, ipcw.point_score)
    @test mean(ipcw.point_score) ≈ ipcw.estimate
    @test all(isnan, ipcw.ic)
    @test isnan(ipcw.se) && isnan(ipcw.lower) && isnan(ipcw.upper)

    @test_throws ArgumentError run_discrete_lmtp(
        df, :A, :Y; common..., fold_groups = [:misaligned], rng = StableRNG(1605),
    )
    @test_throws ArgumentError run_discrete_lmtp_contrast(
        df, :A, :Y; arm_hi = "1", arm_ref = "0", levels = ["0", "1"],
        baseline = [:W], folds = 3, fold_groups = [:misaligned],
        learners_outcome = (:glm, :mean), rng = StableRNG(1606),
    )
end

@testset "discrete bootstrap source groups remain distinct from covariance copies" begin
    fold_groups = repeat([:source1, :source1, :source2, :source2,
                          :source3, :source3, :source4, :source4]; inner = 3)
    cluster_ids = repeat(1:8; inner = 3)
    folds = CausalTargeted._discrete_lmtp_fold_sets(
        length(fold_groups), 2, StableRNG(1610);
        cluster_ids = cluster_ids, fold_groups = fold_groups,
    )
    row_fold = Dict(row => fold for (fold, rows) in enumerate(folds) for row in rows)
    for source in unique(fold_groups)
        rows = findall(==(source), fold_groups)
        @test length(unique(row_fold[row] for row in rows)) == 1
    end
    @test length(unique(cluster_ids)) > length(unique(fold_groups))

    split_group = copy(fold_groups)
    split_group[1] = :other_source
    @test_throws ArgumentError CausalTargeted._discrete_lmtp_fold_sets(
        length(split_group), 2, StableRNG(1611);
        cluster_ids = cluster_ids, fold_groups = split_group,
    )
end
