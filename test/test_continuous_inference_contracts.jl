using Test, StableRNGs, DataFrames

@testset "Smooth bounded logit-shift policy" begin
    policy = CausalTargeted.smooth_bounded_logit_shift_policy(0.0, 1.0)
    exposure = [0.05, 0.2, 0.5, 0.8, 0.95]
    shifted = CausalTargeted.apply_policy_values(exposure, 0.35, policy)
    restored = CausalTargeted.apply_policy_values(shifted, -0.35, policy)

    @test all(0 .< shifted .< 1)
    @test issorted(shifted)
    @test restored ≈ exposure
    mask = BitVector([true, false, true, false, true])
    masked = CausalTargeted.apply_policy_values(exposure, 0.35, policy; stratum_mask = mask)
    @test masked[.!mask] == exposure[.!mask]
    @test masked[mask] ≈ shifted[mask]
    @test_throws ArgumentError CausalTargeted.smooth_bounded_logit_shift_policy(1.0, 1.0)

    data_rng = StableRNG(994)
    z = randn(data_rng, 240)
    a = 1 ./ (1 .+ exp.(-z))
    w = randn(data_rng, length(a))
    y = 1.1 .+ 0.5 .* a .+ 0.6 .* w .+ 0.2 .* randn(data_rng, length(a))
    data = DataFrame(A = a, W = w, Y = y)
    grid = CausalTargeted.run_lmtp_grid(
        data, :A, :Y;
        baseline = [:W], deltas = [0.0, 0.35],
        policy = policy, weight_truncation = false,
        learners_outcome = (:glm,), learners_trt = (:glm,),
        folds = 3, simultaneous = false, parallel = false,
        rng = StableRNG(995),
    )
    smooth = grid[grid.delta .== 0.35, :]
    @test only(smooth.inference_status) === :available
    @test isfinite(only(smooth.se))
    @test isfinite(only(smooth.lwr)) && isfinite(only(smooth.upr))
    @test only(smooth.lwr) < only(smooth.est) < only(smooth.upr)

    uncached = CausalTargeted.run_lmtp_grid(
        data, :A, :Y;
        baseline = [:W], deltas = [0.35],
        policy = policy, weight_truncation = false,
        learners_outcome = (:glm,), learners_trt = (:glm,),
        folds = 3, simultaneous = false, parallel = false,
        cache_nuisances = false, rng = StableRNG(995),
    )
    @test only(uncached.inference_status) === :available
    @test isfinite(only(uncached.se))

    outside = DataFrame(A = [-0.01, 0.2, 0.5, 0.8, 1.01], W = zeros(5), Y = ones(5))
    @test_throws ArgumentError CausalTargeted.run_lmtp_grid(
        outside, :A, :Y;
        baseline = [:W], deltas = [0.2], policy = policy,
        weight_truncation = false, folds = 2, simultaneous = false,
        parallel = false,
    )
end

@testset "Continuous quantile-clamped policies do not claim inference" begin
    data, _ = CausalTargeted.simulate_linear_mtp(100; rng = StableRNG(991))
    grid = @test_logs (:warn, r"Hard-clamped continuous LMTP") CausalTargeted.run_lmtp_grid(
        data, :A, :Y; baseline = [:W],
        deltas = [0.0, 0.4], folds = 2, simultaneous = false,
        parallel = false, rng = StableRNG(992),
    )
    nonzero = grid[grid.delta .!= 0, :]
    @test all(nonzero.inference_status .== :unsupported_clamped_policy)
    @test all(isnan, nonzero.se)
    @test all(isnan, nonzero.lwr)
    @test all(isnan, nonzero.upr)
    @test all(grid.truncation_selection .== :prespecified)
    point = CausalTargeted.run_lmtp_contrast(data, :A, :Y; baseline = [:W],
        folds = 2, rng = StableRNG(993))
    @test point.inference_status === :unsupported_continuous_point_intervention
    @test isnan(point.se) && isnan(point.lower) && isnan(point.upper)
    @test isfinite(point.estimate)
end
