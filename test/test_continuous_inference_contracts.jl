using Test, StableRNGs

@testset "Continuous quantile-clamped policies do not claim inference" begin
    data, _ = CausalTargeted.simulate_linear_mtp(100; rng = StableRNG(991))
    grid = CausalTargeted.run_lmtp_grid(data, :A, :Y; baseline = [:W],
        deltas = [0.0, 0.4], folds = 2, simultaneous = false,
        parallel = false, rng = StableRNG(992))
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
