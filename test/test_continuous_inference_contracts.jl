using Test, StableRNGs

@testset "Continuous policy intervals declare their approximation" begin
    data, _ = CausalTargeted.simulate_linear_mtp(100; rng = StableRNG(991))
    grid = CausalTargeted.run_lmtp_grid(data, :A, :Y; baseline = [:W],
        deltas = [0.0, 0.4], folds = 2, simultaneous = false,
        parallel = false, rng = StableRNG(992))
    @test all(grid[grid.delta .!= 0, :inference_status] .== :approximate_clamped_policy)
    @test all(grid.truncation_selection .== :prespecified)
    point = CausalTargeted.run_lmtp_contrast(data, :A, :Y; baseline = [:W],
        folds = 2, rng = StableRNG(993))
    @test point.inference_status === :unsupported_continuous_point_intervention
    @test isnan(point.se) && isnan(point.lower) && isnan(point.upper)
    @test isfinite(point.estimate)
end
