using Test
using StableRNGs
using Statistics

@testset "weighted influence summary uses sample variance" begin
    values = [1.0, 3.0, 5.0, 7.0]
    summary = CausalTargeted.weighted_influence_summary(values, ones(4))
    @test summary.se ≈ std(values) / sqrt(4)
    weighted = CausalTargeted.weighted_influence_summary(values, [1.0, 2.0, 3.0, 4.0])
    @test weighted.se ≈ std(weighted.ic) / sqrt(4)
end

@testset "multiplier simultaneous critical validation" begin
    scores = [1.0 0.0; -1.0 0.0; 2.0 0.0; -2.0 0.0]
    shifted = scores .+ reshape([100.0, -25.0], 1, :)
    c1 = CausalTargeted.multiplier_simultaneous_critical(
        scores; n_boot = 500, rng = StableRNG(901),
    )
    c2 = CausalTargeted.multiplier_simultaneous_critical(
        shifted; n_boot = 500, rng = StableRNG(901),
    )
    @test c1 ≈ c2
    @test isfinite(c1)
    @test CausalTargeted.multiplier_simultaneous_critical(
        zeros(4, 2); n_boot = 20, rng = StableRNG(902),
    ) == 0.0
    @test_throws ArgumentError CausalTargeted.multiplier_simultaneous_critical(ones(1, 1))
    @test_throws ArgumentError CausalTargeted.multiplier_simultaneous_critical(ones(4, 1); n_boot = 0)
    @test_throws ArgumentError CausalTargeted.multiplier_simultaneous_critical(ones(4, 1); alpha = 1.0)
    @test_throws ArgumentError CausalTargeted.multiplier_simultaneous_critical([1.0; NaN;;])
end
