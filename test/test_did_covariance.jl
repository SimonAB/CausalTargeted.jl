using CausalTargeted
using DataFrames
using StableRNGs
using LinearAlgebra
using Statistics
using Test

@testset "staggered DiD aggregates full unit-level covariance" begin
    df, _ = CausalTargeted.simulate_did_staggered(120; rng = StableRNG(773))
    result = run_did_staggered(df)
    contribution = metadata(result, "causal_targeted_did_unit_influence")
    @test size(contribution.influence) == (120, nrow(result))
    @test contribution.unit_ids == collect(1:120)
    aggregate = aggregate_did(result)
    weights = fill(1 / nrow(result), nrow(result))
    expected_covariance = transpose(contribution.influence) * contribution.influence
    @test aggregate.covariance ≈ expected_covariance
    @test aggregate.se^2 ≈ dot(weights, expected_covariance * weights)
    @test aggregate.covariance_status == :available
    @test diag(aggregate.covariance) ≈ result.se .^ 2
    reversed = result[end:-1:1, :]
    @test aggregate_did(reversed).se ≈ aggregate.se
    subset = result[1:2, :]
    @test aggregate_did(subset).covariance ≈ aggregate.covariance[1:2, 1:2]
end

@testset "cell-only DiD aggregation marks covariance unavailable" begin
    cells = DataFrame(att = [0.8, 1.2], se = [0.2, 0.3])
    aggregate = aggregate_did(cells)
    @test aggregate.att ≈ 1.0
    @test isnan(aggregate.se)
    @test aggregate.independence_se ≈ sqrt(mean(cells.se .^ 2) / 2)
    @test aggregate.covariance === nothing
    @test aggregate.covariance_status == :unavailable
end
