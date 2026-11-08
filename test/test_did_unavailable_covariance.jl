using Test, CausalTargeted, DataFrames

@testset "DiD aggregation does not infer independence without unit information" begin
    cells = DataFrame(att = [1.0, 2.0], se = [0.2, 0.3])
    result = aggregate_did(cells)
    @test result.att == 1.5
    @test result.covariance_status === :unavailable
    @test isnan(result.se)
    @test result.independence_se ≈ sqrt((0.2^2 + 0.3^2) / 4)
end
