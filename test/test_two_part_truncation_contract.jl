using Test, CausalTargeted, DataFrames, StableRNGs

@testset "Two-part runners preserve uncertainty eligibility" begin
    rng = StableRNG(801)
    n = 90
    a = repeat(["a", "b"], n ÷ 2)
    w = randn(rng, n)
    present = Float64.(rand(rng, n) .< 0.6)
    intensity = [present[i] == 1 ? 0.8 + 0.2 * (a[i] == "b") + 0.1w[i] + 0.3randn(rng) : missing for i in 1:n]
    data = DataFrame(A = a, W = w, present = present, intensity = intensity)
    for truncated in (false, true)
        result = run_two_part_discrete_lmtp_contrast(data, :A;
            presence = :present, intensity = :intensity, arm_hi = "b", arm_ref = "a",
            levels = ["a", "b"], baseline = [:W], folds = 2,
            weight_truncation = truncated, rng = StableRNG(802))
        for part in (result.presence, result.intensity)
            @test isfinite(part.estimate)
            @test part.inference_status === (truncated ? :truncated_weights : :available)
            @test truncated ? isnan(part.se) : isfinite(part.se)
        end
    end
end
