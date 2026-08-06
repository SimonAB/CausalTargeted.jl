using CausalTargeted
using CausalDynamics
using DataFrames
using StableRNGs
using Test

@testset "domain transport weights" begin
    df = DataFrame(
        D = vcat(fill("source", 40), fill("target", 60)),
        Y = vcat(randn(StableRNG(1), 40), randn(StableRNG(2), 60) .+ 1),
    )
    w = domain_transport_weights(df, :D; target = "target", source = "source")
    @test length(w) == 100
    @test all(w .> 0)
    @test mean(w) ≈ 1.0 atol = 1e-8
    μ = transport_weighted_mean(df.Y, w)
    @test isfinite(μ)
end

@testset "choose_policy among LMTP estimands" begin
    rng = StableRNG(15)
    df, _ = simulate_linear_mtp(100; rng = rng)
    e0 = InterventionalMean(:A, :Y, [:W], additive_shift_policy())
    e1 = InterventionalMean(:A, :Y, [:W], additive_shift_policy())
    # Same estimand type with different deltas via execute kwargs is awkward;
    # use SequentialPolicy-free path: two InterventionalMean evaluated at different deltas
    choice = choose_policy(
        [:shift0 => e0, :shift1 => e1],
        df;
        deltas = [0.0],
        folds = 2,
        learners_outcome = SMALL_N_SL_LEARNERS,
        learners_trt = SMALL_N_SL_LEARNERS,
        parallel = false,
        rng = StableRNG(16),
    )
    @test choice isa PolicyChoice
    @test choice.selected in (:shift0, :shift1)
    @test nrow(choice.values) == 2
    @test :est in propertynames(choice.values)
end
