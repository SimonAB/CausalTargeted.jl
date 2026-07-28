using CausalTargeted
using CausalDynamics: identify, TotalEffectQuery
using DataFrames
using Graphs
using Random
using StableRNGs
using Test

@testset "CausalTargeted" begin
    @testset "synthetic LMTP" begin
        rng = StableRNG(1)
        df, truth = simulate_linear_mtp(120; rng = rng)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W],
            deltas = [-0.5, 0.0, 0.5],
            folds = 3,
            parallel = false,
            simultaneous = false,
        )
        @test nrow(grid) == 3
    end

    @testset "estimand from query" begin
        q = TotalEffectQuery(:A, :Y)
        est = estimand_from_query(q, [:W])
        @test est isa InterventionalMean
    end

    @testset "identification certificate" begin
        g = SimpleDiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 3)
        add_edge!(g, 2, 3)
        res = identify(g, TotalEffectQuery(2, 3))
        cert = identification_certificate(res, :X, :Y; adjustment = [:Z])
        d = certificate_dict(cert)
        @test d["id_identifiable"] == true
    end
end
