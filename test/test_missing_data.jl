"""Unit tests for Observable missing-data policies and metadata."""

using CausalTargeted
using DataFrames
using StableRNGs
using Test

@testset "handle_missing_data metadata" begin
    df = DataFrame(
        y = [1.0, missing, 3.0, 4.0],
        w = [0.0, 1.0, 2.0, missing],
        a = [0.0, 1.0, 0.0, 1.0],
    )

    @testset ":drop" begin
        result = handle_missing_data(df, :y, [:w, :a], :drop)
        clean, w, extra = result
        @test nrow(clean) == 2
        @test w == ones(2)
        @test isempty(extra)
        @test result.meta.strategy === :drop
        @test result.meta.rung === :L2
        @test result.meta.time_indexed === false
        @test result.meta.n_in == 4
        @test result.meta.n_out == 2
        @test result.meta.miss_rates[:y] ≈ 0.25
        @test result.meta.miss_rates[:w] ≈ 0.25
    end

    @testset ":ipcw" begin
        result = handle_missing_data(
            df, :y, [:w, :a], :ipcw; rng = StableRNG(1),
        )
        @test result.meta.strategy === :ipcw
        @test result.meta.n_out == nrow(result.data)
        @test length(result.weights) == result.meta.n_out
    end

    @testset "complete_numeric_column" begin
        @test complete_numeric_column([1.0, 2.0]; name = :y) == [1.0, 2.0]
        @test_throws ArgumentError complete_numeric_column(
            [1.0, missing]; name = :y,
        )
    end

    @testset "mar_set from IdentificationResult" begin
        using CausalDynamics: DiGraph, add_edge!, TotalEffectQuery, identify, MissingnessSpec
        g = DiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 3)
        add_edge!(g, 2, 3)
        names = Dict(1 => :W, 2 => :A, 3 => :Y)
        id0 = identify(g, TotalEffectQuery(:A, :Y); node_names = names)
        @test mar_set(id0) == Symbol[]
        id1 = identify(
            g, TotalEffectQuery(:A, :Y);
            node_names = names,
            missingness = MissingnessSpec(:Y; regime = :mar, conditioning_set = [:W]),
        )
        @test mar_set(id1) == [:W]
    end
end
