using CausalTargeted
using CausalMediation  # activates mediation façades
using CausalDynamics: identify, TotalEffectQuery, TemporalEffectQuery, TemporalDAGSpec, LaggedEdge, unroll_temporal_dag
using DataFrames
using Graphs
using Random
using StableRNGs
using Statistics
using Test
using EvoTrees
using MLJ
using MLJLinearModels


@testset "CausalTargeted" begin
    include("test_core.jl")
    include("test_mediation.jl")
    include("test_sequential.jl")
    include("test_recovery.jl")
    include("test_capabilities.jl")
    include("test_mlj_ext.jl")
end
