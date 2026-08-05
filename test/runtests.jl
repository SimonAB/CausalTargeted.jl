using CausalTargeted
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

# CausalMediation is re-introduced as a weakdep in 0.3.4 after it lands on General.
# Local / monorepo tests can still exercise façades when the package is available.
const _HAS_CAUSAL_MEDIATION = try
    @eval using CausalMediation
    true
catch
    false
end

@testset "CausalTargeted" begin
    include("test_core.jl")
    if _HAS_CAUSAL_MEDIATION
        include("test_mediation.jl")
    else
        @info "Skipping mediation façade tests (CausalMediation not in test env)"
    end
    include("test_sequential.jl")
    include("test_recovery.jl")
    include("test_capabilities.jl")
    include("test_mlj_ext.jl")
end
