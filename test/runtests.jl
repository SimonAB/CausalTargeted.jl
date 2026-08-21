using CausalTargeted
import CausalDynamics
using CausalDynamics:
    identify, TotalEffectQuery, TemporalEffectQuery, InterventionalPolicyQuery,
    TemporalDAGSpec, LaggedEdge,
    unroll_temporal_dag, DiscreteTimeCDM, intervention_value
using DataFrames
using CategoricalArrays
using Dates
using Graphs
using Random
using StableRNGs
using Statistics
using Test
using EvoTrees
using MLJ
using MLJLinearModels
using MLJDecisionTreeInterface
using MLJXGBoostInterface

import CausalMediation  # load weakdep / extension without clashing CT façade exports
const _HAS_CAUSAL_MEDIATION = true

const _HAS_PANEL_API = isdefined(CausalDynamics, :simulate_panel)

@testset "CausalTargeted" begin
    include("test_covariate_schema.jl")
    include("test_missing_data.jl")
    include("test_missing_strategies_matrix.jl")
    include("test_posterior_imputation.jl")
    include("test_core.jl")
    include("test_nnloglik.jl")
    include("test_metalearners.jl")
    include("test_multinomial_sl.jl")
    include("test_discrete_lmtp.jl")
    include("test_mediation.jl")
    include("test_sequential.jl")
    include("test_survival.jl")
    include("test_transport_decision.jl")
    include("test_recovery.jl")
    include("test_capabilities.jl")
    include("test_mlj_ext.jl")
    include("test_tree_learners.jl")
    # Load Makie only after the façade-unavailable assertion in this file.
    include("test_mtp_plotting.jl")
end
