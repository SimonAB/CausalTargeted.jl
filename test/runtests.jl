using CausalTargeted
import CausalDynamics
using CausalDynamics:
    identify, TotalEffectQuery, TemporalEffectQuery, InterventionalPolicyQuery,
    TemporalDAGSpec, LaggedEdge,
    unroll_temporal_dag, DiscreteTimeCDM, intervention_value, panel_column_name,
    simulate, do_sequence
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
    include("test_project_sources.jl")
    include("test_covariate_schema.jl")
    include("test_missing_data.jl")
    include("test_missing_strategies_matrix.jl")
    include("test_posterior_imputation.jl")
    include("test_missingness_edge_cases.jl")
    include("test_core.jl")
    include("test_parametric_gcomp.jl")
    include("test_nnloglik.jl")
    include("test_metalearners.jl")
    include("test_multinomial_sl.jl")
    include("test_discrete_lmtp.jl")
    include("test_family_outcome.jl")
    include("test_estimation_plan_bridge.jl")
    include("test_panel_path.jl")
    include("test_two_part_lmtp.jl")
    include("test_count_lmtp.jl")
    include("test_bootstrap_lmtp.jl")
    include("test_rxinfer_load.jl")
    include("test_repeated_outcome_msm.jl")
    include("test_parametric_msm.jl")
    include("test_mediation.jl")
    include("test_sequential.jl")
    include("test_semantic_provenance.jl")
    include("test_survival.jl")
    include("test_transport_decision.jl")
    include("test_recovery.jl")
    include("test_capabilities.jl")
    include("test_mlj_ext.jl")
    include("test_tree_learners.jl")
    # Load Makie only after the façade-unavailable assertion in this file.
    include("test_mtp_plotting.jl")
    include("test_hurdle_ci.jl")
    # MixedModels extension (optional parametric LMM / NB2 / mixed g-computation).
    using FastGaussQuadrature, NLopt, SpecialFunctions, MixedModels, QuadGK
    @test Base.get_extension(CausalTargeted, :CausalTargetedMixedModelsExt) !== nothing
    include("test_mixedmodels.jl")
    include("test_profiled_nb2.jl")
    include("test_mmrm.jl")
end
