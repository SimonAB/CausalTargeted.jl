include("test_progress.jl")
println(stderr, "TEST_IMPORTS_START"); flush(stderr)

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

const _UNIT_STRESS = get(ENV, "UNIT_STRESS", get(ENV, "CT_UNIT_STRESS", "")) == "1"

println(stderr, "TEST_IMPORTS_DONE"); flush(stderr)

@testset "CausalTargeted" begin
    @include_test "test_test_progress.jl"
    @include_test "test_project_sources.jl"
    @include_test "test_covariate_schema.jl"
    @include_test "test_bayesian_gcomp.jl"
    @include_test "test_bayesian_learner.jl"
    @include_test "test_missing_data.jl"
    @include_test "test_missingness_edge_cases.jl"
    if _UNIT_STRESS
        @include_test "test_missing_strategies_matrix.jl"
    else
        @info "Skipping missing strategy matrix (set UNIT_STRESS=1)"
    end
    @include_test "test_posterior_imputation.jl"
    @include_test "test_inference_utilities.jl"
    @include_test "test_classifier_orientation.jl"
    @include_test "test_continuous_inference_contracts.jl"
    @include_test "test_lmtp_targeting_contract.jl"
    @include_test "test_discrete_targeting_contract.jl"
    @include_test "test_lmtp_grid_contract.jl"
    @include_test "test_lmtp_execution_options.jl"
    @include_test "test_two_part_truncation_contract.jl"
    @include_test "test_bootstrap_source_groups.jl"
    @include_test "test_did_unavailable_covariance.jl"
    @include_test "test_did_covariance.jl"
    @include_test "test_gcomp_inference.jl"
    @include_test "test_cluster_inference.jl"
    @include_test "test_longitudinal_inference.jl"
    @include_test "test_core.jl"
    @include_test "test_parametric_gcomp.jl"
    @include_test "test_nnloglik.jl"
    @include_test "test_nnls_risk_guard.jl"
    @include_test "test_metalearners.jl"
    @include_test "test_multinomial_sl.jl"
    @include_test "test_discrete_lmtp.jl"
    @include_test "test_family_outcome.jl"
    @include_test "test_estimation_plan_bridge.jl"
    @include_test "test_panel_path.jl"
    @include_test "test_two_part_lmtp.jl"
    @include_test "test_count_lmtp.jl"
    @include_test "test_bootstrap_lmtp.jl"
    @include_test "test_rxinfer_load.jl"
    @include_test "test_logistic_engine.jl"
    @include_test "test_bayesian_logistic.jl"
    @include_test "test_bayesian_policy_support.jl"
    @include_test "test_logistic_quadrature.jl"
    @include_test "test_logistic_tails.jl"
    @include_test "test_policy_calibration_reference.jl"
    @include_test "test_policy_calibration_support.jl"
    @include_test "test_logistic_unloaded.jl"
    @include_test "test_repeated_outcome_msm.jl"
    @include_test "test_parametric_msm.jl"
    @include_test "test_mediation.jl"
    @include_test "test_sequential.jl"
    @include_test "test_semantic_provenance.jl"
    @include_test "test_survival.jl"
    @include_test "test_transport_decision.jl"
    @include_test "test_recovery.jl"
    @include_test "test_capabilities.jl"
    @include_test "test_mlj_ext.jl"
    @include_test "test_tree_learners.jl"
    @include_test "test_hurdle_ci.jl"
    # MixedModels extension (optional parametric LMM / NB2 / mixed g-computation).
    using FastGaussQuadrature, NLopt, SpecialFunctions, MixedModels, QuadGK
    @test Base.get_extension(CausalTargeted, :CausalTargetedMixedModelsExt) !== nothing
    @include_test "test_mixedmodels.jl"
    @include_test "test_parametric_gcomp_mixed.jl"
    @include_test "test_profiled_nb2.jl"
    @include_test "test_mmrm.jl"
end

# Importing CairoMakie from inside the compiled outer testset can expose new
# Makie bindings at an older world age on Julia 1.13. Keep the optional façade
# assertion and backend import at true top level, followed by the plotting tests.
@include_test "test_mtp_plotting.jl"
@include_test "test_mtp_unavailable_intervals.jl"
