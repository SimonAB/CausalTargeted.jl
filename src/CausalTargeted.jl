"""
    CausalTargeted

Cross-fitted targeted inference: LMTP, interventional mediation EIF, nuisance
caching, and grid execution. Identification is delegated to [CausalDynamics.jl](@ref).
"""
module CausalTargeted

using DataFrames
using CausalDynamics

include("config.jl")
include("mtp_common.jl")
include("mtp_inference.jl")
include("mtp_learners.jl")
include("estimand_types.jl")
include("nuisance_interface.jl")
include("synthetic.jl")
include("targeting_diagnostics.jl")
include("lmtp_tmle.jl")
include("fold_nuisance_cache.jl")
include("mediation_eif.jl")
include("lmtp_grid.jl")
include("crumble_grid.jl")
include("id_certificate.jl")
include("mtp_plan.jl")
include("mtp_execution.jl")
include("ppl_mediation.jl")
include("lmtp_contrast.jl")
include("crumble_scalar.jl")
include("tmle3_mediation.jl")

export ShiftPolicy, Estimand
export InterventionalMean, MediationContrast, LongitudinalPolicy, ScalarMediation
export shift_policy_from_settings, estimand_engine, estimand_from_query
export mtp_settings, default_deltas, MTPSettings, resolved_stratify_by
export exposure_bounds, clamp_exposure, make_analysis_strata, crossfit_indices
export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS
export fit_super_learner, predict_super_learner, design_matrix
export run_lmtp_grid, run_crumble_grid, run_crumble_scalar, run_crumble_scalar_ppl
export run_lmtp_contrast, run_tmle3_nde
export lmtp_tmle_contrast, lmtp_tmle_from_components, apply_shift_policy
export execute_estimand, plan_mtp, summarise_plan
export build_run_metadata, attach_run_metadata!, RunMetadata
export build_lmtp_fold_cache, LMTPFoldCache, lmtp_components_from_cache
export tmle_score_diagnostics, optimise_tmle_fluctuation
export prepare_ppl_mediation_spec, conjugate_mediation_bootstrap, run_crumble_scalar_ppl
export _shared_fold_lmtp_components, _crumble_mediation_effects
export _mtp_clever_covariate_gaussian, _mtp_clever_covariate_gaussian_het
export _gaussian_density, _mtp_clever_covariate_clamp_aware
export simulate_linear_mtp, simulate_mediation, simulate_continuous_mtp_mediation
export identification_certificate, certificate_dict
export build_run_metadata, metadata_dict, attach_run_metadata!
export MTPPlan, plan_mtp, summarise_plan
export execute_estimand, _parallel_delta_jobs
export estimand_from_query

end
