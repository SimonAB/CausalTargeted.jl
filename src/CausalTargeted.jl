"""
    CausalTargeted

Cross-fitted targeted inference: LMTP, interventional mediation EIF, nuisance
caching, and grid execution. Identification is delegated to CausalDynamics.jl.

Small-*n* profiles, positivity atlases, and sensitivity helpers target conservation
and other low-sample causal applications.

# Documentation

- Methods ↔ literature: `docs/src/methods.md`
- Full bibliography (DOIs / BibTeX keys): `docs/src/references.md`
- Small-*n* checklist: `docs/src/small_n.md`

Canonical papers: Díaz et al. (2023) LMTP; Díaz & Hejazi (2020) / Liu et al. (2024)
mediation; van der Laan & Rose (2011) TMLE; Cinelli & Hazlett (2020) sensitivity.
"""
module CausalTargeted

using DataFrames
using CausalDynamics

include("config.jl")
include("engines.jl")
include("mtp_common.jl")
include("mtp_inference.jl")
include("mtp_learners.jl")
include("small_n.jl")
include("adaptive_learners.jl")
include("estimand_types.jl")
include("shift_policies.jl")
include("nuisance_interface.jl")
include("synthetic.jl")
include("targeting_diagnostics.jl")
include("lmtp_tmle.jl")
include("fold_nuisance_cache.jl")
include("mediation_fold_cache.jl")
include("mediation_eif.jl")
include("positivity.jl")
include("lmtp_grid.jl")
include("mediation_grid.jl")
include("mediation_diagnostics.jl")
include("sensitivity.jl")
include("discovery_sensitivity.jl")
include("sequential_lmtp.jl")
include("id_certificate.jl")
include("mtp_plan.jl")
include("mtp_execution.jl")
include("ppl_mediation.jl")
include("lmtp_contrast.jl")
include("mediation_scalar.jl")
include("tmle3_mediation.jl")

export ShiftPolicy, Estimand
export InterventionalMean, MediationContrast, LongitudinalPolicy, ScalarMediation
export SequentialPolicy
export shift_policy_from_settings, estimand_engine, estimand_from_query
export additive_shift_policy, multiplicative_shift_policy, threshold_shift_policy
export apply_policy_values
export mtp_settings, default_deltas, MTPSettings, resolved_stratify_by
export exposure_bounds, clamp_exposure, make_analysis_strata, crossfit_indices
export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS, SMALL_N_SL_LEARNERS
export recommend_folds, recommend_learners, recommend_run_options, warn_if_folds_too_large
export adaptive_learners
export fit_super_learner, predict_super_learner, design_matrix
export run_lmtp_grid, run_mediation_grid, run_mediation_scalar, run_crumble_scalar_ppl
export run_crumble_grid, run_crumble_scalar  # legacy aliases
export run_lmtp_contrast, run_tmle3_nde, run_sequential_lmtp
export sequential_identification_certificate
export lmtp_tmle_contrast, lmtp_tmle_from_components, apply_shift_policy
export execute_estimand, plan_mtp, summarise_plan
export build_run_metadata, attach_run_metadata!, RunMetadata
export build_lmtp_fold_cache, LMTPFoldCache, lmtp_components_from_cache
export build_mediation_fold_cache, MediationFoldCache
export build_crumble_fold_cache, CrumbleFoldCache  # legacy
export tmle_score_diagnostics, optimise_tmle_fluctuation
export prepare_ppl_mediation_spec, conjugate_mediation_bootstrap, run_crumble_scalar_ppl
export _shared_fold_lmtp_components, _mediation_effects, _crumble_mediation_effects
export normalize_engine, is_mediation_engine
export _mtp_clever_covariate_gaussian, _mtp_clever_covariate_gaussian_het
export _gaussian_density, _mtp_clever_covariate_clamp_aware
export simulate_linear_mtp, simulate_mediation, simulate_continuous_mtp_mediation
export identification_certificate, certificate_dict
export build_run_metadata, metadata_dict, attach_run_metadata!
export MTPPlan, plan_mtp, summarise_plan
export execute_estimand, _parallel_delta_jobs
export estimand_from_query
export mediation_n_mc_sweep, mediation_stability_summary, mediation_stability_markdown
export positivity_report, positivity_markdown, attach_positivity_summary!
export tipping_point_bias, partial_r2_calibration, sensitivity_report, sensitivity_markdown
export adjustment_set_disagreement, discovery_adjustment_sensitivity, merge_discovery_sensitivity!

end
