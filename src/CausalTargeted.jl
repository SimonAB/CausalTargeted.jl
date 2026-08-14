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
include("covariate_schema.jl")
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
include("positivity.jl")
include("missing_data.jl")
include("lmtp_grid.jl")
include("gcomp.jl")
include("did.jl")
include("sensitivity.jl")
include("discovery_sensitivity.jl")
include("sequential_lmtp.jl")
include("survival_lmtp.jl")
include("sequential_bridge.jl")
include("transport.jl")
include("policy_choice.jl")
include("id_certificate.jl")
include("mtp_plan.jl")
include("mtp_execution.jl")
include("lmtp_contrast.jl")
include("synthetic_recovery.jl")
include("mediation_compat.jl")
include("mtp_plotting.jl")

export ShiftPolicy, Estimand
export InterventionalMean, MediationContrast, LongitudinalPolicy, ScalarMediation
export SequentialPolicy, SurvivalPolicy
export shift_policy_from_settings, estimand_engine, estimand_from_query
export additive_shift_policy, multiplicative_shift_policy, threshold_shift_policy
export apply_policy_values
export mtp_settings, default_deltas, MTPSettings, resolved_stratify_by
export exposure_bounds, clamp_exposure, make_analysis_strata, crossfit_indices
export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS, SMALL_N_SL_LEARNERS
export SuperLearnerFit
export recommend_folds, recommend_learners, recommend_run_options, warn_if_folds_too_large
export adaptive_learners
export fit_super_learner, predict_super_learner, design_matrix
export validate_contrast_learners
export covariate_design_matrix, outcome_design_matrix
export run_lmtp_grid, run_mediation_grid, run_mediation_scalar, run_mediation_scalar_ppl
export run_lmtp_contrast, run_tmle3_nde, run_sequential_lmtp, run_survival_lmtp
export sequential_identification_certificate, survival_identification_certificate
export plan_sequential, sequential_spec_from_identification
export domain_transport_weights, transport_weighted_mean
export PolicyChoice, choose_policy
export lmtp_tmle_contrast, lmtp_tmle_from_components, apply_shift_policy
export execute_estimand, plan_mtp, summarise_plan
export build_run_metadata, attach_run_metadata!, RunMetadata
export build_lmtp_fold_cache, LMTPFoldCache, lmtp_components_from_cache
export build_mediation_fold_cache
export tmle_score_diagnostics, optimise_tmle_fluctuation
export prepare_ppl_mediation_spec, conjugate_mediation_bootstrap
export normalize_engine, is_mediation_engine
export has_makie, mtp_curve!, plot_mtp_curve
# Book / README DGPs (other synthetics stay in-module for tests and benchmarks)
export simulate_linear_mtp, simulate_mediation, simulate_discrete_survival_mtp
export simulate_mixed_baseline_mtp
export impute_covariates_mean!, ipcw_weights, handle_missing_data, weighted_influence_summary
export run_gcomp
export run_did_2x2, run_did_staggered, aggregate_did
export truth_shift_effect, effective_sd_shift, effective_raw_shift
export identification_certificate, certificate_dict
export metadata_dict
export MTPPlan
export mediation_n_mc_sweep, mediation_stability_summary, mediation_stability_markdown
export positivity_report, positivity_markdown, attach_positivity_summary!
export tipping_point_bias, partial_r2_calibration, sensitivity_report, sensitivity_markdown
export adjustment_set_disagreement, discovery_adjustment_sensitivity, merge_discovery_sensitivity!

end
