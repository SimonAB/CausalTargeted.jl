"""
    CausalTargeted

Cross-fitted targeted inference: LMTP, Super Learner, certificates, and grid
execution. Identification is delegated to CausalDynamics.jl. Mediation EIF /
TE / NDE / NIE live in **CausalMediation.jl** (`using CausalMediation`).

Small-*n* profiles, positivity atlases, and sensitivity helpers target conservation
and other low-sample causal applications.

# Documentation

- Methods ↔ literature: `docs/src/methods.md`
- Full bibliography (DOIs / BibTeX keys): `docs/src/references.md`
- Small-*n* checklist: `docs/src/small_n.md`

Canonical papers: Díaz et al. (2023) LMTP; van der Laan & Rose (2011) TMLE;
Cinelli & Hazlett (2020) sensitivity. For mediation, see CausalMediation
(Díaz & Hejazi 2020; Liu et al. 2024).
"""
module CausalTargeted

using DataFrames
using CausalDynamics
using LinearAlgebra
using Random
using Statistics

include("core/config.jl")
include("core/engines.jl")
include("core/mtp_common.jl")
include("core/mtp_inference.jl")
include("core/covariate_schema.jl")
include("learners/mtp_learners.jl")
include("learners/multinomial_sl.jl")
include("learners/small_n.jl")
include("learners/adaptive_learners.jl")
include("core/estimand_types.jl")
include("policies/shift_policies.jl")
include("core/nuisance_interface.jl")
include("synthetic/synthetic.jl")
include("diagnostics/targeting_diagnostics.jl")
include("lmtp/lmtp_density_ratios.jl")
include("lmtp/lmtp_targeting.jl")
include("lmtp/fold_nuisance_cache.jl")
include("lmtp/lmtp_tmle.jl")
include("policies/exponential_tilt.jl")
include("core/cluster_robust.jl")
include("lmtp/discrete_lmtp.jl")
include("lmtp/two_part_discrete_lmtp.jl")
include("diagnostics/positivity.jl")
include("diagnostics/missing_data.jl")
include("diagnostics/imputation/posterior.jl")
include("lmtp/lmtp_grid.jl")
include("gcomp/gcomp.jl")
include("gcomp/parametric_gcomp.jl")
include("bayesian/bayesian_gcomp.jl")
include("bayesian/bayesian_cluster_gaussian.jl")
include("learners/bayesian_learner.jl")
include("bayesian/bayesian_logistic.jl")
include("bayesian/bayesian_cluster_logistic.jl")
include("bayesian/bayesian_cluster_count.jl")

export ConjugateGaussianPrior, ClusterGaussianPrior, BayesianGCompFit, fit_bayesian_gcomp
export BayesianClusterGCompFit, fit_bayesian_cluster_gcomp
export bayesian_policy_contrast, predictive_missing_outcomes, BayesianGaussianLearner
export GaussianCoefficientPrior, BayesianLogisticFit, fit_bayesian_logistic
export ClusterLogisticPrior, BayesianClusterLogisticFit, fit_bayesian_cluster_logistic
export ClusterCountPrior, BayesianClusterCountFit, fit_bayesian_cluster_count
export ClusterHurdlePrior, BayesianClusterHurdleFit, fit_bayesian_cluster_hurdle
include("msm/repeated_outcome_msm.jl")
include("msm/parametric_msm.jl")
include("msm/did.jl")
include("diagnostics/sensitivity.jl")
include("diagnostics/discovery_sensitivity.jl")
include("lmtp/sequential_lmtp.jl")
include("lmtp/survival_lmtp.jl")
include("planning/sequential_bridge.jl")
include("planning/query_bridge.jl")
include("planning/transport.jl")
include("policies/policy_choice.jl")
include("planning/id_certificate.jl")
include("planning/mtp_plan.jl")
include("planning/mtp_execution.jl")
include("planning/temporal_estimation_plan.jl")
include("planning/estimation_plan_bridge.jl")
include("lmtp/bootstrap_lmtp.jl")
include("lmtp/lmtp_contrast.jl")
include("synthetic/synthetic_recovery.jl")
include("planning/mediation_compat.jl")
include("diagnostics/mtp_plotting.jl")
include("diagnostics/hurdle_ci_tests.jl")
include("gcomp/mixedmodels.jl")
include("gcomp/mmrm.jl")

export ShiftPolicy, Estimand
export InterventionalMean, LongitudinalPolicy
export DiscreteTreatmentPolicy, DiscreteInterventionalMean, TwoPartInterventionalMean
export RepeatedOutcomeMSM, run_repeated_outcome_msm, msm_contrast, msm_stratum_contrast
export identify_repeated_outcomes
export ParametricRepeatedOutcomeMSM, run_parametric_repeated_msm
export unstack_repeated_outcomes
export discrete_recode_policy, discrete_static_policy, discrete_shift_policy
export apply_discrete_policy, run_discrete_lmtp, run_discrete_lmtp_contrast, discrete_positivity
export run_two_part_discrete_lmtp, run_two_part_discrete_lmtp_contrast
export SequentialPolicy, SurvivalPolicy
export shift_policy_from_settings, estimand_engine, estimand_from_query
export additive_shift_policy, multiplicative_shift_policy, threshold_shift_policy
export SmoothBoundedLogitShiftPolicy, smooth_bounded_logit_shift_policy
export FeasibleAdditiveShiftPolicy, feasible_additive_shift_policy
export ProportionalHeadroomPolicy, proportional_headroom_policy
export ExponentialTiltPolicy, exponential_tilt_policy, tilt_score, tilt_weight_bounds
export estimate_exponential_tilt, studentised_multiplier_ci
export apply_policy_values
export mtp_settings, default_deltas, MTPSettings, resolved_stratify_by
export exposure_bounds, clamp_exposure, make_analysis_strata, crossfit_indices
export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS, SMALL_N_SL_LEARNERS
export COUNT_SL_LEARNERS, SMALL_COUNT_SL_LEARNERS
export SuperLearnerFit, NestedSLCandidate, nested_sl_candidate
export recommend_folds, recommend_learners, recommend_count_learners, recommend_run_options, warn_if_folds_too_large
export adaptive_learners
export fit_super_learner, predict_super_learner, design_matrix
export validate_family_outcome, suggest_family_outcome
export CovariateSchema, fit_covariate_schema, transform_covariates
export validate_contrast_learners
export covariate_design_matrix, outcome_design_matrix
export run_lmtp_grid
export run_lmtp_contrast, run_sequential_lmtp, run_survival_lmtp
export sequential_identification_certificate, survival_identification_certificate
export plan_sequential, sequential_spec_from_identification
export run_estimation_plan
export bootstrap_discrete_lmtp_contrast, bootstrap_two_part_discrete_lmtp_contrast
export domain_transport_weights, transport_weighted_mean
export PolicyChoice, choose_policy
export lmtp_tmle_contrast, lmtp_tmle_from_components, apply_shift_policy
export execute_estimand, plan_mtp, summarise_plan
export build_run_metadata, attach_run_metadata!, RunMetadata
export build_lmtp_fold_cache, LMTPFoldCache, lmtp_components_from_cache
export tmle_score_diagnostics, optimise_tmle_fluctuation
export normalize_engine
export has_makie, mtp_curve!, plot_mtp_curve
# Book / README DGPs (other synthetics stay in-module for tests and benchmarks)
export simulate_linear_mtp, simulate_discrete_survival_mtp
export simulate_mixed_baseline_mtp
export simulate_binomial_mtp, simulate_multinomial_outcome
export simulate_categorical_treatment_mtp, simulate_sequential_factor_mtp
export simulate_repeated_outcome_ate, simulate_mean_treatment_time_msm
export impute_covariates_mean!, ipcw_weights, handle_missing_data, weighted_influence_summary
export MissingDataResult, complete_numeric_column
export attach_missingness_metadata!, missingness_metadata, with_missingness, MISSINGNESS_META_KEY
export mar_set
export ImputationDraws, impute_posterior, pool_lmtp_grids
export run_gcomp
export ParametricGComputationFit, fit_parametric_gcomp, run_parametric_gcomp
export gcomp_mean, gcomp_contrast, gcomp_interaction, bootstrap_gcomp_interaction
export run_did_2x2, run_did_staggered, aggregate_did
export IdentificationCertificate, identification_certificate, certificate_dict
export metadata_dict
export MTPPlan
export positivity_report, positivity_markdown, attach_positivity_summary!
export tipping_point_bias, partial_r2_calibration, sensitivity_report, sensitivity_markdown

end
