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

using Base.Threads
using CausalDynamics
using DataFrames
using Dates
using Distributions
using GLM
using LinearAlgebra
using Logging
using PrecompileTools
using Random
using SpecialFunctions
using StableRNGs
using Statistics
using StatsModels

# Core utilities, types & nuisances
include("core/config.jl")
include("core/engines.jl")
include("core/mtp_common.jl")
include("core/mtp_inference.jl")
include("core/covariate_schema.jl")

# Super Learner & candidate adapters
include("learners/mtp_learners.jl")
include("learners/mtp_learner_adapters.jl")
include("learners/mtp_metalearners.jl")
include("learners/multinomial_sl.jl")
include("learners/small_n.jl")
include("learners/adaptive_learners.jl")

# Estimands, policies & nuisance interfaces
include("core/estimand_types.jl")
include("policies/shift_policies.jl")
include("core/nuisance_interface.jl")
include("synthetic/synthetic.jl")
include("diagnostics/targeting_diagnostics.jl")

# Continuous, tilted & discrete LMTP estimators
include("lmtp/lmtp_density_ratios.jl")
include("lmtp/lmtp_targeting.jl")
include("lmtp/fold_nuisance_cache.jl")
include("lmtp/lmtp_tmle.jl")
include("policies/exponential_tilt.jl")
include("core/cluster_robust.jl")
include("lmtp/discrete_lmtp.jl")
include("lmtp/two_part_discrete_lmtp.jl")

# Positivity, missing data & grid execution
include("diagnostics/positivity.jl")
include("diagnostics/missing_data.jl")
include("diagnostics/imputation/posterior.jl")
include("lmtp/lmtp_grid.jl")

# G-computation & Bayesian hierarchical models
include("gcomp/gcomp.jl")
include("gcomp/parametric_gcomp.jl")
include("bayesian/bayesian_gcomp.jl")
include("bayesian/bayesian_cluster_gaussian.jl")
include("learners/bayesian_learner.jl")
include("bayesian/bayesian_logistic.jl")
include("bayesian/bayesian_cluster_logistic.jl")
include("bayesian/bayesian_cluster_count.jl")

# MSM, DiD & sensitivity analysis
include("msm/repeated_outcome_msm.jl")
include("msm/parametric_msm.jl")
include("msm/did.jl")
include("diagnostics/sensitivity.jl")
include("diagnostics/discovery_sensitivity.jl")

# Longitudinal LMTP, planning, certificates & execution
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

# Synthetic recovery, compatibility façades, plotting & mixed-model bridges
include("synthetic/synthetic_recovery.jl")
include("planning/mediation_compat.jl")
include("diagnostics/mtp_plotting.jl")
include("diagnostics/hurdle_ci_tests.jl")
include("gcomp/mixedmodels.jl")
include("gcomp/mmrm.jl")

# ---------------------------------------------------------------------------
# Centralised public exports (grouped by subsystem)
# ---------------------------------------------------------------------------

# Core settings, engines, shift/weight utilities, inference & cluster variance
export MTPSettings, mtp_settings, default_deltas, resolved_stratify_by
export normalize_engine
export exposure_bounds, clamp_exposure, shifted_exposure, clamp_diagnostics
export additive_clamp_diagnostics, targeting_weight_from_clamp
export make_analysis_strata, crossfit_indices, wald_ci
export robust_residual_sd, truncate_weights, stabilize_weights, prepare_clever_covariate
export requested_shift, support_diagnostics, get_target_strata
export cv_select_truncation, multiplier_simultaneous_critical, simultaneous_wald_bands
export studentised_multiplier_ci
export CovariateSchema, fit_covariate_schema, transform_covariates
export NuisanceModel, OutcomeRegression, ExposureDensity
export fit_outcome_regression, predict_outcome, fit_exposure_density, predict_exposure_mean
export resolve_cluster_ids, cluster_robust_variance

# Estimands & policies
export Estimand, ShiftPolicy, InterventionalMean, LongitudinalPolicy
export shift_policy_from_settings, estimand_engine, estimand_from_pathway_task
export estimand_from_query, longitudinal_estimand
export additive_shift_policy, multiplicative_shift_policy, threshold_shift_policy
export apply_policy_values
export SmoothBoundedLogitShiftPolicy, smooth_bounded_logit_shift_policy
export FeasibleAdditiveShiftPolicy, feasible_additive_shift_policy
export ProportionalHeadroomPolicy, proportional_headroom_policy
export ExponentialTiltPolicy, exponential_tilt_policy, tilt_score, tilt_weight_bounds
export estimate_exponential_tilt
export PolicyChoice, choose_policy

# Learners & Super Learner
export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS, SMALL_N_SL_LEARNERS
export COUNT_SL_LEARNERS, SMALL_COUNT_SL_LEARNERS
export SuperLearnerFit, NestedSLCandidate, nested_sl_candidate
export validate_contrast_learners, validate_family_outcome, suggest_family_outcome
export design_matrix, covariate_design_matrix, outcome_design_matrix
export sparse_exposure_diagnostic, columns_present
export fit_super_learner, predict_super_learner
export crossfit_outcome_predictions, crossfit_predict_outcome
export crossfit_treatment_mean, crossfit_propensity
export recommend_folds, recommend_learners, recommend_count_learners
export recommend_run_options, warn_if_folds_too_large
export adaptive_learners, BayesianGaussianLearner

# Continuous, discrete, two-part, sequential & survival LMTP
export LMTPFoldCache, build_lmtp_fold_cache, lmtp_components_from_cache
export apply_shift_policy, lmtp_tmle_policy_mean, lmtp_tmle_contrast, lmtp_tmle_from_components
export run_lmtp_grid, run_lmtp_contrast
export DiscreteTreatmentPolicy, DiscreteInterventionalMean, TwoPartInterventionalMean
export discrete_recode_policy, discrete_static_policy, discrete_shift_policy
export apply_discrete_policy, run_discrete_lmtp, run_discrete_lmtp_contrast, discrete_positivity
export run_two_part_discrete_lmtp, run_two_part_discrete_lmtp_contrast
export SequentialPolicy, run_sequential_lmtp, sequential_identification_certificate
export SurvivalPolicy, run_survival_lmtp, survival_identification_certificate
export bootstrap_discrete_lmtp_contrast, bootstrap_two_part_discrete_lmtp_contrast

# G-computation, mixed models & MMRM
export run_gcomp
export ParametricGComputationFit, fit_parametric_gcomp, run_parametric_gcomp
export gcomp_mean, gcomp_contrast, gcomp_interaction, bootstrap_gcomp_interaction
export MixedGComputationResult, StratifiedMixedGComputationResult, MixedModelGComputationFit
export NB2RandomInterceptModel, ProfiledNB2MixedModel, fit_profiled_nb2
export quadrature_diagnostics, validate_fixed_theta_nb2_likelihood
export theta, converged, fitdiagnostics, random_intercept_variance, mixed_g_computation
export MMRMResult, fit_mmrm, run_mmrm

# Bayesian g-computation & hierarchical models
export ConjugateGaussianPrior, ClusterGaussianPrior, GaussianCoefficientPrior
export ClusterLogisticPrior, ClusterCountPrior, ClusterHurdlePrior
export BayesianGCompFit, fit_bayesian_gcomp
export BayesianClusterGCompFit, fit_bayesian_cluster_gcomp
export BayesianLogisticFit, fit_bayesian_logistic
export BayesianClusterLogisticFit, fit_bayesian_cluster_logistic
export BayesianClusterCountFit, fit_bayesian_cluster_count
export BayesianClusterHurdleFit, fit_bayesian_cluster_hurdle
export bayesian_policy_contrast, predictive_missing_outcomes

# MSM & DiD
export RepeatedOutcomeMSM, run_repeated_outcome_msm, msm_contrast, msm_stratum_contrast
export identify_repeated_outcomes, unstack_repeated_outcomes
export ParametricRepeatedOutcomeMSM, run_parametric_repeated_msm
export run_did_2x2, run_did_staggered, aggregate_did

# Planning, certificates, transport & execution
export IdentificationCertificate, RunMetadata
export identification_certificate, certificate_dict
export build_run_metadata, metadata_dict, attach_run_metadata!
export MTPPlan, plan_mtp, summarise_plan
export plan_sequential, sequential_spec_from_identification
export execute_estimand, run_estimation_plan
export domain_transport_weights, transport_weighted_mean

# Diagnostics, missing data, imputation, sensitivity & plotting
export tmle_score_diagnostics, tmle_fluctuation_objective, optimise_tmle_fluctuation
export diagnostics_dict
export positivity_report, positivity_markdown, attach_positivity_summary!
export MissingDataResult, complete_numeric_column
export impute_covariates_mean!, ipcw_weights, handle_missing_data, weighted_influence_summary
export attach_missingness_metadata!, missingness_metadata, with_missingness
export MISSINGNESS_META_KEY, mar_set
export ImputationDraws, impute_posterior, pool_lmtp_grids
export tipping_point_bias, partial_r2_calibration, sensitivity_report, sensitivity_markdown
export has_makie, mtp_curve!, plot_mtp_curve

# Book / README synthetic DGPs
export simulate_linear_mtp, simulate_discrete_survival_mtp, simulate_mixed_baseline_mtp
export simulate_binomial_mtp, simulate_multinomial_outcome, simulate_categorical_treatment_mtp
export simulate_sequential_factor_mtp, simulate_repeated_outcome_ate, simulate_mean_treatment_time_msm

# ---------------------------------------------------------------------------
# Precompilation workload (reduces cold TTFX for LMTP grid & exponential tilt)
# ---------------------------------------------------------------------------

@setup_workload begin
    rng_pc = StableRNG(2026)
    n_pc = 20
    w_pc = randn(rng_pc, n_pc)
    a_pc = clamp.(0.4 .* w_pc .+ randn(rng_pc, n_pc), -2.0, 2.0)
    y_pc = 0.5 .+ 0.6 .* a_pc .+ 0.3 .* w_pc .+ 0.2 .* randn(rng_pc, n_pc)
    df_pc = DataFrame(W = w_pc, A = a_pc, Y = y_pc)
    pol_pc = feasible_additive_shift_policy(-2.5, 2.5; sd_scale = 1.0)
    etp_pc = exponential_tilt_policy(; bound = 2.0)
    @compile_workload begin
        run_lmtp_grid(
            df_pc, :A, :Y;
            baseline = [:W], policy = pol_pc, deltas = [-0.2, 0.0, 0.2],
            folds = 2, parallel = false, simultaneous = true, n_boot_sim = 19,
            weight_truncation = false, rng = StableRNG(1),
        )
        estimate_exponential_tilt(
            df_pc, :A, :Y, [:W], 0.2, etp_pc;
            folds = 2, bootstrap_B = 19, rng = StableRNG(1),
        )
    end
end

end
