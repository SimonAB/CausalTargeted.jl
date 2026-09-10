# Changelog

All notable changes to CausalTargeted.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.29] - 2026-09-05

### Added

- **Parametric formula g-computation ([#45](https://github.com/SimonAB/CausalTargeted.jl/pull/45)):**
  `fit_parametric_gcomp`, `gcomp_mean`, `gcomp_contrast`, `gcomp_interaction`,
  `run_parametric_gcomp`, and `bootstrap_gcomp_interaction`. Complete-case
  StatsModels GLM standardisation for categorical interventions and formula
  interactions (Gaussian / binomial / Gamma / `:negbin`), with HC3 + delta-method
  inference. Separate from cross-fitted `run_gcomp` and LMTP; not an
  `execute_estimand` path. Coverage: `test/test_parametric_gcomp.jl`.

## [0.3.25] - 2026-08-30

### Added

- **ZINB learners ([#40](https://github.com/SimonAB/CausalTargeted.jl/issues/40)):**
  `:zeroinflated_nb` Super Learner and `family_outcome = :zeroinflated_nb` on
  count LMTP paths (logistic inflation + profiled NB composite mean).

## [0.3.24] - 2026-08-30

### Added

- **`run_estimation_plan` structural skip ([#26](https://github.com/SimonAB/CausalDynamics.jl/issues/26)):**
  throws when `EstimationPlan.estimability === :structural_skip`.

## [0.3.23] - 2026-08-30

### Added

- **Hurdle CI lag colmaps ([#43](https://github.com/SimonAB/CausalTargeted.jl/issues/43)):**
  `hurdle_colmap_presence_intensity`, `hurdle_colmap_grid_arm`,
  `hurdle_colmap_lag_panel`.

## [0.3.22] - 2026-08-30

### Added

- **Bootstrap LMTP ([#42](https://github.com/SimonAB/CausalTargeted.jl/issues/42)):**
  `bootstrap_discrete_lmtp_contrast` and `bootstrap_two_part_discrete_lmtp_contrast`
  (row/cluster resample, percentile CI).

## [0.3.21] - 2026-08-30

### Added

- Session-slice integration tests with `plan_session_estimation` / `session_slice`.

## [0.3.20] - 2026-08-30

### Added

- **Continuous count `run_lmtp_grid` ([#41](https://github.com/SimonAB/CausalTargeted.jl/issues/41)):**
  verified count targeting path with `family_outcome = :poisson | :negbin`.
  Closes [#36](https://github.com/SimonAB/CausalTargeted.jl/issues/36) Phase A+C.

## [0.3.19] - 2026-08-30

### Added

- **`run_estimation_plan` family handoff ([#44](https://github.com/SimonAB/CausalTargeted.jl/issues/44)):**
  merges planner `family_outcome` with `recommend_run_options` / count learners.

## [0.3.18] - 2026-08-30

### Added

- **Count outcome LMTP (Phase A, #36):** `:poisson` and `:negbin` Super Learner
  families with `:glm_poisson` / `:glm_nb` learners (profiled NB θ per fold);
  `family_outcome = :poisson | :negbin` on discrete LMTP runners; log-link TMLE
  targeting on the count scale; `suggest_family_outcome` count heuristics;
  `recommend_count_learners` / count-aware `recommend_run_options`. Coverage:
  `test/test_count_lmtp.jl`.

## [0.3.17] - 2026-08-30

### Added

- **`run_estimation_plan`:** execute a CausalDynamics `EstimationPlan` on a wide
  panel (`:discrete_lmtp`, `:two_part_discrete_lmtp`); pairs with Dynamics panel
  bridge ([#32](https://github.com/SimonAB/CausalTargeted.jl/issues/32)).
- **Panel path regression tests:** `test/test_panel_path.jl` — CDM → planner →
  LMTP recovery and hurdle two-part path ([#33](https://github.com/SimonAB/CausalTargeted.jl/issues/33)).
- **Panel bridge docs:** [methods.md](docs/src/methods.md) engine handoff table;
  Apodemus stress uses `run_estimation_plan`.

## [0.3.16] - 2026-08-30

### Added

- **Gaussian MMRM reference path** ([#25](https://github.com/SimonAB/CausalTargeted.jl/issues/25)):
  `fit_mmrm`, `run_mmrm`, `MMRMResult` in `CausalTargetedMixedModelsExt`.
  Coverage: `test/test_mmrm.jl`; stress: `docs/stress/mmrm_stress.qmd`.
- **`family_outcome` on LMTP runners** ([#34](https://github.com/SimonAB/CausalTargeted.jl/issues/34)):
  binomial outcome nuisances for presence endpoints; `suggest_family_outcome`,
  `validate_family_outcome`; `recommend_run_options(...; outcome=)`.
- **Two-part hurdle discrete LMTP** ([#35](https://github.com/SimonAB/CausalTargeted.jl/issues/35)):
  `TwoPartInterventionalMean`, `run_two_part_discrete_lmtp`,
  `run_two_part_discrete_lmtp_contrast`; `execute_estimand` engine
  `:two_part_discrete_lmtp`. Stress: `docs/stress/hurdle_panel_stress.qmd`.
- **Hurdle-aware conditional independence testing** ([#37](https://github.com/SimonAB/CausalTargeted.jl/issues/37)):
  `test_implied_hurdle_independences`, `local_markov_statements`,
  `IndependenceStatement`, `default_hurdle_label_to_col`. Coverage:
  `test/test_hurdle_ci.jl`.
- **Outcome routing docs:** [methods.md](docs/src/methods.md) table linking
  hurdle, MMRM, MSM, NB2 g-comp, and planned count LMTP ([#36](https://github.com/SimonAB/CausalTargeted.jl/issues/36)).

- **`run_discrete_lmtp_contrast`:** static arm contrast (e.g. SS vs R) from two
  discrete LMTP fits ([#28](https://github.com/SimonAB/CausalTargeted.jl/issues/28)).
- **Apodemus panel stress:** `docs/stress/apodemus_panel_stress.qmd` (synthetic
  default; optional private data via `APODEMUS_CAUSAL_HOME`).
- **Gaussian MMRM stress:** `docs/stress/mmrm_stress.qmd`.
- **Hurdle panel stress:** `docs/stress/hurdle_panel_stress.qmd`.
- Regression tests: three-arm cross-fold fixture ([#31](https://github.com/SimonAB/CausalTargeted.jl/issues/31)),
  `DiscreteTimeCDM` → discrete LMTP recovery ([#33](https://github.com/SimonAB/CausalTargeted.jl/issues/33)).

### Fixed

- **`test_mixedmodels.jl`:** qualify `MixedModels.fit` to avoid ambiguity with MLJ
  exports (Julia 1.12 CI).

- **Optional MixedModels / profiled-NB2 backend** (`CausalTargetedMixedModelsExt`):
  `fit_profiled_nb2`, `mixed_g_computation`, and related types for
  static-treatment repeated-outcome standardised contrasts (Gaussian LMM,
  fixed-shape NB2 GLMM, dedicated estimated-shape NB2 random-intercept fitter).
  Weakdeps: MixedModels, FastGaussQuadrature, NLopt, SpecialFunctions.
  Parametric reference path beside MSM / LMTP; not a default estimand swap.
  Coverage: `test/test_mixedmodels.jl`, `test/test_profiled_nb2.jl`.

## [0.3.13] — 2026-08-25

### Added

- **Cluster-robust MSM covariance:** `cluster=` on `run_repeated_outcome_msm`
  and `run_parametric_repeated_msm` (column symbol or id vector). Point
  estimates unchanged; ``\\widehat{\\Sigma}`` uses the cluster sandwich
  (`covariance_kind = :cluster`). Sampling hierarchy ≠ LMM / BLUP.
  Coverage: `test/test_repeated_outcome_msm.jl`. Stress hand-off:
  CausalDynamics `docs/stress/hierarchy_stress.qmd`; Documenter link in
  [stress_validation.md](docs/src/stress_validation.md).

## [0.3.12] — 2026-08-25

### Added

- **Parametric treatment×time MSM:** `ParametricRepeatedOutcomeMSM`,
  `run_parametric_repeated_msm` (GLS projection of unstructured IF estimates;
  designs `:constant`, `:linear_time`, `:factor_time`, custom ``τ`` matrices,
  `:mean_treatment_time` with `target=:mean`). Engine `:parametric_msm`.
  Synthetic gate `simulate_mean_treatment_time_msm`. Coverage:
  `test/test_parametric_msm.jl`.

## [0.3.11] — 2026-08-25

### Added

- **Repeated-outcome MSM (point treatment):** `RepeatedOutcomeMSM`,
  `run_repeated_outcome_msm`, and `msm_contrast` estimate the profile
  ``τ(t) = E[Y_t | do(A=1)] - E[Y_t | do(A=0)]`` with a **joint** influence-function
  covariance (shared cross-fit propensity; per-outcome Q). Engine
  `:repeated_msm`. Synthetic gate `simulate_repeated_outcome_ate`; ID helper
  `identify_repeated_outcomes`. Long tables pivot with `unstack_repeated_outcomes`.
  Default `estimator=:tmle` (per-fold clever-covariate fluctuation); `:eif` is
  untargeted AIPW. Missingness uses a shared complete-profile ``R`` (outcomes are
  not imputed). Coverage: `test/test_repeated_outcome_msm.jl`,
  `test/test_missing_strategies_matrix.jl`, recovery scenario
  `:repeated_outcome_ate`.

- `MissingDataResult` from `handle_missing_data` (destructuring of
  `data, weights, extra` unchanged) with `meta` recording `strategy`, miss
  rates, optional PCH `rung`, and `time_indexed`. `complete_numeric_column`
  refuses silent `Missing` → `Float64`. Coverage: `test/test_missing_data.jl`.
  BOUNDARIES / methods document stratum × rung missingness.
- Phase 2: LMTP / g-comp / sequential / discrete / survival runners attach
  missingness metadata (`missingness_metadata` / `with_missingness`); survival
  docs distinguish censoring IPCW from MAR missing `S_T`. Estimand × strategy
  matrix tests in `test/test_missing_strategies_matrix.jl`.
- `mar_set(id)` reads MAR conditioning sets from CausalDynamics
  `IdentificationResult.missingness` (Phase 3 adapter).
- Phase 4: `ImputationDraws`, `impute_posterior` (Gaussian MAR nested MC),
  `pool_lmtp_grids`, and `run_lmtp_grid(...; imputation=)` with Rubin pooling.
  Coverage: `test/test_posterior_imputation.jl`; stress notebook
  `docs/stress/missingness_posterior_stress.qmd`. Turing/RxInfer backends deferred.
- Phase 5: Deep SCM estimation stress adds raw-assay missing → drop → encode →
  LMTP ledger rows (`chunk-raw-assay-missing-encode`).
- Missingness edge / grid validation: `test/test_missingness_edge_cases.jl`
  (all strategies on g-comp / sequential / survival / discrete / mediation;
  posterior and Dynamics→LMTP bridge); stress notebook
  `docs/stress/missingness_grid_stress.qmd` and Documenter page
  `docs/src/stress_missingness.md`.
- **Documentation:** narrative [Missingness](docs/src/missingness.md) page
  (Observable strategies, estimand families, certificates, posterior imputation).

## [0.3.10] - 2026-08-18

### Added

- `estimand_from_query` maps `InterventionalPolicyQuery` with a
  `DiscreteTreatmentPolicy` to `DiscreteInterventionalMean`. `TemporalEffectQuery`
  still defaults to `LongitudinalPolicy`; pass `policies` and wide `treatments`
  for `SequentialPolicy`.
- Sequential LMTP recovery test against a CausalDynamics integer-coded `Policy`
  recode (`2 → 1`); `CDMPanel` stays `Float64`, treatments are `Int` at the
  estimator boundary.
- `nested_sl_candidate` for Phillips eSL-inside-dSL under `:cv_selector`
  (opt-in; LMTP classifiers stay `:invmse`).
- Stress notebook chunks for nested dSL, sequential factor LMTP, and
  factor-`A` mediation (via CausalMediation).

## [0.3.9] - 2026-08-18

### Added

- Sequential factor treatments: `SequentialPolicy` / `run_sequential_lmtp`
  accept `policies` (`DiscreteTreatmentPolicy` per time, or one policy
  broadcast). Dummy-coded Q, Díaz–Williams classification ratio at ``t = 1``.
  Mixed continuous/discrete `A_t` is rejected.
- `simulate_sequential_factor_mtp` (T=2 string `A_t`, recode `2 → 1`, sample
  g-computation oracle).

### Changed

- Categorical sequential treatments without `policies` still throw, and now
  point at `policies=` for multi-time recodes as well as `run_discrete_lmtp`
  for T=1.

## [0.3.8] - 2026-08-18

### Added

- Super Learner metalearners `:nnls` (R `method.NNLS`) and `:cv_selector`
  (Phillips discrete Super Learner / sl3 `Lrnr_cv_selector`; alias `:winner`).
- `family=:multinomial` Super Learner (simplex predictions; sl3-style mixture
  NLL / Brier stacking).
- Categorical-treatment LMTP: `DiscreteTreatmentPolicy`, `run_discrete_lmtp`,
  and `DiscreteInterventionalMean` using Díaz–Williams 2n classification
  density ratios with dummy-coded `A`. `handle_missing` matches `run_lmtp_grid`.
- Synthetics: `simulate_binomial_mtp`, `simulate_multinomial_outcome`,
  `simulate_categorical_treatment_mtp`.

### Changed

- `fit_super_learner` defaults to `:nnls` for gaussian outcomes and
  `:nnloglik` for binomial. `:discrete` is a deprecated alias of `:nnls`
  (becomes `:cv_selector` in 0.4).
- Under `family=:binomial`, `:glm` / `:glm_interact` / `:glm_quad` fit logistic
  models (R `SL.glm` family-aware behaviour).
- `execute_estimand` records `density_ratio=:classification` and engine
  `:discrete_lmtp` for discrete jobs. `plan_mtp` costs discrete A as a single
  contrast (`folds * 2` fits). Sequential LMTP rejects categorical treatments
  and points at `run_discrete_lmtp` for T=1.

## [0.3.7] - 2026-08-13

### Added

- Internal `CovariateSchema` (StatsModels `DummyCoding`) so string / categorical /
  `Bool` / numeric adjustment covariates encode to a fold-stable `Float64` design
  matrix without manual dummy coding. Wired through g-computation, LMTP (including
  fold cache), sequential and survival LMTP, and missing-data paths. Mediation
  façades are not yet on the fitted-schema path.
- Stress-validation notebook under `docs/stress/` (CDCS spine audit).
- `simulate_mixed_baseline_mtp` — linear MTP DGP with `String` / `Bool` baseline
  covariates (and rare `breed` level) for fold-stable `CovariateSchema` recovery;
  included in `run_julia_synthetic_once(:mixed_baseline_mtp)`.
- Contrast-learner guard: `validate_contrast_learners` rejects `:mean`-only
  libraries for g-comp / LMTP / sequential / survival.
- Sequential and survival LMTP accept `handle_missing` (including MAR terminal
  $S_T$ without complete-casing the outcome before IPCW).

### Changed

- `run_gcomp` bootstrap **refits** the outcome model on each resample
  (`n_boot = 0` → influence-function SE). Fixes under-coverage from ψ-only
  bootstrap ([#13](https://github.com/SimonAB/CausalTargeted.jl/issues/13)).
- IPCW weights from `handle_missing_data` enter LMTP / g-comp influence summaries
  ([#9](https://github.com/SimonAB/CausalTargeted.jl/issues/9)).

### Fixed

- `weighted_influence_summary` returns Hajek IF-scale centred IC and SE from that
  IC ([#15](https://github.com/SimonAB/CausalTargeted.jl/issues/15)).
- Survival LMTP applies censoring IPCW only in `Q`, not again at the summary
  step ([#16](https://github.com/SimonAB/CausalTargeted.jl/issues/16)).

## [0.3.6] - 2026-08-13

### Added

- Optional Super Learner trees via MLJ weakdeps: `:randomforest`
  (`MLJDecisionTreeInterface`) and `:xgboost` (`MLJXGBoostInterface`). Features
  are unscaled (intercept dropped only). `:randomforest` joins `RICH_SL_LEARNERS`;
  `:xgboost` stays explicit opt-in (rich library already has EvoTrees boosting).
  Neither enters `DEFAULT_SL_LEARNERS`, `SMALL_N_SL_LEARNERS`, or
  `adaptive_learners`.

## [0.3.5] - 2026-08-09

### Added

- Optional Makie MTP effect-curve plotting: `plot_mtp_curve`, `mtp_curve!`, and
  `CausalTargetedMakieExt` (load `CairoMakie` to activate). DataFrame defaults
  match `run_lmtp_grid` columns, including optional clamp strips and TE / NDE /
  NIE styling.
- Super Learner metalearner `:nnloglik` for `family=:binomial`: nonnegative
  Bernoulli log-likelihood fitting on trimmed candidate logits, with prediction
  rule `logistic(Σ wⱼ logit(pⱼ))` aligned to R `SuperLearner::method.NNloglik`
  (`dev/qc_nnloglik.R` for manual QC).

### Changed

- `[compat] MLJ` widened from `"0.20"` to `"0.20, 0.21, 0.22, 0.23"` so Super
  Learner weakdeps resolve against current MLJ (0.23.x). No API change; MLJFlow
  is unused.

## [0.3.4] - 2026-08-08

### Added

- Restore **CausalMediation** weakdep and `CausalTargetedCausalMediationExt` now
  that CausalMediation **0.1.0** is on General. Load with `using CausalMediation`
  to activate mediation façades (`run_mediation_grid`, and related).

## [0.3.1] - 2026-07-29

### Changed

- `:glmnet`, `:glmnet_lasso`, and `:glmnet_ridge` are Julia-native aliases over
  **MLJLinearModels** (via `CausalTargetedMLJExt`). Load with
  `using MLJ, MLJLinearModels`. The Fortran **GLMNet** weakdep / extension is removed.

## [0.3.0] - 2026-07-29

### Changed

- **EvoTrees** is a weakdep (`CausalTargetedEvoTreesExt`). Load with `using EvoTrees`.
  **MLJ** / **MLJLinearModels** are weakdeps for `:glmnet*` and `:mlj_*`.
- `DEFAULT_SL_LEARNERS` and `SMALL_N_SL_LEARNERS` are now `(:glm, :mean)` so
  default grids work without optional packages. Use `RICH_SL_LEARNERS` (and load
  MLJ/EvoTrees) for elastic-net / tree candidates.

## [0.2.3] - 2026-07-29

### Changed

- LMTP and mediation δ-grid jobs build typed `NamedTuple` rows (instead of
  `Dict{String,Any}`); simultaneous bands are applied on the result `DataFrame`.
  Public return type remains a `DataFrame` with the same column names.

## [0.2.2] - 2026-07-29

### Changed

- `OutcomeRegression` / `ExposureDensity` cache the full-sample covariate design
  `W`; predictions assemble treatment via `outcome_design_matrix` instead of
  rebuilding covariates from the `DataFrame` each time.
- Learner fit/predict dispatch uses `Val` methods behind Symbol façades
  (`_fit_learner` / `_predict_learner`).

### Added

- `covariate_design_matrix`, `outcome_design_matrix` helpers.

## [0.2.1] - 2026-07-29

### Changed

- `design_matrix`, `_expand_interactions`, and `_expand_quadratic` preallocate
  output matrices instead of building via `hcat` of temporary column vectors.

## [0.2.0] - 2026-07-29

### Changed

- `DEFAULT_SL_LEARNERS` is now `(:glm, :glmnet, :mean)` (leaner grids). Use
  `RICH_SL_LEARNERS` for interactions / EvoTrees (synthetic recovery defaults to rich).
- `fit_super_learner` returns typed `SuperLearnerFit`; fold caches store
  `Vector{SuperLearnerFit}` instead of `Vector{Any}`.
- **MLJ** / **MLJLinearModels** are weakdeps (`CausalTargetedMLJExt`); load with
  `using MLJ, MLJLinearModels`. **MLJFlux** extension now requires `MLJ` as well.
- Mediation grids honour `parallel=true` (default when `nthreads() > 1`), matching LMTP.
- Parallel δ-jobs use per-job `StableRNG` streams derived from the caller seed.
- Underscore-prefixed helpers are no longer exported (still available as
  `CausalTargeted._…` for debugging).
