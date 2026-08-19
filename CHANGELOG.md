# Changelog

All notable changes to CausalTargeted.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
