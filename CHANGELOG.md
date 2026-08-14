# Changelog

All notable changes to CausalTargeted.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

## [0.3.7] - 2026-08-13

### Added

- Internal `CovariateSchema` (StatsModels `DummyCoding`) so string / categorical /
  `Bool` / numeric adjustment covariates encode to a fold-stable `Float64` design
  matrix without manual dummy coding. Wired through g-computation, LMTP (including
  fold cache), sequential and survival LMTP, and missing-data paths. Mediation
  façades are not yet on the fitted-schema path.


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
