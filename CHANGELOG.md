# Changelog

All notable changes to CausalTargeted.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
