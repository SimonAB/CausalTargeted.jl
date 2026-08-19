# CausalTargeted.jl — design principles

This package is the **targeted inference layer**: cross-fitted nuisances, LMTP
estimators, δ-grids, planning, and run provenance. Mediation EIF estimators live
in **CausalMediation.jl** (soft façades remain here for compatibility).

**Shared principles:** [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md)  
**Boundaries:** [BOUNDARIES.md](BOUNDARIES.md)

## Role in the stack

```
IdentificationResult  →  plan_mtp / plan_sequential / execute_estimand  →  grid + metadata
         ↑                        ↑
   CausalDynamics          typed Estimand + nuisances
```

CausalTargeted **consumes** identification; it does not redefine backdoor criteria or parse manuscript DAG strings. Identification is **support-agnostic** (factor vs continuous `A` is not a graph property). Numeric MTP uses `ShiftPolicy`; finite recodes use `DiscreteTreatmentPolicy` on `DiscreteInterventionalMean` or `SequentialPolicy.policies`. CausalMediation interventional specs accept the same two policy kinds (both arms must match).

## Package-specific principles

### Identification stays upstream

- Accept **`IdentificationResult`** or build certificates via **`identification_certificate`**.
- Nuisance column lists come from graph ID + optional **`ColumnResolver`** patterns—not hard-coded pathway names.
- Never duplicate `backdoor_adjustment_set` logic here; call **`CausalDynamics.identify`**.

### Typed estimands, not stringly tasks

- **`InterventionalMean`**, **`MediationContrast`**, **`LongitudinalPolicy`**, **`ScalarMediation`**, **`SequentialPolicy`** encode estimand intent. Sequential factor recodes reuse **`DiscreteTreatmentPolicy`** in `policies` rather than a second estimand type.
- **`estimand_from_query`** bridges `CausalQuery` objects to estimands for composable pipelines. A discrete `InterventionalPolicyQuery` becomes `DiscreteInterventionalMean`; `TemporalEffectQuery` stays `LongitudinalPolicy` unless `policies` and wide `treatments` are set. `MediationQuery` plus a discrete policy throws (use CausalMediation `MediationSpec`).
- Application-specific task structs (e.g. registry TOML rows) are converted at the **application boundary**, not stored in this package.

### Julia-native estimation

- Implement EIF/TMLE steps in Julia (SuperLearner stacks via **GLM**, optional **EvoTrees / MLJ / MLJFlux / DecisionTree / XGBoost** candidates—not `RCall`).
- Match **mathematical notation in code** (`σ`, `δ`, `ψ`, fold indices) where it aids reading against the book.
- Prefer **`Float64` pipelines** with explicit RNG (`StableRNGs`) for reproducible tests.
- Default grid library is lean (`DEFAULT_SL_LEARNERS = (:glm, :mean)`); load MLJ/EvoTrees/DecisionTree and use `RICH_SL_LEARNERS` for recovery / ablation.
- EvoTrees and MLJ candidates are **weakdeps**; `:glmnet*` is an MLJLinearModels alias (no Fortran GLMNet). Linear MLJ fits **standardise features**; `:randomforest` / `:xgboost` keep predictors **unscaled**. `:randomforest` is in `RICH_SL_LEARNERS`; `:xgboost` is opt-in only (EvoTrees already covers boosting). `:mlj_mlp` / `:mlj_nn_binary` require `using MLJFlux` and are **never** in small-*n* / adaptive defaults; trees are likewise absent from `adaptive_learners`.
- Nested eSL-inside-dSL is **opt-in** via `nested_sl_candidate` under `metalearner=:cv_selector`; LMTP density-ratio classifiers stay `:invmse`.
- Richer SuperLearner libraries can improve **single-draw** recovery without guaranteeing better **generalisation**—prefer Monte Carlo / ablation before promoting learners to defaults.

### Efficient grids

- **Parallel δ-jobs** when `nthreads() > 1` and `parallel=true`.
- **`cache_nuisances=true` by default** for grids: reuse fold fits across δ unless the user opts out.
- Separate **computation chunks** from **visualisation** in book code; same discipline applies to internal helpers (`lmtp_components_from_cache` vs grid drivers).

### Explicit provenance

- **`attach_run_metadata!`** adds certificate fields, engine, learners, folds, and package version to result tables.
- **`certificate_dict`** uses string keys for CSV/Parquet export; keep keys stable once published.

### Lean scope

| In scope | Out of scope |
|----------|--------------|
| LMTP grids, scalar helpers | Cohort merge, XLSX loaders, dagitty parsing |
| Nuisance interfaces (`OutcomeRegression`, …) | Hard-wiring neural nets into small-*n* defaults |
| Optional MLJ / MLJFlux learner symbols (`:mlj_*`) | Full Riesz-net / GPU parity with R `crumble` |
| Synthetic DGPs + dual-stack recovery helpers | Promoting flexible learners to defaults from one draw |
| g-computation / DiD utilities | Biological concordance vs R masters |
| Soft façades → CausalMediation | Owning mediation EIF / `moc` / RT (see CausalMediation) |
| Optional Makie MTP curve plots (`plot_mtp_curve`) | DAG figures (see DAGMakie); causal estimation inside plot helpers |

### Composability

- **`execute_estimand`** dispatches on estimand type; lower-level **`run_lmtp_grid`** / **`run_mediation_grid`** remain available for custom workflows.
- **`MTPPlan`** and **`summarise_plan`** support dry-run cost estimation before fitting.
- Targeting diagnostics (`tmle_score_diagnostics`, `optimise_tmle_fluctuation`) are opt-in helpers, not hidden inside every fit.

### Literature

Public APIs are mapped to papers in [`docs/src/methods.md`](docs/src/methods.md).
Keep DOIs and BibTeX keys synchronised with the CDCS book `references.bib` via
[`docs/src/references.md`](docs/src/references.md). Do not rename Pearl/TMLE/LMTP
API terms for process-philosophy gloss (book Table 8 only).

### Testing

- Synthetic recovery tests are the **merge gate**; tolerances should reflect statistical noise, not API drift.
- Do not add dependencies on application repos or book paths in `test/`.

## Adding a feature (workflow)

1. Check [BOUNDARIES.md](BOUNDARIES.md)—is this estimation, not ID or plotting?
2. Extend **`Estimand`** subtypes or grid options only if the estimand class is genuinely new.
3. Thread **`IdentificationResult`** / certificate through any new high-level entry point.
4. Add tests with **`simulate_*`** DGPs; document exports with docstrings and one runnable example.
5. Bump compat on **CausalDynamics** when relying on new identification fields.
