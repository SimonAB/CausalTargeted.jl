# CausalTargeted.jl — design principles

This package is the **targeted inference layer**: cross-fitted nuisances, LMTP and mediation EIF estimators, δ-grids, planning, and run provenance.

**Shared principles:** [../DESIGN_PRINCIPLES.md](../DESIGN_PRINCIPLES.md)  
**Boundaries:** [BOUNDARIES.md](BOUNDARIES.md)

## Role in the stack

```
IdentificationResult  →  plan_mtp / execute_estimand  →  grid + metadata
         ↑                        ↑
   CausalDynamics          typed Estimand + nuisances
```

CausalTargeted **consumes** identification; it does not redefine backdoor criteria or parse manuscript DAG strings.

## Package-specific principles

### Identification stays upstream

- Accept **`IdentificationResult`** or build certificates via **`identification_certificate`**.
- Nuisance column lists come from graph ID + optional **`ColumnResolver`** patterns—not hard-coded pathway names.
- Never duplicate `backdoor_adjustment_set` logic here; call **`CausalDynamics.identify`**.

### Typed estimands, not stringly tasks

- **`InterventionalMean`**, **`MediationContrast`**, **`LongitudinalPolicy`**, **`ScalarMediation`** encode estimand intent.
- **`estimand_from_query`** bridges `CausalQuery` objects to estimands for composable pipelines.
- Application-specific task structs (e.g. registry TOML rows) are converted at the **application boundary**, not stored in this package.

### Julia-native estimation

- Implement EIF/TMLE steps in Julia (SuperLearner stacks via **GLM**, optional **EvoTrees / MLJ / MLJFlux** candidates—not `RCall`).
- Match **mathematical notation in code** (`σ`, `δ`, `ψ`, fold indices) where it aids reading against the book.
- Prefer **`Float64` pipelines** with explicit RNG (`StableRNGs`) for reproducible tests.
- Default grid library is lean (`DEFAULT_SL_LEARNERS = (:glm, :mean)`); load MLJ/EvoTrees and use `RICH_SL_LEARNERS` for recovery / ablation.
- EvoTrees and MLJ candidates are **weakdeps**; `:glmnet*` is an MLJLinearModels alias (no Fortran GLMNet). MLJ fits **standardise features**. `:mlj_mlp` / `:mlj_nn_binary` require `using MLJFlux` and are **never** in small-*n* / adaptive defaults.
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
| LMTP, mediation grids, scalar PPL/bootstrap helpers | Cohort merge, XLSX loaders, dagitty parsing |
| Nuisance interfaces (`OutcomeRegression`, …) | Hard-wiring neural nets into small-*n* defaults |
| Optional MLJ / MLJFlux learner symbols (`:mlj_*`) | Full Riesz-net / GPU parity with R `crumble` |
| Synthetic DGPs + dual-stack recovery helpers | Promoting flexible learners to defaults from one draw |
| g-computation / DiD utilities | Biological concordance vs R masters |

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
