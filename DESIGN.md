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

- Implement EIF/TMLE steps in Julia (SuperLearner stacks via **GLM / GLMNet / EvoTrees**, not `RCall`).
- Match **mathematical notation in code** (`σ`, `δ`, `ψ`, fold indices) where it aids reading against the book.
- Prefer **`Float64` pipelines** with explicit RNG (`StableRNGs`) for reproducible tests.

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
| LMTP, crumble/mediation grids, scalar PPL/bootstrap helpers | Cohort merge, XLSX loaders, dagitty parsing |
| Synthetic DGPs for **`Pkg.test()`** | Biological concordance vs R masters |
| Nuisance interfaces (`OutcomeRegression`, …) | Full MLJ integration (unless added as optional extension) |

### Composability

- **`execute_estimand`** dispatches on estimand type; lower-level **`run_lmtp_grid`** / **`run_crumble_grid`** remain available for custom workflows.
- **`MTPPlan`** and **`summarise_plan`** support dry-run cost estimation before fitting.
- Targeting diagnostics (`tmle_score_diagnostics`, `optimise_tmle_fluctuation`) are opt-in helpers, not hidden inside every fit.

### Testing

- Synthetic recovery tests are the **merge gate**; tolerances should reflect statistical noise, not API drift.
- Do not add dependencies on application repos or book paths in `test/`.

## Adding a feature (workflow)

1. Check [BOUNDARIES.md](BOUNDARIES.md)—is this estimation, not ID or plotting?
2. Extend **`Estimand`** subtypes or grid options only if the estimand class is genuinely new.
3. Thread **`IdentificationResult`** / certificate through any new high-level entry point.
4. Add tests with **`simulate_*`** DGPs; document exports with docstrings and one runnable example.
5. Bump compat on **CausalDynamics** when relying on new identification fields.
