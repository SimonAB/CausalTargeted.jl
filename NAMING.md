# Naming scheme (owned causal stack)

## Packages (keep)

| Package | Role | Why the name |
|---------|------|----------------|
| **CausalDynamics.jl** | Graphs, identification, SCMs / CDMs, `do(·)` | Dynamics = structural + temporal generative layer |
| **CausalTargeted.jl** | Cross-fitted estimation (LMTP, certificates) | Targeted = targeted learning / TMLE lineage |
| **CausalMediation.jl** | Mediation EIF / TE / NDE / NIE / `moc` | Owns interventional and path-specific mediation |
| **DAGMakie.jl** | DAG figures only | Makie backend for DAGs; no identification |

Applications (e.g. SheepVaccineCDCS) stay thin: loaders, registry TOML, concordance with R notebooks.

See also the **Policy taxonomy** in
[DESIGN_PRINCIPLES.md](https://github.com/SimonAB/causal-dynamics-book/blob/main/packages/DESIGN_PRINCIPLES.md).

## Estimation engines (Julia vocabulary)

Prefer **method names**, not R package nicknames:

| Engine symbol | Public APIs | Owner | Meaning |
|---------------|-------------|-------|---------|
| `:lmtp` | `run_lmtp_grid`, `InterventionalMean` | CausalTargeted | Longitudinal / continuous modified treatment policies |
| `:discrete_lmtp` | `run_discrete_lmtp`, `DiscreteInterventionalMean` | CausalTargeted | Categorical-treatment LMTP (classification density ratios) |
| `:mediation` | `run_mediation_grid`, `run_mediation`, `MediationSpec` | **CausalMediation** | Interventional TE / NDE / NIE under MTP shifts |
| `:sequential_lmtp` | `run_sequential_lmtp` | CausalTargeted | Multi-time sequential regression (numeric shift or factor `policies`) |
| `:survival_lmtp` | `run_survival_lmtp`, `SurvivalPolicy` | CausalTargeted | Discrete-time event-time / survival LMTP |
| `:repeated_msm` | `run_repeated_outcome_msm`, `RepeatedOutcomeMSM`, `msm_contrast` | CausalTargeted | Binary point treatment, repeated outcomes, joint IF ``Σ``; optional `cluster=` sandwich |
| `:parametric_msm` | `run_parametric_repeated_msm`, `ParametricRepeatedOutcomeMSM` | CausalTargeted | GLS projection onto treatment×time MSM designs; forwards `cluster=` |

Numeric clamp-aware density-ratio shifts use kwargs `shift_amount` /
`shift_reference` (scalars), not a `ShiftPolicy` object. Construct MTP intent
with `additive_shift_policy` and related constructors.

## Mediation ownership

Use **`using CausalMediation`**. CausalTargeted no longer exports mediation
façades (`run_mediation_grid`, `MediationContrast`, …). Internal
`execute_estimand` may still dispatch mediation when CausalMediation is loaded
as a weakdep; prefer calling CausalMediation directly in new code.

## Legacy R concordance (`crumble`)

The R package [`crumble`](https://cran.r-project.org/package=crumble) (Liu et al.)
inspired the mediation grid. In Julia the brand is CausalMediation:

- Prefer `CausalMediation.run_mediation_grid`, `MediationSpec`, `moc`, …
- Soft-deprecated `crumble` aliases (if present) map to mediation names inside
  CausalMediation / historical Targeted shims.

Cite the papers (`liu2024mediation`, `liu2025crumble`, `diaz2020mediation`); do
not treat the R package name as the Julia API brand.

## Synthetic DGPs

Exported for book / README examples: `simulate_linear_mtp`,
`simulate_discrete_survival_mtp`, `simulate_mixed_baseline_mtp`,
`simulate_binomial_mtp`, `simulate_multinomial_outcome`,
`simulate_categorical_treatment_mtp`, `simulate_sequential_factor_mtp`,
`simulate_repeated_outcome_ate`.
Mediation DGPs live under CausalMediation (or
`CausalTargeted.simulate_mediation` as a qualified in-module helper).
Other scenario builders remain in-module for package tests and
`scripts/synthetic_benchmark/`.
