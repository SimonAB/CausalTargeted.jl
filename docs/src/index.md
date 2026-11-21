# CausalTargeted.jl

```@meta
CurrentModule = CausalTargeted
```

Given data and a known adjustment set, CausalTargeted estimates the causal
effect of shifting a continuous, discrete, or longitudinal exposure, with
small-sample defaults, positivity diagnostics, and omitted-confounder
sensitivity analysis.

**Deciding which estimator fits your study?** See
[Choosing an approach](orientation.md) to match your exposure type, study
design, outcome distribution, and sample size to the right estimator.

CausalTargeted implements cross-fitted targeted estimators for continuous and
longitudinal exposures, including longitudinal modified treatment policies
(LMTP; Díaz et al., 2023,
[JASA](https://doi.org/10.1080/01621459.2021.1955691)), repeated-outcome
profiles (Rosenblum & van der Laan, 2010), parametric regression
standardisation, positivity diagnostics, and omitted-confounder sensitivity
(Cinelli & Hazlett, 2020). Defaults favour small-to-moderate sample sizes.
Method-to-API mapping: [Methods](methods.md); DOIs: [References](references.md).

**Mediation** (TE / NDE / NIE, `moc`, nested Monte Carlo stability) lives in
[CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl)
(`using CausalMediation`). This package supplies Super Learner and LMTP
infrastructure that Mediation reuses.

Identification is delegated to [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).
This package estimates parameters once a query and adjustment set are known.

For longitudinal policy work, CausalDynamics supplies the temporal
specification and its edge provenance, while CausalTargeted consumes the
resulting panel and declared policy estimand. A policy that persists as one
decision rule while its treatment output varies with history should keep that
identity in provenance; repeated applications are not independent policy
instances. `temporal_edge_records` is therefore useful for the hand-off and
audit trail, but is not an estimator input by itself.

## Compared with R and Python

CausalTargeted provides Julia-native LMTP δ-grids, small-*n* Super Learner
profiles, and typed hand-off from `IdentificationResult` (comparable to R `lmtp`
and Python Ananke, while interventional mediation lives in
[CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl)). See
[Comparison](comparison.md) for the full feature matrix and
[ECOSYSTEM_COMPARISON.md](https://github.com/SimonAB/causal-dynamics-book/blob/main/packages/ECOSYSTEM_COMPARISON.md).

## Related packages

| Package | Role |
|---------|------|
| **CausalDynamics** | Graphs, `identify`, temporal unrolling, `IdentificationResult` |
| **CausalTargeted** | Nuisances, LMTP grids, certificates, small-*n* profiles |
| **CausalMediation** | Mediation EIF / TE / NDE / NIE (optional weakdep; `using CausalMediation`) |
| **DAGMakie** | Optional DAG figures (via CausalDynamics plotting façades) |
| Application repos | Cohort data, registries, concordance (thin) |

Design notes:
[DESIGN.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/DESIGN.md) ·
[NAMING.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/NAMING.md) ·
[BOUNDARIES.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/BOUNDARIES.md) ·
[ecosystem principles](https://github.com/SimonAB/causal-dynamics-book/blob/main/packages/DESIGN_PRINCIPLES.md).

## Methods and literature

- **[Choosing an approach](orientation.md)** — match study design, exposure, outcome, and sample size to an estimator
- **[Getting started](getting-started.md)** — executable walk-throughs (LMTP, mediation, sequential, missingness)
- [Methods and literature](methods.md) — LMTP, Super Learner, positivity, sensitivity, outcome paths
- [Bayesian Gaussian & hierarchical estimation](bayesian.md) — conjugate Gaussian and clustered VI g-computation
- [Bayesian logistic estimation (RxInfer)](bayesian_logistic.md) — approximate logistic inference and LRVB
- [Missingness](missingness.md) — Observable policies, certificates, posterior imputation
- [Small-*n* checklist](small_n.md) — defaults for tens to low hundreds of units
- [API overview](api.md) — exported types and functions
- [Comparison](comparison.md) — Julia vs R (`lmtp`, `crumble`) and Python (Ananke, DoubleML)
- [Stress validation](stress_validation.md) — catalogued datasets, papers, packages, smoke freeze, and missingness/deep-SCM ledgers
- **Quarto notebook:** [`docs/stress/stress_validation.qmd`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/stress/stress_validation.qmd) — dataset analyses, expected vs actual, DAGMakie / MTP figures
- [References](references.md) — bibliographic list with DOIs

Canonical sources include Díaz et al. (2023) on LMTP; Díaz & Hejazi (2020) and
Liu et al. (2024) on stochastic / modern mediation; van der Laan & Rose on TMLE /
Super Learner; and Cinelli & Hazlett (2020) on partial-*R*² sensitivity. BibTeX
keys such as `diaz2023lmtp` live in the CDCS book `references.bib`.

## Quick start

For step-by-step walk-throughs (identify → LMTP, mediation, sequential LMTP,
missing data, positivity), see **[Getting started](getting-started.md)**.

Point-treatment LMTP under baseline confounding uses the DAG below
(`W → A → Y`, `W → Y`). Identification sets come from CausalDynamics;
figures use the optional DAGMakie bridge.

```@example home
using CausalTargeted, CausalDynamics, Graphs, DAGMakie, CairoMakie

g = DiGraph(3)
add_edge!(g, 1, 2)  # W → A
add_edge!(g, 1, 3)  # W → Y
add_edge!(g, 2, 3)  # A → Y

fig = plot_backdoor_paths(g, 2, 3; node_labels = ["W", "A", "Y"])
fig
```

```@example home
df, _ = simulate_linear_mtp(200)
opts = recommend_run_options(size(df, 1); engine = :lmtp)
δ = collect(-1.0:0.25:1.0)
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = δ,
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    learners_trt = opts.learners_trt,
    parallel = false,
    positivity = opts.positivity,
    simultaneous = false,
)

fig, ax = plot_mtp_curve(
    grid;
    title = "LMTP grid (synthetic linear MTP)",
    xlabel = "δ (z-scale shift)",
    ylabel = "TE estimate",
)
fig
```

## Installation

```julia
using Pkg
Pkg.add("CausalTargeted")
# Pkg.develop(path="<path-to-local-clone>")
using CausalTargeted
```

## See also

Worked examples appear in the [CDCS book](https://simonab.github.io/causal-dynamics-book/)
and in application repositories. Prefer package APIs over copying application
column names.
