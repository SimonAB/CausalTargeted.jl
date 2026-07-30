# CausalTargeted.jl

Cross-fitted targeted inference for **longitudinal modified treatment policies (LMTP)** and
interventional mediation (TE / NDE / NIE under continuous MTP). Identification is delegated to
[CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).

**Design principles:** [DESIGN.md](DESIGN.md) · [NAMING.md](NAMING.md) · [ecosystem](../DESIGN_PRINCIPLES.md) ·
[boundaries](BOUNDARIES.md)

[![DOI](https://zenodo.org/badge/latestdoi/1314795986.svg)](https://zenodo.org/badge/latestdoi/1314795986)

**Documentation (with literature):**

- [Methods and literature](docs/src/methods.md) — maps APIs to papers (LMTP, mediation, TMLE, sensitivity)
- [Small-*n* checklist](docs/src/small_n.md) — conservation and other low-sample applications
- [References](docs/src/references.md) — DOIs and BibTeX keys shared with the CDCS book `references.bib`

Build Documenter pages locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Installation

```julia
using Pkg
Pkg.develop(path="packages/CausalTargeted.jl")  # from CDCS monorepo
using CausalTargeted
```

## Quick start

```julia
using CausalTargeted, CausalDynamics

df, _ = simulate_linear_mtp(200)
opts = recommend_run_options(nrow(df); engine = :lmtp)
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = [-0.5, 0.0, 0.5],
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    parallel = opts.parallel,
    positivity = opts.positivity,
)
```

## Stack

| Package | Role |
|---------|------|
| **CausalDynamics** | `identify`, graphs, `IdentificationResult` |
| **CausalTargeted** | nuisances, LMTP/mediation grids, certificates, small-*n* profiles |
| **Application repos** | cohort data, registries, concordance (thin) |

## Optional SuperLearner candidates

Default grid library is lean (`:glm`, `:mean`). Use `RICH_SL_LEARNERS` when you want
interactions / elastic-net / EvoTrees — load the matching weakdeps first:

```julia
using CausalTargeted
using MLJ, MLJLinearModels  # :glmnet / :glmnet_lasso / :glmnet_ridge and :mlj_*
using EvoTrees              # :evotree, :evotree_deep

fit_super_learner(X, y; learners = (:glm, :mlj_ridge, :mlj_lasso, :mean))

using MLJFlux  # activates CausalTargetedMLJFluxExt (also needs MLJ)
fit_super_learner(X, y; learners = (:glm, :mlj_mlp, :mean))
```

Features are column-standardised for MLJ fits (leading intercept of ones is dropped). Neural
learners are never included in `SMALL_N_SL_LEARNERS` / `adaptive_learners`.

## Core citations (see [References](docs/src/references.md) for the full list)

- Díaz, Williams, Hoffman & Schenck (2023). Nonparametric causal effects based on longitudinal modified treatment policies. *JASA*. [doi:10.1080/01621459.2021.1955691](https://doi.org/10.1080/01621459.2021.1955691)
- Díaz & Hejazi (2020). Causal mediation analysis for stochastic interventions. *JRSS-B*. [doi:10.1111/rssb.12362](https://doi.org/10.1111/rssb.12362)
- Liu, Williams, Rudolph & Díaz (2024). General targeted machine learning for modern causal mediation analysis. arXiv:2408.14620
- van der Laan & Rose (2011). *Targeted Learning*. Springer
- Cinelli & Hazlett (2020). Making sense of sensitivity. *JRSS-B*. [doi:10.1111/rssb.12348](https://doi.org/10.1111/rssb.12348)

## See also

- [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl)
- [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) for DAG figures
- [CDCS book](https://simonab.github.io/causal-dynamics-book/)
