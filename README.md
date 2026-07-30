# CausalTargeted.jl

Cross-fitted targeted inference for **longitudinal modified treatment policies (LMTP)** and
interventional mediation (TE / NDE / NIE under continuous MTP). Identification is delegated to
[CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).

**Design principles:** [DESIGN.md](DESIGN.md) · [NAMING.md](NAMING.md) · [ecosystem](../DESIGN_PRINCIPLES.md) ·
[boundaries](BOUNDARIES.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21703329.svg)](https://doi.org/10.5281/zenodo.21703329)

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
| **DAGMakie** | DAG figures (optional) |
| **Application repos** | cohort data, registries, concordance (thin) |

## Related packages (Julia causal ecosystem)

This package covers **continuous MTP / LMTP and interventional mediation**. For
point-treatment CM / ATE / AIE with TMLE or one-step estimators, use
[TMLE.jl](https://github.com/TARGENE/TMLE.jl) (TARGENE; JOSS). Identification and
graphs live upstream in CausalDynamics (`prepare_for_tmle` bridges to TMLE.jl).

| Package | Role relative to CausalTargeted |
|---------|----------------------------------|
| [TMLE.jl](https://github.com/TARGENE/TMLE.jl) | Point-treatment CM / ATE / AIE (TMLE, OSE, C-TMLE); prefer this for categorical ATE |
| [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) | Graphs, identification certificates (required upstream) |
| [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) | Publication DAG figures |
| [CausalTables.jl](https://github.com/salbalkus/CausalTables.jl) | SCM-aware tables; interoperates with TMLE.jl |
| [CausalInference.jl](https://github.com/mschauer/CausalInference.jl) | Discovery and classical graphical criteria |

R analogues for the continuous / mediation slice: `lmtp`, `crumble` (conceptual
parity, not API identity) — see [methods](docs/src/methods.md) and [NAMING.md](NAMING.md).

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

- [CDCS book](https://simonab.github.io/causal-dynamics-book/) — worked examples (identify → estimate → display)
