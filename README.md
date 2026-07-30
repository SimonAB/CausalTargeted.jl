# CausalTargeted.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21703329.svg)](https://doi.org/10.5281/zenodo.21703329)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://simonab.github.io/CausalTargeted.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CausalTargeted implements cross-fitted targeted estimators for continuous and
longitudinal exposures: longitudinal modified treatment policies (LMTP),
interventional mediation (TE / NDE / NIE under MTP), positivity diagnostics,
nested Monte Carlo stability checks, and omitted-confounder sensitivity.
Defaults favour small-to-moderate sample sizes. Identification is delegated to
[CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).

**Design principles:** [DESIGN.md](DESIGN.md) · [NAMING.md](NAMING.md) ·
[BOUNDARIES.md](BOUNDARIES.md) · [ecosystem](DESIGN_PRINCIPLES.md)

> Requires Julia **1.12+**. Install from GitHub until the package is on General.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/SimonAB/CausalTargeted.jl.git")
using CausalTargeted
```

From the CDCS monorepo:

```julia
Pkg.develop(path="packages/CausalTargeted.jl")
```

CausalDynamics is a hard dependency and is pulled in automatically when you
`Pkg.add` / `Pkg.develop` this package (path-wired in the monorepo
`Project.toml`).

## Quick start

```julia
using CausalTargeted, CausalDynamics

df, _ = simulate_linear_mtp(200)
opts = recommend_run_options(size(df, 1); engine = :lmtp)
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = [-0.5, 0.0, 0.5],
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    learners_trt = opts.learners_trt,
    parallel = opts.parallel,
    positivity = opts.positivity,
)
```

## Ecosystem

| Package | Role |
|---------|------|
| [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) | Graphs, `identify`, `IdentificationResult` |
| **CausalTargeted** | Nuisances, LMTP / mediation grids, certificates, small-*n* profiles |
| [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) | DAG figures (optional) |
| Application repos | Cohort data, registries, concordance (thin) |

## Optional Super Learner candidates

Default grid library is lean (`:glm`, `:mean`). Use `RICH_SL_LEARNERS` when you
want interactions / elastic-net / EvoTrees; load the matching weakdeps first:

```julia
using CausalTargeted
using MLJ, MLJLinearModels  # :glmnet_* and :mlj_*
using EvoTrees              # :evotree, :evotree_deep

fit_super_learner(X, y; learners = (:glm, :mlj_ridge, :mlj_lasso, :mean))

using MLJFlux  # activates CausalTargetedMLJFluxExt (also needs MLJ)
fit_super_learner(X, y; learners = (:glm, :mlj_mlp, :mean))
```

Features are column-standardised for MLJ fits (leading intercept of ones is
dropped). Neural learners are never included in `SMALL_N_SL_LEARNERS` /
`adaptive_learners`.

## Related packages

This package covers **continuous MTP / LMTP and interventional mediation**. For
point-treatment CM / ATE / AIE, prefer
[TMLE.jl](https://github.com/TARGENE/TMLE.jl). Graphs and identification live
upstream in CausalDynamics (`prepare_for_tmle` bridges to TMLE.jl).

| Package | Role |
|---------|------|
| [TMLE.jl](https://github.com/TARGENE/TMLE.jl) | Point-treatment CM / ATE / AIE (TMLE, OSE, C-TMLE) |
| [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) | Graphs and identification certificates (required upstream) |
| [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) | DAG figures for causal diagrams |
| [CausalTables.jl](https://github.com/salbalkus/CausalTables.jl) | SCM-aware tables; often paired with TMLE.jl |
| [CausalInference.jl](https://github.com/mschauer/CausalInference.jl) | Structure learning and classical graphical criteria |

R analogues for the continuous / mediation slice (`lmtp`, `crumble`) are
conceptual parity, not API identity; see [NAMING.md](NAMING.md).

## Documentation

- [Documenter site](https://simonab.github.io/CausalTargeted.jl/dev/) (methods, small-*n* checklist, live figures)
- [Methods and literature](docs/src/methods.md) — maps APIs to papers
- [References](docs/src/references.md) — DOIs and BibTeX keys shared with the CDCS book
- [CDCS book](https://simonab.github.io/causal-dynamics-book/) — worked identify → estimate → display examples

Build Documenter pages locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

### Core citations

Full list in [References](docs/src/references.md). Highlights:

- Díaz, Williams, Hoffman & Schenck (2023). Nonparametric causal effects based on longitudinal modified treatment policies. *JASA*. [doi:10.1080/01621459.2021.1955691](https://doi.org/10.1080/01621459.2021.1955691)
- Díaz & Hejazi (2020). Causal mediation analysis for stochastic interventions. *JRSS-B*. [doi:10.1111/rssb.12362](https://doi.org/10.1111/rssb.12362)
- Liu, Williams, Rudolph & Díaz (2024). General targeted machine learning for modern causal mediation analysis. arXiv:2408.14620
- van der Laan & Rose (2011). *Targeted Learning*. Springer
- Cinelli & Hazlett (2020). Making sense of sensitivity. *JRSS-B*. [doi:10.1111/rssb.12348](https://doi.org/10.1111/rssb.12348)

## Acknowledgements

Part of the Causal Dynamics for Complex Systems (CDCS) project.
Maintainer: [Simon A. Babayan](https://orcid.org/0000-0002-4949-1117).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff) or:

```bibtex
@software{causaltargeted2026,
  author = {Babayan, Simon A.},
  title  = {CausalTargeted.jl: Cross-fitted LMTP and interventional mediation},
  year   = {2026},
  doi    = {10.5281/zenodo.21703329},
  url    = {https://github.com/SimonAB/CausalTargeted.jl}
}
```
