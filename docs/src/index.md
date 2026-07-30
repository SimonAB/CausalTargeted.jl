# CausalTargeted.jl

```@meta
CurrentModule = CausalTargeted
```

Cross-fitted **targeted inference** for continuous and longitudinal exposures: longitudinal
modified treatment policies (LMTP), interventional mediation (TE / NDE / NIE under MTP), positivity
atlases, nested-MC stability, and omitted-confounder sensitivity—optimised for
**small-to-moderate** sample sizes.

Identification is delegated to [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).
This package estimates parameters once a query and adjustment set are known.

## Stack role

| Package | Role |
|---------|------|
| **CausalDynamics** | Graphs, `identify`, temporal unrolling, `IdentificationResult` |
| **CausalTargeted** | Nuisances, LMTP / mediation grids, certificates, small-*n* profiles |
| Application repos | Cohort data, registries, concordance (thin) |

Shared design rules:
[DESIGN.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/DESIGN.md) ·
[NAMING.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/NAMING.md) ·
[BOUNDARIES.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/BOUNDARIES.md) ·
[ecosystem principles](https://github.com/SimonAB/CausalDynamics.jl/blob/main/DESIGN_PRINCIPLES.md).

## Scientific foundations (start here)

The methods page maps every public API to the research literature:

- [Methods and literature](methods.md) — LMTP, mediation, Super Learner, positivity, sensitivity
- [Small-*n* checklist](small_n.md) — practical defaults for tens to low hundreds of units
- [References](references.md) — full bibliographic list with DOIs

Canonical theory sources include Díaz et al. (2023) on LMTP; Díaz & Hejazi (2020) and
Liu et al. (2024) on stochastic / modern mediation; van der Laan & Rose on TMLE / Super Learner;
and Cinelli & Hazlett (2020) on partial-*R*² sensitivity. BibTeX keys such as `diaz2023lmtp`
live in the CDCS book `references.bib`.

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

## Installation

From the CDCS monorepo:

```julia
using Pkg
Pkg.develop(path="packages/CausalTargeted.jl")
using CausalTargeted
```

## Narrative showcase

Worked examples appear in the [CDCS book](https://simonab.github.io/causal-dynamics-book/)
and in application repos (e.g. sheep vaccine pathways). Prefer package APIs over copying
application column names.
