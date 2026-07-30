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
| **DAGMakie** | Optional DAG figures (via CausalDynamics plotting façades) |
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
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = [-0.5, 0.0, 0.5],
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    learners_trt = opts.learners_trt,
    parallel = false,
    positivity = opts.positivity,
    simultaneous = false,
)

fig = Figure(size = (520, 320))
ax = Axis(fig[1, 1];
    xlabel = "δ (z-scale shift)",
    ylabel = "TE estimate",
    title = "LMTP grid (synthetic linear MTP)",
)
band!(ax, grid.delta, grid.lwr, grid.upr; color = (:steelblue, 0.25))
lines!(ax, grid.delta, grid.est; color = :steelblue, linewidth = 2)
scatter!(ax, grid.delta, grid.est; color = :steelblue, markersize = 10)
hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
fig
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
