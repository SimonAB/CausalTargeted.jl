# CausalTargeted.jl

Cross-fitted targeted inference for **longitudinal modified treatment policies (LMTP)** and interventional mediation (crumble-style EIF). Identification is delegated to [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).

**Design principles:** [DESIGN.md](DESIGN.md) · [ecosystem](../DESIGN_PRINCIPLES.md) · [boundaries](BOUNDARIES.md)

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
grid = run_lmtp_grid(df, :A, :Y; baseline = [:W], deltas = [-0.5, 0.0, 0.5])
```

## Stack

| Package | Role |
|---------|------|
| **CausalDynamics** | `identify`, graphs, `IdentificationResult` |
| **CausalTargeted** | nuisances, LMTP/mediation grids, certificates |
| **Application repos** | cohort data, registries, R concordance |

## See also

- [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl)
- [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) for DAG figures
