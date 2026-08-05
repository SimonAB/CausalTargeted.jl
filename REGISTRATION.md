## Registration status

CausalTargeted.jl is on the Julia **General** registry.

Install: `Pkg.add("CausalTargeted")`. Requires Julia **1.12+**.
Hard dependency: [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) (`0.4` compat from **0.3.3**).

| Version | Status |
|---------|--------|
| **0.3.2** | On General ([#163199](https://github.com/JuliaRegistries/General/pull/163199)) |
| **0.3.3** | Pending — mediation façades → CausalMediation; requires CausalDynamics **0.4** and CausalMediation **0.1** on General |

## 0.3.3 register steps (blocked)

1. CausalDynamics 0.4.0 on General
2. CausalMediation 0.1.0 on General (weakdep)
3. Push `0.3.3`, then `@JuliaRegistrator register`

## Changes in 0.3.3

- Mediation implementation moved to CausalMediation.jl
- Soft façades + `@deprecate` for `run_crumble_*`
- Compat: `CausalDynamics = "0.4"`, weakdep `CausalMediation = "0.1"`

## Zenodo DOI

Deposit metadata is in `.zenodo.json` and `CITATION.cff`.
