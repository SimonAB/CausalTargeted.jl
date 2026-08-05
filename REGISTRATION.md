## Registration status

CausalTargeted.jl is on the Julia **General** registry.

Install: `Pkg.add("CausalTargeted")`. Requires Julia **1.12+**.
Hard dependency: [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) (`0.4` compat from **0.3.3**).

| Version | Status |
|---------|--------|
| **0.3.2** | On General ([#163199](https://github.com/JuliaRegistries/General/pull/163199)) |
| **0.3.3** | Registering — CD **0.4** only (unblocks CausalMediation); CM weakdep deferred to **0.3.4** |
| **0.3.4** | Planned — restore `CausalMediation` weakdep + extension after CM 0.1 is on General |

## Why 0.3.3 omits the CausalMediation weakdep

CausalMediation 0.1 hard-depends on CausalTargeted with CausalDynamics **0.4**.
On General, CT **0.3.2** only allows CD **0.3**, so CM AutoMerge cannot `Pkg.add`.
Register CT **0.3.3** with CD **0.4** first (no CM weakdep — CM is not yet a registered UUID), then retrigger CM, then add the weakdep in **0.3.4**.

## 0.3.3 register steps

1. CausalDynamics 0.4.0 on General — done
2. Push this `0.3.3` (CD 0.4; no CM weakdep)
3. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3)
4. After CT 0.3.3 merges: retrigger CausalMediation Registrator ([General#163653](https://github.com/JuliaRegistries/General/pull/163653))
5. After CM merges: bump **0.3.4** restoring CM weakdep/extension; register

## Changes in 0.3.3

- Mediation implementation moved to CausalMediation.jl (soft façades remain)
- Compat: `CausalDynamics = "0.4"`
- CausalMediation weakdep / extension **not** in this release (see 0.3.4)

## Zenodo DOI

Deposit metadata is in `.zenodo.json` and `CITATION.cff`.
