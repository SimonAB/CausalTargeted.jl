## Registration status

CausalTargeted.jl is on the Julia **General** registry.

Install: `Pkg.add("CausalTargeted")`. Requires Julia **1.12+**.
Hard dependency: [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) (`0.4` compat from **0.3.3**).
Optional weakdep: [CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl) (`0.1`, from **0.3.4**).

| Version | Status |
|---------|--------|
| **0.3.2** | On General ([#163199](https://github.com/JuliaRegistries/General/pull/163199)) |
| **0.3.3** | On General ([#163657](https://github.com/JuliaRegistries/General/pull/163657)) — CD **0.4**; CM weakdep deferred |
| **0.3.4** | On General ([#163904](https://github.com/JuliaRegistries/General/pull/163904), merged 2026-08-08) — restore `CausalMediation` weakdep + extension |
| **0.3.5** | Local tip (not yet registered) — MLJ compat `"0.20, 0.21, 0.22, 0.23"` |

## 0.3.4 register steps

1. CausalMediation 0.1.0 on General — done
2. Push `0.3.4` with CM weakdep / `CausalTargetedCausalMediationExt`
3. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done
4. General AutoMerge — **merged** ([#163904](https://github.com/JuliaRegistries/General/pull/163904)); TagBot tagged `v0.3.4`

## Why 0.3.3 omitted the CausalMediation weakdep

CausalMediation 0.1 hard-depends on CausalTargeted with CausalDynamics **0.4**.
On General, CT **0.3.2** only allowed CD **0.3**, so CM AutoMerge could not `Pkg.add`.
CT **0.3.3** shipped CD **0.4** without a CM weakdep (CM was not yet a registered UUID);
**0.3.4** restores the weakdep now that CM is on General.

## Changes in 0.3.4

- Restore CausalMediation weakdep + `CausalTargetedCausalMediationExt`
- Compat: `CausalMediation = "0.1"`

## Zenodo DOI

Deposit metadata is in `.zenodo.json` and `CITATION.cff`.
