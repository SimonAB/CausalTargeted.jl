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
| **0.3.5** | Skipped on General (MLJ compat widen shipped in **0.3.6**) |
| **0.3.10** | Tip of `main` — `estimand_from_query` discrete / sequential dispatch; nested eSL-inside-dSL; **register after 0.3.9** |
| **0.3.9** | On `main` — sequential factor `policies`; **register after 0.3.8** |
| **0.3.8** | On `main` (`5ae771f`) — metalearners, multinomial SL, T=1 discrete-A LMTP; **not yet on General** |
| **0.3.7** | In history (`bb19381`, `CovariateSchema`); **not on General**. Skipping 0.3.7 on General is acceptable: 0.3.8 includes the schema export |
| **0.3.6** | On General ([#164383](https://github.com/JuliaRegistries/General/pull/164383), merged 2026-08-13) — trees + Makie MTP plot + `:nnloglik` + MLJ compat `"0.20–0.23"` (0.3.5 skipped) |

## Register 0.3.8 then later tips

`main` is already **0.3.10**. `Project.toml` on current `main` cannot register 0.3.7 or 0.3.8.
To put 0.3.8 on General, checkout `5ae771f` (version still 0.3.8) and comment
`@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3),
then register 0.3.9 and 0.3.10 from their commits. Skipping 0.3.7 is fine: Mediation
`[sources]` can drop once **0.3.8+** is on General.

## 0.3.6 register steps

1. Push `0.3.6` on `main` (PR [#6](https://github.com/SimonAB/CausalTargeted.jl/pull/6) + docs) — done (`58247be`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done ([comment](https://github.com/SimonAB/CausalTargeted.jl/issues/3#issuecomment-5276613011))
3. General AutoMerge — **merged** ([#164383](https://github.com/JuliaRegistries/General/pull/164383)); TagBot tagged `v0.3.6`

## Changes in 0.3.6

- Optional Super Learner `:randomforest` (`MLJDecisionTreeInterface`) in `RICH_SL_LEARNERS`
- Optional Super Learner `:xgboost` (`MLJXGBoostInterface`; opt-in only)
- Makie `plot_mtp_curve` / `mtp_curve!` weakdep (from local tip after 0.3.4)
- Super Learner metalearner `:nnloglik` for `family=:binomial`
- `[compat] MLJ` widened to `"0.20, 0.21, 0.22, 0.23"`

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
