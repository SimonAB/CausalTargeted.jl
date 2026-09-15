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
| **0.3.6** | On General ([#164383](https://github.com/JuliaRegistries/General/pull/164383), merged 2026-08-13) — trees + Makie MTP plot + `:nnloglik` + MLJ compat `"0.20–0.23"` |
| **0.3.7** | In history (`bb19381`, `CovariateSchema`); **not on General** (skipped; schema exports are in 0.3.8+) |
| **0.3.8–0.3.9** | Skipped on General (same pattern as 0.3.5) |
| **0.3.10** | On General ([#165016](https://github.com/JuliaRegistries/General/pull/165016), merged 2026-08-19); TagBot tagged `v0.3.10` — metalearners, discrete LMTP, sequential factor `policies`, `estimand_from_query`, nested eSL-inside-dSL |
| **0.3.13** | On General — cluster-robust MSM `cluster=` (+ 0.3.11–0.3.12 MSM tips) |
| **0.3.28** | On General ([#166684](https://github.com/JuliaRegistries/General/pull/166684), merged 2026-08-31) — MSM pooled-nuisance stratification (#26) |
| **0.3.27** | On General ([#166678](https://github.com/JuliaRegistries/General/pull/166678), merged 2026-08-31) — Apodemus stress Julia 1.12 load; docs StatsModels/GraphPPL |
| **0.3.18** | Count LMTP Phase A (local) |
| **0.3.17** | On General — `run_estimation_plan`, panel path tests/docs |
| **0.3.16** | On General ([#166627](https://github.com/JuliaRegistries/General/pull/166627), merged 2026-08-30) |

## 0.3.28 register steps

1. Push tip on `main`
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3)
3. General AutoMerge — done ([#166684](https://github.com/JuliaRegistries/General/pull/166684)); TagBot tagged `v0.3.28`

## 0.3.27 register steps

1. Push tip on `main` — done (`accf7b6`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done
3. General AutoMerge — merged ([#166678](https://github.com/JuliaRegistries/General/pull/166678))

## 0.3.26 register steps

1. Push tip on `main` — done (`bc4899a`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done
3. General AutoMerge — merged ([#166670](https://github.com/JuliaRegistries/General/pull/166670))

## 0.3.25 register steps

1. Push tip on `main`
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) (after CausalDynamics **0.4.6** merges if General resolution requires it)
3. General AutoMerge — merged ([#166645](https://github.com/JuliaRegistries/General/pull/166645))

## 0.3.17 register steps

1. Push tip on `main` — done (`1d106bc`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done
3. General AutoMerge — pending (after CausalDynamics 0.4.3 if compat resolution requires it)

## 0.3.16 register steps

1. Merge PR [#39](https://github.com/SimonAB/CausalTargeted.jl/pull/39) to `main` — done (`3afa515`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done ([comment](https://github.com/SimonAB/CausalTargeted.jl/issues/3#issuecomment-5469404617))
3. General AutoMerge — pending

## 0.3.13 register steps

1. Push tip on `main` — done (`ca1d113`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done ([comment](https://github.com/SimonAB/CausalTargeted.jl/issues/3#issuecomment-5414846138))
3. General AutoMerge — pending (skips 0.3.11–0.3.12, same pattern as 0.3.7–0.3.9)

## 0.3.10 register steps

1. Push tip on `main` — done (`4fe6ba7`)
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) — done
3. General AutoMerge — **merged** ([#165016](https://github.com/JuliaRegistries/General/pull/165016)); TagBot tagged `v0.3.10`

## Changes in 0.3.10 (includes 0.3.7–0.3.9 tips)

- `CovariateSchema` exports (from 0.3.7)
- Super Learner metalearners `:nnls` / `:nnloglik` / `:cv_selector`; multinomial SL (from 0.3.8)
- Categorical-treatment LMTP: `DiscreteTreatmentPolicy`, `run_discrete_lmtp` (from 0.3.8)
- Sequential factor recodes via `SequentialPolicy.policies` (from 0.3.9)
- `estimand_from_query` maps discrete `InterventionalPolicyQuery` to `DiscreteInterventionalMean`; `TemporalEffectQuery` stays `LongitudinalPolicy` unless `policies` and wide `treatments` are set
- Opt-in `nested_sl_candidate` (Phillips eSL-inside-dSL under `:cv_selector`)

## Earlier register notes

See git history for 0.3.4 / 0.3.6 step lists. Mediation may drop its `[sources]` pin now that **0.3.10** is on General.
