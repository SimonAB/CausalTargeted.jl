## Registration status

Version **0.5.2** is prepared locally on `main` (pending push and General
registration). It delivers performance and TTFX improvements (Gram-matrix NNLS,
weighted GLM Armijo guard and zero-`Core.Box` closures, `δ`-invariant
exponential-tilt grid caching with `compute_bootstrap = false`, precomputed
`LMTPFoldCache` nuisances, zero-allocation `multiplier_simultaneous_critical`,
safeguarded Newton `_weighted_intercept_tilt`, single-pass
`studentised_multiplier_ci` and `_feasible_gaussian_ratio`, concrete
`CovariateSchema{T}`, and a `PrecompileTools` workload), bug fixes (`:policy`
passthrough in `execute_estimand`, fold-salted classifier RNG in
`lmtp_components_from_cache`, `:eif` count-prediction clamping in
`lmtp_tmle_policy_mean`, and unstratified MAR IPCW TMLE fluctuation weighting in
`repeated_outcome_msm.jl`), and internal organisation (`AbstractMTPPolicy` /
`AbstractContinuousPolicy` dispatch, 10-subdirectory `src/` layout, centralised
exports and `using`s, decomposed mega-files, typed MixedModels result/model
structs, `StatsBase` removal, and `SpecialFunctions` / `PrecompileTools` in
`[deps]`).

Version **0.5.1** is on `main` (commit `e258b94`; not registered on General —
superseded by **0.5.2**). It adds continuous LMTP feasible additive shifts
(`FeasibleAdditiveShiftPolicy`, PR #48), proportional headroom shifts
(`ProportionalHeadroomPolicy`), bounded exponential-tilting stochastic
interventions (`ExponentialTiltPolicy`, calibration-balanced TMLE,
`studentised_multiplier_ci`, and `run_lmtp_grid` integration), EvoTrees
0.19/0.20 support, qualified `CausalTargeted.EstimationPlan` /
`plan_targeted_estimation`, Bayesian Phases 3A–5 validation archives, and the
overhauled Documenter manual.

Version **0.5.0** is on General ([#177025](https://github.com/JuliaRegistries/General/pull/177025),
merged 2026-11-20) from commit `88af1bc`. It adds one-level hierarchical
Bayesian Gaussian, logistic, count (`:poisson`, `:nb2`), and two-part hurdle
models, optional `CausalTargetedPSISExt` (`correction = :psis`), `endpoint_mc_bounds`,
the Stage 2 1,000-replication validation archive, and the Julia 1.13 nested-`quadgk`
test fix.

Version **0.4.2** is on General ([#176144](https://github.com/JuliaRegistries/General/pull/176144),
merged 2026-11-13; tree `73549af8b58b61270f13ad407676157a3c1c44c3`)
from commit `08dd269`. It raises the default Bayesian logistic iteration
budget and strengthens independent posterior-reference validation, including
a prior-only reference normalisation repair. Tag
[`v0.4.2`](https://github.com/SimonAB/CausalTargeted.jl/releases/tag/v0.4.2)
identifies that registered commit.

Version **0.4.1** is on General ([#175973](https://github.com/JuliaRegistries/General/pull/175973),
merged 2026-11-12; tree `819b70968f4e3443a9d15b34d22eaba16db9c223`)
and identified by tag [`v0.4.1`](https://github.com/SimonAB/CausalTargeted.jl/releases/tag/v0.4.1)
at commit `dfa4187`. It adds exact Gaussian Bayesian estimation,
optional GraphPPL/RxInfer logistic inference and smooth bounded LMTP policies.

Source release **0.4.0** is on General ([#175636](https://github.com/JuliaRegistries/General/pull/175636), merged 2026-11-10; tree `8eb2c628...`) and identified by tag `v0.4.0`.

CausalTargeted.jl is on the Julia **General** registry.

Install: `Pkg.add("CausalTargeted")`. Requires Julia **1.12+**.
Hard dependency: [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) (`0.5` since **0.4.0**).
Optional weakdep: [CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl). CausalMediation **0.3.x** supports CausalTargeted **0.5.x**. CausalMediation **0.2.x** requires CausalTargeted **0.4.x**.

| Version | Status |
|---------|--------|
| **0.5.2** | Prepared locally on `main` (pending push and General registration) |
| **0.5.1** | On `main` (`e258b94`; superseded on General by **0.5.2**) |
| **0.5.0** | On General ([#177025](https://github.com/JuliaRegistries/General/pull/177025), merged 2026-11-20), tagged [`v0.5.0`](https://github.com/SimonAB/CausalTargeted.jl/releases/tag/v0.5.0) |
| **0.4.2** | On General ([#176144](https://github.com/JuliaRegistries/General/pull/176144), merged 2026-11-13); tagged `v0.4.2` |
| **0.4.1** | On General ([#175973](https://github.com/JuliaRegistries/General/pull/175973), merged 2026-11-12); TagBot tagged `v0.4.1` |
| **0.4.0** | On General ([#175636](https://github.com/JuliaRegistries/General/pull/175636), merged 2026-11-10); TagBot tagged `v0.4.0` |
| **0.3.2** | On General ([#163199](https://github.com/JuliaRegistries/General/pull/163199)) |
| **0.3.3** | On General ([#163657](https://github.com/JuliaRegistries/General/pull/163657)) — CD **0.4**; CM weakdep deferred |
| **0.3.4** | On General ([#163904](https://github.com/JuliaRegistries/General/pull/163904), merged 2026-08-08) — restore `CausalMediation` weakdep + extension |
| **0.3.5** | Skipped on General (MLJ compat widen shipped in **0.3.6**) |
| **0.3.6** | On General ([#164383](https://github.com/JuliaRegistries/General/pull/164383), merged 2026-08-13) — trees + Makie MTP plot + `:nnloglik` + MLJ compat `"0.20–0.23"` |
| **0.3.7** | In history (`bb19381`, `CovariateSchema`); **not on General** (skipped; schema exports are in 0.3.8+) |
| **0.3.8–0.3.9** | Skipped on General (same pattern as 0.3.5) |
| **0.3.10** | On General ([#165016](https://github.com/JuliaRegistries/General/pull/165016), merged 2026-08-19); TagBot tagged `v0.3.10` — metalearners, discrete LMTP, sequential factor `policies`, `estimand_from_query`, nested eSL-inside-dSL |
| **0.3.29** | On General (tree `968dd5112bc0f6a06f872cb471a16cdbfd8fdcbe`); TagBot tagged `v0.3.29` at `a7cd555`; requires CausalDynamics **0.4.7+** |
| **0.3.28** | On General ([#166684](https://github.com/JuliaRegistries/General/pull/166684), merged 2026-08-31) — MSM pooled-nuisance stratification (#26) |
| **0.3.27** | On General ([#166678](https://github.com/JuliaRegistries/General/pull/166678), merged 2026-08-31) — Apodemus stress Julia 1.12 load; docs StatsModels/GraphPPL |
| **0.3.13** | On General — cluster-robust MSM `cluster=` (+ 0.3.11–0.3.12 MSM tips) |
| **0.3.18** | Count LMTP Phase A (local) |
| **0.3.17** | On General — `run_estimation_plan`, panel path tests/docs |
| **0.3.16** | On General ([#166627](https://github.com/JuliaRegistries/General/pull/166627), merged 2026-08-30) |

## 0.3.28 register steps

1. Push tip on `main`
2. `@JuliaRegistrator register` on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3)
3. General AutoMerge — done ([#166684](https://github.com/JuliaRegistries/General/pull/166684)); TagBot tagged `v0.3.28`

## 0.3.29 register steps

1. Push `main` with `version = "0.3.29"`
2. Update registration on [issue #3](https://github.com/SimonAB/CausalTargeted.jl/issues/3) after CausalDynamics **0.4.7** is on General — done; PR updated to `a7cd555`
3. General AutoMerge — complete ([#169589](https://github.com/JuliaRegistries/General/pull/169589)); TagBot tagged `v0.3.29`

Version 0.4.0 includes the inference-contract corrections, API changes and
posterior-imputation fix listed in the changelog. It requires CausalDynamics
0.5 and accepts the CausalMediation 0.2 extension.

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
