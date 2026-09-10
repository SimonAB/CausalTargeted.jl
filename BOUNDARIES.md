# Package boundaries

**Design principles:** [DESIGN.md](DESIGN.md) · [shared](DESIGN_PRINCIPLES.md)

## CausalTargeted.jl (this package)

- Cross-fitted nuisances and SuperLearner stacks
- LMTP / sequential LMTP / thin survival LMTP, δ-grids, planning, parallel execution, run metadata
- **Repeated-outcome MSM** (`RepeatedOutcomeMSM` / `run_repeated_outcome_msm`):
  binary point treatment, several outcomes on the same units, joint IF
  covariance for unstructured ``τ(t)``
- **Parametric treatment×time MSM** (`ParametricRepeatedOutcomeMSM` /
  `run_parametric_repeated_msm`): GLS projection onto `:constant`,
  `:linear_time`, `:factor_time`, custom ``τ`` designs, or
  `:mean_treatment_time` (`target=:mean`)
- Sequential factor recodes via `SequentialPolicy.policies` (`DiscreteTreatmentPolicy`); mixed continuous/discrete `A_t` rejected
- Sequential certificate bridges (`plan_sequential`, `sequential_spec_from_identification`)
- Survival / event-time path (`SurvivalPolicy`, `run_survival_lmtp`; competing risks and factor `A_t` recodes deferred)
- Domain transport weights (`domain_transport_weights`); policy choice (`choose_policy`)
- Synthetic DGPs for **package** tests
- Soft façades for mediation APIs (implementation in **CausalMediation.jl**)
- Optional Makie MTP effect-curve plotting (`plot_mtp_curve` / `mtp_curve!` via `CausalTargetedMakieExt`); visualises already-estimated grids, does not estimate
- **Cluster-aware estimation:** `cluster=` on `run_repeated_outcome_msm` /
  `run_parametric_repeated_msm` for cluster-robust IF sandwich; consume
  CausalDynamics hierarchical DGPs (`RandomEffectSpec` / `:cluster` column).
  Sampling hierarchy ≠ LMM estimand
- **Optional MixedModels / profiled-NB2 backend** (weakdep extension
  `CausalTargetedMixedModelsExt`): `fit_profiled_nb2`, `mixed_g_computation`
  for static-treatment repeated outcomes. Parametric reference path; not the
  default for LMTP or MSM
- **Gaussian MMRM reference** (`fit_mmrm`, `run_mmrm`, `MMRMResult`): static
  treatment × visit Gaussian LMM; see [#25](https://github.com/SimonAB/CausalTargeted.jl/issues/25)
- **Outcome family plumbing** (`family_outcome` on LMTP runners;
  `suggest_family_outcome`, `validate_family_outcome`): binomial presence nuisances ([#34](https://github.com/SimonAB/CausalTargeted.jl/issues/34));
  Poisson / NB count nuisances and log-link TMLE on marginal mean ([#36](https://github.com/SimonAB/CausalTargeted.jl/issues/36))
- **Two-part hurdle LMTP** (`TwoPartInterventionalMean`,
  `run_two_part_discrete_lmtp`, `run_two_part_discrete_lmtp_contrast`): binomial
  presence + conditional Gaussian intensity ([#35](https://github.com/SimonAB/CausalTargeted.jl/issues/35))
- **Hurdle-aware CI testing** (`test_implied_hurdle_independences`,
  `local_markov_statements`): faithfulness checks for zero-inflated nodes ([#37](https://github.com/SimonAB/CausalTargeted.jl/issues/37))
- **Parametric formula g-computation** (`fit_parametric_gcomp`, `gcomp_mean`,
  `gcomp_contrast`, `gcomp_interaction`, `run_parametric_gcomp`): complete-case
  StatsModels GLM standardisation for categorical interventions and interactions
  ([#45](https://github.com/SimonAB/CausalTargeted.jl/pull/45)). Reference path
  beside cross-fitted `run_gcomp` / LMTP; not the default engine and not wired
  through `execute_estimand`

## CausalMediation.jl

- Interventional / natural / organic / controlled / recanting-twin mediation
- Numeric MTP and factor-`A` recodes (continuous `M`); `moc` on the numeric path
- `moc` intermediate confounding; full continuous-MTP EIF

## CausalDynamics.jl

- Graphs, identification, `identify`, `IdentificationResult`, CDMs
- Temporal unrolling and temporal backdoor ID (support-agnostic: factor vs continuous `A` is an estimation-policy choice in CausalTargeted)
- Generative duals: `DoSequence` / `Policy`, not `DiscreteTreatmentPolicy`
- No cohort data, no R parity

## Application layers (e.g. Sheep_VaccineCDCS)

- Data merge, registry TOML, dagitty strings from manuscripts
- Concordance vs reference implementations
- Manuscript drivers

## Missingness (Observable policies)

- Own cheap `handle_missing_data` strategies (`:drop`, `:ipcw`, `:impute`,
  `:ipcw_impute`) and record strategy / miss rates / optional PCH rung in
  result metadata
- Treat missingness as **stratum × rung**: Structural claims ($R$, MAR/MNAR)
  come from CausalDynamics certificates; Dynamical sequential/survival gaps
  are not the same object as static MAR-$Y$; this package implements the
  numerical Observable policy for the chosen estimand
- Survival *censoring* IPCW ≠ MAR missing terminal $S_T$
- Do not silently coerce `Missing` to `Float64`; call a documented strategy first
- Opt-in **posterior** path: `impute_posterior` (Gaussian MAR nested MC) and
  `run_lmtp_grid(...; imputation=draws)` with Rubin pooling. Not a default.
  Turing / RxInfer imputation backends deferred

## Out of scope (for now)

- Random-effects Super Learner as LMTP default (opt-in candidates may come later)
- Silent replacement of MSM ``τ(t)`` by BLUP partial pooling
- Mixed continuous/discrete sequential `A_t` (rejected, not a planned path)
- Survival-time factor recodes (`SurvivalPolicy` + `policies`)
- Nested discrete Super Learner as the LMTP classifier default (opt-in `nested_sl_candidate` only)
- Automatic MNAR identification; Turing / RxInfer posterior imputation backends
  (Gaussian MAR nested MC ships as opt-in `impute_posterior`)

Do not add paper-specific pathway names or biological concordance here.
