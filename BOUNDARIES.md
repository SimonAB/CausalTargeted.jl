# Package boundaries

**Design principles:** [DESIGN.md](DESIGN.md) · [shared](DESIGN_PRINCIPLES.md)

## CausalTargeted.jl (this package)

- Cross-fitted nuisances and SuperLearner stacks
- LMTP / sequential LMTP / thin survival LMTP, δ-grids, planning, parallel execution, run metadata
- Sequential factor recodes via `SequentialPolicy.policies` (`DiscreteTreatmentPolicy`); mixed continuous/discrete `A_t` rejected
- Sequential certificate bridges (`plan_sequential`, `sequential_spec_from_identification`)
- Survival / event-time path (`SurvivalPolicy`, `run_survival_lmtp`; competing risks and factor `A_t` recodes deferred)
- Domain transport weights (`domain_transport_weights`); policy choice (`choose_policy`)
- Synthetic DGPs for **package** tests
- Soft façades for mediation APIs (implementation in **CausalMediation.jl**)
- Optional Makie MTP effect-curve plotting (`plot_mtp_curve` / `mtp_curve!` via `CausalTargetedMakieExt`); visualises already-estimated grids, does not estimate

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

## Out of scope (for now)

- Mixed continuous/discrete sequential `A_t` (rejected, not a planned path)
- Survival-time factor recodes (`SurvivalPolicy` + `policies`)
- Nested discrete Super Learner as the LMTP classifier default (opt-in `nested_sl_candidate` only)
- Automatic MNAR identification; posterior causal imputation (planned, opt-in)

Do not add paper-specific pathway names or biological concordance here.
