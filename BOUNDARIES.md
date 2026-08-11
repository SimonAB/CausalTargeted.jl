# Package boundaries

**Design principles:** [DESIGN.md](DESIGN.md) · [shared](DESIGN_PRINCIPLES.md)

## CausalTargeted.jl (this package)

- Cross-fitted nuisances and SuperLearner stacks
- LMTP / sequential LMTP / thin survival LMTP, δ-grids, planning, parallel execution, run metadata
- Sequential certificate bridges (`plan_sequential`, `sequential_spec_from_identification`)
- Survival / event-time path (`SurvivalPolicy`, `run_survival_lmtp`; competing risks deferred)
- Domain transport weights (`domain_transport_weights`); policy choice (`choose_policy`)
- Synthetic DGPs for **package** tests
- Soft façades for mediation APIs (implementation in **CausalMediation.jl**)
- Optional Makie MTP effect-curve plotting (`plot_mtp_curve` / `mtp_curve!` via `CausalTargetedMakieExt`); visualises already-estimated grids, does not estimate

## CausalMediation.jl

- Interventional / natural / organic / controlled / recanting-twin mediation
- `moc` intermediate confounding; full continuous-MTP EIF

## CausalDynamics.jl

- Graphs, identification, `identify`, `IdentificationResult`, CDMs
- Temporal unrolling and temporal backdoor ID
- No cohort data, no R parity

## Application layers (e.g. Sheep_VaccineCDCS)

- Data merge, registry TOML, dagitty strings from manuscripts
- Concordance vs reference implementations
- Manuscript drivers

Do not add paper-specific pathway names or biological concordance here.
