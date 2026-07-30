# Package boundaries

**Design principles:** [DESIGN.md](DESIGN.md) · [shared](DESIGN_PRINCIPLES.md)

## CausalTargeted.jl (this package)

- Cross-fitted nuisances and SuperLearner stacks
- LMTP / interventional mediation EIF estimators
- δ-grids, planning, parallel execution, run metadata
- Synthetic DGPs for **package** tests

## CausalDynamics.jl

- Graphs, identification, `identify`, `IdentificationResult`, CDMs
- Temporal unrolling and temporal backdoor ID
- No cohort data, no R parity

## Application layers (e.g. Sheep_VaccineCDCS)

- Data merge, registry TOML, dagitty strings from manuscripts
- Concordance vs reference implementations
- Manuscript drivers

Do not add paper-specific pathway names or biological concordance here.
