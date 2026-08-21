# Missingness grid stress

Cross-stratum ledger for incomplete observation. Narrative API catalogue:
[Missingness](missingness.md).

- **Structural** — `MissingnessSpec` / `certify_missingness` / `identify(...; missingness=)`
- **Dynamical** — sequential and survival under `handle_missing`; generative masks via CausalDynamics
- **Observable** — `:drop` / `:ipcw` / `:impute` / `:ipcw_impute` on LMTP, g-comp, mediation; opt-in `impute_posterior`

**Quarto notebook:**
[`docs/stress/missingness_grid_stress.qmd`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/stress/missingness_grid_stress.qmd)

Posterior-only sibling:
[`docs/stress/missingness_posterior_stress.qmd`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/stress/missingness_posterior_stress.qmd)

```bash
cd docs/stress && quarto render missingness_grid_stress.qmd
```

From the CDCS book repo, run the focused edge suites plus smoke harness:

```bash
julia --project=. --threads=auto scripts/stress_harness/run_missingness_validation.jl
```
