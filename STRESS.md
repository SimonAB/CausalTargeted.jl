# Stress validation

Stress notebook for the owned Julia causal stack, organised as a CDCS
**analysis path**: Structural → Dynamical → Observable (across Pearl L1–L3),
then audit. The distinctive claim is typed integration with certificates — see
[ECOSYSTEM_COMPARISON.md](ECOSYSTEM_COMPARISON.md).

**Quarto notebook:**
[`docs/stress/stress_validation.qmd`](docs/stress/stress_validation.qmd)

**Deep SCM estimation sibling:**
[`docs/stress/deep_scm_estimation_stress.qmd`](docs/stress/deep_scm_estimation_stress.qmd)
(codes → mediation / LMTP; mechanisms in
[CausalDynamics STRESS.md](https://github.com/SimonAB/CausalDynamics.jl/blob/main/STRESS.md)).

**Missingness grid sibling:**
[`docs/stress/missingness_grid_stress.qmd`](docs/stress/missingness_grid_stress.qmd)
(Structural certificates, Dynamical sequential/survival, Observable strategies +
posterior). Posterior-only notebook:
[`docs/stress/missingness_posterior_stress.qmd`](docs/stress/missingness_posterior_stress.qmd).

```bash
cd docs/stress && quarto render stress_validation.qmd
cd docs/stress && quarto render deep_scm_estimation_stress.qmd
cd docs/stress && quarto render missingness_grid_stress.qmd
open stress_validation.html
```

Focused CDCS harness entry (edge unit suites + smoke functionality):

```bash
julia --project=. --threads=auto scripts/stress_harness/run_missingness_validation.jl
RENDER_STRESS=1 julia --project=. --threads=auto scripts/stress_harness/run_missingness_validation.jl
```

Activates the parent CDCS book `Project.toml` when present (Turing / RxInfer).
Chunks are split so figures and tables are the last value returned.

**Documenter summary:** [stress_validation.md](docs/src/stress_validation.md) ·
[online](https://simonab.github.io/CausalTargeted.jl/dev/stress_validation/)

**Harness:** [causal-dynamics-book/scripts/stress_harness](https://github.com/SimonAB/causal-dynamics-book/tree/main/scripts/stress_harness)

Fixtures: [`docs/data/`](docs/data/).

**2026-08-21:** Deep SCM estimation sibling rendered green
(`deep_scm_estimation_stress.qmd`; codes → mediation/LMTP/missing $Y$).
Mechanisms/L3 notebook: CausalDynamics `docs/stress/deep_scm_stress.qmd`.

**2026-08-19:** Categorical-treatment LMTP, sequential factor recodes, nested
eSL-inside-dSL, and interventional factor-`A` mediation (continuous `M`) have
recovery chunks in the notebook. Remaining: mediation PPL complete-case
bootstrap, sequential/survival Monte Carlo oracles, R `lmtp` concordance (see
`scripts/synthetic_benchmark/` in the book repo).
