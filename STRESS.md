# Stress validation

Stress notebook for the owned Julia causal stack, organised as a CDCS
**analysis path**: Structural → Dynamical → Observable (across Pearl L1–L3),
then audit. The distinctive claim is typed integration with certificates — see
[ECOSYSTEM_COMPARISON.md](ECOSYSTEM_COMPARISON.md).

**Quarto notebook:**
[`docs/stress/stress_validation.qmd`](docs/stress/stress_validation.qmd)

```bash
cd docs/stress && quarto render stress_validation.qmd
open stress_validation.html
```

Activates the parent CDCS book `Project.toml` when present (Turing / RxInfer).
Chunks are split so figures and tables are the last value returned.

**Documenter summary:** [stress_validation.md](docs/src/stress_validation.md) ·
[online](https://simonab.github.io/CausalTargeted.jl/dev/stress_validation/)

**Harness:** [causal-dynamics-book/scripts/stress_harness](https://github.com/SimonAB/causal-dynamics-book/tree/main/scripts/stress_harness)

Fixtures: [`docs/data/`](docs/data/).

**2026-08-14:** IPCW weights enter LMTP/g-comp summaries; sequential/survival
accept `handle_missing`; `:mean`-only contrast libraries are rejected;
mediation fold caches reuse a fitted covariate schema. Remaining: mediation PPL
complete-case bootstrap, sequential/survival Monte Carlo oracles.
