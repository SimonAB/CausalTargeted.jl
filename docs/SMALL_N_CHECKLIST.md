# Small-*n* checklist

Canonical copy lives in Documenter: [`docs/src/small_n.md`](src/small_n.md).

Continuous MTP and mediation with **tens to low hundreds** of units need different defaults
from large-cohort epi software. Theory: Díaz et al. (2023) LMTP; van der Laan & Rose (2011)
TMLE; Cinelli & Hazlett (2020) sensitivity; Petersen et al. (2012) positivity. Full citations:
[`docs/src/references.md`](src/references.md).

## Before you estimate

1. Call `recommend_run_options(n; engine, n_mediators)` and pass kwargs into the grids.
2. Prefer `SMALL_N_SL_LEARNERS` or `adaptive_learners(n)` when `n < 80`.
3. Keep `parallel=false` unless you have measured headroom.
4. Run `positivity_report` (or `positivity=true`) before interpreting TE curves.
5. For mediation, raise `n_mc` until `mediation_stability_summary` stabilises.
6. Attach `sensitivity_report` for policy-relevant claims.
7. Treat discovery graphs as sensitivity via `merge_discovery_sensitivity!`.

## Policy library

- `additive_shift_policy` · `multiplicative_shift_policy` · `threshold_shift_policy`

## Longitudinal

`SequentialPolicy` / `run_sequential_lmtp` with CausalDynamics `TemporalEffectQuery` certificates.
