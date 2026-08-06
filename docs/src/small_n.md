# Small-*n* checklist

Continuous MTP and mediation with **tens to low hundreds** of units (conservation,
ecology, early trials) need different defaults from large-cohort epi software. Theory still
comes from LMTP and targeted learning (Díaz et al., 2023; van der Laan & Rose, 2011); the
checklist below is about *finite-sample practice*. See [Methods](methods.md) and
[References](references.md).

## Before you estimate

1. Call `recommend_run_options(n; engine, n_mediators)` and pass the kwargs into
   `run_lmtp_grid` / `run_mediation_grid` / `execute_estimand`.
2. Prefer `SMALL_N_SL_LEARNERS` or `adaptive_learners(n)` over `RICH_SL_LEARNERS` when
   `n < 80` (Super Learner library must be estimable at the given *n*;
   van der Laan, Polley & Hubbard, 2007).
3. Keep `parallel=false` unless you have measured headroom (δ-grid threading can thrash
   memory on multi-core laptops).
4. Run `positivity_report` (or `positivity=true` on the grid) and inspect
   `weak_support` / `unsupported_shift` cells before interpreting TE curves
   (Petersen et al., 2012; Díaz et al., 2023 on MTP-designed positivity).
5. For mediation, run `mediation_n_mc_sweep` and `mediation_stability_summary`; raise
   `n_mc` until TE SE and signs stabilise (nested MC is part of the estimand’s
   approximation error under continuous MTP mediation; cf. Liu et al., 2024).
6. Attach `sensitivity_report` (tipping-point / partial-*R*²) for any claim that is
   policy-relevant under possible omitted confounding (Cinelli & Hazlett, 2020).
7. Treat discovery graphs as **sensitivity**, not oracles:
   `discovery_adjustment_sensitivity` / `merge_discovery_sensitivity!`
   (Pearl, 2009; Spirtes et al., 2000).

## Policy library

Policies implement scientifically interpretable MTPs (Díaz & van der Laan, 2012;
Díaz et al., 2023):

- Additive *z*-shift: `additive_shift_policy` (default)
- Multiplicative: `multiplicative_shift_policy`
- Threshold (“raise if below *c*”): `threshold_shift_policy`

## Longitudinal

Use `SequentialPolicy` / `run_sequential_lmtp` for multi-time exposures, with ID from
CausalDynamics `TemporalEffectQuery` when a lag DAG is available (Díaz et al., 2023).
For discrete-time event-free probability under MTP, use `SurvivalPolicy` /
`run_survival_lmtp` (Díaz–Hoffman–Hejazi 2024 spirit; competing risks deferred).
Point-treatment `LongitudinalPolicy` carries temporal ID metadata for single-time
pathway→AUC analyses.

## Sysimage (optional)

See
[`scripts/build_mtp_sysimage.jl`](https://github.com/SimonAB/causal-dynamics-book/blob/main/scripts/build_mtp_sysimage.jl)
in the CDCS monorepo for a PackageCompiler recipe (community convenience; not required for CI).
