# Small-*n* checklist

```@meta
CurrentModule = CausalTargeted
```

Continuous MTP and mediation with **tens to low hundreds** of units (conservation,
ecology, early trials) need different defaults from large-cohort epi software. Theory still
comes from LMTP and targeted learning (Díaz et al., 2023; van der Laan & Rose, 2011); the
checklist below is about *finite-sample practice*. See [Methods](methods.md) and
[References](references.md).

## Before you estimate

1. Call `recommend_run_options(n; engine, n_mediators)` and pass the kwargs into
   `run_lmtp_grid` / `execute_estimand` (or into `CausalMediation.run_mediation`
   for mediation; `run_mediation_grid` is a compatibility façade delegating to
   `CausalMediation.jl`).
2. Prefer `SMALL_N_SL_LEARNERS` or `adaptive_learners(n)` over `RICH_SL_LEARNERS` when
   `n < 80` (Super Learner library must be estimable at the given *n*;
   van der Laan, Polley & Hubbard, 2007).
3. Keep `parallel=false` unless you have measured headroom (δ-grid threading can thrash
   memory on multi-core laptops).
4. Run `positivity_report` (or `positivity=true` on the grid) and inspect
   `weak_support` / `unsupported_shift` cells before interpreting TE curves
   (Petersen et al., 2012; Díaz et al., 2023 on MTP-designed positivity).
5. Avoid hard quantile clamping when you need EIF confidence intervals: clamping creates boundary
   point masses and gates ordinary EIF limits (`inference_status = :unsupported_clamped_policy`),
   and finite density-ratio caps likewise report unavailable EIF limits (`:truncated_weights`).
   For uncapped EIF inference on continuous exposures, choose one of the four regular policies
   (see [Continuous shift & tilting](continuous_policies.md)):
   `feasible_additive_shift_policy`, `proportional_headroom_policy`, or
   `smooth_bounded_logit_shift_policy` with `weight_truncation = false` and
   `density_ratio = :gaussian`, or—**preferred under tight conditional overlap at small *n***—
   bounded exponential tilting via `exponential_tilt_policy(; bound = 2.0)` (which guarantees exact
   density-ratio bounds ``e^{-2|\delta|M} \le H_i \le e^{2|\delta|M}`` without ad-hoc truncation
   and maintains near-nominal 95% coverage down to ``n = 30``). For supported finite-support discrete
   policies, `weight_truncation = false` likewise requests the uncapped EIF route. The small-*n*
   profile never selects a truncation cap by minimising empirical variance.
6. For mediation (owned by `CausalMediation.jl`, with compatibility façades in
   `CausalTargeted.jl`), run `mediation_n_mc_sweep` and
   `mediation_stability_summary`; raise `n_mc` until TE SE and signs stabilise
   (nested MC is part of the estimand’s approximation error under continuous
   MTP mediation; cf. Liu et al., 2024).
7. Attach `sensitivity_report` (tipping-point / partial-*R*²) for any claim that is
   policy-relevant under possible omitted confounding (Cinelli & Hazlett, 2020).
8. Treat discovery graphs as **sensitivity**, not oracles:
   `discovery_adjustment_sensitivity` / `merge_discovery_sensitivity!`
   (Pearl, 2009; Spirtes et al., 2000).

## Policy library

Continuous-exposure policies implement scientifically interpretable modified treatment policies
and stochastic interventions (Díaz & van der Laan, 2012; Díaz & Hejazi, 2020; Díaz et al., 2023;
Schindl, Shen & Kennedy, 2024; see [Continuous shift & tilting](continuous_policies.md) for a
full comparison):

**Regular continuous policies (uncapped EIF inference available):**

- **Bounded exponential tilting** (recommended under weak conditional overlap and small ``n``):
  `exponential_tilt_policy(; bound = 2.0, center = 0.0)` with `estimate_exponential_tilt` or
  `run_lmtp_grid` (linear tilting via `bound = nothing` is also supported)
- **Feasible additive shift** (prespecified bounds ``[L, U]`` and `sd_scale`):
  `feasible_additive_shift_policy(L, U; sd_scale)` with `weight_truncation = false`
- **Proportional headroom contraction** (raw-scale fraction ``\delta \in (-1, 1)`` toward ``U`` or ``L``):
  `proportional_headroom_policy(L, U)` with `weight_truncation = false`
- **Smooth bounded logit shift** (strictly interior ``(L, U)``, logit-scale shift):
  `smooth_bounded_logit_shift_policy(L, U)` with `weight_truncation = false`

**Exploratory quantile-clamped shifts (point estimates only; EIF intervals gated):**

- Additive *z*-shift: `additive_shift_policy`
- Multiplicative: `multiplicative_shift_policy`
- Threshold (“raise if below median”): `threshold_shift_policy`

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
