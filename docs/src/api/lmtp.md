# LMTP, sequential & survival

```@meta
CurrentModule = CausalTargeted
```

Continuous, sequential, and survival longitudinal modified treatment policy
(LMTP) estimators implementing Díaz & van der Laan (2012) shifts and Díaz et al.
(2023) LMTP variants (see [Methods and estimators](../methods.md)).

Continuous `lmtp_tmle_contrast` uses Gaussian density ratios for uncapped
influence-curve inference with constant additive, smooth bounded logit,
feasible additive, and proportional headroom shifts.

For fixed SD shifts on an already standardised exposure, use
`feasible_additive_shift_policy(L, U; sd_scale=1)` as the `policy` argument
to `run_lmtp_grid` with `weight_truncation=false`. On the original scale,
supply a prespecified SD in the exposure's units. The grid reports
`effective_shift` in the exposure's units. Bounds determine eligibility
for the shift. Inference still relies on overlap conditional on covariates.

For a deterministic, raw-scale, range-adaptive contraction without boundary
clamping, use `proportional_headroom_policy(L, U)` with grid values `delta` in
`(-1, 1)` and `weight_truncation=false`. Each exposure in `[L, U]` moves by a
fraction `delta` of its remaining headroom toward `U` (when `delta >= 0`) or
`abs(delta)` toward `L` (when `delta < 0`); exposures outside `[L, U]` remain
unchanged. The policy induces a single-branch Gaussian density ratio with
Jacobian ``1 / (1 - |\delta|)`` on its image.

For a stochastic exponential-tilting intervention that remains well-conditioned
under weak overlap without ad-hoc weight truncation, use
`exponential_tilt_policy(; bound = 2.0, center = 0.0)` with
`estimate_exponential_tilt` or as the `policy` argument to `run_lmtp_grid`.
Setting `bound = M > 0` defines the smooth tilt score
``\phi(a) = M \tanh((a - c) / M)``, guaranteeing density-ratio bounds
``\exp(-2|\delta|M) \le H_\delta(A, W) \le \exp(2|\delta|M)`` almost surely.
Normalisers ``\log m_0(W)`` are fitted by strictly convex calibration balancing,
outcome targeting reuses canonical-link TMLE, and dual-axis EIF inference
reports both the outcome contrast ``\tau(\delta)`` and the realised exposure
shift ``\Delta(\delta)`` with Wald and studentised multiplier bootstrap
intervals (`studentised_multiplier_ci`).

## Continuous LMTP grids, contrasts & core TMLE

```@docs
run_lmtp_grid
run_lmtp_contrast
lmtp_tmle_contrast
lmtp_tmle_from_components
lmtp_tmle_policy_mean
estimate_exponential_tilt
apply_shift_policy
```

## Shift policies, bounds & clamping diagnostics

```@docs
ShiftPolicy
shift_policy_from_settings
additive_shift_policy
multiplicative_shift_policy
threshold_shift_policy
SmoothBoundedLogitShiftPolicy
smooth_bounded_logit_shift_policy
FeasibleAdditiveShiftPolicy
feasible_additive_shift_policy
ProportionalHeadroomPolicy
proportional_headroom_policy
ExponentialTiltPolicy
exponential_tilt_policy
tilt_score
tilt_weight_bounds
apply_policy_values
exposure_bounds
clamp_exposure
shifted_exposure
requested_shift
clamp_diagnostics
additive_clamp_diagnostics
targeting_weight_from_clamp
```

## Fold nuisance cache, targeting & inference

```@docs
LMTPFoldCache
build_lmtp_fold_cache
lmtp_components_from_cache
tmle_score_diagnostics
tmle_fluctuation_objective
optimise_tmle_fluctuation
diagnostics_dict
prepare_clever_covariate
truncate_weights
stabilize_weights
cv_select_truncation
wald_ci
studentised_multiplier_ci
simultaneous_wald_bands
multiplier_simultaneous_critical
robust_residual_sd
support_diagnostics
cluster_robust_variance
resolve_cluster_ids
```

## Sequential & survival LMTP

```@docs
SequentialPolicy
run_sequential_lmtp
sequential_identification_certificate
SurvivalPolicy
run_survival_lmtp
survival_identification_certificate
```
