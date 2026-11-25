# LMTP, sequential & survival

```@meta
CurrentModule = CausalTargeted
```

Continuous, sequential, and survival longitudinal modified treatment policy
(LMTP) estimators implementing Díaz & van der Laan (2012) shifts and Díaz et al.
(2023) LMTP variants (see [Methods and estimators](../methods.md)).

Continuous `lmtp_tmle_contrast` uses Gaussian density ratios for uncapped
influence-curve inference with constant additive, smooth bounded logit and
feasible additive shifts.

For fixed SD shifts on an already standardised exposure, use
`feasible_additive_shift_policy(L, U; sd_scale=1)` as the `policy` argument
to `run_lmtp_grid` with `weight_truncation=false`. On the original scale,
supply a prespecified SD in the exposure's units. The grid reports
`effective_shift` in the exposure's units. Bounds determine eligibility
for the shift. Inference still relies on overlap conditional on covariates.

## Continuous LMTP grids, contrasts & core TMLE

```@docs
run_lmtp_grid
run_lmtp_contrast
lmtp_tmle_contrast
lmtp_tmle_from_components
lmtp_tmle_policy_mean
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
