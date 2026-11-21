# Discrete, hurdle, MSM & DiD

```@meta
CurrentModule = CausalTargeted
```

Discrete-treatment LMTP (Díaz–Williams classification density ratios), two-part
hurdle LMTP, repeated-outcome and parametric marginal structural models (MSMs),
mixed-model repeated measures (MMRM), and difference-in-differences (DiD).

Discrete runners default to capped classifier ratios and return point estimates.
Their ordinary EIF standard errors, limits, and influence curves are unavailable
with `inference_status=:truncated_weights`. Pass `weight_truncation=false` to
use uncapped ratios and request EIF inference for supported policies;
positivity, missingness, imputation, and targeting-status gates still apply.

## Discrete LMTP

```@docs
DiscreteTreatmentPolicy
DiscreteInterventionalMean
discrete_recode_policy
discrete_static_policy
discrete_shift_policy
apply_discrete_policy
run_discrete_lmtp
run_discrete_lmtp_contrast
discrete_positivity
bootstrap_discrete_lmtp_contrast
```

## Two-part hurdle LMTP

```@docs
TwoPartInterventionalMean
run_two_part_discrete_lmtp
run_two_part_discrete_lmtp_contrast
bootstrap_two_part_discrete_lmtp_contrast
```

## Repeated-outcome & parametric MSMs

```@docs
RepeatedOutcomeMSM
run_repeated_outcome_msm
msm_contrast
msm_stratum_contrast
identify_repeated_outcomes
unstack_repeated_outcomes
ParametricRepeatedOutcomeMSM
run_parametric_repeated_msm
simulate_mean_treatment_time_msm
```

## Mixed-model repeated measures (MMRM)

Requires `using MixedModels` to load the `CausalTargetedMixedModelsExt`
extension.

```@docs
MMRMResult
fit_mmrm
run_mmrm
```

## Difference-in-differences (DiD)

```@docs
run_did_2x2
run_did_staggered
aggregate_did
```
