# API overview

```@meta
CurrentModule = CausalTargeted
```

Literature mapping lives in [Methods](methods.md); DOIs in [References](references.md).
Docstrings on the symbols below also carry `# References` sections.

## Small-*n* profile

- `recommend_folds` · `recommend_run_options` · `recommend_learners`
- `SMALL_N_SL_LEARNERS` · `adaptive_learners` · `warn_if_folds_too_large`

## LMTP and policies

- `run_lmtp_grid` · `lmtp_tmle_contrast` · `ShiftPolicy`
- `additive_shift_policy` · `multiplicative_shift_policy` · `threshold_shift_policy`
- `SequentialPolicy` · `run_sequential_lmtp` · `sequential_identification_certificate`
- `build_lmtp_fold_cache`

## Mediation

- `run_mediation_grid` · `run_mediation_scalar`
- `mediation_n_mc_sweep` · `mediation_stability_summary` · `mediation_stability_markdown`
- `build_mediation_fold_cache` · `MediationFoldCache`
- Soft-deprecated aliases (emit `DeprecationWarning`): `run_crumble_grid`, `run_crumble_scalar`, `build_crumble_fold_cache`, `CrumbleFoldCache`, `run_crumble_scalar_ppl`
- Prefer: `run_mediation_grid` · `run_mediation_scalar` · `run_mediation_scalar_ppl` · `MediationFoldCache`

## Positivity and sensitivity

- `positivity_report` · `positivity_markdown` · `attach_positivity_summary!`
- `tipping_point_bias` · `partial_r2_calibration` · `sensitivity_report` · `sensitivity_markdown`
- `adjustment_set_disagreement` · `discovery_adjustment_sensitivity` · `merge_discovery_sensitivity!`

## Planning and certificates

- `plan_mtp` · `summarise_plan` · `execute_estimand`
- `identification_certificate` · `certificate_dict`
- `build_run_metadata` · `attach_run_metadata!`

```@docs
CausalTargeted
recommend_run_options
run_lmtp_grid
run_mediation_grid
positivity_report
sensitivity_report
SequentialPolicy
```
