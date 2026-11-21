# Missingness & diagnostics

```@meta
CurrentModule = CausalTargeted
```

Missing-data handling (complete-case, IPCW, mean imputation, and posterior
multiple imputation with Rubin pooling), domain transportability weights,
positivity diagnostics, and sensitivity calibration.

Unexported diagnostic helpers for hurdle conditional-independence tests
(`CausalTargeted.test_implied_hurdle_independences`,
`CausalTargeted.local_markov_statements`, `CausalTargeted.IndependenceStatement`,
`CausalTargeted.default_hurdle_label_to_col`) and discovery adjustment-set
sensitivity (`CausalTargeted.adjustment_set_disagreement`,
`CausalTargeted.discovery_adjustment_sensitivity`,
`CausalTargeted.merge_discovery_sensitivity!`) are also available qualified.

## Missingness & IPCW

```@docs
MissingDataResult
MISSINGNESS_META_KEY
handle_missing_data
impute_covariates_mean!
complete_numeric_column
ipcw_weights
weighted_influence_summary
attach_missingness_metadata!
missingness_metadata
with_missingness
mar_set
```

## Posterior imputation & pooling

```@docs
ImputationDraws
impute_posterior
pool_lmtp_grids
```

## Domain transportability

```@docs
domain_transport_weights
transport_weighted_mean
```

## Positivity & sensitivity analysis

```@docs
positivity_report
positivity_markdown
attach_positivity_summary!
tipping_point_bias
partial_r2_calibration
sensitivity_report
sensitivity_markdown
```
