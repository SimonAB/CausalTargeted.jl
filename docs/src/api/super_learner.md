# Super Learner & schemas

```@meta
CurrentModule = CausalTargeted
```

Super Learner ensemble fitting, candidate libraries, small-*n* profile helpers,
cross-fitted nuisance models, fold-stable covariate schemas, and typed estimand
structures.

Categorical adjustment variables are handled automatically across g-computation,
LMTP (Gaussian, classification, and hybrid density ratios), LMTP fold-cache,
sequential LMTP, survival LMTP, repeated-outcome MSM, and missing-data nuisance
pathways. Numeric, `Bool`, `String`, and explicit categorical-array columns may
be passed without manual dummy coding. A stable internal numeric representation
is fitted on the cleaned analysis data and reused across folds. Integer columns
are treated as numeric unless explicitly converted to categorical.

Metalearners include `:nnls` (default for Gaussian), `:nnloglik` (default for
binomial), `:cv_selector` (discrete Super Learner; alias `:winner`), and
`:invmse` (`:discrete` is a deprecated alias for `:nnls`). Opt-in tree learners
`:randomforest` and `:xgboost` require loading `MLJDecisionTreeInterface` and
`MLJXGBoostInterface` (see [Methods and estimators](../methods.md)).

## Super Learner & candidate libraries

```@docs
SuperLearnerFit
fit_super_learner
predict_super_learner
NestedSLCandidate
nested_sl_candidate
DEFAULT_SL_LEARNERS
RICH_SL_LEARNERS
SMALL_N_SL_LEARNERS
COUNT_SL_LEARNERS
SMALL_COUNT_SL_LEARNERS
adaptive_learners
validate_contrast_learners
validate_family_outcome
suggest_family_outcome
```

## Small-*n* profile & cross-fitting

```@docs
recommend_folds
recommend_learners
recommend_count_learners
recommend_run_options
warn_if_folds_too_large
crossfit_indices
make_analysis_strata
get_target_strata
```

## Nuisance models & cross-fitted helpers

```@docs
NuisanceModel
OutcomeRegression
ExposureDensity
fit_outcome_regression
predict_outcome
fit_exposure_density
predict_exposure_mean
crossfit_outcome_predictions
crossfit_predict_outcome
crossfit_propensity
crossfit_treatment_mean
sparse_exposure_diagnostic
```

## Covariate schemas & design matrices

```@docs
CovariateSchema
fit_covariate_schema
transform_covariates
design_matrix
covariate_design_matrix
outcome_design_matrix
columns_present
```

## Estimand & engine types

```@docs
Estimand
InterventionalMean
LongitudinalPolicy
CausalTargeted.MediationContrast
estimand_engine
estimand_from_query
estimand_from_pathway_task
longitudinal_estimand
normalize_engine
```
