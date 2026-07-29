"""Typed nuisance-model interface wrapping SuperLearner fits."""

using DataFrames
using Random

abstract type NuisanceModel end

"""
    OutcomeRegression

Cross-fitted outcome regression `Q(A, W)`.

Stores the full-sample covariate design `W` (intercept + covariates) so
predictions can swap the treatment column without rebuilding covariates.
"""
mutable struct OutcomeRegression <: NuisanceModel
    treatment::Symbol
    covariates::Vector{Symbol}
    learners::Tuple
    fold_models::Vector{SuperLearnerFit}
    fold_test_idx::Vector{Vector{Int}}
    W::Matrix{Float64}
end

"""
    ExposureDensity

Cross-fitted exposure mean model for Gaussian density ratios.

Stores the full-sample covariate design `W` for fold-sliced prediction.
"""
mutable struct ExposureDensity <: NuisanceModel
    covariates::Vector{Symbol}
    learners::Tuple
    fold_models::Vector{SuperLearnerFit}
    fold_test_idx::Vector{Vector{Int}}
    W::Matrix{Float64}
end

"""
    fit_outcome_regression(df, outcome, treatment, covariates, folds, rng; learners) -> OutcomeRegression
"""
function fit_outcome_regression(
    df::DataFrame,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng::AbstractRNG;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, treatment])
    W = covariate_design_matrix(df, covariates)
    fold_sets = crossfit_indices(n, folds, rng)
    models = SuperLearnerFit[]
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        Xtr = outcome_design_matrix(W[train_idx, :], a[train_idx])
        push!(models, fit_super_learner(Xtr, y[train_idx]; learners = learners, rng = rng))
    end
    return OutcomeRegression(treatment, covariates, Tuple(learners), models, fold_sets, W)
end

"""
    predict_outcome(model, df, treatment_values=nothing) -> Vector{Float64}

Predict using the cached covariate design from fit. `df` must have the same
number of rows as the training frame; treatment may be overridden via
`treatment_values`.
"""
function predict_outcome(
    model::OutcomeRegression,
    df::DataFrame;
    treatment_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = nrow(df)
    size(model.W, 1) == n || throw(DimensionMismatch(
        "predict_outcome expected $(size(model.W, 1)) rows (fit design), got $n",
    ))
    a = treatment_values === nothing ? Float64.(df[!, model.treatment]) : Float64.(treatment_values)
    length(a) == n || throw(DimensionMismatch(
        "treatment_values length $(length(a)) does not match $n rows",
    ))
    preds = zeros(n)
    for (sl, test_idx) in zip(model.fold_models, model.fold_test_idx)
        preds[test_idx] = predict_super_learner(
            sl,
            outcome_design_matrix(model.W[test_idx, :], a[test_idx]),
        )
    end
    return preds
end

"""
    fit_exposure_density(df, treatment, covariates, folds, rng; learners) -> ExposureDensity
"""
function fit_exposure_density(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng::AbstractRNG;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    a = Float64.(df[!, treatment])
    W = covariate_design_matrix(df, covariates)
    fold_sets = crossfit_indices(n, folds, rng)
    models = SuperLearnerFit[]
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        push!(models, fit_super_learner(W[train_idx, :], a[train_idx]; learners = learners, rng = rng))
    end
    return ExposureDensity(covariates, Tuple(learners), models, fold_sets, W)
end

"""
    predict_exposure_mean(model, df) -> Vector{Float64}
"""
function predict_exposure_mean(model::ExposureDensity, df::DataFrame)
    n = nrow(df)
    size(model.W, 1) == n || throw(DimensionMismatch(
        "predict_exposure_mean expected $(size(model.W, 1)) rows (fit design), got $n",
    ))
    preds = zeros(n)
    for (sl, test_idx) in zip(model.fold_models, model.fold_test_idx)
        preds[test_idx] = predict_super_learner(sl, model.W[test_idx, :])
    end
    return preds
end

export NuisanceModel, OutcomeRegression, ExposureDensity
export fit_outcome_regression, predict_outcome
export fit_exposure_density, predict_exposure_mean
