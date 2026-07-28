"""Typed nuisance-model interface wrapping SuperLearner fits."""

using DataFrames
using Random

abstract type NuisanceModel end

"""
    OutcomeRegression

Cross-fitted outcome regression `Q(A, W)`.
"""
mutable struct OutcomeRegression <: NuisanceModel
    treatment::Symbol
    covariates::Vector{Symbol}
    learners::Tuple
    fold_models::Vector{Any}
    fold_test_idx::Vector{Vector{Int}}
end

"""
    ExposureDensity

Cross-fitted exposure mean model for Gaussian density ratios.
"""
mutable struct ExposureDensity <: NuisanceModel
    covariates::Vector{Symbol}
    learners::Tuple
    fold_models::Vector{Any}
    fold_test_idx::Vector{Vector{Int}}
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
    fold_sets = crossfit_indices(n, folds, rng)
    models = Any[]
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        Xtr = design_matrix(train, covariates; treatment = treatment)
        push!(models, fit_super_learner(Xtr, y[train_idx]; learners = learners, rng = rng))
    end
    return OutcomeRegression(treatment, covariates, learners, models, fold_sets)
end

"""
    predict_outcome(model, df, treatment_values=nothing) -> Vector{Float64}
"""
function predict_outcome(
    model::OutcomeRegression,
    df::DataFrame;
    treatment_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = nrow(df)
    preds = zeros(n)
    for (sl, test_idx) in zip(model.fold_models, model.fold_test_idx)
        test = df[test_idx, :]
        tv = treatment_values === nothing ? nothing : treatment_values[test_idx]
        preds[test_idx] = predict_super_learner(
            sl,
            design_matrix(test, model.covariates; treatment = model.treatment, treatment_values = tv),
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
    fold_sets = crossfit_indices(n, folds, rng)
    models = Any[]
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        Xtr = design_matrix(train, covariates)
        push!(models, fit_super_learner(Xtr, a[train_idx]; learners = learners, rng = rng))
    end
    return ExposureDensity(covariates, learners, models, fold_sets)
end

"""
    predict_exposure_mean(model, df) -> Vector{Float64}
"""
function predict_exposure_mean(model::ExposureDensity, df::DataFrame)
    n = nrow(df)
    preds = zeros(n)
    for (sl, test_idx) in zip(model.fold_models, model.fold_test_idx)
        test = df[test_idx, :]
        preds[test_idx] = predict_super_learner(sl, design_matrix(test, model.covariates))
    end
    return preds
end

export NuisanceModel, OutcomeRegression, ExposureDensity
export fit_outcome_regression, predict_outcome
export fit_exposure_density, predict_exposure_mean
