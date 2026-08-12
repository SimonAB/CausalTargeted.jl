"""
Optional XGBoost integration through MLJ and XGBoost.jl.

Activated by `using MLJ, MLJXGBoostInterface`. The public Super Learner symbol
is `:xgboost`. A leading artificial intercept is removed and predictors are
passed to XGBoost without standardisation.
"""
module CausalTargetedMLJXGBoostExt

using CausalTargeted
using DataFrames: DataFrame
using Logging: NullLogger, with_logger
using MLJ: categorical, classes, fit!, machine, pdf, predict
using MLJXGBoostInterface

const XGBOOST_SEED = 42

"""Construct the package's deterministic XGBoost regression/classification model."""
function xgboost_model(family::Symbol)
    family in (:gaussian, :binomial) || throw(ArgumentError(
        ":xgboost supports family=:gaussian or family=:binomial, got $family",
    ))
    Model = family == :gaussian ? XGBoostRegressor : XGBoostClassifier
    return Model(
        num_round = 100,
        max_depth = 2,
        eta = 0.05,
        min_child_weight = 5.0,
        subsample = 0.8,
        colsample_bytree = 0.8,
        lambda = 1.0,
        alpha = 0.0,
        gamma = 0.0,
        nthread = 1,
        seed = XGBOOST_SEED,
    )
end

function CausalTargeted._fit_mlj_tree(
    ::Val{:xgboost},
    X::Matrix{Float64},
    y::Vector{Float64};
    family::Symbol,
)
    Xt = CausalTargeted._prepare_mlj_tree_features(X)
    if size(Xt, 2) == 0 || length(y) < 2 || all(==(first(y)), y)
        μ = family == :binomial ? clamp(CausalTargeted._safe_mean(y), 1e-6, 1 - 1e-6) :
            CausalTargeted._safe_mean(y)
        return (mach = nothing, family = family, mean = μ)
    end
    model = xgboost_model(family)
    target = family == :binomial ? categorical(round.(Int, clamp.(y, 0.0, 1.0))) : y
    mach = machine(model, DataFrame(Xt, :auto), target)
    with_logger(NullLogger()) do
        fit!(mach, verbosity = 0)
    end
    return (mach = mach, family = family, mean = nothing)
end

function CausalTargeted._predict_mlj_tree(
    ::Val{:xgboost},
    fit,
    X::Matrix{Float64},
)
    fit.mach === nothing && return fill(Float64(fit.mean), size(X, 1))
    Xt = CausalTargeted._prepare_mlj_tree_features(X)
    pred = predict(fit.mach, DataFrame(Xt, :auto))
    if fit.family == :gaussian
        return vec(Float64.(pred))
    end
    levels = collect(classes(first(pred)))
    positive = 1 in levels ? 1 : levels[end]
    return clamp.(vec(Float64.(pdf.(pred, positive))), 1e-6, 1 - 1e-6)
end

end # module
