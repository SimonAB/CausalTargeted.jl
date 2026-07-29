"""
Optional MLJ / MLJLinearModels integration for linear SuperLearner candidates.

Activated by `using MLJ, MLJLinearModels`. Provides fit/predict helpers for
`:mlj_ridge`, `:mlj_lasso`, `:mlj_elasticnet`, and `:mlj_logistic`.
"""
module CausalTargetedMLJExt

using CausalTargeted
using DataFrames: DataFrame
using Distributions: pdf
using Logging: NullLogger, with_logger
using MLJ: machine, fit!, predict, categorical, classes
using MLJLinearModels

"""Rebuild the feature matrix used at fit time for prediction."""
function _mlj_predict_matrix(fit, X::Matrix{Float64})
    mach, μ, σ = CausalTargeted._unpack_mlj_fit(fit)
    Xc = CausalTargeted._drop_intercept_column(X)
    Xs = μ === nothing ? Xc : CausalTargeted._apply_feature_standardise(Xc, μ, σ)
    return mach, DataFrame(Xs, :auto)
end

function CausalTargeted._fit_mlj_regressor(
    name::Symbol,
    X::Matrix{Float64},
    y::Vector{Float64},
)
    Xs, μ, σ = CausalTargeted._prepare_mlj_features(X)
    Xdf = DataFrame(Xs, :auto)
    model = if name == :mlj_ridge
        MLJLinearModels.RidgeRegressor(solver = MLJLinearModels.Analytical())
    elseif name == :mlj_lasso
        MLJLinearModels.LassoRegressor(
            solver = MLJLinearModels.ProxGrad(accel = true, max_iter = 5000),
        )
    elseif name == :mlj_elasticnet
        MLJLinearModels.ElasticNetRegressor(
            solver = MLJLinearModels.ProxGrad(accel = true, max_iter = 5000),
        )
    else
        error("Unknown MLJ regressor learner: $name")
    end
    mach = machine(model, Xdf, y)
    with_logger(NullLogger()) do
        fit!(mach, verbosity = 0)
    end
    return (mach = mach, μ = μ, σ = σ)
end

function CausalTargeted._predict_mlj_regressor(fit, X::Matrix{Float64})
    mach, Xdf = _mlj_predict_matrix(fit, X)
    pred = predict(mach, Xdf)
    return vec(Float64.(pred))
end

function CausalTargeted._fit_mlj_logistic(X::Matrix{Float64}, y::Vector{Float64})
    Xs, μ, σ = CausalTargeted._prepare_mlj_features(X)
    Xdf = DataFrame(Xs, :auto)
    yb = categorical(round.(Int, clamp.(y, 0.0, 1.0)))
    model = MLJLinearModels.LogisticClassifier(
        solver = MLJLinearModels.ProxGrad(accel = true, max_iter = 5000),
    )
    mach = machine(model, Xdf, yb)
    with_logger(NullLogger()) do
        fit!(mach, verbosity = 0)
    end
    return (mach = mach, μ = μ, σ = σ)
end

function CausalTargeted._predict_mlj_logistic(fit, X::Matrix{Float64})
    mach, Xdf = _mlj_predict_matrix(fit, X)
    pred_dist = predict(mach, Xdf)
    lv = collect(classes(first(pred_dist)))
    pos = 1 in lv ? 1 : lv[end]
    return clamp.(vec(Float64.(pdf.(pred_dist, pos))), 1e-6, 1 - 1e-6)
end

end # module
