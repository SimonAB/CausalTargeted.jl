"""
Optional MLJ integration for nuisance learners.

Activated by `using MLJ, MLJLinearModels`. Provides MLJ-backed learner
implementations for `:mlj_ridge`, `:mlj_lasso`, `:mlj_elasticnet`,
`:mlj_logistic`, and the `:glmnet*` aliases inside
`CausalTargeted.fit_super_learner`.

Features are column-standardised (after dropping a leading intercept column)
before fitting, matching the package design note for MLJ candidates.
"""
module CausalTargetedMLJExt

using CausalTargeted
using DataFrames: DataFrame
using Logging: NullLogger, with_logger

using MLJ: machine, fit!, predict, categorical, classes, pdf
using MLJLinearModels

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
    mach, Xs = CausalTargeted._mlj_feature_matrix(fit, X)
    pred = predict(mach, DataFrame(Xs, :auto))
    return vec(Float64.(pred))
end

function CausalTargeted._fit_mlj_logistic(X::Matrix{Float64}, y::Vector{Float64})
    Xs, μ, σ = CausalTargeted._prepare_mlj_features(X)
    Xdf = DataFrame(Xs, :auto)
    yb = categorical(round.(Int, clamp.(y, 0.0, 1.0)))
    # Default solver handles L2; ProxGrad is only valid with L1-style penalties.
    model = MLJLinearModels.LogisticClassifier()
    mach = machine(model, Xdf, yb)
    with_logger(NullLogger()) do
        fit!(mach, verbosity = 0)
    end
    return (mach = mach, μ = μ, σ = σ)
end

function CausalTargeted._predict_mlj_logistic(fit, X::Matrix{Float64})
    mach, Xs = CausalTargeted._mlj_feature_matrix(fit, X)
    pred_dist = predict(mach, DataFrame(Xs, :auto))
    lv = collect(classes(first(pred_dist)))
    pos = 1 in lv ? 1 : lv[end]
    return clamp.(vec(Float64.(pdf.(pred_dist, pos))), 1e-6, 1 - 1e-6)
end

end # module
