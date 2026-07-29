"""
MLJ-backed nuisance learner helpers (linear models).

Always available when CausalTargeted loads (`MLJ` / `MLJLinearModels` are hard
dependencies). Neural learners live in the optional `CausalTargetedMLJFluxExt`
extension (`using MLJFlux`).

All MLJ fits drop a leading intercept column of ones (MLJ models fit their own
intercept), then column-standardise remaining features. This avoids proximal-GD
failures on unscaled or double-intercept designs.
"""

using DataFrames: DataFrame
using Distributions: pdf
using Logging: Logging, NullLogger, with_logger
using Statistics: mean, std

using MLJ: machine, fit!, predict
using MLJ: categorical
using MLJLinearModels

"""
    _drop_intercept_column(X) -> Matrix{Float64}

Remove a leading column of ones if present (design-matrix intercept).
"""
function _drop_intercept_column(X::Matrix{Float64})
    if size(X, 2) >= 1 && all(abs.(view(X, :, 1) .- 1) .< 1e-12)
        return X[:, 2:end]
    end
    return X
end

"""
    _standardise_features(X) -> (Xs, μ, σ)

Column-wise z-score. Constant columns are left unchanged (`μ ← 0`, `σ ← 1`).
"""
function _standardise_features(X::Matrix{Float64})
    n, p = size(X)
    μ = vec(mean(X; dims = 1))
    σ = vec(std(X; dims = 1))
    Xs = Matrix{Float64}(undef, n, p)
    for j in 1:p
        if isfinite(σ[j]) && σ[j] > 1e-8
            Xs[:, j] .= (view(X, :, j) .- μ[j]) ./ σ[j]
        else
            μ[j] = 0.0
            σ[j] = 1.0
            Xs[:, j] .= view(X, :, j)
        end
    end
    return Xs, μ, σ
end

"""
    _apply_feature_standardise(X, μ, σ) -> Matrix{Float64}

Apply a previously fitted column standardisation.
"""
function _apply_feature_standardise(X::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    return (X .- μ') ./ σ'
end

"""
    _prepare_mlj_features(X) -> (Xs, μ, σ)

Drop intercept, then standardise remaining columns.
"""
function _prepare_mlj_features(X::Matrix{Float64})
    Xc = _drop_intercept_column(X)
    size(Xc, 2) == 0 && return ones(size(X, 1), 1), [0.0], [1.0]
    return _standardise_features(Xc)
end

"""
    _unpack_mlj_fit(fit) -> (mach, μ, σ)

Accept either a standardised fit NamedTuple or a bare MLJ machine (legacy).
"""
function _unpack_mlj_fit(fit)
    if fit isa NamedTuple && haskey(fit, :mach)
        return fit.mach, fit.μ, fit.σ
    end
    return fit, nothing, nothing
end

"""
    _mlj_predict_matrix(fit, X) -> DataFrame

Rebuild the feature matrix used at fit time for prediction.
"""
function _mlj_predict_matrix(fit, X::Matrix{Float64})
    mach, μ, σ = _unpack_mlj_fit(fit)
    Xc = _drop_intercept_column(X)
    Xs = μ === nothing ? Xc : _apply_feature_standardise(Xc, μ, σ)
    return mach, DataFrame(Xs, :auto)
end

function _fit_mlj_regressor(
    name::Symbol,
    X::Matrix{Float64},
    y::Vector{Float64},
)
    Xs, μ, σ = _prepare_mlj_features(X)
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

function _predict_mlj_regressor(fit, X::Matrix{Float64})
    mach, Xdf = _mlj_predict_matrix(fit, X)
    pred = predict(mach, Xdf)
    return vec(Float64.(pred))
end

function _fit_mlj_logistic(X::Matrix{Float64}, y::Vector{Float64})
    Xs, μ, σ = _prepare_mlj_features(X)
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

function _predict_mlj_logistic(fit, X::Matrix{Float64})
    mach, Xdf = _mlj_predict_matrix(fit, X)
    pred_dist = predict(mach, Xdf)
    levels = collect(MLJ.classes(mach))
    pos = 1 in levels ? 1 : levels[end]
    return clamp.(vec(Float64.(pdf.(pred_dist, pos))), 1e-6, 1 - 1e-6)
end
