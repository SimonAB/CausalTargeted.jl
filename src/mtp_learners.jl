"""SuperLearner-style nuisance models for MTP estimators.

Uses discrete SuperLearner (cross-validated nonnegative weights) by default,
matching the spirit of R SuperLearner / nnls metalearning.
"""

using DataFrames
using Statistics
using GLM
using GLMNet
using EvoTrees
using Distributions
using Random
using StableRNGs

const DEFAULT_SL_LEARNERS = (:glm, :glm_interact, :glmnet, :evotree, :mean)
const RICH_SL_LEARNERS = (:glm, :glm_interact, :glm_quad, :glmnet, :glmnet_lasso, :glmnet_ridge, :evotree, :evotree_deep, :mean)

"""
    design_matrix(df, covariates; treatment=nothing, treatment_values=nothing) -> Matrix{Float64}

Build a model matrix with intercept, optional treatment column, and covariates.
"""
function design_matrix(
    df::DataFrame,
    covariates::Vector{Symbol};
    treatment::Union{Symbol, Nothing} = nothing,
    treatment_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = nrow(df)
    parts = Vector{Vector{Float64}}()
    push!(parts, ones(n))
    if treatment !== nothing
        tv = treatment_values === nothing ? Float64.(df[!, treatment]) : Float64.(treatment_values)
        push!(parts, tv)
    end
    for c in covariates
        hasproperty(df, c) || continue
        push!(parts, Float64.(df[!, c]))
    end
    return hcat(parts...)
end

"""
    sparse_exposure_diagnostic(a; modal_prop=0.5, min_nonmodal=20) -> NamedTuple
"""
function sparse_exposure_diagnostic(
    a::AbstractVector{<:Real};
    modal_prop::Real = 0.5,
    min_nonmodal::Int = 20,
)
    x = collect(skipmissing(Float64.(a)))
    n = length(x)
    isempty(x) && return (sparse = false, modal_prop = 0.0, modal_count = 0, n = n, n_unique = 0)
    counts = Dict{Float64, Int}()
    for v in x
        counts[v] = get(counts, v, 0) + 1
    end
    modal_count = maximum(values(counts))
    prop = modal_count / n
    nonmodal = n - modal_count
    return (
        sparse = prop >= modal_prop && nonmodal <= min_nonmodal,
        modal_prop = prop,
        modal_count = modal_count,
        n = n,
        n_unique = length(counts),
    )
end

function _safe_mean(y::AbstractVector{<:Real})
    xv = collect(skipmissing(Float64.(y)))
    isempty(xv) && return 0.0
    return mean(xv)
end

"""
    _fit_mlj_regressor(name, X, y)

Internal hook point for MLJ-backed nuisance regression learners.

The default implementation throws; the MLJ implementation is provided by an
optional MLJ integration (via `Requires.jl`).
"""
function _fit_mlj_regressor(
    name::Symbol,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
)
    error("Requested MLJ learner $name, but MLJ integration is not available. Ensure MLJ is installed and loaded.")
end

"""
    _predict_mlj_regressor(mach, X)

Internal hook point for MLJ-backed nuisance regression predictions.
"""
function _predict_mlj_regressor(::Nothing, X::Matrix{Float64})
    error("MLJ regression prediction requested, but MLJ integration is not available.")
end

"""
    _fit_mlj_logistic(X, y)

Internal hook point for MLJ-backed binomial logistic nuisance learners.
"""
function _fit_mlj_logistic(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    error("Requested MLJ binomial learner, but MLJ integration is not available. Ensure MLJ is installed and loaded.")
end

"""
    _predict_mlj_logistic(mach, X)

Internal hook point for MLJ-backed binomial logistic predictions.
"""
function _predict_mlj_logistic(::Nothing, X::Matrix{Float64})
    error("MLJ logistic prediction requested, but MLJ integration is not available.")
end

"""
    _mljflux_ext()

Return the optional `CausalTargetedMLJFluxExt` module, or `nothing` if MLJFlux
is not loaded.
"""
function _mljflux_ext()
    return Base.get_extension(@__MODULE__, :CausalTargetedMLJFluxExt)
end

"""
    _fit_mlj_mlp(X, y)

Tiny MLP regressor via optional MLJFlux extension (`:mlj_mlp`).
Features are column-standardised before fitting.
"""
function _fit_mlj_mlp(X::Matrix{Float64}, y::Vector{Float64})
    ext = _mljflux_ext()
    ext === nothing && error(
        "Requested :mlj_mlp but MLJFlux is not available. " *
        "Install/load MLJFlux (`using MLJFlux`) to enable neural nuisance learners.",
    )
    Xs, μ, σ = _prepare_mlj_features(X)
    mach = ext.fit_mlp(Xs, y)
    return (mach = mach, μ = μ, σ = σ)
end

"""
    _predict_mlj_mlp(fit, X)

Predict with a fitted `:mlj_mlp` machine (applies train-time standardisation).
"""
function _predict_mlj_mlp(fit, X::Matrix{Float64})
    ext = _mljflux_ext()
    ext === nothing && error("Requested :mlj_mlp prediction but MLJFlux extension is not loaded.")
    mach, μ, σ = _unpack_mlj_fit(fit)
    Xc = _drop_intercept_column(X)
    Xs = μ === nothing ? Xc : _apply_feature_standardise(Xc, μ, σ)
    return ext.predict_mlp(mach, Xs)
end

"""
    _fit_mlj_nn_binary(X, y)

Tiny binary neural classifier via optional MLJFlux extension (`:mlj_nn_binary`).
Features are column-standardised before fitting.
"""
function _fit_mlj_nn_binary(X::Matrix{Float64}, y::Vector{Float64})
    ext = _mljflux_ext()
    ext === nothing && error(
        "Requested :mlj_nn_binary but MLJFlux is not available. " *
        "Install/load MLJFlux (`using MLJFlux`) to enable neural nuisance learners.",
    )
    Xs, μ, σ = _prepare_mlj_features(X)
    mach = ext.fit_nn_binary(Xs, y)
    return (mach = mach, μ = μ, σ = σ)
end

"""
    _predict_mlj_nn_binary(fit, X)

Class-1 probabilities from `:mlj_nn_binary` (applies train-time standardisation).
"""
function _predict_mlj_nn_binary(fit, X::Matrix{Float64})
    ext = _mljflux_ext()
    ext === nothing && error("Requested :mlj_nn_binary prediction but MLJFlux extension is not loaded.")
    mach, μ, σ = _unpack_mlj_fit(fit)
    Xc = _drop_intercept_column(X)
    Xs = μ === nothing ? Xc : _apply_feature_standardise(Xc, μ, σ)
    return ext.predict_nn_binary(mach, Xs)
end

function _fit_glmnet_safe(X::Matrix{Float64}, y::Vector{Float64}; α = 0.5, nfolds = 3)
    n, p = size(X)
    pred_sd = p >= 2 ? std(X[:, 2:end]; dims = 1) : [0.0]
    if n < 5 || p < 2 || std(y) < 1e-12 || all(pred_sd .< 1e-12)
        return (:mean, _safe_mean(y))
    end
    try
        cv = glmnetcv(X, y; α = α, nfolds = min(nfolds, n))
        λ = cv.lambda[cv.index_1se]
        return (:glmnet, (cv = cv, λ = λ))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _predict_glmnet(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return vec(GLMNet.predict(obj.cv.path, X, obj.λ))
end

function _fit_evotree_safe(
    X::Matrix{Float64},
    y::Vector{Float64};
    max_depth::Int = 2,
    nrounds::Int = 100,
)
    n = size(X, 1)
    if n < 8 || std(y) < 1e-12
        return (:mean, _safe_mean(y))
    end
    try
        mw = n < 50 ? 5.0 : max(3.0, ceil(n / 100))
        cfg = EvoTreeRegressor(
            nrounds = nrounds,
            max_depth = max_depth,
            eta = max_depth >= 4 ? 0.05 : 0.1,
            min_weight = mw,
            rowsample = 0.8,
            colsample = 0.8,
            lambda = 0.5,
            gamma = 0.0,
            seed = 42,
        )
        m = EvoTrees.fit(cfg, X, y)
        return (:evotree, m)
    catch
        return (:mean, _safe_mean(y))
    end
end

function _predict_evotree(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return vec(EvoTrees.predict(obj, X))
end

function _fit_glm_safe(X::Matrix{Float64}, y::Vector{Float64})
    try
        return (:glm, lm(X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _predict_glm(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return vec(GLM.predict(obj, X))
end

function _fit_logistic_safe(X::Matrix{Float64}, y::Vector{Float64})
    yb = clamp.(Float64.(y), 0.0, 1.0)
    try
        return (:logistic, glm(X, yb, Binomial(), LogitLink()))
    catch
        μ = clamp(_safe_mean(yb), 1e-3, 1 - 1e-3)
        return (:mean, μ)
    end
end

function _predict_logistic(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return clamp.(vec(GLM.predict(obj, X)), 1e-6, 1 - 1e-6)
end

"""
    _expand_interactions(X) -> Matrix{Float64}

Append all pairwise interaction columns X_i·X_j (i < j) to the design matrix.
"""
function _expand_interactions(X::Matrix{Float64})
    n, p = size(X)
    pairs = [(i, j) for i in 1:p for j in (i+1):p]
    isempty(pairs) && return X
    Xint = hcat([X[:, i] .* X[:, j] for (i, j) in pairs]...)
    return hcat(X, Xint)
end

"""
    _expand_quadratic(X) -> Matrix{Float64}

Append X_i² columns to the design matrix.
"""
function _expand_quadratic(X::Matrix{Float64})
    return hcat(X, X .^ 2)
end

function _fit_learner(name::Symbol, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    if name == :logistic
        family == :binomial || return (:mean, _safe_mean(y))
        return _fit_logistic_safe(X, y)
    elseif family == :binomial && name == :mean
        # Preserve legacy behaviour: when fitting with `family=:binomial`, the
        # `:mean` candidate is treated as a logistic classifier as a stable
        # probability estimator on {0,1}.
        return _fit_logistic_safe(X, y)
    elseif name == :mlj_logistic
        family == :binomial || return (:mean, _safe_mean(y))
        try
            return (:mlj_logistic, _fit_mlj_logistic(X, y))
        catch
            return _fit_logistic_safe(X, y)
        end
    elseif name == :mlj_mlp
        family == :binomial && return (:mean, _safe_mean(y))
        try
            return (:mlj_mlp, _fit_mlj_mlp(X, y))
        catch
            return (:mean, _safe_mean(y))
        end
    elseif name == :mlj_nn_binary
        family == :binomial || return (:mean, _safe_mean(y))
        try
            return (:mlj_nn_binary, _fit_mlj_nn_binary(X, y))
        catch
            return _fit_logistic_safe(X, y)
        end
    elseif name in (:mlj_ridge, :mlj_lasso, :mlj_elasticnet)
        family == :binomial && return (:mean, _safe_mean(y))
        try
            return (name, _fit_mlj_regressor(name, X, y))
        catch
            return (:mean, _safe_mean(y))
        end
    elseif name == :glm
        return _fit_glm_safe(X, y)
    elseif name == :glm_interact
        return (:glm_interact, _fit_glm_safe(_expand_interactions(X), y))
    elseif name == :glm_quad
        return (:glm_quad, _fit_glm_safe(_expand_quadratic(X), y))
    elseif name == :glmnet
        return _fit_glmnet_safe(X, y; α = 0.5)
    elseif name == :glmnet_lasso
        return _fit_glmnet_safe(X, y; α = 1.0)
    elseif name == :glmnet_ridge
        return _fit_glmnet_safe(X, y; α = 0.0)
    elseif name == :evotree
        return _fit_evotree_safe(X, y; max_depth = 2)
    elseif name == :evotree_deep
        return _fit_evotree_safe(X, y; max_depth = 4, nrounds = 150)
    elseif name == :mean
        return (:mean, _safe_mean(y))
    else
        error("Unknown learner $name")
    end
end

function _predict_learner(model, X::Matrix{Float64})
    typ = model[1]
    if typ == :glm
        return _predict_glm(model, X)
    elseif typ == :glm_interact
        return _predict_glm(model[2], _expand_interactions(X))
    elseif typ == :glm_quad
        return _predict_glm(model[2], _expand_quadratic(X))
    elseif typ == :glmnet
        return _predict_glmnet(model, X)
    elseif typ == :evotree
        return _predict_evotree(model, X)
    elseif typ in (:mlj_ridge, :mlj_lasso, :mlj_elasticnet)
        return _predict_mlj_regressor(model[2], X)
    elseif typ == :mlj_mlp
        return _predict_mlj_mlp(model[2], X)
    elseif typ == :logistic
        return _predict_logistic(model, X)
    elseif typ == :mlj_logistic
        return _predict_mlj_logistic(model[2], X)
    elseif typ == :mlj_nn_binary
        return _predict_mlj_nn_binary(model[2], X)
    else
        return fill(Float64(model[2]), size(X, 1))
    end
end

"""
    _nonneg_ls_weights(Z, y) -> Vector{Float64}

Nonnegative least-squares metalearner (discrete SuperLearner). Falls back to
uniform weights if the system is degenerate.
"""
function _nonneg_ls_weights(Z::Matrix{Float64}, y::Vector{Float64})
    k = size(Z, 2)
    k == 0 && return Float64[]
    # Coordinate descent NNLS on min ||y - Z w||², w ≥ 0, then renormalise.
    w = fill(1 / k, k)
    for _ in 1:200
        r = y .- Z * w
        for j in 1:k
            zj = Z[:, j]
            denom = sum(abs2, zj)
            denom < 1e-12 && continue
            w[j] = max(0.0, w[j] + sum(zj .* r) / denom)
            r = y .- Z * w
        end
    end
    s = sum(w)
    return s > 0 ? w ./ s : fill(1 / k, k)
end

function _invmse_weights(Z::Matrix{Float64}, y::Vector{Float64})
    k = size(Z, 2)
    mse = [mean((Z[:, j] .- y) .^ 2) for j in 1:k]
    inv = [isfinite(m) && m > 1e-12 ? 1 / m : 0.0 for m in mse]
    s = sum(inv)
    return s > 0 ? inv ./ s : fill(1 / k, k)
end

"""
    fit_super_learner(X, y; learners, metalearner, folds, rng) -> NamedTuple

Fit candidate learners and combine with a metalearner.
- `:discrete` — nested CV OOS predictions + nonnegative LS (default)
- `:invmse` — inverse training MSE weights (fast fallback)
"""
function fit_super_learner(
    X::Matrix{Float64},
    y::Vector{Float64};
    learners = DEFAULT_SL_LEARNERS,
    metalearner::Symbol = :discrete,
    folds::Int = 3,
    rng = StableRNG(42),
    family::Symbol = :gaussian,
)
    n = length(y)
    names = collect(learners)
    k = length(names)
    # Cross-validated library predictions for discrete SL
    Z = zeros(n, k)
    if metalearner == :discrete && n >= 2 * folds
        fold_sets = crossfit_indices(n, folds, rng)
        for test_idx in fold_sets
            train_idx = setdiff(1:n, test_idx)
            Xtr = X[train_idx, :]
            ytr = y[train_idx]
            Xte = X[test_idx, :]
            for (j, lrn) in enumerate(names)
                m = _fit_learner(lrn, Xtr, ytr; family = family)
                Z[test_idx, j] = _predict_learner(m, Xte)
            end
        end
        weights = _nonneg_ls_weights(Z, y)
    else
        for (j, lrn) in enumerate(names)
            m = _fit_learner(lrn, X, y; family = family)
            Z[:, j] = _predict_learner(m, X)
        end
        weights = _invmse_weights(Z, y)
    end
    # Refit on full data for prediction
    fits = Dict{Symbol, Any}()
    for lrn in names
        fits[lrn] = _fit_learner(lrn, X, y; family = family)
    end
    return (fits = fits, weights = weights, learners = names, metalearner = metalearner)
end

"""
    predict_super_learner(sl, X) -> Vector{Float64}
"""
function predict_super_learner(sl, X::Matrix{Float64})
    n = size(X, 1)
    out = zeros(n)
    for (j, lrn) in enumerate(sl.learners)
        out .+= sl.weights[j] .* _predict_learner(sl.fits[lrn], X)
    end
    return out
end

"""
    crossfit_outcome_predictions(df, outcome, treatment, covariates, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_outcome_predictions(
    df::DataFrame,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    preds = zeros(n)
    for test_idx in crossfit_indices(n, folds, rng)
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(train, covariates; treatment = treatment)
        Xte = design_matrix(test, covariates; treatment = treatment)
        sl = fit_super_learner(Xtr, y[train_idx]; learners = learners, rng = rng)
        preds[test_idx] = predict_super_learner(sl, Xte)
    end
    return preds
end

"""
    crossfit_predict_outcome(df, outcome, treatment, covariates, treatment_values, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_predict_outcome(
    df::DataFrame,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol},
    treatment_values::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    preds = zeros(n)
    a_cf = Float64.(treatment_values)
    for test_idx in crossfit_indices(n, folds, rng)
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(train, covariates; treatment = treatment)
        Xte = design_matrix(test, covariates; treatment = treatment, treatment_values = a_cf[test_idx])
        sl = fit_super_learner(Xtr, y[train_idx]; learners = learners, rng = rng)
        preds[test_idx] = predict_super_learner(sl, Xte)
    end
    return preds
end

"""
    crossfit_treatment_mean(df, treatment, covariates, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_treatment_mean(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    a = Float64.(df[!, treatment])
    preds = zeros(n)
    for test_idx in crossfit_indices(n, folds, rng)
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(train, covariates)
        Xte = design_matrix(test, covariates)
        sl = fit_super_learner(Xtr, a[train_idx]; learners = learners, rng = rng)
        preds[test_idx] = predict_super_learner(sl, Xte)
    end
    return preds
end

"""
    crossfit_propensity(df, treatment, covariates, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_propensity(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    a = Float64.(df[!, treatment])
    preds = zeros(n)
    for test_idx in crossfit_indices(n, folds, rng)
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(train, covariates)
        Xte = design_matrix(test, covariates)
        sl = fit_super_learner(
            Xtr, a[train_idx];
            learners = (:logistic, :mean),
            family = :binomial,
            metalearner = :invmse,
            rng = rng,
        )
        raw = predict_super_learner(sl, Xte)
        preds[test_idx] = clamp.(raw, 1e-3, 1 - 1e-3)
    end
    return preds
end

"""
    columns_present(df, cols) -> Vector{Symbol}
"""
function columns_present(df::DataFrame, cols)
    return [c for c in cols if hasproperty(df, c)]
end

export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS
export design_matrix, sparse_exposure_diagnostic
export fit_super_learner, predict_super_learner
export crossfit_outcome_predictions, crossfit_predict_outcome
export crossfit_treatment_mean, crossfit_propensity, columns_present
