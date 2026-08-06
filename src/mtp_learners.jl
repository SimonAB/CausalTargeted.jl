"""SuperLearner-style nuisance models for MTP estimators.

Uses discrete SuperLearner (cross-validated nonnegative weights) by default,
matching the spirit of R SuperLearner / nnls metalearning.
"""

using DataFrames
using Statistics
using GLM
using Distributions
using Random
using StableRNGs

const DEFAULT_SL_LEARNERS = (:glm, :mean)
const RICH_SL_LEARNERS = (:glm, :glm_interact, :glm_quad, :glmnet, :glmnet_lasso, :glmnet_ridge, :evotree, :evotree_deep, :mean)

"""
    SuperLearnerFit

Typed SuperLearner ensemble: candidate fits, metalearner weights, and library names.
"""
struct SuperLearnerFit
    fits::Dict{Symbol, Any}
    weights::Vector{Float64}
    learners::Vector{Symbol}
    metalearner::Symbol
end

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
    present = [c for c in covariates if hasproperty(df, c)]
    p = 1 + Int(treatment !== nothing) + length(present)
    X = Matrix{Float64}(undef, n, p)
    X[:, 1] .= 1.0
    col = 2
    if treatment !== nothing
        if treatment_values === nothing
            @inbounds for i in 1:n
                X[i, col] = Float64(df[i, treatment])
            end
        else
            @inbounds for i in 1:n
                X[i, col] = Float64(treatment_values[i])
            end
        end
        col += 1
    end
    for c in present
        @inbounds for i in 1:n
            X[i, col] = Float64(df[i, c])
        end
        col += 1
    end
    return X
end

"""
    covariate_design_matrix(df, covariates) -> Matrix{Float64}

Intercept plus covariates only (no treatment column). Reused across δ / counterfactual
predictions via [`outcome_design_matrix`](@ref).
"""
function covariate_design_matrix(df::DataFrame, covariates::Vector{Symbol})
    return design_matrix(df, covariates; treatment = nothing)
end

"""
    outcome_design_matrix(W, a) -> Matrix{Float64}

Assemble `[1 | A | covariates]` from a cached covariate design `W` (intercept +
covariates) and a treatment vector `a` of matching length.
"""
function outcome_design_matrix(W::AbstractMatrix{<:Real}, a::AbstractVector{<:Real})
    n, p = size(W)
    length(a) == n || throw(DimensionMismatch(
        "treatment length $(length(a)) does not match design rows $n",
    ))
    X = Matrix{Float64}(undef, n, p + 1)
    @inbounds for i in 1:n
        X[i, 1] = Float64(W[i, 1])
        X[i, 2] = Float64(a[i])
    end
    if p > 1
        @inbounds for j in 2:p
            for i in 1:n
                X[i, j + 1] = Float64(W[i, j])
            end
        end
    end
    return X
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

The default implementation throws; load `MLJ` and `MLJLinearModels` to activate
`CausalTargetedMLJExt`.
"""
function _fit_mlj_regressor(
    name::Symbol,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
)
    error(
        "Requested MLJ learner $name, but MLJ integration is not loaded. " *
        "Run `using MLJ, MLJLinearModels` to activate CausalTargetedMLJExt.",
    )
end

"""
    _predict_mlj_regressor(fit, X)

Internal hook point for MLJ-backed nuisance regression predictions.
Real methods are provided by `CausalTargetedMLJExt`.
"""
function _predict_mlj_regressor(fit, X::AbstractMatrix{<:Real})
    error(
        "MLJ regression prediction requested, but MLJ integration is not loaded. " *
        "Run `using MLJ, MLJLinearModels` to activate CausalTargetedMLJExt.",
    )
end

"""
    _fit_mlj_logistic(X, y)

Internal hook point for MLJ-backed binomial logistic nuisance learners.
"""
function _fit_mlj_logistic(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    error(
        "Requested MLJ binomial learner, but MLJ integration is not loaded. " *
        "Run `using MLJ, MLJLinearModels` to activate CausalTargetedMLJExt.",
    )
end

"""
    _predict_mlj_logistic(fit, X)

Internal hook point for MLJ-backed binomial logistic predictions.
Real methods are provided by `CausalTargetedMLJExt`.
"""
function _predict_mlj_logistic(fit, X::AbstractMatrix{<:Real})
    error(
        "MLJ logistic prediction requested, but MLJ integration is not loaded. " *
        "Run `using MLJ, MLJLinearModels` to activate CausalTargetedMLJExt.",
    )
end

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
    _mlj_feature_matrix(fit, X) -> (mach, Xs)

Rebuild the standardised feature matrix used at fit time (no DataFrame wrap).
"""
function _mlj_feature_matrix(fit, X::Matrix{Float64})
    mach, μ, σ = _unpack_mlj_fit(fit)
    Xc = _drop_intercept_column(X)
    Xs = μ === nothing ? Xc : _apply_feature_standardise(Xc, μ, σ)
    return mach, Xs
end

"""
    _mlj_ext()

Return the optional `CausalTargetedMLJExt` module, or `nothing` if MLJ is not loaded.
"""
function _mlj_ext()
    return Base.get_extension(@__MODULE__, :CausalTargetedMLJExt)
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

"""
    _fit_evotree_safe(X, y; max_depth, nrounds)

Internal EvoTrees fit. Load `EvoTrees` to activate `CausalTargetedEvoTreesExt`.
"""
function _fit_evotree_safe(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    max_depth::Int = 2,
    nrounds::Int = 100,
)
    error(
        "Requested an EvoTrees learner, but EvoTrees is not loaded. " *
        "Run `using EvoTrees` to activate CausalTargetedEvoTreesExt.",
    )
end

"""
    _predict_evotree(model, X)

Internal EvoTrees prediction. Real method provided by `CausalTargetedEvoTreesExt`.
"""
function _predict_evotree(model, X::AbstractMatrix{<:Real})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    error(
        "EvoTrees prediction requested, but EvoTrees is not loaded. " *
        "Run `using EvoTrees` to activate CausalTargetedEvoTreesExt.",
    )
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
    n_pairs = p * (p - 1) ÷ 2
    n_pairs == 0 && return X
    out = Matrix{Float64}(undef, n, p + n_pairs)
    @inbounds out[:, 1:p] .= X
    col = p + 1
    @inbounds for i in 1:p
        for j in (i + 1):p
            for r in 1:n
                out[r, col] = X[r, i] * X[r, j]
            end
            col += 1
        end
    end
    return out
end

"""
    _expand_quadratic(X) -> Matrix{Float64}

Append X_i² columns to the design matrix.
"""
function _expand_quadratic(X::Matrix{Float64})
    n, p = size(X)
    out = Matrix{Float64}(undef, n, 2p)
    @inbounds out[:, 1:p] .= X
    @inbounds for j in 1:p
        for r in 1:n
            out[r, p + j] = X[r, j]^2
        end
    end
    return out
end

function _fit_learner(name::Symbol, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    # Preserve legacy behaviour: under `family=:binomial`, `:mean` is a logistic
    # probability estimator on {0,1}.
    if family == :binomial && name == :mean
        return _fit_logistic_safe(X, y)
    end
    return _fit_learner(Val(name), X, y; family = family)
end

function _fit_learner(::Val{name}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian) where {name}
    error("Unknown learner $name")
end

function _fit_learner(::Val{:logistic}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial || return (:mean, _safe_mean(y))
    return _fit_logistic_safe(X, y)
end

function _fit_learner(::Val{:mlj_logistic}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial || return (:mean, _safe_mean(y))
    try
        return (:mlj_logistic, _fit_mlj_logistic(X, y))
    catch
        return _fit_logistic_safe(X, y)
    end
end

function _fit_learner(::Val{:mlj_mlp}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:mlj_mlp, _fit_mlj_mlp(X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:mlj_nn_binary}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial || return (:mean, _safe_mean(y))
    try
        return (:mlj_nn_binary, _fit_mlj_nn_binary(X, y))
    catch
        return _fit_logistic_safe(X, y)
    end
end

function _fit_learner(::Val{:mlj_ridge}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:mlj_ridge, _fit_mlj_regressor(:mlj_ridge, X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:mlj_lasso}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:mlj_lasso, _fit_mlj_regressor(:mlj_lasso, X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:mlj_elasticnet}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:mlj_elasticnet, _fit_mlj_regressor(:mlj_elasticnet, X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:glm}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_glm_safe(X, y)
end

function _fit_learner(::Val{:glm_interact}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return (:glm_interact, _fit_glm_safe(_expand_interactions(X), y))
end

function _fit_learner(::Val{:glm_quad}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return (:glm_quad, _fit_glm_safe(_expand_quadratic(X), y))
end

function _fit_learner(::Val{:glmnet}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:glmnet, _fit_mlj_regressor(:mlj_elasticnet, X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:glmnet_lasso}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:glmnet_lasso, _fit_mlj_regressor(:mlj_lasso, X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:glmnet_ridge}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family == :binomial && return (:mean, _safe_mean(y))
    try
        return (:glmnet_ridge, _fit_mlj_regressor(:mlj_ridge, X, y))
    catch
        return (:mean, _safe_mean(y))
    end
end

function _fit_learner(::Val{:evotree}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_evotree_safe(X, y; max_depth = 2)
end

function _fit_learner(::Val{:evotree_deep}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_evotree_safe(X, y; max_depth = 4, nrounds = 150)
end

function _fit_learner(::Val{:mean}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return (:mean, _safe_mean(y))
end

function _predict_learner(model, X::Matrix{Float64})
    return _predict_learner(Val(model[1]), model, X)
end

function _predict_learner(::Val{:glm}, model, X::Matrix{Float64})
    return _predict_glm(model, X)
end

function _predict_learner(::Val{:glm_interact}, model, X::Matrix{Float64})
    return _predict_glm(model[2], _expand_interactions(X))
end

function _predict_learner(::Val{:glm_quad}, model, X::Matrix{Float64})
    return _predict_glm(model[2], _expand_quadratic(X))
end

function _predict_learner(::Val{:glmnet}, model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return _predict_mlj_regressor(obj, X)
end

function _predict_learner(::Val{:glmnet_lasso}, model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return _predict_mlj_regressor(obj, X)
end

function _predict_learner(::Val{:glmnet_ridge}, model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return _predict_mlj_regressor(obj, X)
end

function _predict_learner(::Val{:evotree}, model, X::Matrix{Float64})
    return _predict_evotree(model, X)
end

function _predict_learner(::Val{:mlj_ridge}, model, X::Matrix{Float64})
    return _predict_mlj_regressor(model[2], X)
end

function _predict_learner(::Val{:mlj_lasso}, model, X::Matrix{Float64})
    return _predict_mlj_regressor(model[2], X)
end

function _predict_learner(::Val{:mlj_elasticnet}, model, X::Matrix{Float64})
    return _predict_mlj_regressor(model[2], X)
end

function _predict_learner(::Val{:mlj_mlp}, model, X::Matrix{Float64})
    return _predict_mlj_mlp(model[2], X)
end

function _predict_learner(::Val{:logistic}, model, X::Matrix{Float64})
    return _predict_logistic(model, X)
end

function _predict_learner(::Val{:mlj_logistic}, model, X::Matrix{Float64})
    return _predict_mlj_logistic(model[2], X)
end

function _predict_learner(::Val{:mlj_nn_binary}, model, X::Matrix{Float64})
    return _predict_mlj_nn_binary(model[2], X)
end

function _predict_learner(::Val{typ}, model, X::Matrix{Float64}) where {typ}
    return fill(Float64(model[2]), size(X, 1))
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
    _fit_sl_outcome(df, cols, y; treatment, learners, rng) -> SuperLearnerFit

Fit a Super Learner on `design_matrix(df, cols; treatment)`. Used by g-computation,
sequential LMTP, and (via CausalMediation) mediation nuisances.
"""
function _fit_sl_outcome(
    df::DataFrame,
    cols::Vector{Symbol},
    y::AbstractVector{<:Real};
    treatment = nothing,
    learners = DEFAULT_SL_LEARNERS,
    rng = StableRNG(1),
)
    X = design_matrix(df, cols; treatment = treatment)
    return fit_super_learner(X, Float64.(y); learners = learners, rng = rng)
end

"""Predict from a Super Learner fit on a design matrix for `cols`."""
function _predict_sl(
    sl,
    df::DataFrame,
    cols::Vector{Symbol};
    treatment = nothing,
    treatment_values = nothing,
)
    X = design_matrix(df, cols; treatment = treatment, treatment_values = treatment_values)
    return predict_super_learner(sl, X)
end

"""
    fit_super_learner(X, y; learners, metalearner, folds, rng) -> SuperLearnerFit

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
    names = collect(Symbol, learners)
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
    return SuperLearnerFit(fits, weights, names, metalearner)
end

"""
    predict_super_learner(sl, X) -> Vector{Float64}
"""
function predict_super_learner(sl::SuperLearnerFit, X::Matrix{Float64})
    n = size(X, 1)
    out = zeros(n)
    for (j, lrn) in enumerate(sl.learners)
        out .+= sl.weights[j] .* _predict_learner(sl.fits[lrn], X)
    end
    return out
end

# Compat for NamedTuple fits from older call sites / notebooks
function predict_super_learner(sl::NamedTuple, X::Matrix{Float64})
    return predict_super_learner(
        SuperLearnerFit(sl.fits, collect(Float64, sl.weights), collect(Symbol, sl.learners), sl.metalearner),
        X,
    )
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
export SuperLearnerFit
export design_matrix, covariate_design_matrix, outcome_design_matrix, sparse_exposure_diagnostic
export fit_super_learner, predict_super_learner
export crossfit_outcome_predictions, crossfit_predict_outcome
export crossfit_treatment_mean, crossfit_propensity, columns_present
