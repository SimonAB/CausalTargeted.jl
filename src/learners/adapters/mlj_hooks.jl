# ---------------------------------------------------------------------------
# MLJ / MLJFlux / EvoTrees extension hooks & feature standardisation helpers
# ---------------------------------------------------------------------------

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
    _fit_mlj_tree(name, X, y; family)

Internal hook for the controlled MLJ tree learners `:randomforest` and
`:xgboost`. Their package extensions prepare unscaled features, construct the
corresponding regression or probabilistic classification model, and fit it.
"""
function _fit_mlj_tree(
    name::Symbol,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    family::Symbol,
)
    return _fit_mlj_tree(Val(name), X, y; family = family)
end

function _fit_mlj_tree(
    ::Val{name},
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    family::Symbol,
) where {name}
    packages = name == :randomforest ? "MLJ, MLJDecisionTreeInterface" :
               name == :xgboost ? "MLJ, MLJXGBoostInterface" : "the required MLJ interface"
    error(
        "Requested :$name, but its optional MLJ backend is not loaded. " *
        "Install and run `using $packages` before fitting this learner.",
    )
end

"""
    _predict_mlj_tree(name, fit, X)

Internal prediction hook for optional MLJ tree learners.
"""
function _predict_mlj_tree(name::Symbol, fit, X::AbstractMatrix{<:Real})
    return _predict_mlj_tree(Val(name), fit, X)
end

function _predict_mlj_tree(::Val{name}, fit, X::AbstractMatrix{<:Real}) where {name}
    error(
        "Prediction for :$name requires its optional MLJ backend to remain loaded.",
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
    _prepare_mlj_tree_features(X) -> Matrix{Float64}

Drop the artificial design-matrix intercept without centring or scaling. Tree
backends receive predictors in their original numeric units.
"""
function _prepare_mlj_tree_features(X::Matrix{Float64})
    return _drop_intercept_column(X)
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
    mach, Xs = _mlj_feature_matrix(fit, X)
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
    mach, Xs = _mlj_feature_matrix(fit, X)
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
