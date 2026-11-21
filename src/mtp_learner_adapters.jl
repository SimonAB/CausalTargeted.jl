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

"""Nonnegative mean fallback for count learners."""
function _safe_mean_count(y::AbstractVector{<:Real})
    return max(_safe_mean(y), 1e-8)
end

function _validate_count_outcome(y::AbstractVector{<:Real})
    all(isfinite, y) ||
        throw(ArgumentError("count outcomes must be finite"))
    any(<(0), y) &&
        throw(ArgumentError("count outcomes must be non-negative"))
    return nothing
end

function _fit_poisson_safe(X::Matrix{Float64}, y::Vector{Float64})
    yc = max.(Float64.(y), 0.0)
    try
        return (:poisson, glm(X, yc, Poisson(), LogLink()))
    catch
        return (:mean, _safe_mean_count(yc))
    end
end

function _predict_poisson(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return max.(vec(GLM.predict(obj, X)), 1e-8)
end

"""
    _profile_nb_theta(X, y; grid) -> NamedTuple

Select a NegativeBinomial shape parameter by minimum deviance on the training
fold (NB2-style GLM with log link).
"""
function _profile_nb_theta(
    X::Matrix{Float64},
    y::Vector{Float64};
    grid = DEFAULT_NB_THETA_GRID,
)
    yc = max.(Float64.(y), 0.0)
    best_dev = Inf
    best_model = nothing
    best_θ = 2.0
    for θ in grid
        try
            m = glm(X, yc, NegativeBinomial(θ), LogLink())
            dev = deviance(m)
            if dev < best_dev
                best_dev = dev
                best_model = m
                best_θ = θ
            end
        catch
        end
    end
    best_model === nothing && return _fit_poisson_safe(X, y)
    return (:negbin, (model = best_model, θ = best_θ))
end

function _fit_negbin_safe(
    X::Matrix{Float64},
    y::Vector{Float64};
    θ::Union{Nothing, Real} = nothing,
)
    yc = max.(Float64.(y), 0.0)
    if θ !== nothing
        try
            m = glm(X, yc, NegativeBinomial(Float64(θ)), LogLink())
            return (:negbin, (model = m, θ = Float64(θ)))
        catch
            return (:mean, _safe_mean_count(yc))
        end
    end
    return _profile_nb_theta(X, y)
end

"""
    _fit_zeroinflated_nb_safe(X, y) -> Tuple

Zero-inflated NB learner: logistic inflation on ``I(y=0)`` plus profiled NB
on counts. Predictions use ``(1 - \\hat\\pi) \\hat\\mu`` (composite mean).
"""
function _fit_zeroinflated_nb_safe(X::Matrix{Float64}, y::Vector{Float64})
    yc = max.(Float64.(y), 0.0)
    z = Float64.(yc .== 0)
    pi_model = try
        m = glm(X, z, Binomial(), LogitLink())
        (:logistic, m)
    catch
        (:mean, _safe_mean(z))
    end
    nb_model = _fit_negbin_safe(X, yc)
    return (:zeroinflated_nb, (pi = pi_model, nb = nb_model))
end

function _predict_zeroinflated_nb(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    pi_pred = if obj.pi[1] == :mean
        fill(Float64(obj.pi[2]), size(X, 1))
    else
        clamp.(vec(GLM.predict(obj.pi[2], X)), 1e-8, 1 - 1e-8)
    end
    mu = _predict_negbin(obj.nb, X)
    return max.((1 .- pi_pred) .* mu, 0.0)
end

function _predict_negbin(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return max.(vec(GLM.predict(obj.model, X)), 1e-8)
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

"""
    NestedSLCandidate(name, learners, metalearner)

An ensemble Super Learner treated as one discrete-SL candidate (Phillips
eSL-inside-dSL). Outer `fit_super_learner` must use `metalearner=:cv_selector`.
Inner ensembles may not themselves be `:cv_selector` or contain nested candidates.
LMTP density-ratio classifiers stay on `:invmse` and should not use this type.
"""
struct NestedSLCandidate
    name::Symbol
    learners::Vector{Symbol}
    metalearner::Symbol
end

"""
    nested_sl_candidate(learners; name=:esl, metalearner=:nnls) -> NestedSLCandidate

Build a nested ensemble candidate for `metalearner=:cv_selector`.
"""
function nested_sl_candidate(
    learners;
    name::Symbol = :esl,
    metalearner::Symbol = :nnls,
)
    inner = collect(Symbol, learners)
    isempty(inner) && throw(ArgumentError("nested_sl_candidate requires at least one learner"))
    metalearner === :cv_selector && throw(ArgumentError(
        "nested SL inner metalearner cannot be :cv_selector; use :nnls or :nnloglik",
    ))
    return NestedSLCandidate(name, inner, metalearner)
end

_candidate_key(c::Symbol) = c
_candidate_key(c::NestedSLCandidate) = c.name

function _fit_any_candidate(c::Symbol, X, y; family)
    return _fit_learner(c, X, y; family = family)
end

function _fit_any_candidate(c::NestedSLCandidate, X, y; family)
    return fit_super_learner(
        X, y;
        learners = c.learners,
        metalearner = c.metalearner,
        family = family,
    )
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

function _fit_learner(::Val{:randomforest}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family in (:gaussian, :binomial, :multinomial, :poisson, :negbin, :zeroinflated_nb) || throw(ArgumentError(
        ":randomforest supports family=:gaussian, :binomial, :multinomial, :poisson, :negbin, or :zeroinflated_nb; got $family",
    ))
    tree_family = family in COUNT_OUTCOME_FAMILIES ? :gaussian : family
    return (:randomforest, _fit_mlj_tree(:randomforest, X, y; family = tree_family))
end

function _fit_learner(::Val{:xgboost}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family in (:gaussian, :binomial, :multinomial, :poisson, :negbin, :zeroinflated_nb) || throw(ArgumentError(
        ":xgboost supports family=:gaussian, :binomial, :multinomial, :poisson, :negbin, or :zeroinflated_nb; got $family",
    ))
    tree_family = family in COUNT_OUTCOME_FAMILIES ? :gaussian : family
    return (:xgboost, _fit_mlj_tree(:xgboost, X, y; family = tree_family))
end

"""Fit a GLM on an already-expanded design (Gaussian, logistic, Poisson, or NB)."""
function _fit_glm_design(X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    family === :binomial && return _fit_logistic_safe(X, y)
    family === :poisson && return _fit_poisson_safe(X, y)
    family === :negbin && return _fit_negbin_safe(X, y)
    family === :zeroinflated_nb && return _fit_zeroinflated_nb_safe(X, y)
    return _fit_glm_safe(X, y)
end

"""Predict from a nested GLM / count / logistic / `mean` fallback tuple."""
function _predict_glm_design(inner, X::Matrix{Float64})
    typ = inner[1]
    typ === :logistic && return _predict_logistic(inner, X)
    typ === :poisson && return _predict_poisson(inner, X)
    typ === :negbin && return _predict_negbin(inner, X)
    typ === :zeroinflated_nb && return _predict_zeroinflated_nb(inner, X)
    return _predict_glm(inner, X)
end

function _fit_learner(::Val{:glm}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_glm_design(X, y; family = family)
end

function _fit_learner(::Val{:glm_poisson}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_poisson_safe(X, y)
end

function _fit_learner(::Val{:glm_nb}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_negbin_safe(X, y)
end

function _fit_learner(::Val{:zeroinflated_nb}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return _fit_zeroinflated_nb_safe(X, y)
end

function _fit_learner(::Val{:glm_interact}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return (:glm_interact, _fit_glm_design(_expand_interactions(X), y; family = family))
end

function _fit_learner(::Val{:glm_quad}, X::Matrix{Float64}, y::Vector{Float64}; family = :gaussian)
    return (:glm_quad, _fit_glm_design(_expand_quadratic(X), y; family = family))
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
    model isa SuperLearnerFit && return predict_super_learner(model, X)
    return _predict_learner(Val(model[1]), model, X)
end

function _predict_learner(::Val{:glm}, model, X::Matrix{Float64})
    return _predict_glm_design(model, X)
end

function _predict_learner(::Val{:glm_poisson}, model, X::Matrix{Float64})
    return _predict_poisson(model, X)
end

function _predict_learner(::Val{:glm_nb}, model, X::Matrix{Float64})
    return _predict_negbin(model, X)
end

function _predict_learner(::Val{:zeroinflated_nb}, model, X::Matrix{Float64})
    return _predict_zeroinflated_nb(model, X)
end

function _predict_learner(::Val{:glm_interact}, model, X::Matrix{Float64})
    return _predict_glm_design(model[2], _expand_interactions(X))
end

function _predict_learner(::Val{:glm_quad}, model, X::Matrix{Float64})
    return _predict_glm_design(model[2], _expand_quadratic(X))
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

function _predict_learner(::Val{:randomforest}, model, X::Matrix{Float64})
    return _predict_mlj_tree(:randomforest, model[2], X)
end

function _predict_learner(::Val{:xgboost}, model, X::Matrix{Float64})
    return _predict_mlj_tree(:xgboost, model[2], X)
end

function _predict_learner(::Val{:poisson}, model, X::Matrix{Float64})
    return _predict_poisson(model, X)
end

function _predict_learner(::Val{:negbin}, model, X::Matrix{Float64})
    return _predict_negbin(model, X)
end

function _predict_learner(::Val{typ}, model, X::Matrix{Float64}) where {typ}
    return fill(Float64(model[2]), size(X, 1))
end
