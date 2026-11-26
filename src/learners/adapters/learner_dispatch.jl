# ---------------------------------------------------------------------------
# NestedSLCandidate & Val-dispatched learner fit / predict table
# ---------------------------------------------------------------------------

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
