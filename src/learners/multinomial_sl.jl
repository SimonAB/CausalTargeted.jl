"""Multinomial Super Learner (sl3-style simplex stacking)."""

"""
    _multinomial_levels(y; levels=nothing) -> Vector

Canonical class labels. Integers `1:K` or arbitrary labels; order is sorted
when possible so `K=2` matches binomial `P(Y=1)` on the second column.
"""
function _multinomial_levels(y::AbstractVector; levels = nothing)
    if levels !== nothing
        lev = collect(levels)
        isempty(lev) && throw(ArgumentError("multinomial levels must be non-empty"))
        return lev
    end
    observed = collect(unique(y))
    isempty(observed) && throw(ArgumentError("multinomial outcome is empty"))
    try
        return sort(observed)
    catch
        return observed
    end
end

function _onehot_outcome(y::AbstractVector, levels::AbstractVector)
    n = length(y)
    K = length(levels)
    Y = zeros(n, K)
    lookup = Dict{Any, Int}(levels[k] => k for k in 1:K)
    @inbounds for i in 1:n
        k = get(lookup, y[i], 0)
        k == 0 && throw(ArgumentError(
            "outcome value $(repr(y[i])) is not among multinomial levels $(repr(levels))",
        ))
        Y[i, k] = 1.0
    end
    return Y
end

function _row_simplex!(P::AbstractMatrix{<:Real})
    @inbounds for i in 1:size(P, 1)
        s = 0.0
        for k in 1:size(P, 2)
            P[i, k] = max(Float64(P[i, k]), 1e-12)
            s += P[i, k]
        end
        invs = s > 0 ? 1 / s : 1 / size(P, 2)
        for k in 1:size(P, 2)
            P[i, k] *= invs
        end
    end
    return P
end

function _class_frequencies(Y::AbstractMatrix{<:Real})
    n = size(Y, 1)
    n == 0 && return fill(1 / size(Y, 2), size(Y, 2))
    p = vec(sum(Y; dims = 1)) ./ n
    p .= max.(p, 1e-12)
    return p ./ sum(p)
end

function _predict_multinomial_mean(fit, n::Int)
    p = fit[2]
    return repeat(reshape(p, 1, :), n, 1)
end

"""One-vs-rest logistic, then row-softmax. `K=2` uses a single binomial GLM."""
function _fit_multinomial_glm(X::Matrix{Float64}, Y::AbstractMatrix{<:Real})
    n, K = size(Y)
    K == 2 && return (:multinomial_binomial, _fit_logistic_safe(X, Y[:, 2]))
    models = Vector{Any}(undef, K)
    for k in 1:K
        models[k] = _fit_logistic_safe(X, Y[:, k])
    end
    return (:multinomial_ovr, models)
end

function _predict_multinomial_glm(fit, X::Matrix{Float64})
    tag, obj = fit
    n = size(X, 1)
    if tag === :multinomial_binomial
        p1 = _predict_logistic(obj, X)
        return hcat(1 .- p1, p1)
    end
    K = length(obj)
    P = Matrix{Float64}(undef, n, K)
    for k in 1:K
        P[:, k] = _predict_logistic(obj[k], X)
    end
    return _row_simplex!(P)
end

function _fit_multinomial_learner(
    name::Symbol,
    X::Matrix{Float64},
    Y::AbstractMatrix{<:Real},
    y_labels::AbstractVector,
    levels::AbstractVector,
)
    name === :mean && return (:multinomial_mean, _class_frequencies(Y))
    name === :glm && return _fit_multinomial_glm(X, Y)
    # Trees: K independent binomial fits when the extension supports binomial.
    if name in (:randomforest, :xgboost, :evotree, :evotree_deep, :glm_interact, :glm_quad)
        K = size(Y, 2)
        if K == 2
            try
                m = _fit_learner(name, X, Y[:, 2]; family = :binomial)
                return (:multinomial_binomial_wrapped, m)
            catch
                return (:multinomial_mean, _class_frequencies(Y))
            end
        end
        models = Vector{Any}(undef, K)
        for k in 1:K
            try
                models[k] = _fit_learner(name, X, Y[:, k]; family = :binomial)
            catch
                return (:multinomial_mean, _class_frequencies(Y))
            end
        end
        return (:multinomial_ovr_wrapped, models)
    end
    return (:multinomial_mean, _class_frequencies(Y))
end

function _predict_multinomial_learner(fit, X::Matrix{Float64})
    tag = fit[1]
    tag === :multinomial_mean && return _predict_multinomial_mean(fit, size(X, 1))
    if tag === :multinomial_binomial || tag === :multinomial_ovr
        return _predict_multinomial_glm(fit, X)
    end
    if tag === :multinomial_binomial_wrapped
        p1 = _predict_learner(fit[2], X)
        return hcat(1 .- p1, p1)
    end
    if tag === :multinomial_ovr_wrapped
        models = fit[2]
        P = Matrix{Float64}(undef, size(X, 1), length(models))
        for k in eachindex(models)
            P[:, k] = _predict_learner(models[k], X)
        end
        return _row_simplex!(P)
    end
    return _predict_multinomial_mean((:multinomial_mean, fill(1 / 2, 2)), size(X, 1))
end

"""Mix class-probability tensors with learner weights; rows on the simplex."""
function _mix_multinomial_probs(
    preds::Vector{Matrix{Float64}},
    weights::AbstractVector{<:Real},
)
    n, K = size(preds[1])
    P = zeros(n, K)
    @inbounds for j in eachindex(weights)
        w = Float64(weights[j])
        w == 0 && continue
        Pj = preds[j]
        for i in 1:n, k in 1:K
            P[i, k] += w * Pj[i, k]
        end
    end
    return _row_simplex!(P)
end

function _multinomial_brier(P::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    return mean(sum((P .- Y) .^ 2; dims = 2))
end

function _multinomial_nll(P::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    loss = 0.0
    @inbounds for i in 1:size(Y, 1)
        for k in 1:size(Y, 2)
            Y[i, k] == 0 && continue
            loss -= Y[i, k] * log(max(P[i, k], 1e-12))
        end
    end
    return loss / size(Y, 1)
end

"""Nonnegative simplex weights minimising multinomial NLL of the mixture."""
function _multinomial_nnloglik_weights(
    preds::Vector{Matrix{Float64}},
    Y::AbstractMatrix{<:Real};
    maxiter::Int = 400,
)
    L = length(preds)
    w = fill(1 / L, L)
    step = 0.2
    for _ in 1:maxiter
        P = _mix_multinomial_probs(preds, w)
        g = zeros(L)
        @inbounds for j in 1:L
            Pj = preds[j]
            acc = 0.0
            for i in 1:size(Y, 1), k in 1:size(Y, 2)
                Y[i, k] == 0 && continue
                acc -= Y[i, k] * (Pj[i, k] / max(P[i, k], 1e-12))
            end
            g[j] = acc / size(Y, 1)
        end
        w = max.(0.0, w .- step .* g)
        s = sum(w)
        w = s > 0 ? w ./ s : fill(1 / L, L)
        step *= 0.995
    end
    return w
end

function _multinomial_nnls_weights(preds::Vector{Matrix{Float64}}, Y::AbstractMatrix{<:Real})
    n, K = size(Y)
    L = length(preds)
    Z = Matrix{Float64}(undef, n * K, L)
    yv = vec(Y)
    for j in 1:L
        Z[:, j] = vec(preds[j])
    end
    return _nonneg_ls_weights(Z, yv)
end

function _multinomial_cv_selector_weights(preds::Vector{Matrix{Float64}}, Y::AbstractMatrix{<:Real})
    L = length(preds)
    brier = [_multinomial_brier(preds[j], Y) for j in 1:L]
    w = zeros(L)
    w[argmin(brier)] = 1.0
    return w
end

function _cv_multinomial_predictions(
    X::Matrix{Float64},
    Y::AbstractMatrix{<:Real},
    y_labels::AbstractVector,
    levels::AbstractVector,
    names::Vector{Symbol},
    folds::Int,
    rng,
)
    n = size(X, 1)
    L = length(names)
    out = [zeros(n, size(Y, 2)) for _ in 1:L]
    fold_sets = crossfit_indices(n, folds, rng)
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        for (j, lrn) in enumerate(names)
            m = _fit_multinomial_learner(
                lrn, X[train_idx, :], Y[train_idx, :], y_labels[train_idx], levels,
            )
            out[j][test_idx, :] .= _predict_multinomial_learner(m, X[test_idx, :])
        end
    end
    return out
end

function _fit_multinomial_super_learner(
    X::Matrix{Float64},
    y::AbstractVector;
    learners = DEFAULT_SL_LEARNERS,
    metalearner::Symbol = :nnloglik,
    folds::Int = 3,
    rng = StableRNG(42),
    levels = nothing,
)
    lev = _multinomial_levels(y; levels = levels)
    length(lev) < 2 && throw(ArgumentError("family=:multinomial requires at least two classes"))
    Y = _onehot_outcome(y, lev)
    y_labels = collect(y)
    names = collect(Symbol, learners)
    n = length(y)
    use_cv = metalearner in (:nnls, :nnloglik, :cv_selector) && n >= 2 * folds
    cv_preds = if use_cv
        _cv_multinomial_predictions(X, Y, y_labels, lev, names, folds, rng)
    else
        [_predict_multinomial_learner(
            _fit_multinomial_learner(lrn, X, Y, y_labels, lev), X,
        ) for lrn in names]
    end
    weights = if metalearner === :nnloglik && use_cv
        _multinomial_nnloglik_weights(cv_preds, Y)
    elseif metalearner === :cv_selector && use_cv
        _multinomial_cv_selector_weights(cv_preds, Y)
    elseif metalearner === :nnls && use_cv
        _multinomial_nnls_weights(cv_preds, Y)
    else
        brier = [_multinomial_brier(cv_preds[j], Y) for j in eachindex(names)]
        inv = [isfinite(b) && b > 1e-12 ? 1 / b : 0.0 for b in brier]
        s = sum(inv)
        s > 0 ? inv ./ s : fill(1 / length(names), length(names))
    end
    fits = Dict{Symbol, Any}()
    for lrn in names
        fits[lrn] = _fit_multinomial_learner(lrn, X, Y, y_labels, lev)
    end
    return SuperLearnerFit(fits, weights, names, metalearner, :multinomial, collect(Any, lev))
end

function _predict_multinomial_super_learner(sl::SuperLearnerFit, X::Matrix{Float64})
    preds = [_predict_multinomial_learner(sl.fits[lrn], X) for lrn in sl.learners]
    return _mix_multinomial_probs(preds, sl.weights)
end
