# ---------------------------------------------------------------------------
# GLM, logistic, Poisson, Negative Binomial & polynomial feature adapters
# ---------------------------------------------------------------------------

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
