"""
    _cv_selector_weights(Z, y) -> Vector{Float64}

Discrete Super Learner (Phillips dSL / sl3 `Lrnr_cv_selector`): one-hot weight
on the candidate with lowest cross-validated squared error (Brier).
"""
function _cv_selector_weights(Z::Matrix{Float64}, y::Vector{Float64})
    k = size(Z, 2)
    k == 0 && return Float64[]
    mse = [mean((Z[:, j] .- y) .^ 2) for j in 1:k]
    for j in 1:k
        isfinite(mse[j]) || (mse[j] = Inf)
    end
    w = zeros(k)
    w[argmin(mse)] = 1.0
    return w
end

"""
    _nonneg_ls_weights(Z, y) -> Vector{Float64}

Coordinate-descent nonnegative least squares followed by normalisation to
simplex weights. Because normalising the unconstrained NNLS coefficients can
increase squared prediction loss, compare that blend with each single
candidate on the same predictions and return the best candidate if it has lower
loss. This package-specific safeguard is not equivalent to R
`SuperLearner::method.NNLS` or sl3 `Lrnr_nnls`.
"""
function _nonneg_ls_weights(Z::Matrix{Float64}, y::Vector{Float64})
    n, k = size(Z)
    k == 0 && return Float64[]
    if n == 0
        fallback = zeros(k)
        fallback[1] = 1.0
        return fallback
    end
    # Precompute k×k Gram matrix G = Z'Z, c = Z'y, and yy = y'y once so
    # coordinate descent runs in O(k²) per sweep with zero n-sized allocations.
    G = Matrix{Float64}(undef, k, k)
    mul!(G, transpose(Z), Z)
    c = Vector{Float64}(undef, k)
    mul!(c, transpose(Z), y)
    yy = dot(y, y)
    # Coordinate descent NNLS on min ||y - Z w||², w ≥ 0, then renormalise.
    w = fill(1 / k, k)
    @inbounds for _ in 1:200
        max_diff = 0.0
        for j in 1:k
            denom = G[j, j]
            denom < 1e-12 && continue
            gw = 0.0
            for l in 1:k
                gw += G[l, j] * w[l]
            end
            wj_old = w[j]
            wj_new = max(0.0, wj_old + (c[j] - gw) / denom)
            w[j] = wj_new
            diff = abs(wj_new - wj_old)
            diff > max_diff && (max_diff = diff)
        end
        max_diff <= 1e-12 && break
    end
    s = sum(w)
    normalised = s > 0 ? w ./ s : fill(1 / k, k)
    inv_n = 1.0 / n
    q_blend = 0.0
    @inbounds for j in 1:k
        col_dot = 0.0
        for l in 1:k
            col_dot += G[l, j] * normalised[l]
        end
        q_blend += normalised[j] * col_dot
    end
    blend_risk = (q_blend - 2.0 * dot(normalised, c) + yy) * inv_n
    best_candidate = 1
    best_risk = Inf
    @inbounds for j in 1:k
        r_j = (G[j, j] - 2.0 * c[j] + yy) * inv_n
        if j == 1 || isless(r_j, best_risk)
            best_risk = r_j
            best_candidate = j
        end
    end
    if !(blend_risk <= best_risk)
        fallback = zeros(k)
        fallback[best_candidate] = 1.0
        return fallback
    end
    return normalised
end

function _invmse_weights(Z::Matrix{Float64}, y::Vector{Float64})
    k = size(Z, 2)
    mse = [mean((Z[:, j] .- y) .^ 2) for j in 1:k]
    inv = [isfinite(m) && m > 1e-12 ? 1 / m : 0.0 for m in mse]
    s = sum(inv)
    return s > 0 ? inv ./ s : fill(1 / k, k)
end

function _validate_binary_outcome(y::AbstractVector{<:Real})
    all(isfinite, y) || throw(ArgumentError("NNloglik outcomes must be finite and equal to 0 or 1"))
    all(v -> v == 0 || v == 1, y) ||
        throw(ArgumentError("NNloglik requires a binary outcome containing only 0 and 1"))
    return nothing
end

"""
    validate_family_outcome(y, family) -> nothing

Check that `family` is a supported Super Learner outcome family and that
`:binomial` outcomes lie in ``\\{0, 1\\}``.
"""
function validate_family_outcome(y::AbstractVector, family::Symbol)
    family in (:gaussian, :binomial, :multinomial, :poisson, :negbin, :zeroinflated_nb) ||
        throw(ArgumentError(
        "family_outcome must be :gaussian, :binomial, :multinomial, :poisson, " *
        ":negbin, or :zeroinflated_nb; got $family",
    ))
    family === :binomial && _validate_binary_outcome(collect(Float64, y))
    family in COUNT_OUTCOME_FAMILIES && _validate_count_outcome(collect(Float64, y))
    return nothing
end

"""Return true when values look like nonnegative integer counts (not binary)."""
function _looks_like_count_vals(vals::AbstractVector{<:Real})
    isempty(vals) && return false
    all(v -> v >= 0, vals) || return false
    all(v -> abs(v - round(v)) < 1e-8, vals) || return false
    return !(maximum(vals) <= 1 && all(v -> v == 0 || v == 1, vals))
end

"""
    suggest_family_outcome(y) -> Symbol

Return `:binomial` when all non-missing values lie in ``\\{0, 1\\}``;
`:negbin` or `:poisson` for nonnegative integer counts (overdispersion heuristic);
otherwise `:gaussian`. Use with `recommend_run_options(...; outcome=)` or pass
the result as `family_outcome` on LMTP runners
([#34](https://github.com/SimonAB/CausalTargeted.jl/issues/34),
[#36](https://github.com/SimonAB/CausalTargeted.jl/issues/36)).
"""
function suggest_family_outcome(y::AbstractVector)
    vals = Float64[]
    for v in y
        ismissing(v) && continue
        push!(vals, Float64(v))
    end
    isempty(vals) && return :gaussian
    if maximum(vals) <= 1 && all(v -> v == 0.0 || v == 1.0, vals)
        return :binomial
    end
    if _looks_like_count_vals(vals)
        μ = mean(vals)
        σ² = var(vals; corrected = true)
        return σ² > μ * 1.25 + 1e-8 ? :negbin : :poisson
    end
    return :gaussian
end

"""Trim a probability matrix and return its elementwise logits."""
function _trim_logit_predictions(
    Z::AbstractMatrix{<:Real};
    trim::Real = NNLOGLIK_TRIM,
)
    0 < trim < 0.5 || throw(ArgumentError("trim must lie strictly between 0 and 0.5"))
    all(isfinite, Z) || throw(ArgumentError("NNloglik learner predictions must be finite"))
    all(p -> 0 <= p <= 1, Z) ||
        throw(ArgumentError("NNloglik learner predictions must lie in [0, 1]"))
    lower = Float64(trim)
    upper = 1.0 - lower
    Xlogit = Matrix{Float64}(undef, size(Z))
    @inbounds for idx in eachindex(Z)
        p = clamp(Float64(Z[idx]), lower, upper)
        Xlogit[idx] = log(p) - log1p(-p)
    end
    return Xlogit
end

@inline function _logistic(eta::Float64)
    if eta >= 0
        z = exp(-eta)
        return 1 / (1 + z)
    end
    z = exp(eta)
    return z / (1 + z)
end

@inline _log1pexp(x::Float64) = max(x, 0.0) + log1p(exp(-abs(x)))

"""Weighted mean Bernoulli negative log likelihood from a logit design."""
function _nnloglik_objective(
    beta::AbstractVector{<:Real},
    Xlogit::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    obs_weights::AbstractVector{<:Real},
)
    eta = Xlogit * beta
    total_weight = sum(obs_weights)
    loss = 0.0
    @inbounds for i in eachindex(y)
        loss += Float64(obs_weights[i]) *
                (_log1pexp(Float64(eta[i])) - Float64(y[i]) * Float64(eta[i]))
    end
    return loss / total_weight
end

"""Analytic gradient of `_nnloglik_objective`."""
function _nnloglik_gradient(
    beta::AbstractVector{<:Real},
    Xlogit::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    obs_weights::AbstractVector{<:Real},
)
    eta = Xlogit * beta
    residual = Vector{Float64}(undef, length(y))
    @inbounds for i in eachindex(y)
        residual[i] = Float64(obs_weights[i]) *
                      (_logistic(Float64(eta[i])) - Float64(y[i]))
    end
    return vec(transpose(Xlogit) * residual) ./ sum(obs_weights)
end

"""
    _nnloglik_fit(Z, y; obs_weights, trim, tol, maxiter)

Fit non-negative logit-combination coefficients with deterministic projected
gradient descent and Armijo backtracking. The returned `weights` are normalised
when the fitted coefficient sum is positive; `raw_weights` retain the optimiser
scale for numerical QC.
"""
function _nnloglik_fit(
    Z::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    obs_weights::AbstractVector{<:Real} = ones(length(y)),
    trim::Real = NNLOGLIK_TRIM,
    tol::Real = 1e-8,
    maxiter::Int = 10_000,
)
    n, k = size(Z)
    n == length(y) || throw(DimensionMismatch("prediction rows must match outcome length"))
    length(obs_weights) == n ||
        throw(DimensionMismatch("observation weights must match outcome length"))
    k > 0 || throw(ArgumentError("NNloglik requires at least one candidate learner"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    _validate_binary_outcome(y)
    all(isfinite, obs_weights) ||
        throw(ArgumentError("NNloglik observation weights must be finite"))
    all(w -> w >= 0, obs_weights) ||
        throw(ArgumentError("NNloglik observation weights must be non-negative"))
    sum(obs_weights) > 0 ||
        throw(ArgumentError("NNloglik observation weights must have a positive sum"))

    Xlogit = _trim_logit_predictions(Z; trim = trim)
    beta = fill(1 / k, k)
    objective = _nnloglik_objective(beta, Xlogit, y, obs_weights)
    isfinite(objective) || error("NNloglik optimisation started from a non-finite objective")
    gradient = _nnloglik_gradient(beta, Xlogit, y, obs_weights)
    previous_beta = copy(beta)
    previous_gradient = copy(gradient)
    step = 1.0
    converged = false
    iterations = 0

    for iteration in 1:maxiter
        iterations = iteration
        projected_gradient = beta .- max.(0.0, beta .- gradient)
        if maximum(abs, projected_gradient) <= tol * (1 + maximum(abs, beta))
            converged = true
            break
        end

        if iteration > 1
            delta_beta = beta .- previous_beta
            delta_gradient = gradient .- previous_gradient
            curvature = sum(delta_beta .* delta_gradient)
            if curvature > 0
                step = clamp(sum(abs2, delta_beta) / curvature, 1e-12, 1e6)
            else
                step = 1.0
            end
        end

        previous_beta .= beta
        previous_gradient .= gradient
        accepted = false
        trial_beta = similar(beta)
        trial_objective = objective
        for _ in 1:60
            trial_beta .= max.(0.0, beta .- step .* gradient)
            direction = trial_beta .- beta
            trial_objective = _nnloglik_objective(trial_beta, Xlogit, y, obs_weights)
            if isfinite(trial_objective) &&
               trial_objective <= objective + 1e-4 * sum(gradient .* direction)
                accepted = true
                break
            end
            step *= 0.5
        end
        accepted || error(
            "NNloglik optimisation failed to find a finite descent step at iteration $iteration",
        )

        beta .= trial_beta
        objective = trial_objective
        gradient = _nnloglik_gradient(beta, Xlogit, y, obs_weights)
        all(isfinite, gradient) ||
            error("NNloglik optimisation produced a non-finite gradient")
    end

    converged || error(
        "NNloglik optimisation did not converge within $maxiter iterations " *
        "(projected-gradient tolerance $tol)",
    )
    all(isfinite, beta) || error("NNloglik optimisation produced non-finite coefficients")
    beta .= max.(beta, 0.0)
    coefficient_sum = sum(beta)
    weights = coefficient_sum > 0 ? beta ./ coefficient_sum : copy(beta)
    return (
        weights = weights,
        raw_weights = beta,
        objective = objective,
        iterations = iterations,
        converged = converged,
    )
end

function _nnloglik_weights(
    Z::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    kwargs...,
)
    return _nnloglik_fit(Z, y; kwargs...).weights
end

"""Combine probability columns as a convex combination of their logits."""
function _predict_nnloglik(
    Z::AbstractMatrix{<:Real},
    weights::AbstractVector{<:Real};
    trim::Real = NNLOGLIK_TRIM,
)
    size(Z, 2) == length(weights) ||
        throw(DimensionMismatch("prediction columns must match NNloglik weights"))
    all(isfinite, weights) || throw(ArgumentError("NNloglik weights must be finite"))
    Xlogit = _trim_logit_predictions(Z; trim = trim)
    eta = Xlogit * weights
    return [_logistic(Float64(value)) for value in eta]
end
