# ---------------------------------------------------------------------------
# Internal numerical helpers, fold standardisation & convex calibration normaliser
# ---------------------------------------------------------------------------

function _logsumexp(x::AbstractVector{Float64})
    isempty(x) && return -Inf
    m = maximum(x)
    !isfinite(m) && return m
    s = 0.0
    @inbounds for xi in x
        s += exp(xi - m)
    end
    return m + log(s)
end

@inline _expit(x::Float64) = x >= 0.0 ? inv(1.0 + exp(-x)) : (ex = exp(x); ex / (1.0 + ex))

"""
Standardise non-constant columns of `W` and prepend an intercept column.
Constant columns (`sd <= 1e-12`) are omitted from the active basis to prevent
rank deficiency on small folds.
"""
function _fit_standardised_basis(W::AbstractMatrix{<:Real})
    n, d = size(W)
    active_cols = Int[]
    col_means = Float64[]
    col_sds = Float64[]
    for j in 1:d
        col = view(W, :, j)
        all(isfinite, col) || throw(ArgumentError("covariate matrix W contains non-finite values"))
        μ = mean(col)
        σ = std(col; corrected = false)
        if isfinite(σ) && σ > 1e-12
            push!(active_cols, j)
            push!(col_means, Float64(μ))
            push!(col_sds, Float64(σ))
        end
    end
    k = length(active_cols)
    Psi = Matrix{Float64}(undef, n, 1 + k)
    Psi[:, 1] .= 1.0
    @inbounds for (idx, j) in enumerate(active_cols)
        μ = col_means[idx]
        inv_σ = inv(col_sds[idx])
        for i in 1:n
            Psi[i, idx + 1] = (Float64(W[i, j]) - μ) * inv_σ
        end
    end
    return Psi, active_cols, col_means, col_sds
end

function _apply_standardised_basis(
    W::AbstractMatrix{<:Real},
    active_cols::Vector{Int},
    col_means::Vector{Float64},
    col_sds::Vector{Float64},
    n_raw_cols::Int,
)
    n, d = size(W)
    d == n_raw_cols || throw(DimensionMismatch(
        "covariate matrix width $d does not match fitted width $n_raw_cols",
    ))
    k = length(active_cols)
    Psi = Matrix{Float64}(undef, n, 1 + k)
    Psi[:, 1] .= 1.0
    @inbounds for (idx, j) in enumerate(active_cols)
        μ = col_means[idx]
        inv_σ = inv(col_sds[idx])
        for i in 1:n
            val = Float64(W[i, j])
            isfinite(val) || throw(ArgumentError("covariate matrix W contains non-finite values"))
            Psi[i, idx + 1] = (val - μ) * inv_σ
        end
    end
    return Psi
end

"""Solve `(H + τI) d = -g` via Cholesky with adaptive Levenberg-Marquardt damping."""
function _solve_damped_newton(H::Matrix{Float64}, g::Vector{Float64})
    p = length(g)
    F = cholesky(Symmetric(H); check = false)
    if issuccess(F)
        d = -(F \ g)
        if all(isfinite, d) && dot(g, d) < 0.0
            return d
        end
    end
    scale = max(1.0, tr(H) / p)
    τ = 1e-10 * scale
    H_damped = copy(H)
    for _ in 1:24
        @inbounds for j in 1:p
            H_damped[j, j] = H[j, j] + τ
        end
        F = cholesky(Symmetric(H_damped); check = false)
        if issuccess(F)
            d = -(F \ g)
            if all(isfinite, d) && dot(g, d) < 0.0
                return d
            end
        end
        τ *= 10.0
    end
    return -g
end

# ---------------------------------------------------------------------------
# Convex calibration-balanced normaliser estimator for log m_0(W) = β'ψ(W)
# ---------------------------------------------------------------------------

"""
    TiltNormalizerFit

Fitted convex calibration-balancing model for ``\\log m_0(W) = \\beta^\\top \\psi(W)``,
stored on a fold-standardised active covariate basis.
"""
struct TiltNormalizerFit
    beta::Vector{Float64}
    active_cols::Vector{Int}
    col_means::Vector{Float64}
    col_sds::Vector{Float64}
    n_raw_cols::Int
    delta::Float64
    policy::ExponentialTiltPolicy
    lambda::Float64
    converged::Bool
    iterations::Int
    grad_inf_norm::Float64
    objective::Float64
end

function _calibration_objective!(
    eta::Vector{Float64},
    beta::Vector{Float64},
    Psi::Matrix{Float64},
    log_w::AbstractVector{Float64},
    lambda::Float64,
)
    n = length(log_w)
    mul!(eta, Psi, beta)
    s = 0.0
    @inbounds for i in 1:n
        arg = log_w[i] - eta[i]
        arg > 500.0 && return Inf
        s += exp(arg) + eta[i]
    end
    val = s / n
    if lambda > 0.0 && length(beta) > 1
        pen = 0.0
        @inbounds for j in 2:length(beta)
            pen += beta[j]^2
        end
        val += 0.5 * lambda * pen
    end
    return val
end

function _calibration_objective(
    beta::Vector{Float64},
    Psi::Matrix{Float64},
    log_w::AbstractVector{Float64},
    lambda::Float64,
)
    return _calibration_objective!(
        Vector{Float64}(undef, length(log_w)), beta, Psi, log_w, lambda,
    )
end

function _calibration_gradient!(
    g::Vector{Float64},
    rhs::Vector{Float64},
    Psi::Matrix{Float64},
    eta::Vector{Float64},
    beta::Vector{Float64},
    log_w::AbstractVector{Float64},
    lambda::Float64;
    h::Union{Nothing, Vector{Float64}} = nothing,
)
    n, p = size(Psi)
    @inbounds for i in 1:n
        hi = exp(log_w[i] - eta[i])
        h !== nothing && (h[i] = hi)
        rhs[i] = 1.0 - hi
    end
    mul!(g, Psi', rhs)
    g ./= n
    if lambda > 0.0
        @inbounds for j in 2:p
            g[j] += lambda * beta[j]
        end
    end
    return maximum(abs, g)
end

function _fit_tilt_normalizer_from_basis(
    Psi::Matrix{Float64},
    active_cols::Vector{Int},
    col_means::Vector{Float64},
    col_sds::Vector{Float64},
    n_raw_cols::Int,
    log_w::AbstractVector{Float64},
    δ::Float64,
    policy::ExponentialTiltPolicy;
    lambda::Float64 = 0.0,
    tol::Float64 = 1e-11,
    maxiter::Int = 100,
)
    n, p = size(Psi)

    # Trivial tilt δ = 0: exact optimum is β = 0 (m_0(W) ≡ 1).
    if iszero(δ)
        beta = zeros(Float64, p)
        return TiltNormalizerFit(
            beta, active_cols, col_means, col_sds, n_raw_cols, δ, policy, lambda,
            true, 0, 0.0, 1.0,
        )
    end

    # Intercept-only closed form: β₁ = log(mean(w_δ(A))).
    beta = zeros(Float64, p)
    beta[1] = _logsumexp(log_w) - log(n)
    if p == 1
        obj = 1.0 + beta[1]
        return TiltNormalizerFit(
            beta, active_cols, col_means, col_sds, n_raw_cols, δ, policy, lambda,
            true, 0, 0.0, obj,
        )
    end

    eta = Vector{Float64}(undef, n)
    obj = _calibration_objective!(eta, beta, Psi, log_w, lambda)
    # If initial log-weights are extreme, shift intercept slightly until finite.
    if !isfinite(obj)
        beta[1] = maximum(log_w)
        obj = _calibration_objective!(eta, beta, Psi, log_w, lambda)
    end

    converged = false
    iters = 0
    gnorm = Inf
    h = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, n)
    g = Vector{Float64}(undef, p)
    g_try = Vector{Float64}(undef, p)
    Psi_w = Matrix{Float64}(undef, n, p)
    Hmat = Matrix{Float64}(undef, p, p)
    beta_try = similar(beta)

    for iter in 1:maxiter
        iters = iter
        mul!(eta, Psi, beta)
        gnorm = _calibration_gradient!(g, rhs, Psi, eta, beta, log_w, lambda; h = h)
        if gnorm <= tol
            converged = true
            break
        end

        # Hessian: (1/n) Ψ' diag(h) Ψ + λ D (reusing preallocated Psi_w buffer)
        @inbounds for j in 1:p
            for i in 1:n
                Psi_w[i, j] = h[i] * Psi[i, j]
            end
        end
        mul!(Hmat, Psi', Psi_w)
        Hmat ./= n
        if lambda > 0.0
            @inbounds for j in 2:p
                Hmat[j, j] += lambda
            end
        end

        dir = _solve_damped_newton(Hmat, g)
        slope = dot(g, dir)

        # Armijo backtracking line search (with near-machine-epsilon gradient-decrease acceptance
        # so the final quadratic Newton step is not rejected by Float64 rounding in the objective)
        step = 1.0
        accepted = false
        c1 = 1e-4
        obj_eps = 1e-14 * max(1.0, abs(obj))
        while step >= 1e-12
            @. beta_try = beta + step * dir
            obj_try = _calibration_objective!(eta, beta_try, Psi, log_w, lambda)
            if isfinite(obj_try) && obj_try <= obj + c1 * step * slope
                copyto!(beta, beta_try)
                obj = obj_try
                accepted = true
                break
            elseif step == 1.0 && isfinite(obj_try) && obj_try <= obj + obj_eps
                if _calibration_gradient!(g_try, rhs, Psi, eta, beta_try, log_w, lambda) < gnorm
                    copyto!(beta, beta_try)
                    obj = obj_try
                    accepted = true
                    break
                end
            end
            step *= 0.5
        end
        if !accepted
            # Re-check gradient norm; if already tight, mark converged.
            converged = gnorm <= max(tol, 1e-8)
            break
        end
    end

    # Final gradient check if loop exhausted on an accepted step
    if !converged
        mul!(eta, Psi, beta)
        gnorm = _calibration_gradient!(g, rhs, Psi, eta, beta, log_w, lambda)
        converged = gnorm <= tol
    end

    return TiltNormalizerFit(
        beta, active_cols, col_means, col_sds, n_raw_cols, δ, policy, lambda,
        converged, iters, gnorm, obj,
    )
end

"""
    fit_tilt_normalizer(W, a, delta, policy; lambda = 0.0, tol = 1e-11, maxiter = 100) -> TiltNormalizerFit

Minimise the strictly convex calibration-balancing loss:

``L(\\beta) = \\frac{1}{n} \\sum_{i=1}^n \\left(w_\\delta(A_i) \\exp(-\\beta^\\top \\psi(W_i)) + \\beta^\\top \\psi(W_i)\\right) + \\frac{\\lambda}{2} \\|\\beta_{-1}\\|_2^2``

via damped Newton–Raphson with Armijo backtracking line search on the standardised
covariate basis ``\\psi(W) = [1, \\tilde{W}]``. At `lambda = 0.0`, the stationary point
satisfies exact finite-sample balance ``\\frac{1}{n}\\sum_i H_i \\psi(W_i) = \\frac{1}{n}\\sum_i \\psi(W_i)``,
including ``\\bar{H} = 1``.
"""
function fit_tilt_normalizer(
    W::AbstractMatrix{<:Real},
    a::AbstractVector{<:Real},
    delta::Real,
    policy::ExponentialTiltPolicy;
    lambda::Real = 0.0,
    tol::Real = 1e-11,
    maxiter::Int = 100,
)
    n, d = size(W)
    n > 0 || throw(ArgumentError("fit_tilt_normalizer requires at least one observation"))
    length(a) == n || throw(DimensionMismatch(
        "exposure length $(length(a)) does not match W rows $n",
    ))
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("delta must be finite; got $delta"))
    λ = Float64(lambda)
    (isfinite(λ) && λ >= 0.0) || throw(ArgumentError(
        "ridge penalty lambda must be finite and >= 0; got $lambda",
    ))

    Psi, active_cols, col_means, col_sds = _fit_standardised_basis(W)
    log_w = tilt_log_weight(a, δ, policy)
    return _fit_tilt_normalizer_from_basis(
        Psi, active_cols, col_means, col_sds, d, log_w, δ, policy;
        lambda = λ, tol = Float64(tol), maxiter = maxiter,
    )
end

function _predict_tilt_weights_from_basis(
    fit::TiltNormalizerFit,
    Psi_te::Matrix{Float64},
    log_w_te::AbstractVector{Float64};
    enforce_bounds::Bool = true,
)
    eta = Psi_te * fit.beta
    if enforce_bounds && fit.policy.bound !== nothing
        lo, hi = tilt_normalizer_log_bounds(fit.delta, fit.policy)
        @. eta = clamp(eta, lo, hi)
    end
    n = length(log_w_te)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = exp(log_w_te[i] - eta[i])
    end
    return out
end

"""
    predict_log_normalizer(fit::TiltNormalizerFit, W; enforce_bounds = true) -> Vector{Float64}

Predict ``\\log \\hat{m}_0(W)``. When `enforce_bounds = true` and `fit.policy.bound !== nothing`,
predictions are projected onto the exact theoretical range ``[-|\\delta|M, |\\delta|M]`` of
``\\log \\mathbb{E}[w_\\delta(A) \\mid W]``.
"""
function predict_log_normalizer(
    fit::TiltNormalizerFit,
    W::AbstractMatrix{<:Real};
    enforce_bounds::Bool = true,
)
    Psi = _apply_standardised_basis(
        W, fit.active_cols, fit.col_means, fit.col_sds, fit.n_raw_cols,
    )
    eta = Psi * fit.beta
    if enforce_bounds && fit.policy.bound !== nothing
        lo, hi = tilt_normalizer_log_bounds(fit.delta, fit.policy)
        @. eta = clamp(eta, lo, hi)
    end
    return eta
end

"""
    predict_tilt_weights(fit::TiltNormalizerFit, W, a; enforce_bounds = true) -> Vector{Float64}

Evaluate estimated density ratios ``\\hat{H}_\\delta(A_i, W_i) = w_\\delta(A_i) / \\hat{m}_0(W_i)``.
When `enforce_bounds = true` and `fit.policy.bound = M`, every weight satisfies
``\\exp(-2|\\delta|M) \\le \\hat{H}_i \\le \\exp(2|\\delta|M)`` strictly.
"""
function predict_tilt_weights(
    fit::TiltNormalizerFit,
    W::AbstractMatrix{<:Real},
    a::AbstractVector{<:Real};
    enforce_bounds::Bool = true,
)
    length(a) == size(W, 1) || throw(DimensionMismatch(
        "exposure length $(length(a)) does not match W rows $(size(W, 1))",
    ))
    log_w = tilt_log_weight(a, fit.delta, fit.policy)
    log_m0 = predict_log_normalizer(fit, W; enforce_bounds = enforce_bounds)
    return exp.(log_w .- log_m0)
end
