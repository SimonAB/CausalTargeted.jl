"""Bounded exponential-tilting stochastic interventions with calibration-balanced TMLE.

Defines exponential-tilting policies ``g_\\delta(a \\mid W) = w_\\delta(a) g(a \\mid W) / m_0(W)``
with ``w_\\delta(a) = \\exp(\\delta \\phi(a))``, where the tilt score ``\\phi(a)`` is either
linear (``a - c``) or smoothly bounded via ``M \\tanh((a - c) / M)``. When bounded by ``M``,
the true density ratio satisfies ``\\exp(-2|\\delta|M) \\le H_\\delta(A, W) \\le \\exp(2|\\delta|M)``
almost surely without ad-hoc weight truncation.

The normaliser ``\\log m_0(W) = \\beta^\\top \\psi(W)`` is estimated via a strictly convex
calibration-balancing objective that enforces exact finite-sample covariate balance
``\\frac{1}{n}\\sum_i H_i \\psi(W_i) = \\frac{1}{n}\\sum_i \\psi(W_i)`` at ``\\lambda = 0``.
Targeting reuses `CausalTargeted._weighted_intercept_tilt`, yielding canonical-link TMLE
estimates and dual-axis EIF inference for both the outcome contrast ``\\tau(\\delta)`` and
the realised mean exposure shift ``\\Delta(\\delta)``.

# References

- Díaz & Hejazi (2020), *JRSS-B* — incremental and exponential-tilting stochastic interventions
- Schindl, Shen & Kennedy (2024) — bounded exponential-tilting interventions
- Tan (2020), *Biometrika* — regularised calibrated estimation of propensity/normaliser weights
- Mammen (1993), *Annals of Statistics* — wild / multiplier bootstrap for skewed functionals
"""

using DataFrames
using Distributions
using GLM
using LinearAlgebra
using Random
using StableRNGs
using Statistics

# ---------------------------------------------------------------------------
# 1. Policy struct, tilt score, weights, and analytical support bounds
# ---------------------------------------------------------------------------

"""
    ExponentialTiltPolicy(; bound = nothing, center = 0.0)

Stochastic exponential-tilting intervention policy with score function:

- Unbounded (`bound = nothing`): ``\\phi(a) = a - c``
- Bounded (`bound = M > 0`): ``\\phi(a) = M \\tanh((a - c) / M)``

The tilt weight is ``w_\\delta(a) = \\exp(\\delta \\phi(a))``. For `bound = M`,
``|\\phi(a)| \\le M``, so ``\\exp(-2|\\delta|M) \\le w_\\delta(A) / m_0(W) \\le \\exp(2|\\delta|M)``
almost surely without ad-hoc truncation.
"""
struct ExponentialTiltPolicy
    bound::Union{Nothing, Float64}
    center::Float64

    function ExponentialTiltPolicy(bound::Union{Nothing, Real}, center::Real)
        c = Float64(center)
        isfinite(c) || throw(ArgumentError(
            "exponential tilt policy requires a finite center; got $center",
        ))
        if bound === nothing
            return new(nothing, c)
        end
        b = Float64(bound)
        (isfinite(b) && b > 0) || throw(ArgumentError(
            "exponential tilt policy bound must be nothing or finite and > 0; got $bound",
        ))
        return new(b, c)
    end
end

ExponentialTiltPolicy(; bound::Union{Nothing, Real} = nothing, center::Real = 0.0) =
    ExponentialTiltPolicy(bound, center)

"""
    exponential_tilt_policy(; bound = nothing, center = 0.0) -> ExponentialTiltPolicy

Construct an [`ExponentialTiltPolicy`](@ref). Set `bound = M > 0` for the smooth
bounded tilt ``\\phi(a) = M \\tanh((a - c) / M)`` or leave `bound = nothing` for the
linear tilt ``\\phi(a) = a - c``.
"""
exponential_tilt_policy(; bound::Union{Nothing, Real} = nothing, center::Real = 0.0) =
    ExponentialTiltPolicy(bound, center)

"""
    tilt_score(a, policy::ExponentialTiltPolicy)

Evaluate the tilt score ``\\phi(a)`` at a scalar or vector of finite exposures.
"""
function tilt_score(a::Real, policy::ExponentialTiltPolicy)
    x = Float64(a)
    isfinite(x) || throw(ArgumentError("tilt_score requires finite exposure values; got $a"))
    d = x - policy.center
    return policy.bound === nothing ? d : policy.bound * tanh(d / policy.bound)
end

function tilt_score(a::AbstractVector{<:Real}, policy::ExponentialTiltPolicy)
    out = Vector{Float64}(undef, length(a))
    @inbounds for i in eachindex(a)
        out[i] = tilt_score(a[i], policy)
    end
    return out
end

"""
    tilt_log_weight(a, delta, policy::ExponentialTiltPolicy)

Compute ``\\log w_\\delta(a) = \\delta \\phi(a)`` for a finite tilt strength `delta`.
"""
function tilt_log_weight(a::Real, delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    return δ * tilt_score(a, policy)
end

function tilt_log_weight(a::AbstractVector{<:Real}, delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    scores = tilt_score(a, policy)
    return δ .* scores
end

"""
    tilt_weight(a, delta, policy::ExponentialTiltPolicy)

Compute the unnormalised tilt weight ``w_\\delta(a) = \\exp(\\delta \\phi(a))``.
"""
tilt_weight(a::Real, delta::Real, policy::ExponentialTiltPolicy) =
    exp(tilt_log_weight(a, delta, policy))

tilt_weight(a::AbstractVector{<:Real}, delta::Real, policy::ExponentialTiltPolicy) =
    exp.(tilt_log_weight(a, delta, policy))

"""
    tilt_normalizer_log_bounds(delta, policy::ExponentialTiltPolicy) -> (lo, hi)

Exact theoretical bounds ``[-|\\delta|M, |\\delta|M]`` on ``\\log m_0(W) = \\log \\mathbb{E}[w_\\delta(A) \\mid W]``
for a bounded policy (`(-Inf, Inf)` when unbounded).
"""
function tilt_normalizer_log_bounds(delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    policy.bound === nothing && return (-Inf, Inf)
    lim = abs(δ) * policy.bound
    return (-lim, lim)
end

"""
    tilt_weight_bounds(delta, policy::ExponentialTiltPolicy) -> (lo, hi)

Exact theoretical bounds ``[\\exp(-2|\\delta|M), \\exp(2|\\delta|M)]`` on the density ratio
``H_\\delta(A, W) = w_\\delta(A) / m_0(W)`` for a bounded policy (`(0.0, Inf)` when unbounded).
"""
function tilt_weight_bounds(delta::Real, policy::ExponentialTiltPolicy)
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("exponential tilt delta must be finite; got $delta"))
    policy.bound === nothing && return (0.0, Inf)
    lim = 2.0 * abs(δ) * policy.bound
    return (exp(-lim), exp(lim))
end

# ---------------------------------------------------------------------------
# 2. Internal numerical helpers & fold standardisation
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
# 3. Convex calibration-balanced normaliser estimator for log m_0(W) = β'ψ(W)
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

function _tilt_complement_indices(n::Int, test_idx::AbstractVector{Int})
    mask = trues(n)
    @inbounds for idx in test_idx
        mask[idx] = false
    end
    out = Vector{Int}(undef, count(mask))
    k = 0
    @inbounds for i in 1:n
        if mask[i]
            k += 1
            out[k] = i
        end
    end
    return out
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
        @inbounds for i in 1:n
            hi = exp(log_w[i] - eta[i])
            h[i] = hi
            rhs[i] = 1.0 - hi
        end

        # Gradient: (1/n) Ψ'(1 - h) + λ D β
        mul!(g, Psi', rhs)
        g ./= n
        if lambda > 0.0
            @inbounds for j in 2:p
                g[j] += lambda * beta[j]
            end
        end
        gnorm = maximum(abs, g)
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
                @inbounds for i in 1:n
                    rhs[i] = 1.0 - exp(log_w[i] - eta[i])
                end
                mul!(g_try, Psi', rhs)
                g_try ./= n
                if lambda > 0.0
                    @inbounds for j in 2:p
                        g_try[j] += lambda * beta_try[j]
                    end
                end
                if maximum(abs, g_try) < gnorm
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
        @inbounds for i in 1:n
            rhs[i] = 1.0 - exp(log_w[i] - eta[i])
        end
        mul!(g, Psi', rhs)
        g ./= n
        if lambda > 0.0
            @inbounds for j in 2:p
                g[j] += lambda * beta[j]
            end
        end
        gnorm = maximum(abs, g)
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

# ---------------------------------------------------------------------------
# 4. Weighted canonical-link GLM & cross-fitted outcome nuisance θ_δ(W)
# ---------------------------------------------------------------------------

"""
Weighted OLS on standardised design matrix `Psi` (intercept in column 1), with tiny
non-intercept ridge guard if rank-deficient.
"""
function _fit_weighted_gaussian_glm(
    Psi::Matrix{Float64},
    y::AbstractVector{Float64},
    w::Vector{Float64};
    ridge::Float64 = 1e-8,
)
    n, p = size(Psi)
    wn = w ./ mean(w)
    if p == 1
        return [sum(wn .* y) / sum(wn)]
    end
    XtW = Psi' .* wn'
    A = XtW * Psi
    b = XtW * y
    F = cholesky(Symmetric(A); check = false)
    if issuccess(F)
        coef = F \ b
        all(isfinite, coef) && return coef
    end
    @inbounds for j in 2:p
        A[j, j] += ridge * n
    end
    F = cholesky(Symmetric(A); check = false)
    if issuccess(F)
        coef = F \ b
        all(isfinite, coef) && return coef
    end
    return vcat(sum(wn .* y) / sum(wn), zeros(Float64, p - 1))
end

function _binomial_neg_loglik!(
    eta::Vector{Float64},
    Psi::Matrix{Float64},
    gvec::Vector{Float64},
    yc::Vector{Float64},
    wn::Vector{Float64},
    ridge::Float64,
)
    n = length(yc)
    p = length(gvec)
    mul!(eta, Psi, gvec)
    s = 0.0
    @inbounds for i in 1:n
        ei = eta[i]
        # Stable log(1 + exp(ei)) - yc[i] * ei
        l1p = ei >= 0.0 ? ei + log1p(exp(-ei)) : log1p(exp(ei))
        s += wn[i] * (l1p - yc[i] * ei)
    end
    val = s / n
    if ridge > 0.0
        pen = 0.0
        @inbounds for j in 2:p
            pen += gvec[j]^2
        end
        val += 0.5 * ridge * pen
    end
    return val
end

"""
Weighted canonical-link logistic regression maximising
`mean(wn * (y * η - log(1 + exp(η)))) - (ridge/2) * ||γ_{-1}||^2`.
Accepts binary `y ∈ {0, 1}` as well as continuous first-stage probabilities `y ∈ (0, 1)`.
"""
function _fit_weighted_binomial_glm(
    Psi::Matrix{Float64},
    y::AbstractVector{Float64},
    w::Vector{Float64};
    ridge::Float64 = 1e-6,
    tol::Float64 = 1e-10,
    maxiter::Int = 60,
)
    n, p = size(Psi)
    wn = w ./ mean(w)
    yc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        yc[i] = clamp(y[i], 0.0, 1.0)
    end
    ybar = clamp(dot(wn, yc) / sum(wn), 1e-6, 1.0 - 1e-6)
    gamma = zeros(Float64, p)
    gamma[1] = log(ybar / (1.0 - ybar))
    p == 1 && return gamma

    eta = Vector{Float64}(undef, n)
    wt = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, n)
    grad = Vector{Float64}(undef, p)
    grad_try = Vector{Float64}(undef, p)
    Psi_w = Matrix{Float64}(undef, n, p)
    Hmat = Matrix{Float64}(undef, p, p)
    gamma_try = similar(gamma)

    obj = _binomial_neg_loglik!(eta, Psi, gamma, yc, wn, ridge)

    for _ in 1:maxiter
        mul!(eta, Psi, gamma)
        @inbounds for i in 1:n
            mi = _expit(eta[i])
            rhs[i] = wn[i] * (mi - yc[i])
            wt[i] = wn[i] * max(mi * (1.0 - mi), 1e-12)
        end
        mul!(grad, Psi', rhs)
        grad ./= n
        if ridge > 0.0
            @inbounds for j in 2:p
                grad[j] += ridge * gamma[j]
            end
        end
        gnorm = maximum(abs, grad)
        gnorm <= tol && break

        @inbounds for j in 1:p
            for i in 1:n
                Psi_w[i, j] = wt[i] * Psi[i, j]
            end
        end
        mul!(Hmat, Psi', Psi_w)
        Hmat ./= n
        if ridge > 0.0
            @inbounds for j in 2:p
                Hmat[j, j] += ridge
            end
        end
        dir = _solve_damped_newton(Hmat, grad)
        slope = dot(grad, dir)
        step = 1.0
        accepted = false
        obj_eps = 1e-14 * max(1.0, abs(obj))
        while step >= 1e-10
            @. gamma_try = gamma + step * dir
            obj_try = _binomial_neg_loglik!(eta, Psi, gamma_try, yc, wn, ridge)
            if isfinite(obj_try) && obj_try <= obj + 1e-4 * step * slope
                copyto!(gamma, gamma_try)
                obj = obj_try
                accepted = true
                break
            elseif step == 1.0 && isfinite(obj_try) && obj_try <= obj + obj_eps
                @inbounds for i in 1:n
                    rhs[i] = wn[i] * (_expit(eta[i]) - yc[i])
                end
                mul!(grad_try, Psi', rhs)
                grad_try ./= n
                if ridge > 0.0
                    @inbounds for j in 2:p
                        grad_try[j] += ridge * gamma_try[j]
                    end
                end
                if maximum(abs, grad_try) < gnorm
                    copyto!(gamma, gamma_try)
                    obj = obj_try
                    accepted = true
                    break
                end
            end
            step *= 0.5
        end
        !accepted && break
    end
    return gamma
end

function _poisson_neg_loglik!(
    eta::Vector{Float64},
    Psi::Matrix{Float64},
    gvec::Vector{Float64},
    ys::Vector{Float64},
    wn::Vector{Float64},
    ridge::Float64,
)
    n = length(ys)
    p = length(gvec)
    mul!(eta, Psi, gvec)
    s = 0.0
    @inbounds for i in 1:n
        ei = eta[i]
        ei > 50.0 && return Inf
        s += wn[i] * (exp(ei) - ys[i] * ei)
    end
    val = s / n
    if ridge > 0.0
        pen = 0.0
        @inbounds for j in 2:p
            pen += gvec[j]^2
        end
        val += 0.5 * ridge * pen
    end
    return val
end

"""
Weighted canonical log-link Poisson quasi-likelihood maximising
`mean(wn * (y * η - exp(η))) - (ridge/2) * ||γ_{-1}||^2`.
Accepts arbitrary nonnegative real targets (counts or positive stage-1 predictions).
"""
function _fit_weighted_poisson_glm(
    Psi::Matrix{Float64},
    y::AbstractVector{Float64},
    w::Vector{Float64};
    ridge::Float64 = 1e-6,
    tol::Float64 = 1e-10,
    maxiter::Int = 60,
)
    n, p = size(Psi)
    wn = w ./ mean(w)
    yc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        yc[i] = max(y[i], 0.0)
    end
    ybar = dot(wn, yc) / sum(wn)
    if !(isfinite(ybar) && ybar > 0.0)
        return vcat(log(1e-8), zeros(Float64, p - 1))
    end
    # Scale targets by ybar so intercept starts at 0 and ridge is scale-invariant.
    ys = yc ./ ybar
    gamma = zeros(Float64, p)
    p == 1 && return [log(ybar)]

    eta = Vector{Float64}(undef, n)
    wt = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, n)
    grad = Vector{Float64}(undef, p)
    grad_try = Vector{Float64}(undef, p)
    Psi_w = Matrix{Float64}(undef, n, p)
    Hmat = Matrix{Float64}(undef, p, p)
    gamma_try = similar(gamma)

    obj = _poisson_neg_loglik!(eta, Psi, gamma, ys, wn, ridge)

    for _ in 1:maxiter
        mul!(eta, Psi, gamma)
        @inbounds for i in 1:n
            mi = exp(clamp(eta[i], -50.0, 50.0))
            rhs[i] = wn[i] * (mi - ys[i])
            wt[i] = wn[i] * mi
        end
        mul!(grad, Psi', rhs)
        grad ./= n
        if ridge > 0.0
            @inbounds for j in 2:p
                grad[j] += ridge * gamma[j]
            end
        end
        gnorm = maximum(abs, grad)
        gnorm <= tol && break

        @inbounds for j in 1:p
            for i in 1:n
                Psi_w[i, j] = wt[i] * Psi[i, j]
            end
        end
        mul!(Hmat, Psi', Psi_w)
        Hmat ./= n
        if ridge > 0.0
            @inbounds for j in 2:p
                Hmat[j, j] += ridge
            end
        end
        dir = _solve_damped_newton(Hmat, grad)
        slope = dot(grad, dir)
        step = 1.0
        accepted = false
        obj_eps = 1e-14 * max(1.0, abs(obj))
        while step >= 1e-10
            @. gamma_try = gamma + step * dir
            obj_try = _poisson_neg_loglik!(eta, Psi, gamma_try, ys, wn, ridge)
            if isfinite(obj_try) && obj_try <= obj + 1e-4 * step * slope
                copyto!(gamma, gamma_try)
                obj = obj_try
                accepted = true
                break
            elseif step == 1.0 && isfinite(obj_try) && obj_try <= obj + obj_eps
                @inbounds for i in 1:n
                    mi = exp(clamp(eta[i], -50.0, 50.0))
                    rhs[i] = wn[i] * (mi - ys[i])
                end
                mul!(grad_try, Psi', rhs)
                grad_try ./= n
                if ridge > 0.0
                    @inbounds for j in 2:p
                        grad_try[j] += ridge * gamma_try[j]
                    end
                end
                if maximum(abs, grad_try) < gnorm
                    copyto!(gamma, gamma_try)
                    obj = obj_try
                    accepted = true
                    break
                end
            end
            step *= 0.5
        end
        !accepted && break
    end
    gamma[1] += log(ybar)
    return gamma
end

"""
Weighted log-link Negative Binomial GLM (profiled over dispersion grid on `wn`-weighted
log-likelihood), falling back to weighted Poisson quasi-likelihood if GLM fails.
"""
function _fit_weighted_negbin_glm(
    Psi::Matrix{Float64},
    y::AbstractVector{Float64},
    w::Vector{Float64};
    ridge::Float64 = 1e-6,
)
    n, p = size(Psi)
    wn = w ./ mean(w)
    yc = max.(y, 0.0)
    ybar = sum(wn .* yc) / sum(wn)
    if !(isfinite(ybar) && ybar > 0.0)
        return vcat(log(1e-8), zeros(Float64, p - 1))
    end
    p == 1 && return [log(ybar)]

    best_coef = nothing
    best_ll = -Inf
    for θ in CausalTargeted.DEFAULT_NB_THETA_GRID
        try
            m = GLM.glm(Psi, yc, NegativeBinomial(Float64(θ)), LogLink(); wts = wn)
            c = Float64.(GLM.coef(m))
            all(isfinite, c) || continue
            eta = Psi * c
            any(x -> !isfinite(x) || x > 50.0 || x < -50.0, eta) && continue
            ll = 0.0
            @inbounds for i in 1:n
                μi = exp(eta[i])
                # Weighted NB2 log-likelihood up to constants independent of (c, θ)
                ll += wn[i] * logpdf(NegativeBinomial(Float64(θ), θ / (θ + μi)), round(Int, yc[i]))
            end
            if isfinite(ll) && ll > best_ll
                best_ll = ll
                best_coef = c
            end
        catch
            continue
        end
    end
    if best_coef !== nothing
        return best_coef
    end
    return _fit_weighted_poisson_glm(Psi, yc, wn; ridge = ridge)
end

function _gamma_log_neg_loglik!(
    eta::Vector{Float64},
    Psi::Matrix{Float64},
    gvec::Vector{Float64},
    ys::Vector{Float64},
    wn::Vector{Float64},
    ridge::Float64,
)
    n = length(ys)
    p = length(gvec)
    mul!(eta, Psi, gvec)
    s = 0.0
    @inbounds for i in 1:n
        ei = eta[i]
        (ei < -50.0 || ei > 50.0) && return Inf
        s += wn[i] * (ys[i] * exp(-ei) + ei)
    end
    val = s / n
    if ridge > 0.0
        pen = 0.0
        @inbounds for j in 2:p
            pen += gvec[j]^2
        end
        val += 0.5 * ridge * pen
    end
    return val
end

"""
Weighted log-link Gamma quasi-likelihood maximising
`mean(wn * (-y * exp(-η) - η)) - (ridge/2) * ||γ_{-1}||^2`.
Well-suited for Stage-2 projection of positive predicted means `μ̂(A, W)` with roughly
constant coefficient of variation.
"""
function _fit_weighted_gamma_log_glm(
    Psi::Matrix{Float64},
    y::AbstractVector{Float64},
    w::Vector{Float64};
    ridge::Float64 = 1e-6,
    tol::Float64 = 1e-10,
    maxiter::Int = 60,
)
    n, p = size(Psi)
    wn = w ./ mean(w)
    yc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        yc[i] = max(y[i], 1e-8)
    end
    ybar = dot(wn, yc) / sum(wn)
    if !(isfinite(ybar) && ybar > 0.0)
        return vcat(log(1e-8), zeros(Float64, p - 1))
    end
    ys = yc ./ ybar
    gamma = zeros(Float64, p)
    p == 1 && return [log(ybar)]

    eta = Vector{Float64}(undef, n)
    wt = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, n)
    grad = Vector{Float64}(undef, p)
    grad_try = Vector{Float64}(undef, p)
    Psi_w = Matrix{Float64}(undef, n, p)
    Hmat = Matrix{Float64}(undef, p, p)
    gamma_try = similar(gamma)

    obj = _gamma_log_neg_loglik!(eta, Psi, gamma, ys, wn, ridge)

    for _ in 1:maxiter
        mul!(eta, Psi, gamma)
        @inbounds for i in 1:n
            ri = ys[i] * exp(-clamp(eta[i], -50.0, 50.0))
            rhs[i] = wn[i] * (1.0 - ri)
            wt[i] = wn[i] * ri
        end
        mul!(grad, Psi', rhs)
        grad ./= n
        if ridge > 0.0
            @inbounds for j in 2:p
                grad[j] += ridge * gamma[j]
            end
        end
        gnorm = maximum(abs, grad)
        gnorm <= tol && break

        @inbounds for j in 1:p
            for i in 1:n
                Psi_w[i, j] = wt[i] * Psi[i, j]
            end
        end
        mul!(Hmat, Psi', Psi_w)
        Hmat ./= n
        if ridge > 0.0
            @inbounds for j in 2:p
                Hmat[j, j] += ridge
            end
        end
        dir = _solve_damped_newton(Hmat, grad)
        slope = dot(grad, dir)
        step = 1.0
        accepted = false
        obj_eps = 1e-14 * max(1.0, abs(obj))
        while step >= 1e-10
            @. gamma_try = gamma + step * dir
            obj_try = _gamma_log_neg_loglik!(eta, Psi, gamma_try, ys, wn, ridge)
            if isfinite(obj_try) && obj_try <= obj + 1e-4 * step * slope
                copyto!(gamma, gamma_try)
                obj = obj_try
                accepted = true
                break
            elseif step == 1.0 && isfinite(obj_try) && obj_try <= obj + obj_eps
                @inbounds for i in 1:n
                    ri = ys[i] * exp(-clamp(eta[i], -50.0, 50.0))
                    rhs[i] = wn[i] * (1.0 - ri)
                end
                mul!(grad_try, Psi', rhs)
                grad_try ./= n
                if ridge > 0.0
                    @inbounds for j in 2:p
                        grad_try[j] += ridge * gamma_try[j]
                    end
                end
                if maximum(abs, grad_try) < gnorm
                    copyto!(gamma, gamma_try)
                    obj = obj_try
                    accepted = true
                    break
                end
            end
            step *= 0.5
        end
        !accepted && break
    end
    gamma[1] += log(ybar)
    return gamma
end

function _predict_canonical_glm(
    Psi::Matrix{Float64},
    gamma::Vector{Float64},
    family::Symbol,
)
    eta = Psi * gamma
    if family === :gaussian
        return eta
    elseif family === :binomial
        return clamp.(_expit.(eta), 1e-8, 1.0 - 1e-8)
    elseif family in (:poisson, :negbin)
        return exp.(clamp.(eta, -30.0, 30.0))
    end
    throw(ArgumentError("unsupported family_outcome=$family for exponential tilt"))
end

"""
Fit fold outcome nuisance ``\\hat{\\theta}_\\delta(W)`` on `(W_tr, a_tr, y_tr)` and predict
on `W_te`. Supports `outcome_method`:
- `:plugin` (default) — fit unweighted Stage-1 GLM ``\\hat{\\mu}(A, W)`` on `[1, A, W]`,
  then project ``\\hat{\\mu}(A_i, W_i)`` onto `W` with weights ``w_\\delta(A_i)`` via
  canonical-link GLM (for `:negbin`, Stage-2 uses Gamma log-link on positive ``\\hat{\\mu}``
  to match constant coefficient of variation). Rao-Blackwellising out ``\\mathrm{Var}(Y \\mid A, W)``
  before tilt weighting substantially lowers RMSE and improves coverage on overdispersed
  count outcomes compared with `:direct`.
- `:plugin_poisson` — same Stage-1 ``\\hat{\\mu}(A, W)``, with Stage-2 Poisson log-link for counts.
- `:direct` — fit ``w_\\delta(A)``-weighted canonical GLM directly from `y` on `W`.
"""
function _fit_stage1_mu_hat(
    Psi_tr::Matrix{Float64},
    a_tr::AbstractVector{Float64},
    y_tr::AbstractVector{Float64},
    family::Symbol,
)
    a_mean = mean(a_tr)
    a_sd = std(a_tr; corrected = false)
    a_std = (isfinite(a_sd) && a_sd > 1e-12) ? (a_tr .- a_mean) ./ a_sd : zeros(Float64, length(a_tr))
    X_stage1 = hcat(Psi_tr[:, 1:1], reshape(a_std, :, 1), Psi_tr[:, 2:end])
    ones_w = ones(Float64, length(y_tr))
    if family === :gaussian
        c1 = _fit_weighted_gaussian_glm(X_stage1, y_tr, ones_w)
        return _predict_canonical_glm(X_stage1, c1, :gaussian)
    elseif family === :binomial
        c1 = _fit_weighted_binomial_glm(X_stage1, y_tr, ones_w)
        return _predict_canonical_glm(X_stage1, c1, :binomial)
    elseif family === :poisson
        c1 = _fit_weighted_poisson_glm(X_stage1, y_tr, ones_w)
        return _predict_canonical_glm(X_stage1, c1, :poisson)
    elseif family === :negbin
        c1 = _fit_weighted_negbin_glm(X_stage1, y_tr, ones_w)
        return _predict_canonical_glm(X_stage1, c1, :negbin)
    end
    throw(ArgumentError("unsupported family_outcome=$family; expected :gaussian, :binomial, :poisson, or :negbin"))
end

function _fit_predict_tilt_outcome_from_basis(
    Psi_tr::Matrix{Float64},
    Psi_te::Matrix{Float64},
    a_tr::AbstractVector{Float64},
    y_tr::AbstractVector{Float64},
    w_tr::Vector{Float64};
    family::Symbol,
    outcome_method::Symbol,
    mu_hat::Union{Nothing, Vector{Float64}} = nothing,
)
    if outcome_method === :direct
        gamma = if family === :gaussian
            _fit_weighted_gaussian_glm(Psi_tr, y_tr, w_tr)
        elseif family === :binomial
            _fit_weighted_binomial_glm(Psi_tr, y_tr, w_tr)
        elseif family === :poisson
            _fit_weighted_poisson_glm(Psi_tr, y_tr, w_tr)
        elseif family === :negbin
            _fit_weighted_negbin_glm(Psi_tr, y_tr, w_tr)
        else
            throw(ArgumentError("unsupported family_outcome=$family; expected :gaussian, :binomial, :poisson, or :negbin"))
        end
        return _predict_canonical_glm(Psi_te, gamma, family)
    elseif outcome_method in (:plugin, :plugin_poisson)
        # Stage 1: unweighted outcome model μ̂(A, W) on standardised [1, A, W]
        mu_s1 = mu_hat === nothing ? _fit_stage1_mu_hat(Psi_tr, a_tr, y_tr, family) : mu_hat

        # Stage 2: regress μ̂(A_i, W_i) on W_i with tilt weights w_δ(A_i)
        gamma = if family === :gaussian
            _fit_weighted_gaussian_glm(Psi_tr, mu_s1, w_tr)
        elseif family === :binomial
            _fit_weighted_binomial_glm(Psi_tr, mu_s1, w_tr)
        elseif family in (:poisson, :negbin)
            if outcome_method === :plugin && family === :negbin
                _fit_weighted_gamma_log_glm(Psi_tr, mu_s1, w_tr)
            else
                _fit_weighted_poisson_glm(Psi_tr, mu_s1, w_tr)
            end
        else
            throw(ArgumentError("unsupported family_outcome=$family; expected :gaussian, :binomial, :poisson, or :negbin"))
        end
        return _predict_canonical_glm(Psi_te, gamma, family)
    end
    throw(ArgumentError(
        "unsupported outcome_method=$outcome_method; expected :plugin, :plugin_poisson, or :direct",
    ))
end

function _fit_predict_tilt_outcome(
    W_tr::Matrix{Float64},
    a_tr::AbstractVector{Float64},
    y_tr::AbstractVector{Float64},
    w_tr::Vector{Float64},
    W_te::Matrix{Float64};
    family::Symbol,
    outcome_method::Symbol,
)
    Psi_tr, active_cols, col_means, col_sds = _fit_standardised_basis(W_tr)
    Psi_te = _apply_standardised_basis(W_te, active_cols, col_means, col_sds, size(W_tr, 2))
    return _fit_predict_tilt_outcome_from_basis(
        Psi_tr, Psi_te, a_tr, y_tr, w_tr;
        family = family, outcome_method = outcome_method,
    )
end

# ---------------------------------------------------------------------------
# 5. Studentised multiplier / wild bootstrap CI
# ---------------------------------------------------------------------------

"""
    studentised_multiplier_ci(estimate, ic; alpha = 0.05, B = 999, weights = :rademacher, rng = StableRNG(2026))

Compute a studentised multiplier (wild) bootstrap confidence interval for an
asymptotically linear estimator with influence function vector `ic` (mean zero).

For each bootstrap draw ``b = 1, \\dots, B``:
1. Draw i.i.d. multipliers ``\\xi_i^{(b)}`` (`:rademacher` on ``\\{-1, +1\\}``, default, or
   `:mammen` two-point distribution matching ``\\mathbb{E}[\\xi]=0, \\mathbb{E}[\\xi^2]=1,
   \\mathbb{E}[\\xi^3]=1``).
2. Form ``z_i^{(b)} = \\xi_i^{(b)} \\hat{\\varphi}_i``, bootstrap shift
   ``\\bar{z}^{(b)} = \\frac{1}{n}\\sum_i z_i^{(b)}``, and bootstrap SE
   ``\\widehat{\\mathrm{SE}}^{(b)} = \\mathrm{sd}(z^{(b)}) / \\sqrt{n}``.
3. Studentise ``T^{(b)} = \\bar{z}^{(b)} / \\widehat{\\mathrm{SE}}^{(b)}``.
4. Invert quantiles ``[q_{\\alpha/2}, q_{1-\\alpha/2}]``:
   ``[\\hat{\\tau} - q_{1-\\alpha/2} \\widehat{\\mathrm{SE}},\\; \\hat{\\tau} - q_{\\alpha/2} \\widehat{\\mathrm{SE}}]``.

`:rademacher` is the default because ``\\xi_i^2 \\equiv 1`` yields strictly heavier-than-normal
``t``-like finite-sample quantiles, whereas `:mammen` satisfies ``\\xi_i^2 = \\xi_i + 1`` so
large positive draws inflate ``\\widehat{\\mathrm{SE}}^{(b)}`` and can compress the lower bound
in small samples.
"""
function studentised_multiplier_ci(
    estimate::Real,
    ic::AbstractVector{<:Real};
    alpha::Real = 0.05,
    B::Int = 999,
    weights::Symbol = :rademacher,
    rng::AbstractRNG = StableRNG(2026),
)
    est = Float64(estimate)
    phi = Float64.(ic)
    n = length(phi)
    n >= 2 || throw(ArgumentError("studentised_multiplier_ci requires n >= 2"))
    (0.0 < alpha < 1.0) || throw(ArgumentError("alpha must lie in (0, 1); got $alpha"))
    B >= 19 || throw(ArgumentError("bootstrap replicates B must be >= 19; got $B"))
    weights in (:mammen, :rademacher) || throw(ArgumentError(
        "bootstrap weights must be :mammen or :rademacher; got $weights",
    ))

    # Centre EIF defensively in case of tiny numerical drift
    phi_c = phi .- mean(phi)
    se = std(phi_c) / sqrt(n)
    if !isfinite(se) || se <= 1e-15
        return (
            lower = est, upper = est, se = se,
            q_lower = 0.0, q_upper = 0.0, weights = weights, B = B,
        )
    end

    t_stats = Vector{Float64}(undef, B)
    inv_n = inv(Float64(n))
    inv_nm1 = inv(Float64(n - 1))
    inv_sqrt_n = inv(sqrt(n))
    valid = 0

    if weights === :rademacher
        # For Rademacher multipliers ξ_i ∈ {-1, +1}, z_i^2 ≡ phi_c[i]^2 is invariant
        # across draws, so ∑(z_i - z̄)^2 = ∑phi_c^2 - n z̄^2 in a single pass.
        sum_phi2 = sum(abs2, phi_c)
        for _ in 1:B
            s1 = 0.0
            @inbounds for i in 1:n
                s1 += ifelse(rand(rng, Bool), phi_c[i], -phi_c[i])
            end
            zbar = s1 * inv_n
            ssq = max(sum_phi2 - n * zbar * zbar, 0.0)
            se_b = sqrt(ssq * inv_nm1) * inv_sqrt_n
            if isfinite(se_b) && se_b > 1e-15 * se
                valid += 1
                t_stats[valid] = zbar / se_b
            end
        end
    else
        # Mammen's two-point distribution: E[ξ]=0, E[ξ²]=1, E[ξ³]=1
        sq5 = sqrt(5.0)
        m_lo = -(sq5 - 1.0) / 2.0
        m_hi = (sq5 + 1.0) / 2.0
        p_lo = (sq5 + 1.0) / (2.0 * sq5)
        z = Vector{Float64}(undef, n)

        for _ in 1:B
            s1 = 0.0
            @inbounds for i in 1:n
                ξ = rand(rng) < p_lo ? m_lo : m_hi
                zi = ξ * phi_c[i]
                z[i] = zi
                s1 += zi
            end
            zbar = s1 * inv_n
            ssq = 0.0
            @inbounds for i in 1:n
                d = z[i] - zbar
                ssq += d * d
            end
            se_b = sqrt(ssq * inv_nm1) * inv_sqrt_n
            if isfinite(se_b) && se_b > 1e-15 * se
                valid += 1
                t_stats[valid] = zbar / se_b
            end
        end
    end

    if valid < 10
        lwr, upr = CausalTargeted.wald_ci(est, se; alpha = alpha)
        zcrit = quantile(Normal(0, 1), 1 - alpha / 2)
        return (
            lower = lwr, upper = upr, se = se,
            q_lower = -zcrit, q_upper = zcrit, weights = weights, B = valid,
        )
    end

    resize!(t_stats, valid)
    q_lo = quantile(t_stats, alpha / 2)
    q_hi = quantile(t_stats, 1.0 - alpha / 2)
    # Pivot inversion: P(q_lo <= (est - truth) / se <= q_hi) ≈ 1 - alpha
    lower = est - q_hi * se
    upper = est - q_lo * se
    return (
        lower = lower, upper = upper, se = se,
        q_lower = q_lo, q_upper = q_hi, weights = weights, B = valid,
    )
end

# ---------------------------------------------------------------------------
# 6. δ-invariant grid cache & main cross-fitted exponential-tilting TMLE estimator
# ---------------------------------------------------------------------------

struct _StandardisedFoldBasis
    Psi_tr::Matrix{Float64}
    Psi_te::Matrix{Float64}
    active_cols::Vector{Int}
    col_means::Vector{Float64}
    col_sds::Vector{Float64}
    n_raw_cols::Int
end

function _build_fold_bases(
    W::Matrix{Float64},
    train_sets::Vector{Vector{Int}},
    fold_sets::Vector{Vector{Int}},
)
    d = size(W, 2)
    bases = Vector{_StandardisedFoldBasis}(undef, length(fold_sets))
    for k in eachindex(fold_sets)
        Psi_tr, active_cols, col_means, col_sds = _fit_standardised_basis(@view W[train_sets[k], :])
        Psi_te = _apply_standardised_basis(@view(W[fold_sets[k], :]), active_cols, col_means, col_sds, d)
        bases[k] = _StandardisedFoldBasis(Psi_tr, Psi_te, active_cols, col_means, col_sds, d)
    end
    return bases
end

struct _TiltGridCache
    n::Int
    trt::Symbol
    outcome::Symbol
    normalizer_covariates::Vector{Symbol}
    outcome_covariates::Vector{Symbol}
    family_outcome::Symbol
    outcome_method::Symbol
    folds::Int
    independent_folds::Bool
    a::Vector{Float64}
    y::Vector{Float64}
    fold_sets_m0::Vector{Vector{Int}}
    train_sets_m0::Vector{Vector{Int}}
    basis_m0::Vector{_StandardisedFoldBasis}
    fold_sets_y::Vector{Vector{Int}}
    train_sets_y::Vector{Vector{Int}}
    basis_y::Vector{_StandardisedFoldBasis}
    basis_a::Vector{_StandardisedFoldBasis}
    mu_hat_folds::Union{Nothing, Vector{Vector{Float64}}}
end

"""
    _build_tilt_grid_cache(df, trt, outcome, covariates; kwargs...) -> _TiltGridCache

Precompute the δ-invariant components shared across a grid of exponential-tilt
strengths: numeric `a`/`y` vectors, `CovariateSchema`-encoded design matrices,
cross-fitting fold partitions and standardised fold bases, and (for `:plugin` /
`:plugin_poisson`) the δ-invariant Stage-1 outcome regression `μ̂(A, W)` on each
training fold. Thread-safe (read-only after construction).
"""
function _build_tilt_grid_cache(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::AbstractVector{Symbol};
    family_outcome::Symbol = :gaussian,
    folds::Int = 5,
    independent_folds::Bool = true,
    outcome_method::Symbol = :plugin,
    normalizer_covariates::AbstractVector{Symbol} = covariates,
    outcome_covariates::AbstractVector{Symbol} = covariates,
    rng::AbstractRNG = StableRNG(2026),
)
    family_outcome in (:gaussian, :binomial, :poisson, :negbin) || throw(ArgumentError(
        "family_outcome must be :gaussian, :binomial, :poisson, or :negbin; got $family_outcome",
    ))
    outcome_method in (:plugin, :plugin_poisson, :direct) || throw(ArgumentError(
        "unsupported outcome_method=$outcome_method; expected :plugin, :plugin_poisson, or :direct",
    ))
    n = nrow(df)
    n >= 4 || throw(ArgumentError("estimate_exponential_tilt requires at least 4 observations"))

    a = collect(Float64, df[!, trt])::Vector{Float64}
    y = collect(Float64, df[!, outcome])::Vector{Float64}
    all(isfinite, a) || throw(ArgumentError("treatment :$trt contains non-finite values"))
    all(isfinite, y) || throw(ArgumentError("outcome :$outcome contains non-finite values"))

    norm_covs = collect(Symbol, normalizer_covariates)
    out_covs = collect(Symbol, outcome_covariates)
    schema_m0 = CausalTargeted.fit_covariate_schema(df, norm_covs)
    W_m0 = CausalTargeted.transform_covariates(schema_m0, df)::Matrix{Float64}
    same_covs = (out_covs == norm_covs)
    W_y = if same_covs
        W_m0
    else
        schema_y = CausalTargeted.fit_covariate_schema(df, out_covs)
        CausalTargeted.transform_covariates(schema_y, df)::Matrix{Float64}
    end

    eff_folds = clamp(folds, 2, max(2, n ÷ 2))
    fold_sets_m0 = CausalTargeted.crossfit_indices(n, eff_folds, rng)
    fold_sets_y = independent_folds ?
        CausalTargeted.crossfit_indices(n, eff_folds, rng) : fold_sets_m0
    train_sets_m0 = [_tilt_complement_indices(n, test_idx) for test_idx in fold_sets_m0]
    train_sets_y = independent_folds ?
        [_tilt_complement_indices(n, test_idx) for test_idx in fold_sets_y] : train_sets_m0

    basis_m0 = _build_fold_bases(W_m0, train_sets_m0, fold_sets_m0)
    basis_a = independent_folds ? _build_fold_bases(W_m0, train_sets_y, fold_sets_y) : basis_m0
    basis_y = same_covs ? basis_a : _build_fold_bases(W_y, train_sets_y, fold_sets_y)

    mu_hat_folds = if outcome_method in (:plugin, :plugin_poisson)
        Vector{Float64}[
            _fit_stage1_mu_hat(
                basis_y[k].Psi_tr,
                @view(a[train_sets_y[k]]),
                @view(y[train_sets_y[k]]),
                family_outcome,
            )
            for k in eachindex(fold_sets_y)
        ]
    else
        nothing
    end

    return _TiltGridCache(
        n, trt, outcome, norm_covs, out_covs, family_outcome, outcome_method,
        eff_folds, independent_folds, a, y,
        fold_sets_m0, train_sets_m0, basis_m0,
        fold_sets_y, train_sets_y, basis_y, basis_a, mu_hat_folds,
    )
end

"""
    estimate_exponential_tilt(df, trt, outcome, covariates, delta, policy; kwargs...) -> NamedTuple

Cross-fitted calibration-balanced TMLE for an exponential-tilting stochastic
intervention ``g_\\delta(a \\mid W) \\propto \\exp(\\delta \\phi(a)) g(a \\mid W)``.

Estimates both:
- Outcome contrast ``\\tau(\\delta) = \\psi(\\delta) - \\mathbb{E}[Y]`` with canonical-link
  TMLE targeting via `CausalTargeted._weighted_intercept_tilt` and exact EIF
  ``\\varphi_\\delta(O) = H_\\delta(A, W)(Y - \\theta^*_\\delta(W)) + \\theta^*_\\delta(W) - Y - \\hat{\\tau}(\\delta)``.
- Realised mean exposure shift ``\\Delta(\\delta) = \\mathbb{E}_{g_\\delta}[A] - \\mathbb{E}[A]``
  with the matching Gaussian EIF.

Returns both Wald CIs (`lower`, `upper`) and studentised multiplier bootstrap CIs
(`boot_lower`, `boot_upper`), alongside weight ESS and TMLE score diagnostics.

# Keyword arguments
- `family_outcome::Symbol = :gaussian`: `:gaussian`, `:binomial`, `:poisson`, or `:negbin`.
- `folds::Int = 5`: number of cross-fitting folds (automatically capped to `n ÷ 2` if needed).
- `independent_folds::Bool = true`: use separate cross-fit partitions for ``\\hat{m}_0(W)`` and
  ``\\hat{\\theta}_\\delta(W)`` to attenuate finite-sample correlation in the second-order DR
  remainder (Newey & Robins 2018); set `false` for a single shared partition.
- `lambda::Real = 0.0`: optional L2 ridge penalty on standardised non-intercept normaliser coefficients.
- `outcome_method::Symbol = :plugin`: `:plugin` (two-stage Rao-Blackwellised), `:plugin_poisson`, or `:direct`.
- `enforce_bounds::Bool = true`: project bounded-policy ``\\log \\hat{m}_0(W)`` onto ``[-|\\delta|M, |\\delta|M]``.
- `normalizer_covariates = covariates`: override covariate columns used for ``\\hat{m}_0(W)`` (e.g. for DR tests).
- `outcome_covariates = covariates`: override covariate columns used for ``\\hat{\\theta}_\\delta(W)`` (e.g. for DR tests).
- `alpha::Real = 0.05`: two-sided significance level.
- `bootstrap_B::Int = 999`: replicates for studentised multiplier bootstrap.
- `bootstrap_weights::Symbol = :rademacher`: `:rademacher` (default) or `:mammen`.
- `compute_bootstrap::Bool = true`: compute studentised multiplier bootstrap CIs; when `false`,
  skips `studentised_multiplier_ci` and returns `NaN` for `boot_lower`, `boot_upper`,
  `shift_boot_lower`, and `shift_boot_upper`.
- `tilt_cache = nothing`: optional internal `_TiltGridCache` from `_build_tilt_grid_cache`
  to reuse δ-invariant covariate encodings, folds, and Stage-1 outcome fits across a δ grid.
- `rng::AbstractRNG = StableRNG(2026)`: RNG for cross-fitting and multiplier bootstrap.
"""
function estimate_exponential_tilt(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::AbstractVector{Symbol},
    delta::Real,
    policy::ExponentialTiltPolicy = exponential_tilt_policy();
    family_outcome::Symbol = :gaussian,
    folds::Int = 5,
    independent_folds::Bool = true,
    lambda::Real = 0.0,
    outcome_method::Symbol = :plugin,
    enforce_bounds::Bool = true,
    normalizer_covariates::AbstractVector{Symbol} = covariates,
    outcome_covariates::AbstractVector{Symbol} = covariates,
    alpha::Real = 0.05,
    bootstrap_B::Int = 999,
    bootstrap_weights::Symbol = :rademacher,
    compute_bootstrap::Bool = true,
    tilt_cache::Union{Nothing, _TiltGridCache} = nothing,
    rng::AbstractRNG = StableRNG(2026),
)
    family_outcome in (:gaussian, :binomial, :poisson, :negbin) || throw(ArgumentError(
        "family_outcome must be :gaussian, :binomial, :poisson, or :negbin; got $family_outcome",
    ))
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("delta must be finite; got $delta"))
    λ = Float64(lambda)
    (isfinite(λ) && λ >= 0.0) || throw(ArgumentError(
        "ridge penalty lambda must be finite and >= 0; got $lambda",
    ))
    n = nrow(df)
    n >= 4 || throw(ArgumentError("estimate_exponential_tilt requires at least 4 observations"))

    if tilt_cache !== nothing
        tilt_cache.n == n || throw(ArgumentError(
            "tilt_cache row count $(tilt_cache.n) does not match data nrow $n",
        ))
        (tilt_cache.trt === trt && tilt_cache.outcome === outcome) || throw(ArgumentError(
            "tilt_cache columns (:$(tilt_cache.trt), :$(tilt_cache.outcome)) do not match (:$trt, :$outcome)",
        ))
        (tilt_cache.family_outcome === family_outcome && tilt_cache.outcome_method === outcome_method) || throw(ArgumentError(
            "tilt_cache family/method ($(tilt_cache.family_outcome), $(tilt_cache.outcome_method)) does not match ($family_outcome, $outcome_method)",
        ))
        (tilt_cache.normalizer_covariates == normalizer_covariates && tilt_cache.outcome_covariates == outcome_covariates) || throw(ArgumentError(
            "tilt_cache covariates do not match requested normalizer/outcome covariates",
        ))
        a = tilt_cache.a
        y = tilt_cache.y
    else
        a = collect(Float64, df[!, trt])::Vector{Float64}
        y = collect(Float64, df[!, outcome])::Vector{Float64}
        all(isfinite, a) || throw(ArgumentError("treatment :$trt contains non-finite values"))
        all(isfinite, y) || throw(ArgumentError("outcome :$outcome contains non-finite values"))
    end

    w_bounds = tilt_weight_bounds(δ, policy)

    # Trivial zero-tilt shortcut: g_0 = g, τ(0) = 0, Δ(0) = 0
    if iszero(δ)
        if tilt_cache === nothing
            norm_covs = collect(Symbol, normalizer_covariates)
            out_covs = collect(Symbol, outcome_covariates)
            schema_m0 = CausalTargeted.fit_covariate_schema(df, norm_covs)
            CausalTargeted.transform_covariates(schema_m0, df)::Matrix{Float64}
            if out_covs != norm_covs
                schema_y = CausalTargeted.fit_covariate_schema(df, out_covs)
                CausalTargeted.transform_covariates(schema_y, df)::Matrix{Float64}
            end
        end
        H = ones(Float64, n)
        zeros_n = zeros(Float64, n)
        boot_val = compute_bootstrap ? 0.0 : NaN
        return (
            delta = 0.0,
            policy = policy,
            family_outcome = family_outcome,
            outcome_method = outcome_method,
            estimate = 0.0,
            psi = mean(y),
            psi0 = mean(y),
            se = 0.0,
            lower = 0.0,
            upper = 0.0,
            boot_lower = boot_val,
            boot_upper = boot_val,
            ic = zeros_n,
            tmle_score = 0.0,
            shift_estimate = 0.0,
            shift_se = 0.0,
            shift_lower = 0.0,
            shift_upper = 0.0,
            shift_boot_lower = boot_val,
            shift_boot_upper = boot_val,
            shift_ic = copy(zeros_n),
            shift_tmle_score = 0.0,
            H = H,
            theta_init = fill(mean(y), n),
            theta_star = fill(mean(y), n),
            theta_a_star = fill(mean(a), n),
            ess = Float64(n),
            ess_ratio = 1.0,
            mean_H = 1.0,
            min_H = 1.0,
            max_H = 1.0,
            weight_bounds = w_bounds,
            fold_fits = TiltNormalizerFit[],
            inference_status = :trivial,
        )
    end

    H = Vector{Float64}(undef, n)
    theta_init = Vector{Float64}(undef, n)
    theta_a_init = Vector{Float64}(undef, n)
    fold_fits = TiltNormalizerFit[]
    log_w_all = tilt_log_weight(a, δ, policy)

    if tilt_cache !== nothing
        # 1. Cross-fitted convex calibration-balanced normaliser m̂_0(W) using cached fold bases
        for k in eachindex(tilt_cache.fold_sets_m0)
            test_idx = tilt_cache.fold_sets_m0[k]
            train_idx = tilt_cache.train_sets_m0[k]
            b = tilt_cache.basis_m0[k]
            fit_m0 = _fit_tilt_normalizer_from_basis(
                b.Psi_tr, b.active_cols, b.col_means, b.col_sds, b.n_raw_cols,
                @view(log_w_all[train_idx]), δ, policy;
                lambda = λ,
            )
            push!(fold_fits, fit_m0)
            H[test_idx] = _predict_tilt_weights_from_basis(
                fit_m0, b.Psi_te, @view(log_w_all[test_idx]);
                enforce_bounds = enforce_bounds,
            )
        end

        # 2. Cross-fitted outcome nuisance θ̂_δ(W) and exposure nuisance θ̂^A_δ(W) using cached bases & Stage-1 μ̂
        for k in eachindex(tilt_cache.fold_sets_y)
            test_idx = tilt_cache.fold_sets_y[k]
            train_idx = tilt_cache.train_sets_y[k]
            log_w_tr = @view log_w_all[train_idx]
            max_lw = maximum(log_w_tr)
            w_tr = Vector{Float64}(undef, length(train_idx))
            @inbounds for i in eachindex(log_w_tr)
                w_tr[i] = exp(log_w_tr[i] - max_lw)
            end

            by = tilt_cache.basis_y[k]
            ba = tilt_cache.basis_a[k]
            mu_hat_k = tilt_cache.mu_hat_folds === nothing ? nothing : tilt_cache.mu_hat_folds[k]
            theta_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                by.Psi_tr, by.Psi_te, @view(a[train_idx]), @view(y[train_idx]), w_tr;
                family = family_outcome, outcome_method = outcome_method, mu_hat = mu_hat_k,
            )
            theta_a_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                ba.Psi_tr, ba.Psi_te, @view(a[train_idx]), @view(a[train_idx]), w_tr;
                family = :gaussian, outcome_method = :direct,
            )
        end
    else
        norm_covs = collect(Symbol, normalizer_covariates)
        out_covs = collect(Symbol, outcome_covariates)
        schema_m0 = CausalTargeted.fit_covariate_schema(df, norm_covs)
        W_m0 = CausalTargeted.transform_covariates(schema_m0, df)::Matrix{Float64}
        W_y = if out_covs == norm_covs
            W_m0
        else
            schema_y = CausalTargeted.fit_covariate_schema(df, out_covs)
            CausalTargeted.transform_covariates(schema_y, df)::Matrix{Float64}
        end

        eff_folds = clamp(folds, 2, max(2, n ÷ 2))
        fold_sets_m0 = CausalTargeted.crossfit_indices(n, eff_folds, rng)
        fold_sets_y = independent_folds ?
            CausalTargeted.crossfit_indices(n, eff_folds, rng) : fold_sets_m0

        # 1. Cross-fitted convex calibration-balanced normaliser m̂_0(W)
        for test_idx in fold_sets_m0
            train_idx = _tilt_complement_indices(n, test_idx)
            Psi_tr, active_cols, col_means, col_sds = _fit_standardised_basis(@view W_m0[train_idx, :])
            Psi_te = _apply_standardised_basis(@view(W_m0[test_idx, :]), active_cols, col_means, col_sds, size(W_m0, 2))
            fit_m0 = _fit_tilt_normalizer_from_basis(
                Psi_tr, active_cols, col_means, col_sds, size(W_m0, 2),
                @view(log_w_all[train_idx]), δ, policy;
                lambda = λ,
            )
            push!(fold_fits, fit_m0)
            H[test_idx] = _predict_tilt_weights_from_basis(
                fit_m0, Psi_te, @view(log_w_all[test_idx]);
                enforce_bounds = enforce_bounds,
            )
        end

        # 2. Cross-fitted outcome nuisance θ̂_δ(W) and exposure nuisance θ̂^A_δ(W).
        # Shift log_w by its fold maximum before exponentiating so unbounded linear
        # tilts with large `center` never overflow or underflow; weighted GLMs only
        # depend on relative weights `w / mean(w)`.
        same_covs = (out_covs == norm_covs)
        for test_idx in fold_sets_y
            train_idx = _tilt_complement_indices(n, test_idx)
            log_w_tr = @view log_w_all[train_idx]
            max_lw = maximum(log_w_tr)
            w_tr = Vector{Float64}(undef, length(train_idx))
            @inbounds for i in eachindex(log_w_tr)
                w_tr[i] = exp(log_w_tr[i] - max_lw)
            end

            Psi_y_tr, ac_y, cm_y, cs_y = _fit_standardised_basis(@view W_y[train_idx, :])
            Psi_y_te = _apply_standardised_basis(@view(W_y[test_idx, :]), ac_y, cm_y, cs_y, size(W_y, 2))
            theta_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                Psi_y_tr, Psi_y_te, @view(a[train_idx]), @view(y[train_idx]), w_tr;
                family = family_outcome, outcome_method = outcome_method,
            )

            if same_covs
                theta_a_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                    Psi_y_tr, Psi_y_te, @view(a[train_idx]), @view(a[train_idx]), w_tr;
                    family = :gaussian, outcome_method = :direct,
                )
            else
                Psi_a_tr, ac_a, cm_a, cs_a = _fit_standardised_basis(@view W_m0[train_idx, :])
                Psi_a_te = _apply_standardised_basis(@view(W_m0[test_idx, :]), ac_a, cm_a, cs_a, size(W_m0, 2))
                theta_a_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                    Psi_a_tr, Psi_a_te, @view(a[train_idx]), @view(a[train_idx]), w_tr;
                    family = :gaussian, outcome_method = :direct,
                )
            end
        end
    end

    # 4. Canonical-link one-step TMLE targeting via CausalTargeted._weighted_intercept_tilt
    _, theta_star, _, boundary_y = CausalTargeted._weighted_intercept_tilt(
        y, theta_init, theta_init, H, family_outcome, 1.0,
    )
    _, theta_a_star, _, boundary_a = CausalTargeted._weighted_intercept_tilt(
        a, theta_a_init, theta_a_init, H, :gaussian, 1.0,
    )

    # 5. Outcome contrast τ(δ) = ψ(δ) - E[Y] and exact EIF
    psi = mean(theta_star)
    psi0 = mean(y)
    est = psi - psi0
    tmle_score = mean(H .* (y .- theta_star))
    ic = H .* (y .- theta_star) .+ theta_star .- y .- est
    se = std(ic) / sqrt(n)
    lower, upper = CausalTargeted.wald_ci(est, se; alpha = alpha)
    if compute_bootstrap
        boot_y = studentised_multiplier_ci(
            est, ic; alpha = alpha, B = bootstrap_B, weights = bootstrap_weights, rng = rng,
        )
        boot_lower, boot_upper = boot_y.lower, boot_y.upper
    else
        boot_lower, boot_upper = NaN, NaN
    end

    # 6. Realised exposure shift Δ(δ) = E_{g_δ}[A] - E[A] and exact EIF
    psi_a = mean(theta_a_star)
    psi0_a = mean(a)
    shift_est = psi_a - psi0_a
    shift_tmle_score = mean(H .* (a .- theta_a_star))
    shift_ic = H .* (a .- theta_a_star) .+ theta_a_star .- a .- shift_est
    shift_se = std(shift_ic) / sqrt(n)
    shift_lower, shift_upper = CausalTargeted.wald_ci(shift_est, shift_se; alpha = alpha)
    if compute_bootstrap
        boot_a = studentised_multiplier_ci(
            shift_est, shift_ic; alpha = alpha, B = bootstrap_B, weights = bootstrap_weights, rng = rng,
        )
        shift_boot_lower, shift_boot_upper = boot_a.lower, boot_a.upper
    else
        shift_boot_lower, shift_boot_upper = NaN, NaN
    end

    sum_H = sum(H)
    sum_H2 = sum(abs2, H)
    ess = sum_H2 > 0.0 ? (sum_H^2) / sum_H2 : 0.0
    inference_status = (boundary_y || boundary_a) ? :boundary_outcome : :available

    return (
        delta = δ,
        policy = policy,
        family_outcome = family_outcome,
        outcome_method = outcome_method,
        estimate = est,
        psi = psi,
        psi0 = psi0,
        se = se,
        lower = lower,
        upper = upper,
        boot_lower = boot_lower,
        boot_upper = boot_upper,
        ic = ic,
        tmle_score = tmle_score,
        shift_estimate = shift_est,
        shift_se = shift_se,
        shift_lower = shift_lower,
        shift_upper = shift_upper,
        shift_boot_lower = shift_boot_lower,
        shift_boot_upper = shift_boot_upper,
        shift_ic = shift_ic,
        shift_tmle_score = shift_tmle_score,
        H = H,
        theta_init = theta_init,
        theta_star = theta_star,
        theta_a_star = theta_a_star,
        ess = ess,
        ess_ratio = ess / n,
        mean_H = mean(H),
        min_H = minimum(H),
        max_H = maximum(H),
        weight_bounds = w_bounds,
        fold_fits = fold_fits,
        inference_status = inference_status,
    )
end
