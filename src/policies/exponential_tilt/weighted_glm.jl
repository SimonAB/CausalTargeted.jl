# ---------------------------------------------------------------------------
# Weighted canonical-link GLMs & cross-fitted outcome nuisance θ_δ(W)
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

# Singleton link tags for zero-overhead specialisation inside _fit_weighted_irls_newton.
struct _BinomialLink end
struct _PoissonLink end
struct _GammaLogLink end

@inline function _glm_neg_loglik_term(::_BinomialLink, ei::Float64, yi::Float64)
    l1p = ei >= 0.0 ? ei + log1p(exp(-ei)) : log1p(exp(ei))
    return l1p - yi * ei
end
@inline function _glm_grad_weight_terms(::_BinomialLink, ei::Float64, yi::Float64)
    mi = _expit(ei)
    return (mi - yi, max(mi * (1.0 - mi), 1e-12))
end
@inline _glm_grad_term(::_BinomialLink, ei::Float64, yi::Float64) = _expit(ei) - yi

@inline function _glm_neg_loglik_term(::_PoissonLink, ei::Float64, yi::Float64)
    ei > 50.0 && return Inf
    return exp(ei) - yi * ei
end
@inline function _glm_grad_weight_terms(::_PoissonLink, ei::Float64, yi::Float64)
    mi = exp(clamp(ei, -50.0, 50.0))
    return (mi - yi, mi)
end
@inline _glm_grad_term(::_PoissonLink, ei::Float64, yi::Float64) =
    exp(clamp(ei, -50.0, 50.0)) - yi

@inline function _glm_neg_loglik_term(::_GammaLogLink, ei::Float64, yi::Float64)
    (ei < -50.0 || ei > 50.0) && return Inf
    return yi * exp(-ei) + ei
end
@inline function _glm_grad_weight_terms(::_GammaLogLink, ei::Float64, yi::Float64)
    ri = yi * exp(-clamp(ei, -50.0, 50.0))
    return (1.0 - ri, ri)
end
@inline _glm_grad_term(::_GammaLogLink, ei::Float64, yi::Float64) =
    1.0 - yi * exp(-clamp(ei, -50.0, 50.0))

function _glm_neg_loglik!(
    link::L,
    eta::Vector{Float64},
    Psi::Matrix{Float64},
    gvec::Vector{Float64},
    yt::Vector{Float64},
    wn::Vector{Float64},
    ridge::Float64,
) where {L}
    n = length(yt)
    p = length(gvec)
    mul!(eta, Psi, gvec)
    s = 0.0
    @inbounds for i in 1:n
        t = _glm_neg_loglik_term(link, eta[i], yt[i])
        !isfinite(t) && return Inf
        s += wn[i] * t
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
Shared ridge-penalised damped Newton / IRLS solver with Armijo backtracking and
near-machine-epsilon gradient-decrease acceptance (preventing Float64 rounding
stalls on the final quadratic Newton step). Specialised statically on `link::L`
with preallocated work buffers and zero closure captures.
"""
function _fit_weighted_irls_newton(
    link::L,
    Psi::Matrix{Float64},
    yt::Vector{Float64},
    wn::Vector{Float64},
    gamma::Vector{Float64};
    ridge::Float64 = 1e-6,
    tol::Float64 = 1e-10,
    maxiter::Int = 60,
) where {L}
    n, p = size(Psi)
    eta = Vector{Float64}(undef, n)
    wt = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, n)
    grad = Vector{Float64}(undef, p)
    grad_try = Vector{Float64}(undef, p)
    Psi_w = Matrix{Float64}(undef, n, p)
    Hmat = Matrix{Float64}(undef, p, p)
    gamma_try = similar(gamma)

    obj = _glm_neg_loglik!(link, eta, Psi, gamma, yt, wn, ridge)

    for _ in 1:maxiter
        mul!(eta, Psi, gamma)
        @inbounds for i in 1:n
            gi, wi = _glm_grad_weight_terms(link, eta[i], yt[i])
            rhs[i] = wn[i] * gi
            wt[i] = wn[i] * wi
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
            obj_try = _glm_neg_loglik!(link, eta, Psi, gamma_try, yt, wn, ridge)
            if isfinite(obj_try) && obj_try <= obj + 1e-4 * step * slope
                copyto!(gamma, gamma_try)
                obj = obj_try
                accepted = true
                break
            elseif step == 1.0 && isfinite(obj_try) && obj_try <= obj + obj_eps
                @inbounds for i in 1:n
                    rhs[i] = wn[i] * _glm_grad_term(link, eta[i], yt[i])
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
    return _fit_weighted_irls_newton(
        _BinomialLink(), Psi, yc, wn, gamma;
        ridge = ridge, tol = tol, maxiter = maxiter,
    )
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
    p == 1 && return [log(ybar)]
    # Scale targets by ybar so intercept starts at 0 and ridge is scale-invariant.
    ys = yc ./ ybar
    gamma = zeros(Float64, p)
    _fit_weighted_irls_newton(
        _PoissonLink(), Psi, ys, wn, gamma;
        ridge = ridge, tol = tol, maxiter = maxiter,
    )
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
    for θ in DEFAULT_NB_THETA_GRID
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
    p == 1 && return [log(ybar)]
    ys = yc ./ ybar
    gamma = zeros(Float64, p)
    _fit_weighted_irls_newton(
        _GammaLogLink(), Psi, ys, wn, gamma;
        ridge = ridge, tol = tol, maxiter = maxiter,
    )
    gamma[1] += log(ybar)
    return gamma
end

function _fit_weighted_glm_by_family(
    Psi::Matrix{Float64},
    y::AbstractVector{Float64},
    w::Vector{Float64},
    family::Symbol;
    stage2_method::Symbol = :direct,
)
    if family === :gaussian
        return _fit_weighted_gaussian_glm(Psi, y, w)
    elseif family === :binomial
        return _fit_weighted_binomial_glm(Psi, y, w)
    elseif family === :poisson
        return _fit_weighted_poisson_glm(Psi, y, w)
    elseif family === :negbin
        if stage2_method === :plugin
            return _fit_weighted_gamma_log_glm(Psi, y, w)
        elseif stage2_method === :plugin_poisson
            return _fit_weighted_poisson_glm(Psi, y, w)
        end
        return _fit_weighted_negbin_glm(Psi, y, w)
    end
    throw(ArgumentError(
        "unsupported family_outcome=$family; expected :gaussian, :binomial, :poisson, or :negbin",
    ))
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
    c1 = _fit_weighted_glm_by_family(X_stage1, y_tr, ones_w, family)
    return _predict_canonical_glm(X_stage1, c1, family)
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
        gamma = _fit_weighted_glm_by_family(Psi_tr, y_tr, w_tr, family)
        return _predict_canonical_glm(Psi_te, gamma, family)
    elseif outcome_method in (:plugin, :plugin_poisson)
        # Stage 1: unweighted outcome model μ̂(A, W) on standardised [1, A, W]
        mu_s1 = mu_hat === nothing ? _fit_stage1_mu_hat(Psi_tr, a_tr, y_tr, family) : mu_hat
        # Stage 2: regress μ̂(A_i, W_i) on W_i with tilt weights w_δ(A_i)
        gamma = _fit_weighted_glm_by_family(
            Psi_tr, mu_s1, w_tr, family; stage2_method = outcome_method,
        )
        return _predict_canonical_glm(Psi_te, gamma, family)
    end
    throw(ArgumentError(
        "unsupported outcome_method=$outcome_method; expected :plugin, :plugin_poisson, or :direct",
    ))
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
