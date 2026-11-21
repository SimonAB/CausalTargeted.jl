using Distributions: MvNormal, MvTDist, logpdf
using LinearAlgebra
using PSIS
using Random
using StableRNGs: StableRNG
using Statistics: mean

@inline _psis_logistic(x::Real) = x >= 0 ? inv(1.0 + exp(-x)) : (ex = exp(x); ex / (1.0 + ex))
@inline _psis_log1pexp(x::Real) = x <= 0 ? log1p(exp(x)) : x + log1p(exp(-x))

function _psis_log_posterior(
    X::Matrix{Float64},
    y::Vector{Float64},
    prior::MvNormal,
    β::AbstractVector{<:Real},
)
    lp = Float64(logpdf(prior, β))
    isfinite(lp) || return -Inf
    n = size(X, 1)
    if n > 0
        η = X * β
        @inbounds for i in 1:n
            ηi = Float64(η[i])
            lp += y[i] == 1.0 ? -_psis_log1pexp(-ηi) : -_psis_log1pexp(ηi)
        end
    end
    return lp
end

function _psis_newton_map(
    X::Matrix{Float64},
    y::Vector{Float64},
    μ0::Vector{Float64},
    inv_Σ0::Matrix{Float64},
    prior::MvNormal;
    max_iter::Int = 200,
)
    n, p = size(X)
    β = copy(μ0)
    f_cur = _psis_log_posterior(X, y, prior, β)
    isfinite(f_cur) || return (; converged = false, reason = "newton_failed", β, H_chol = nothing)

    η = Vector{Float64}(undef, n)
    prob = Vector{Float64}(undef, n)
    w_diag = Vector{Float64}(undef, n)
    residual = Vector{Float64}(undef, n)
    g = Vector{Float64}(undef, p)
    H = Matrix{Float64}(undef, p, p)

    for _ in 1:max_iter
        if n > 0
            mul!(η, X, β)
            @inbounds for i in 1:n
                pi = _psis_logistic(η[i])
                prob[i] = pi
                w_diag[i] = pi * (1.0 - pi)
                residual[i] = y[i] - pi
            end
            mul!(g, transpose(X), residual)
            g .-= inv_Σ0 * (β .- μ0)
            # X' * Diagonal(w_diag) * X + inv_Σ0
            Xw = X .* w_diag
            mul!(H, transpose(X), Xw)
            H .+= inv_Σ0
        else
            g .= -inv_Σ0 * (β .- μ0)
            H .= inv_Σ0
        end

        H_sym = Symmetric(0.5 .* (H .+ transpose(H)))
        all(isfinite, H_sym) || return (; converged = false, reason = "hessian_not_pd", β, H_chol = nothing)
        H_chol = cholesky(H_sym; check = false)
        issuccess(H_chol) || return (; converged = false, reason = "hessian_not_pd", β, H_chol = nothing)

        d = H_chol \ g
        all(isfinite, d) || return (; converged = false, reason = "newton_failed", β, H_chol = nothing)

        if norm(d, Inf) < 1e-10 && norm(g, Inf) < 1e-8
            return (; converged = true, reason = "", β, H_chol)
        end

        slope = dot(g, d)
        if slope <= 0.0
            norm(g, Inf) < 1e-6 && return (; converged = true, reason = "", β, H_chol)
            return (; converged = false, reason = "newton_failed", β, H_chol = nothing)
        end

        α = 1.0
        accepted = false
        while α >= 1e-12
            β_cand = β .+ α .* d
            f_cand = _psis_log_posterior(X, y, prior, β_cand)
            if isfinite(f_cand) && f_cand >= f_cur + 1e-4 * α * slope
                β = β_cand
                f_cur = f_cand
                accepted = true
                break
            end
            α *= 0.5
        end

        if !accepted
            norm(g, Inf) < 1e-6 && return (; converged = true, reason = "", β, H_chol)
            return (; converged = false, reason = "newton_failed", β, H_chol = nothing)
        end
    end

    # Final check at last iterate
    if n > 0
        mul!(η, X, β)
        @inbounds for i in 1:n
            pi = _psis_logistic(η[i])
            w_diag[i] = pi * (1.0 - pi)
            residual[i] = y[i] - pi
        end
        mul!(g, transpose(X), residual)
        g .-= inv_Σ0 * (β .- μ0)
        mul!(H, transpose(X), X .* w_diag)
        H .+= inv_Σ0
    else
        g .= -inv_Σ0 * (β .- μ0)
        H .= inv_Σ0
    end
    H_sym = Symmetric(0.5 .* (H .+ transpose(H)))
    H_chol = cholesky(H_sym; check = false)
    issuccess(H_chol) || return (; converged = false, reason = "hessian_not_pd", β, H_chol = nothing)
    if norm(g, Inf) < 1e-6
        return (; converged = true, reason = "", β, H_chol)
    end
    return (; converged = false, reason = "newton_failed", β, H_chol = nothing)
end

function _psis_weighted_quantile(values::AbstractVector{Float64}, weights::AbstractVector{Float64}, q::Float64)
    perm = sortperm(values)
    cum = 0.0
    @inbounds for idx in perm
        cum += weights[idx]
        if cum >= q
            return values[idx]
        end
    end
    return values[perm[end]]
end

function _psis_failure_result(
    p::Int,
    failure_reason::String,
    t0::UInt64;
    khat::Float64 = NaN,
    ess::Float64 = NaN,
    mcse::Float64 = NaN,
    mcse_to_sd::Float64 = NaN,
    beta_mean::Vector{Float64} = fill(NaN, p),
    beta_cov::Matrix{Float64} = fill(NaN, p, p),
    policy_mean::Float64 = NaN,
    policy_sd::Float64 = NaN,
    policy_lower::Float64 = NaN,
    policy_upper::Float64 = NaN,
    policy_draws::Vector{Float64} = Float64[],
    weights::Vector{Float64} = Float64[],
)
    seconds = (time_ns() - t0) * 1e-9
    return (;
        available = false,
        failure_reason,
        khat,
        ess,
        mcse,
        mcse_to_sd,
        beta_mean,
        beta_cov,
        policy_mean,
        policy_sd,
        policy_lower,
        policy_upper,
        policy_draws,
        weights,
        seconds,
    )
end

"""
    weighted_interval_content(draws, weights, lo, hi) -> Float64

Return the normalised weighted mass of `draws` lying in the closed interval `[lo, hi]`,
or `NaN` if `!isfinite(lo) || !isfinite(hi)` (or if `draws` is empty or `weights` have
non-positive total).
"""
function weighted_interval_content(
    draws::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
    lo::Real,
    hi::Real,
)::Float64
    length(draws) == length(weights) || throw(DimensionMismatch("draws and weights must have equal length"))
    (!isfinite(lo) || !isfinite(hi) || isempty(draws)) && return NaN
    lo > hi && return 0.0
    mass = 0.0
    total = 0.0
    @inbounds for i in eachindex(draws, weights)
        wi = Float64(weights[i])
        di = Float64(draws[i])
        if isfinite(wi)
            total += wi
            if isfinite(di) && lo <= di <= hi
                mass += wi
            end
        end
    end
    (!isfinite(total) || total <= 0.0) && return NaN
    return clamp(mass / total, 0.0, 1.0)
end

"""
    evaluate_logistic_psis_reference(X, y, prior_mean, prior_cov, X_alt, X_obs; ...)

Compute the Tier 1 Laplace-centred Student-t PSIS reference for multi-covariate
Bayesian logistic regression under a Gaussian prior `MvNormal(prior_mean, prior_cov)`
and cohort marginal policy contrast `mean(logistic.(X_alt * β) .- logistic.(X_obs * β))`.
"""
function evaluate_logistic_psis_reference(
    X::AbstractMatrix,
    y::AbstractVector,
    prior_mean::AbstractVector,
    prior_cov::AbstractMatrix,
    X_alt::AbstractMatrix,
    X_obs::AbstractMatrix;
    n_draws::Integer = 20_000,
    df::Real = 5.0,
    rng::AbstractRNG = StableRNG(20261121),
    khat_max::Real = 0.7,
    ess_min::Real = 400.0,
    mcse_sd_max::Real = 0.05,
    level::Real = 0.95,
)
    t0 = time_ns()

    X_m = Matrix{Float64}(X)
    y_v = Float64.(y)
    μ0 = Vector{Float64}(prior_mean)
    Σ0 = Matrix{Float64}(prior_cov)
    X_alt_m = Matrix{Float64}(X_alt)
    X_obs_m = Matrix{Float64}(X_obs)

    n, p = size(X_m)
    length(y_v) == n || throw(DimensionMismatch("X and y row counts must match"))
    length(μ0) == p || throw(DimensionMismatch("prior_mean length must match columns of X"))
    size(Σ0) == (p, p) || throw(DimensionMismatch("prior_cov dimensions must match columns of X"))
    size(X_alt_m, 2) == p || throw(DimensionMismatch("X_alt columns must match X"))
    size(X_obs_m, 2) == p || throw(DimensionMismatch("X_obs columns must match X"))
    size(X_alt_m, 1) == size(X_obs_m, 1) > 0 || throw(DimensionMismatch("X_alt and X_obs must have equal positive row counts"))
    all(yi -> yi == 0.0 || yi == 1.0, y_v) || throw(ArgumentError("y must contain only binary 0/1 values"))
    n_draws >= 2 || throw(ArgumentError("n_draws must be at least 2"))
    df > 0 || throw(ArgumentError("df must be positive"))
    0.0 < level < 1.0 || throw(ArgumentError("level must lie strictly in (0, 1)"))

    all(isfinite, μ0) && all(isfinite, Σ0) && all(isfinite, X_m) &&
        all(isfinite, X_alt_m) && all(isfinite, X_obs_m) ||
        return _psis_failure_result(p, "newton_failed", t0)

    Σ0_sym = Symmetric(0.5 .* (Σ0 .+ transpose(Σ0)))
    prior_chol = cholesky(Σ0_sym; check = false)
    issuccess(prior_chol) || return _psis_failure_result(p, "hessian_not_pd", t0)

    inv_Σ0 = Matrix{Float64}(Symmetric(inv(prior_chol)))
    prior = MvNormal(μ0, Matrix(Σ0_sym))

    map_res = _psis_newton_map(X_m, y_v, μ0, inv_Σ0, prior)
    map_res.converged || return _psis_failure_result(p, map_res.reason, t0)

    β_hat = map_res.β
    H_chol = map_res.H_chol
    Σ_prop = Matrix{Float64}(Symmetric(inv(H_chol)))
    prop_chol = cholesky(Symmetric(Σ_prop); check = false)
    issuccess(prop_chol) || return _psis_failure_result(p, "hessian_not_pd", t0)

    prop = MvTDist(Float64(df), β_hat, Σ_prop)
    S = Int(n_draws)
    β_draws = rand(rng, prop, S)

    log_r = Vector{Float64}(undef, S)
    @inbounds for s in 1:S
        β_s = view(β_draws, :, s)
        log_r[s] = _psis_log_posterior(X_m, y_v, prior, β_s) - Float64(logpdf(prop, β_s))
    end
    all(isfinite, log_r) || return _psis_failure_result(p, "khat_gate", t0)

    psis_res = PSIS.psis(log_r)
    khat = Float64(psis_res.pareto_shape)
    w = Vector{Float64}(psis_res.weights)
    if !(all(isfinite, w) && isfinite(khat))
        return _psis_failure_result(p, "khat_gate", t0; khat)
    end
    w_sum = sum(w)
    if !(isfinite(w_sum) && w_sum > 0.0)
        return _psis_failure_result(p, "khat_gate", t0; khat)
    end
    w ./= w_sum
    sum_w2 = sum(abs2, w)
    ess = 1.0 / sum_w2

    n_eval = size(X_alt_m, 1)
    η_alt = Vector{Float64}(undef, n_eval)
    η_obs = Vector{Float64}(undef, n_eval)
    policy_draws = Vector{Float64}(undef, S)
    @inbounds for s in 1:S
        β_s = view(β_draws, :, s)
        mul!(η_alt, X_alt_m, β_s)
        mul!(η_obs, X_obs_m, β_s)
        acc = 0.0
        for i in 1:n_eval
            acc += _psis_logistic(η_alt[i]) - _psis_logistic(η_obs[i])
        end
        policy_draws[s] = acc / n_eval
    end

    beta_mean = β_draws * w
    centred_beta = β_draws .- beta_mean
    denom = max(1.0 - sum_w2, eps(Float64))
    beta_cov = Matrix{Float64}(Symmetric(((centred_beta .* transpose(w)) * transpose(centred_beta)) ./ denom))

    policy_mean = dot(w, policy_draws)
    policy_var = 0.0
    @inbounds for s in 1:S
        d = policy_draws[s] - policy_mean
        policy_var += w[s] * d * d
    end
    policy_sd = sqrt(max(0.0, policy_var / denom))

    α = (1.0 - Float64(level)) / 2.0
    policy_lower = _psis_weighted_quantile(policy_draws, w, α)
    policy_upper = _psis_weighted_quantile(policy_draws, w, 1.0 - α)

    mcse = policy_sd / sqrt(ess)
    mcse_to_sd = policy_sd > 0.0 ? (mcse / policy_sd) : (mcse == 0.0 ? 0.0 : Inf)

    if !(isfinite(khat) && khat < khat_max)
        return _psis_failure_result(
            p, "khat_gate", t0;
            khat, ess, mcse, mcse_to_sd, beta_mean, beta_cov,
            policy_mean, policy_sd, policy_lower, policy_upper,
            policy_draws, weights = w,
        )
    end
    if !(isfinite(ess) && ess >= ess_min)
        return _psis_failure_result(
            p, "ess_gate", t0;
            khat, ess, mcse, mcse_to_sd, beta_mean, beta_cov,
            policy_mean, policy_sd, policy_lower, policy_upper,
            policy_draws, weights = w,
        )
    end
    if !(isfinite(mcse_to_sd) && mcse_to_sd <= mcse_sd_max)
        return _psis_failure_result(
            p, "mcse_gate", t0;
            khat, ess, mcse, mcse_to_sd, beta_mean, beta_cov,
            policy_mean, policy_sd, policy_lower, policy_upper,
            policy_draws, weights = w,
        )
    end

    seconds = (time_ns() - t0) * 1e-9
    return (;
        available = true,
        failure_reason = "",
        khat,
        ess,
        mcse,
        mcse_to_sd,
        beta_mean,
        beta_cov,
        policy_mean,
        policy_sd,
        policy_lower,
        policy_upper,
        policy_draws,
        weights = w,
        seconds,
    )
end
