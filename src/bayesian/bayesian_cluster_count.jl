# One-level random-intercept hierarchical Bayesian count (Poisson / NB2) and
# two-part hurdle (Bernoulli-logit + zero-truncated Poisson / NB2) models.
# Reuses weighted Schur-complement block algebra, Golub-Welsch 1D Gauss-Hermite
# quadrature, and CausalTargetedPSISExt (zero dense N×N or K×K allocations).

"""
    ClusterCountPrior(; mean, covariance, cluster_shape, cluster_scale,
        log_dispersion_mean=0.0, log_dispersion_sd=1.5)
    ClusterCountPrior(coefficient_prior::GaussianCoefficientPrior; cluster_shape,
        cluster_scale, log_dispersion_mean=0.0, log_dispersion_sd=1.5)

Explicit hierarchical count prior:
`β ~ Normal(mean, covariance)`, `σ_b² ~ InverseGamma(cluster_shape, cluster_scale)`,
and (when `family = :nb2`) `log(θ) ~ Normal(log_dispersion_mean, log_dispersion_sd²)`.
"""
struct ClusterCountPrior
    coefficient_prior::GaussianCoefficientPrior
    cluster_shape::Float64
    cluster_scale::Float64
    log_dispersion_mean::Float64
    log_dispersion_sd::Float64

    function ClusterCountPrior(coefficient_prior::GaussianCoefficientPrior;
            cluster_shape::Real, cluster_scale::Real,
            log_dispersion_mean::Real = 0.0, log_dispersion_sd::Real = 1.5)
        ab, bb = Float64(cluster_shape), Float64(cluster_scale)
        mϕ, sϕ = Float64(log_dispersion_mean), Float64(log_dispersion_sd)
        isfinite(ab) && ab > 1 ||
            throw(ArgumentError("cluster variance prior shape must be finite and greater than one"))
        isfinite(bb) && bb > 0 ||
            throw(ArgumentError("cluster variance prior scale must be finite and positive"))
        isfinite(mϕ) && isfinite(sϕ) && sϕ > 0 ||
            throw(ArgumentError("log_dispersion_mean must be finite and log_dispersion_sd positive"))
        new(coefficient_prior, ab, bb, mϕ, sϕ)
    end
end

function ClusterCountPrior(; mean::AbstractVector, covariance::AbstractMatrix,
        cluster_shape::Real, cluster_scale::Real,
        log_dispersion_mean::Real = 0.0, log_dispersion_sd::Real = 1.5)
    return ClusterCountPrior(GaussianCoefficientPrior(; mean, covariance);
        cluster_shape, cluster_scale, log_dispersion_mean, log_dispersion_sd)
end

Base.propertynames(::ClusterCountPrior) = (
    :coefficient_prior, :mean, :covariance, :cluster_shape, :cluster_scale,
    :log_dispersion_mean, :log_dispersion_sd,
)

function Base.getproperty(p::ClusterCountPrior, name::Symbol)
    cp = getfield(p, :coefficient_prior)
    if name === :mean
        return cp.mean
    elseif name === :covariance
        return cp.covariance
    else
        return getfield(p, name)
    end
end

_transform_prior(prior::ClusterCountPrior, t::FrozenAffineDesignTransform) =
    ClusterCountPrior(_transform_prior(prior.coefficient_prior, t);
        cluster_shape = prior.cluster_shape, cluster_scale = prior.cluster_scale,
        log_dispersion_mean = prior.log_dispersion_mean,
        log_dispersion_sd = prior.log_dispersion_sd)

"""
    ClusterHurdlePrior(; presence::ClusterLogisticPrior, intensity::ClusterCountPrior)

Explicit two-part hierarchical hurdle prior combining a presence-part
`ClusterLogisticPrior` and a positive-intensity `ClusterCountPrior`.
"""
struct ClusterHurdlePrior
    presence::ClusterLogisticPrior
    intensity::ClusterCountPrior
end

ClusterHurdlePrior(; presence::ClusterLogisticPrior, intensity::ClusterCountPrior) =
    ClusterHurdlePrior(presence, intensity)

struct _CountClusterLayout{T}
    n::Int
    p::Int
    K::Int
    X::Matrix{Float64}
    y::Vector{Float64}
    cluster_levels::Vector{T}
    cluster_index::Vector{Int}
    cluster_rows::Vector{Vector{Int}}
    cluster_sizes::Vector{Int}
end

function _count_cluster_layout(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
        cluster_ids::AbstractVector; positive_only::Bool = false)
    n, p = size(X)
    p > 0 || throw(ArgumentError("design matrix must contain at least one column"))
    length(y) == n || throw(DimensionMismatch("outcome length must match design rows"))
    length(cluster_ids) == n || throw(DimensionMismatch("cluster_ids length must match design rows"))
    all(isfinite, X) && all(yi -> isfinite(yi) && isinteger(yi) && (positive_only ? yi >= 1 : yi >= 0), y) ||
        throw(ArgumentError(positive_only ?
            "positive intensity counts must be finite integers >= 1" :
            "count outcomes must be finite non-negative integers"))
    any(ismissing, cluster_ids) && throw(ArgumentError("cluster_ids must not contain missing values"))
    Xf = Matrix{Float64}(X)
    yf = Float64.(y)
    T = nonmissingtype(eltype(cluster_ids))
    levels = Vector{T}()
    level_to_idx = Dict{T, Int}()
    cidx = Vector{Int}(undef, n)
    rows_by_cluster = Vector{Vector{Int}}()
    @inbounds for i in 1:n
        cid = cluster_ids[i]
        idx = get!(level_to_idx, cid) do
            push!(levels, cid)
            push!(rows_by_cluster, Int[])
            length(levels)
        end
        cidx[i] = idx
        push!(rows_by_cluster[idx], i)
    end
    K = length(levels)
    sizes = [length(r) for r in rows_by_cluster]
    return _CountClusterLayout{T}(n, p, K, Xf, yf, levels, cidx, rows_by_cluster, sizes)
end

# Observation-level log-likelihood, 1st derivative (grad), and negative 2nd derivative (weight)
# wrt linear predictor η = x'β + b, for :poisson, :nb2, and zero-truncated variants.
@inline function _count_obs_loglik_grad_weight(y::Float64, η::Float64,
        family::Symbol, θ::Float64, zero_truncated::Bool)
    ηc = clamp(η, -20.0, 20.0)
    μ = exp(ηc)
    if family === :poisson
        ll = y * ηc - μ - SpecialFunctions.loggamma(y + 1.0)
        g = y - μ
        w = μ
        if zero_truncated
            # log(1 - exp(-μ)) = log(-expm1(-μ))
            e_m = exp(-μ)
            one_m = max(-expm1(-μ), 1e-15)
            ll -= log(one_m)
            # d/dη [-log(1 - e^{-μ})] = -μ e^{-μ} / (1 - e^{-μ})
            r = (μ * e_m) / one_m
            g -= r
            # -d²/dη² [-log(1 - e^{-μ})] = r * (1 - μ / one_m)
            w += r * (1.0 - μ / one_m)
        end
        return (ll, g, max(w, 1e-8))
    else
        # NB2: Var = μ + μ² / θ
        s = θ + μ
        ll = SpecialFunctions.loggamma(y + θ) -
            SpecialFunctions.loggamma(θ) -
            SpecialFunctions.loggamma(y + 1.0) +
            θ * log(θ) + y * ηc - (y + θ) * log(s)
        g = y - (y + θ) * μ / s
        w = (y + θ) * θ * μ / (s * s)
        if zero_truncated
            # p0 = (θ / (θ + μ))^θ
            log_p0 = θ * (log(θ) - log(s))
            p0 = exp(log_p0)
            one_m = max(-expm1(log_p0), 1e-15)
            ll -= log(one_m)
            # q = -d(log p0)/dη = θ μ / s; d/dη[-log(1-p0)] = -p0 * q / (1 - p0)
            q = θ * μ / s
            dq = θ * θ * μ / (s * s)
            r = (p0 * q) / one_m
            g -= r
            # -d²/dη²[-log(1-p0)] = (p0 / one_m) * (dq - q² / one_m)
            w += (p0 / one_m) * (dq - (q * q) / one_m)
        end
        return (ll, g, max(w, 1e-8))
    end
end

"""
Weighted Schur-complement Newton-IRLS Gaussian block solve for general GLM
working weights `W` and working linear RHS `z_rhs[i] = g[i] + W[i] * η[i]`.
"""
function _solve_weighted_glm_block(layout::_CountClusterLayout,
        cp::GaussianCoefficientPrior, W::AbstractVector{Float64},
        z_rhs::AbstractVector{Float64}; cluster_precision::Real)
    p, K, n = layout.p, layout.K, layout.n
    τb = Float64(cluster_precision)
    prior_chol = cholesky(Symmetric(cp.covariance))
    V0_inv = prior_chol \ Matrix{Float64}(I, p, p)
    hβ = (prior_chol \ cp.mean) .+ (n == 0 ? zeros(Float64, p) : Vector{Float64}(layout.X' * z_rhs))

    Λβ = copy(V0_inv)
    if n > 0
        XtW = layout.X' .* reshape(W, 1, n)
        mul!(Λβ, XtW, layout.X, 1.0, 1.0)
    end

    inv_d = zeros(Float64, K)
    s_Wx = zeros(Float64, p, K)
    hk_vec = zeros(Float64, K)
    if isfinite(τb) && K > 0
        @inbounds for k in 1:K
            rows = layout.cluster_rows[k]
            w_sum = 0.0
            hk = 0.0
            for i in rows
                wi = W[i]
                w_sum += wi
                hk += z_rhs[i]
                for j in 1:p
                    s_Wx[j, k] += wi * layout.X[i, j]
                end
            end
            hk_vec[k] = hk
            dk = τb + w_sum
            inv_dk = 1.0 / dk
            inv_d[k] = inv_dk
            for j2 in 1:p
                sxj2 = s_Wx[j2, k]
                hβ[j2] -= (hk * inv_dk) * sxj2
                for j1 in 1:p
                    Λβ[j1, j2] -= inv_dk * s_Wx[j1, k] * sxj2
                end
            end
        end
    end

    Λβ_chol = cholesky(Symmetric(0.5 .* (Λβ .+ Λβ')))
    Σβ = Matrix(Symmetric(Λβ_chol \ Matrix{Float64}(I, p, p)))
    mβ = Λβ_chol \ hβ

    mb = zeros(Float64, K)
    vb = zeros(Float64, K)
    cov_βb = zeros(Float64, p, K)
    expected_cluster_ss = 0.0
    if isfinite(τb) && K > 0
        Σβ_sx = Vector{Float64}(undef, p)
        @inbounds for k in 1:K
            inv_dk = inv_d[k]
            sx = view(s_Wx, :, k)
            mb[k] = inv_dk * (hk_vec[k] - dot(sx, mβ))
            mul!(Σβ_sx, Σβ, sx)
            for j in 1:p
                cov_βb[j, k] = -inv_dk * Σβ_sx[j]
            end
            vb[k] = inv_dk + (inv_dk * inv_dk) * dot(sx, Σβ_sx)
            expected_cluster_ss += mb[k]^2 + vb[k]
        end
    end

    return HierarchicalGaussianBlockState(
        mβ, Σβ, Λβ_chol, mb, vb, inv_d, cov_βb, 1.0, τb, 0.0, expected_cluster_ss)
end

# 1D Newton Laplace update for log(θ) in NB2 / ZTNB2 given current η vector
function _update_nb2_log_dispersion(layout::_CountClusterLayout, η::Vector{Float64},
        prior::ClusterCountPrior, ϕ_init::Float64, zero_truncated::Bool)
    μ_ϕ, s2_ϕ = prior.log_dispersion_mean, prior.log_dispersion_sd^2
    function obj(ϕ)
        θ = exp(clamp(ϕ, -8.0, 8.0))
        ll = -0.5 * ((ϕ - μ_ϕ)^2) / s2_ϕ
        @inbounds for i in 1:layout.n
            ll += _count_obs_loglik_grad_weight(layout.y[i], η[i], :nb2, θ, zero_truncated)[1]
        end
        return ll
    end
    ϕ = clamp(ϕ_init, -6.0, 6.0)
    h = 1e-4
    neg_hess = 1.0 / s2_ϕ
    for _ in 1:15
        f0 = obj(ϕ)
        fp = obj(ϕ + h)
        fm = obj(ϕ - h)
        g = (fp - fm) / (2.0 * h)
        hess = (fp - 2.0 * f0 + fm) / (h * h)
        neg_hess = max(-hess, 1.0 / s2_ϕ)
        step = clamp(g / neg_hess, -1.0, 1.0)
        # Backtracking line search
        α = 1.0
        while α > 1e-4 && obj(clamp(ϕ + α * step, -6.0, 6.0)) < f0 - 1e-12
            α *= 0.5
        end
        ϕ_new = clamp(ϕ + α * step, -6.0, 6.0)
        abs(ϕ_new - ϕ) <= 1e-7 && (ϕ = ϕ_new; break)
        ϕ = ϕ_new
    end
    return (; log_disp_mean = ϕ, log_disp_var = 1.0 / neg_hess)
end

function _structured_vi_hierarchical_count(layout::_CountClusterLayout,
        prior::ClusterCountPrior; family::Symbol = :poisson,
        zero_truncated::Bool = false, maxiter::Integer = 1000, tol::Real = 1e-8)
    family in (:poisson, :nb2) || throw(ArgumentError("family must be :poisson or :nb2"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    cp = prior.coefficient_prior
    p, K, n = layout.p, layout.K, layout.n

    a_b = prior.cluster_shape + 0.5 * K
    b_b = prior.cluster_scale * (a_b / prior.cluster_shape)
    mβ = copy(cp.mean)
    mb = zeros(Float64, K)
    log_disp_mean = prior.log_dispersion_mean
    log_disp_var = prior.log_dispersion_sd^2

    W = Vector{Float64}(undef, n)
    z_rhs = Vector{Float64}(undef, n)
    η = Vector{Float64}(undef, n)
    converged = false
    iters = 0
    local state::HierarchicalGaussianBlockState

    for iter in 1:Int(maxiter)
        iters = iter
        τ_b = a_b / b_b
        θ = exp(log_disp_mean)
        @inbounds for i in 1:n
            k = layout.cluster_index[i]
            ηi = dot(view(layout.X, i, :), mβ) + mb[k]
            η[i] = ηi
            _, gi, wi = _count_obs_loglik_grad_weight(layout.y[i], ηi, family, θ, zero_truncated)
            W[i] = wi
            z_rhs[i] = gi + wi * ηi
        end

        state = _solve_weighted_glm_block(layout, cp, W, z_rhs; cluster_precision = τ_b)
        # Damped step for global stability under extreme counts
        step_β = state.beta_mean - mβ
        step_b = state.cluster_mean - mb
        max_step = max(maximum(abs, step_β; init = 0.0), maximum(abs, step_b; init = 0.0))
        damp = max_step > 2.0 ? (2.0 / max_step) : 1.0
        mβ_new = mβ .+ damp .* step_β
        mb_new = mb .+ damp .* step_b

        b_b_new = prior.cluster_scale + 0.5 * (sum(abs2, mb_new) + sum(state.cluster_var))
        if family === :nb2 && (iter <= 5 || iter % 3 == 0)
            @inbounds for i in 1:n
                η[i] = dot(view(layout.X, i, :), mβ_new) + mb_new[layout.cluster_index[i]]
            end
            disp = _update_nb2_log_dispersion(layout, η, prior, log_disp_mean, zero_truncated)
            log_disp_mean = disp.log_disp_mean
            log_disp_var = disp.log_disp_var
        end

        rel_β = norm(mβ_new - mβ) / max(1.0, norm(mβ_new))
        rel_b = abs(b_b_new - b_b) / max(1.0, abs(b_b_new))
        mβ = mβ_new
        mb = mb_new
        b_b = b_b_new

        if iter > 1 && max(rel_β, rel_b) <= tol
            # Final converged block + dispersion refresh
            θ = exp(log_disp_mean)
            @inbounds for i in 1:n
                ηi = dot(view(layout.X, i, :), mβ) + mb[layout.cluster_index[i]]
                η[i] = ηi
                _, gi, wi = _count_obs_loglik_grad_weight(layout.y[i], ηi, family, θ, zero_truncated)
                W[i] = wi
                z_rhs[i] = gi + wi * ηi
            end
            if family === :nb2
                disp = _update_nb2_log_dispersion(layout, η, prior, log_disp_mean, zero_truncated)
                log_disp_mean = disp.log_disp_mean
                log_disp_var = disp.log_disp_var
            end
            state = _solve_weighted_glm_block(layout, cp, W, z_rhs; cluster_precision = a_b / b_b)
            converged = true
            break
        end
    end

    status = converged ? :converged : :maxiter_reached
    return (; state, cluster_shape = a_b, cluster_scale = b_b,
        log_dispersion_mean = family === :nb2 ? log_disp_mean : NaN,
        log_dispersion_var = family === :nb2 ? log_disp_var : NaN,
        iterations = iters, converged, status)
end

"""
Adaptive Laplace-centred 1D Gauss-Hermite marginalised log-likelihood for one
cluster under `:poisson` or `:nb2` (with optional zero truncation).
"""
function _cluster_count_marginal_loglik(η0::AbstractVector{Float64},
        y::AbstractVector{Float64}, rows::Vector{Int}, σ2_b::Float64,
        family::Symbol, θ::Float64, zero_truncated::Bool)
    τ_b = 1.0 / σ2_b
    b = 0.0
    neg_hess = τ_b
    for _ in 1:12
        grad = -τ_b * b
        neg_hess = τ_b
        @inbounds for i in rows
            _, gi, wi = _count_obs_loglik_grad_weight(y[i], η0[i] + b, family, θ, zero_truncated)
            grad += gi
            neg_hess += wi
        end
        step = clamp(grad / neg_hess, -2.0, 2.0)
        b += step
        abs(step) <= 1e-10 && break
    end
    s = 1.0 / sqrt(neg_hess)
    gh = _GH_21
    Q = length(gh.nodes)
    log_terms = Vector{Float64}(undef, Q)
    max_lt = -Inf
    @inbounds for q in 1:Q
        uq = gh.nodes[q]
        bq = b + s * uq
        ll = -0.5 * τ_b * (bq * bq) - 0.5 * log(2π * σ2_b)
        for i in rows
            ll += _count_obs_loglik_grad_weight(y[i], η0[i] + bq, family, θ, zero_truncated)[1]
        end
        lt = gh.log_weights[q] + ll + 0.5 * (uq * uq)
        log_terms[q] = lt
        lt > max_lt && (max_lt = lt)
    end
    acc = 0.0
    @inbounds for q in 1:Q
        acc += exp(log_terms[q] - max_lt)
    end
    return 0.5 * log(2π) + log(s) + max_lt + log(acc)
end

function _marginal_cluster_count_log_posterior(layout::_CountClusterLayout,
        prior::ClusterCountPrior, family::Symbol, zero_truncated::Bool,
        β::AbstractVector{<:Real}, log_sigma2_b::Real, log_disp::Real = 0.0)
    cp = prior.coefficient_prior
    p, K = layout.p, layout.K
    all(isfinite, β) && isfinite(log_sigma2_b) && isfinite(log_disp) || return -Inf
    σ2_b = exp(Float64(log_sigma2_b))
    θ = family === :nb2 ? exp(Float64(log_disp)) : 1.0
    (σ2_b > 0 && isfinite(σ2_b) && θ > 0 && isfinite(θ)) || return -Inf

    η0 = layout.X * Vector{Float64}(β)
    log_lik = 0.0
    for k in 1:K
        log_lik += _cluster_count_marginal_loglik(η0, layout.y, layout.cluster_rows[k],
            σ2_b, family, θ, zero_truncated)
    end
    prior_chol = cholesky(Symmetric(cp.covariance))
    diff = Vector{Float64}(β) - cp.mean
    log_p_β = -0.5 * p * log(2π) - 0.5 * logdet(prior_chol) - 0.5 * dot(diff, prior_chol \ diff)
    log_p_σ2_b = prior.cluster_shape * log(prior.cluster_scale) -
        SpecialFunctions.loggamma(prior.cluster_shape) -
        prior.cluster_shape * log_sigma2_b - prior.cluster_scale / σ2_b
    log_p_disp = family === :nb2 ?
        logpdf(Normal(prior.log_dispersion_mean, prior.log_dispersion_sd), Float64(log_disp)) : 0.0
    return log_lik + log_p_β + log_p_σ2_b + log_p_disp
end

"""
Typed hierarchical count fit (`:poisson` or `:nb2`, optionally zero-truncated)
from structured Laplace-Gaussian VI `q(β, b) q(σ_b⁻²) [q(log θ)]`.
"""
struct BayesianClusterCountFit
    posterior_mean::Vector{Float64}
    posterior_covariance::Matrix{Float64}
    cluster_mean::Vector{Float64}
    cluster_var::Vector{Float64}
    beta_cluster_cov::Matrix{Float64}
    cluster_shape::Float64
    cluster_scale::Float64
    log_dispersion_mean::Float64
    log_dispersion_var::Float64
    family::Symbol
    zero_truncated::Bool
    status::Symbol
    converged::Bool
    iterations::Int
    prior::ClusterCountPrior
    transform::FrozenAffineDesignTransform
    layout_raw::_CountClusterLayout
    block_state::HierarchicalGaussianBlockState
    outcome::Symbol
    treatment::Symbol
    covariates::Vector{Symbol}
    cluster::Symbol
    schema::CovariateSchema
    covariate_schema::CovariateSchema
    basis::Any
    design_metadata::NamedTuple
    metadata::NamedTuple
end

"""
    fit_bayesian_cluster_count(data, outcome, treatment, covariates; cluster, prior,
        family=:poisson, zero_truncated=false, basis=nothing, identification=nothing,
        handle_missing=:error, maxiter=1000, tol=1e-8, standardize=true)

Fit a one-level random-intercept hierarchical Bayesian count model
(`family = :poisson` or `:nb2`, optionally `zero_truncated = true`) via
weighted Schur-complement structured Laplace-Gaussian VI.
"""
function fit_bayesian_cluster_count(data::AbstractDataFrame, outcome::Symbol,
        treatment::Symbol, covariates::AbstractVector{Symbol};
        cluster::Symbol, prior::ClusterCountPrior, family::Symbol = :poisson,
        zero_truncated::Bool = false, basis = nothing,
        identification = nothing, handle_missing::Symbol = :error,
        maxiter::Integer = 1000, tol::Real = 1e-8, standardize::Bool = true)
    family in (:poisson, :nb2) || throw(ArgumentError("family must be :poisson or :nb2"))
    handle_missing in (:error, :mar) || throw(ArgumentError("handle_missing must be :error or :mar"))
    covariates = collect(covariates)
    length(unique(covariates)) == length(covariates) || throw(ArgumentError("covariates must be unique"))
    treatment ∉ covariates && outcome ∉ [treatment; covariates] ||
        throw(ArgumentError("treatment, outcome and covariates must be distinct"))
    cluster ∉ [outcome; treatment; covariates] ||
        throw(ArgumentError("cluster column must be distinct from outcome, treatment and covariates"))
    _validate_requested_columns(data, [outcome, cluster])
    frame = _bayesian_predictor_frame(data, treatment, covariates)
    nrow(frame) > 0 || throw(ArgumentError("fitting data must contain at least one row"))

    observed = findall(!ismissing, data[!, outcome])
    length(observed) == nrow(data) || handle_missing == :mar || throw(ArgumentError(
        "missing outcomes require explicit handle_missing=:mar"))
    y = try
        Float64.(data[observed, outcome])
    catch
        throw(ArgumentError("count outcome must be real integer-valued"))
    end
    cids = data[observed, cluster]
    any(ismissing, cids) && throw(ArgumentError("cluster column must not contain missing values"))

    graph_status = _bayesian_identification(identification, treatment, outcome, covariates)
    id_status = basis === nothing ? graph_status : :user_assumed
    schema = fit_covariate_schema(frame, [treatment; covariates])
    covariate_schema = fit_covariate_schema(frame, covariates)
    p = length(prior.mean)
    X_all = _bayesian_design(schema, basis, frame, treatment, covariates, frame[!, treatment], p)
    X_obs = X_all[observed, :]

    transform = standardize ? _freeze_affine_transform(X_obs) :
        FrozenAffineDesignTransform(Matrix{Float64}(I, p, p), Matrix{Float64}(I, p, p),
            _detect_intercept_column(X_obs), zeros(p), ones(p))
    X_std = _apply_affine_transform(X_obs, transform)
    prior_std = _transform_prior(prior, transform)

    layout_std = _count_cluster_layout(X_std, y, cids; positive_only = zero_truncated)
    layout_raw = _count_cluster_layout(X_obs, y, cids; positive_only = zero_truncated)
    vi = _structured_vi_hierarchical_count(layout_std, prior_std; family, zero_truncated, maxiter, tol)
    state_orig = _untransform_block_state(vi.state, transform)

    feature_names = basis === nothing ? ["(Intercept)"; schema.feature_names] : ["basis_$j" for j in 1:p]
    design_metadata = (; feature_names, n_columns = p,
        basis_source = basis === nothing ? :covariate_schema : :user_function,
        adjustment = copy(covariates), categorical_levels = _bayesian_categorical_levels(frame),
        treatment_levels = collect(unique(frame[!, treatment])),
        standardized = standardize, outcome_scaling = :none)
    metadata = (; identification_status = id_status,
        graph_identification_status = graph_status,
        outcome_model_status = :user_assumed, identification,
        missingness_status = handle_missing == :mar ? :user_assumed_mar : :complete,
        n_observed = length(observed), n_missing = nrow(data) - length(observed),
        n_clusters = layout_raw.K, cluster_sizes = copy(layout_raw.cluster_sizes),
        likelihood = Symbol(zero_truncated ? "hierarchical_zt_$(family)" : "hierarchical_$(family)"),
        family, zero_truncated, approximation = :structured_laplace_vi,
        convergence_status = vi.status,
        cluster_variance_mean = vi.cluster_scale / (vi.cluster_shape - 1.0),
        dispersion_mode = family === :nb2 ? exp(vi.log_dispersion_mean) : NaN)

    return BayesianClusterCountFit(
        state_orig.beta_mean, state_orig.beta_cov,
        state_orig.cluster_mean, state_orig.cluster_var, state_orig.beta_cluster_cov,
        vi.cluster_shape, vi.cluster_scale,
        vi.log_dispersion_mean, vi.log_dispersion_var,
        family, zero_truncated, vi.status, vi.converged, vi.iterations,
        prior, transform, layout_raw, state_orig,
        outcome, treatment, covariates, cluster, schema, covariate_schema, basis,
        design_metadata, metadata)
end

@inline function _marginal_count_mean(η::Float64, σ_b::Float64, family::Symbol,
        θ::Float64, zero_truncated::Bool)
    if !zero_truncated
        # Exact closed-form log-normal expectation: E[exp(η + b)] = exp(η + 0.5 σ_b²)
        return exp(clamp(η + 0.5 * σ_b * σ_b, -20.0, 20.0))
    end
    # Zero-truncated positive mean E[μ / (1 - p0(μ, θ))] via 25-node GH
    if σ_b <= 0.0
        μ = exp(clamp(η, -20.0, 20.0))
        p0 = family === :poisson ? exp(-μ) : exp(θ * (log(θ) - log(θ + μ)))
        return μ / max(1.0 - p0, 1e-15)
    end
    gh = _GH_25
    acc = 0.0
    @inbounds for q in eachindex(gh.nodes)
        μ = exp(clamp(η + σ_b * gh.nodes[q], -20.0, 20.0))
        p0 = family === :poisson ? exp(-μ) : exp(θ * (log(θ) - log(θ + μ)))
        acc += gh.weights[q] * (μ / max(1.0 - p0, 1e-15))
    end
    return acc
end

function _sample_count_global_draws(fit::BayesianClusterCountFit, S::Int, rng::AbstractRNG,
        use_student_t::Bool)
    if use_student_t
        prop = _draw_student_t_logistic_proposal(fit.block_state, fit.cluster_shape, fit.cluster_scale, S, rng)
        if fit.family === :nb2
            t_disp = LocationScale(fit.log_dispersion_mean, 1.25 * sqrt(fit.log_dispersion_var), TDist(7.0))
            ld = [rand(rng, t_disp) for _ in 1:S]
            lq = prop.log_q .+ [logpdf(t_disp, ld[s]) for s in 1:S]
            return (; beta_draws = prop.beta_draws, σ2_b_draws = prop.σ2_b_draws,
                log_σ2_b_draws = prop.log_σ2_b_draws, log_disp_draws = ld, log_q = lq)
        end
        return (; beta_draws = prop.beta_draws, σ2_b_draws = prop.σ2_b_draws,
            log_σ2_b_draws = prop.log_σ2_b_draws, log_disp_draws = zeros(Float64, S), log_q = prop.log_q)
    end
    p = length(fit.posterior_mean)
    Zβ = randn(rng, p, S)
    ldiv!(fit.block_state.beta_precision_chol.U, Zβ)
    beta_draws = Zβ .+ fit.posterior_mean
    σ2_b_draws = [rand(rng, InverseGamma(fit.cluster_shape, fit.cluster_scale)) for _ in 1:S]
    ld = fit.family === :nb2 ?
        [fit.log_dispersion_mean + sqrt(fit.log_dispersion_var) * randn(rng) for _ in 1:S] :
        zeros(Float64, S)
    return (; beta_draws, σ2_b_draws, log_σ2_b_draws = log.(σ2_b_draws),
        log_disp_draws = ld, log_q = Float64[])
end

"""
    bayesian_policy_contrast(fit::BayesianClusterCountFit, target, policy;
        deltas, n_draws=2000, rng=Random.default_rng(), level=0.95,
        mc_level=0.95, weighting=:individual, cluster=fit.cluster,
        correction=:none, marginalize_random_effects=true)

Marginal `:policy_vs_observed` expected-count contrasts from a fitted
hierarchical Bayesian count model (`:poisson` or `:nb2`, standard or
zero-truncated). When `marginalize_random_effects=true` (default), integrates
over `b_k ~ Normal(0, σ_b²)` (exact `exp(η + 0.5 σ_b²)` for standard counts;
25-node GH for zero-truncated counts).
"""
function bayesian_policy_contrast(fit::BayesianClusterCountFit, target::AbstractDataFrame, policy;
        deltas, n_draws::Integer = 2000, rng::AbstractRNG = Random.default_rng(),
        level::Real = 0.95, mc_level::Real = 0.95, weighting::Symbol = :individual,
        cluster::Union{Nothing, Symbol} = fit.cluster, correction::Symbol = :none,
        marginalize_random_effects::Bool = true)
    policy isa Union{SmoothBoundedLogitShiftPolicy, DiscreteTreatmentPolicy} ||
        throw(ArgumentError("use a fixed-bound smooth policy or DiscreteTreatmentPolicy; clamped ShiftPolicy is unsupported"))
    correction in (:none, :psis) || throw(ArgumentError("correction must be :none or :psis"))
    n_draws >= 2 || throw(ArgumentError("n_draws must be at least 2"))
    isfinite(level) && 0 < level < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
    isfinite(mc_level) && 0 < mc_level < 1 || throw(ArgumentError("mc_level must lie strictly between zero and one"))
    deltas = Float64.(collect(deltas))
    !isempty(deltas) && all(isfinite, deltas) || throw(ArgumentError("deltas must be non-empty and finite"))

    frame = _bayesian_predictor_frame(target, fit.treatment, fit.covariates)
    n_target = nrow(frame)
    n_target > 0 || throw(ArgumentError("target cohort must contain at least one row"))
    w = _target_row_weights(target, weighting, cluster)

    a = frame[!, fit.treatment]
    p = length(fit.posterior_mean)
    categorical_levels = fit.design_metadata.categorical_levels
    X_obs = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, a, p; categorical_levels)
    W_cov = transform_covariates(fit.covariate_schema, frame)
    discrete_values = policy isa DiscreteTreatmentPolicy ?
        _bayesian_discrete_values(a, W_cov, policy, fit.design_metadata.treatment_levels) : nothing

    m = length(deltas)
    X_policies = Vector{Matrix{Float64}}(undef, m)
    for (j, δ) in enumerate(deltas)
        shifted = policy isa DiscreteTreatmentPolicy ? discrete_values :
            apply_policy_values(Float64.(a), δ, policy)
        X_policies[j] = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, shifted, p; categorical_levels)
    end

    policy_constants = policy isa SmoothBoundedLogitShiftPolicy ?
        (; kind = :smooth_bounded_logit_shift, lower = policy.lower, upper = policy.upper, deltas = copy(deltas)) :
        (; kind = :discrete, policy = deepcopy(policy), deltas = copy(deltas))
    target_kind = marginalize_random_effects ? :fixed_cohort_marginal : :fixed_cohort_conditional_b0

    if fit.status !== :converged
        σ_b_pt = marginalize_random_effects ? sqrt(fit.cluster_scale / (fit.cluster_shape - 1.0)) : 0.0
        θ_pt = fit.family === :nb2 ? exp(fit.log_dispersion_mean) : 1.0
        η_obs = X_obs * fit.posterior_mean
        μ_obs = sum(i -> w[i] * _marginal_count_mean(η_obs[i], σ_b_pt, fit.family, θ_pt, fit.zero_truncated), 1:n_target)
        μ_pt = [begin
            η_p = X_policies[j] * fit.posterior_mean
            sum(i -> w[i] * _marginal_count_mean(η_p[i], σ_b_pt, fit.family, θ_pt, fit.zero_truncated), 1:n_target) - μ_obs
        end for j in 1:m]
        nan_vec = fill(NaN, m)
        nan_mat = fill(NaN, m, m)
        metadata = (; fit.metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level = Float64(level), mc_level = Float64(mc_level), n_draws = Int(n_draws),
            correction, uncertainty = :unavailable, intervals = :unavailable,
            intervals_available = false, policy_constants, design = fit.design_metadata)
        return (; deltas, mean = μ_pt, sd = nan_vec, covariance = nan_mat,
            lower = nan_vec, upper = nan_vec, mcse = nan_vec, metadata)
    end

    S = Int(n_draws)
    draws_g = _sample_count_global_draws(fit, S, rng, correction === :psis)
    η_obs_mat = X_obs * draws_g.beta_draws
    η_pol_mats = [X_policies[j] * draws_g.beta_draws for j in 1:m]
    contrast_draws = Matrix{Float64}(undef, m, S)

    @inbounds for s in 1:S
        σ_b = marginalize_random_effects ? sqrt(draws_g.σ2_b_draws[s]) : 0.0
        θs = fit.family === :nb2 ? exp(draws_g.log_disp_draws[s]) : 1.0
        obs_mean = 0.0
        for i in 1:n_target
            obs_mean += w[i] * _marginal_count_mean(η_obs_mat[i, s], σ_b, fit.family, θs, fit.zero_truncated)
        end
        for j in 1:m
            pol_mat = η_pol_mats[j]
            pol_mean = 0.0
            for i in 1:n_target
                pol_mean += w[i] * _marginal_count_mean(pol_mat[i, s], σ_b, fit.family, θs, fit.zero_truncated)
            end
            contrast_draws[j, s] = pol_mean - obs_mean
        end
    end

    if correction === :psis
        log_ratios = Vector{Float64}(undef, S)
        @inbounds for s in 1:S
            βs = view(draws_g.beta_draws, :, s)
            lp = _marginal_cluster_count_log_posterior(fit.layout_raw, fit.prior,
                fit.family, fit.zero_truncated, βs, draws_g.log_σ2_b_draws[s],
                draws_g.log_disp_draws[s])
            log_ratios[s] = lp - draws_g.log_q[s]
        end
        psis_res = _psis_weighted_summary(contrast_draws, log_ratios; level = Float64(level))
        psis_diagnostics = (; passed = psis_res.passed, failures = psis_res.failures,
            khat = psis_res.khat, ess = psis_res.ess, max_mcse_to_sd = psis_res.max_mcse_to_sd)
        metadata = (; fit.metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level = Float64(level), mc_level = Float64(mc_level), n_draws = S, correction,
            uncertainty = psis_res.passed ? :psis_refined_moments : :unavailable,
            intervals = psis_res.passed ? :pointwise : :unavailable,
            intervals_available = psis_res.passed, psis_diagnostics,
            policy_constants, design = fit.design_metadata)
        return (; deltas, mean = psis_res.mean, sd = psis_res.sd,
            covariance = psis_res.covariance, lower = psis_res.lower,
            upper = psis_res.upper, mcse = psis_res.mcse, metadata)
    end

    α = (1.0 - Float64(level)) / 2.0
    μ = vec(mean(contrast_draws; dims = 2))
    sd = [std(view(contrast_draws, j, :)) for j in 1:m]
    mcse = sd ./ sqrt(S)
    cov_mat = Matrix(Symmetric(cov(contrast_draws; dims = 2)))
    lower = [quantile(view(contrast_draws, j, :), α) for j in 1:m]
    upper = [quantile(view(contrast_draws, j, :), 1.0 - α) for j in 1:m]

    metadata = (; fit.metadata..., functional = :policy_vs_observed,
        factual_reference = :observed, target = target_kind, weighting, n_target,
        level = Float64(level), mc_level = Float64(mc_level), n_draws = S, correction,
        uncertainty = :structured_laplace_vi, intervals = :pointwise,
        intervals_available = true, policy_constants, design = fit.design_metadata)
    return (; deltas, mean = μ, sd, covariance = cov_mat, lower, upper, mcse, metadata)
end

"""
Typed two-part hierarchical hurdle fit combining a Bernoulli-logit presence fit
(`BayesianClusterLogisticFit`) and a zero-truncated count intensity fit
(`BayesianClusterCountFit`).
"""
struct BayesianClusterHurdleFit
    presence_fit::BayesianClusterLogisticFit
    intensity_fit::BayesianClusterCountFit
    family::Symbol
    status::Symbol
    converged::Bool
    prior::ClusterHurdlePrior
    outcome::Symbol
    treatment::Symbol
    covariates::Vector{Symbol}
    cluster::Symbol
    metadata::NamedTuple
end

"""
    fit_bayesian_cluster_hurdle(data, outcome, treatment, covariates; cluster,
        prior::ClusterHurdlePrior, family=:nb2, basis=nothing,
        identification=nothing, handle_missing=:error, maxiter=1000,
        tol=1e-8, standardize=true)

Fit a two-part hierarchical Bayesian hurdle model for zero-inflated counts:
Part 1 fits binary presence `I(Y > 0)` via `fit_bayesian_cluster_logistic`, and
Part 2 fits positive intensity `Y | Y > 0` via `fit_bayesian_cluster_count`
with `zero_truncated = true` (`family = :poisson` or `:nb2`).
"""
function fit_bayesian_cluster_hurdle(data::AbstractDataFrame, outcome::Symbol,
        treatment::Symbol, covariates::AbstractVector{Symbol};
        cluster::Symbol, prior::ClusterHurdlePrior, family::Symbol = :nb2,
        basis = nothing, identification = nothing, handle_missing::Symbol = :error,
        maxiter::Integer = 1000, tol::Real = 1e-8, standardize::Bool = true)
    family in (:poisson, :nb2) || throw(ArgumentError("family must be :poisson or :nb2"))
    _validate_requested_columns(data, [outcome, cluster])
    observed = findall(!ismissing, data[!, outcome])
    length(observed) == nrow(data) || handle_missing == :mar || throw(ArgumentError(
        "missing outcomes require explicit handle_missing=:mar"))

    # Build binary presence column cleanly without mutating caller DataFrame
    df_pres = copy(data)
    pres_col = Vector{Union{Missing, Int}}(undef, nrow(data))
    @inbounds for i in 1:nrow(data)
        v = data[i, outcome]
        if ismissing(v)
            pres_col[i] = missing
        else
            (v isa Real && isfinite(v) && isinteger(v) && v >= 0) ||
                throw(ArgumentError("hurdle outcome must be non-negative integer counts"))
            pres_col[i] = v > 0 ? 1 : 0
        end
    end
    df_pres[!, outcome] = pres_col

    pos_rows = [i for i in observed if data[i, outcome] > 0]
    !isempty(pos_rows) || throw(ArgumentError("hurdle fitting requires at least one positive outcome"))
    df_pos = data[pos_rows, :]

    pres_fit = fit_bayesian_cluster_logistic(df_pres, outcome, treatment, covariates;
        cluster, prior = prior.presence, basis, identification, handle_missing,
        maxiter, tol, standardize)
    int_fit = fit_bayesian_cluster_count(df_pos, outcome, treatment, covariates;
        cluster, prior = prior.intensity, family, zero_truncated = true,
        basis, identification, handle_missing = :error, maxiter, tol, standardize)

    converged = pres_fit.converged && int_fit.converged
    status = converged ? :converged : :maxiter_reached
    metadata = (; identification_status = pres_fit.metadata.identification_status,
        outcome_model_status = :user_assumed,
        likelihood = Symbol("hierarchical_hurdle_$(family)"),
        family, n_observed = length(observed), n_positive = length(pos_rows),
        zero_fraction = 1.0 - length(pos_rows) / length(observed),
        part_statuses = (; presence = pres_fit.status, intensity = int_fit.status),
        independent_part_random_effects = true)
    return BayesianClusterHurdleFit(pres_fit, int_fit, family, status, converged,
        prior, outcome, treatment, collect(covariates), cluster, metadata)
end

"""
    bayesian_policy_contrast(fit::BayesianClusterHurdleFit, target, policy;
        deltas, n_draws=2000, rng=Random.default_rng(), level=0.95,
        mc_level=0.95, weighting=:individual, cluster=fit.cluster,
        correction=:none, marginalize_random_effects=true)

Joint marginal `:policy_vs_observed` contrasts from a fitted two-part
hierarchical hurdle model, returning `overall` (combined expected count
contrast), `presence` (binary prevalence contrast), and `intensity`
(positive-part intensity contrast) evaluated on shared posterior draws.
"""
function bayesian_policy_contrast(fit::BayesianClusterHurdleFit, target::AbstractDataFrame, policy;
        deltas, n_draws::Integer = 2000, rng::AbstractRNG = Random.default_rng(),
        level::Real = 0.95, mc_level::Real = 0.95, weighting::Symbol = :individual,
        cluster::Union{Nothing, Symbol} = fit.cluster, correction::Symbol = :none,
        marginalize_random_effects::Bool = true)
    policy isa Union{SmoothBoundedLogitShiftPolicy, DiscreteTreatmentPolicy} ||
        throw(ArgumentError("use a fixed-bound smooth policy or DiscreteTreatmentPolicy; clamped ShiftPolicy is unsupported"))
    correction in (:none, :psis) || throw(ArgumentError("correction must be :none or :psis"))
    n_draws >= 2 || throw(ArgumentError("n_draws must be at least 2"))
    isfinite(level) && 0 < level < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
    deltas = Float64.(collect(deltas))
    !isempty(deltas) && all(isfinite, deltas) || throw(ArgumentError("deltas must be non-empty and finite"))

    pf, inf = fit.presence_fit, fit.intensity_fit
    frame = _bayesian_predictor_frame(target, pf.treatment, pf.covariates)
    n_target = nrow(frame)
    n_target > 0 || throw(ArgumentError("target cohort must contain at least one row"))
    w = _target_row_weights(target, weighting, cluster)

    a = frame[!, pf.treatment]
    p1 = length(pf.posterior_mean)
    p2 = length(inf.posterior_mean)
    X1_obs = _bayesian_design(pf.schema, pf.basis, frame, pf.treatment, pf.covariates, a, p1;
        categorical_levels = pf.design_metadata.categorical_levels)
    X2_obs = _bayesian_design(inf.schema, inf.basis, frame, inf.treatment, inf.covariates, a, p2;
        categorical_levels = inf.design_metadata.categorical_levels)
    W_cov = transform_covariates(pf.covariate_schema, frame)
    discrete_values = policy isa DiscreteTreatmentPolicy ?
        _bayesian_discrete_values(a, W_cov, policy, pf.design_metadata.treatment_levels) : nothing

    m = length(deltas)
    X1_pols = Vector{Matrix{Float64}}(undef, m)
    X2_pols = Vector{Matrix{Float64}}(undef, m)
    for (j, δ) in enumerate(deltas)
        shifted = policy isa DiscreteTreatmentPolicy ? discrete_values :
            apply_policy_values(Float64.(a), δ, policy)
        X1_pols[j] = _bayesian_design(pf.schema, pf.basis, frame, pf.treatment, pf.covariates, shifted, p1;
            categorical_levels = pf.design_metadata.categorical_levels)
        X2_pols[j] = _bayesian_design(inf.schema, inf.basis, frame, inf.treatment, inf.covariates, shifted, p2;
            categorical_levels = inf.design_metadata.categorical_levels)
    end

    target_kind = marginalize_random_effects ? :fixed_cohort_marginal : :fixed_cohort_conditional_b0
    if fit.status !== :converged
        nan_vec = fill(NaN, m)
        nan_mat = fill(NaN, m, m)
        metadata = (; fit.metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level = Float64(level), n_draws = Int(n_draws), correction,
            uncertainty = :unavailable, intervals = :unavailable, intervals_available = false)
        part_na = (; mean = nan_vec, sd = nan_vec, lower = nan_vec, upper = nan_vec,
            mcse = nan_vec, covariance = nan_mat)
        return (; deltas, mean = nan_vec, sd = nan_vec, lower = nan_vec, upper = nan_vec,
            mcse = nan_vec, covariance = nan_mat,
            overall = part_na, presence = part_na, intensity = part_na, metadata)
    end

    S = Int(n_draws)
    use_t = correction === :psis
    if use_t
        d1 = _draw_student_t_logistic_proposal(pf.block_state, pf.cluster_shape, pf.cluster_scale, S, rng)
    else
        Z1 = randn(rng, p1, S)
        ldiv!(pf.block_state.beta_precision_chol.U, Z1)
        s2_1 = [rand(rng, InverseGamma(pf.cluster_shape, pf.cluster_scale)) for _ in 1:S]
        d1 = (; beta_draws = Z1 .+ pf.posterior_mean, σ2_b_draws = s2_1,
            log_σ2_b_draws = log.(s2_1), log_q = Float64[])
    end
    d2 = _sample_count_global_draws(inf, S, rng, use_t)

    η1_obs = X1_obs * d1.beta_draws
    η2_obs = X2_obs * d2.beta_draws
    η1_pols = [X1_pols[j] * d1.beta_draws for j in 1:m]
    η2_pols = [X2_pols[j] * d2.beta_draws for j in 1:m]

    # Stack [overall (1:m); presence (m+1:2m); intensity (2m+1:3m)]
    all_draws = Matrix{Float64}(undef, 3 * m, S)
    @inbounds for s in 1:S
        σ1 = marginalize_random_effects ? sqrt(d1.σ2_b_draws[s]) : 0.0
        σ2 = marginalize_random_effects ? sqrt(d2.σ2_b_draws[s]) : 0.0
        θs = inf.family === :nb2 ? exp(d2.log_disp_draws[s]) : 1.0
        obs_p = 0.0; obs_int = 0.0; obs_ov = 0.0
        for i in 1:n_target
            pi_ = _marginal_logistic_prob(η1_obs[i, s], σ1)
            mi_ = _marginal_count_mean(η2_obs[i, s], σ2, inf.family, θs, true)
            wi = w[i]
            obs_p += wi * pi_
            obs_int += wi * mi_
            obs_ov += wi * (pi_ * mi_)
        end
        for j in 1:m
            p_mat = η1_pols[j]; i_mat = η2_pols[j]
            pol_p = 0.0; pol_int = 0.0; pol_ov = 0.0
            for i in 1:n_target
                pi_ = _marginal_logistic_prob(p_mat[i, s], σ1)
                mi_ = _marginal_count_mean(i_mat[i, s], σ2, inf.family, θs, true)
                wi = w[i]
                pol_p += wi * pi_
                pol_int += wi * mi_
                pol_ov += wi * (pi_ * mi_)
            end
            all_draws[j, s] = pol_ov - obs_ov
            all_draws[m + j, s] = pol_p - obs_p
            all_draws[2 * m + j, s] = pol_int - obs_int
        end
    end

    function _slice_part(μ, sd, lo, hi, mc, cov_full, rng_idx)
        return (; mean = μ[rng_idx], sd = sd[rng_idx], lower = lo[rng_idx],
            upper = hi[rng_idx], mcse = mc[rng_idx],
            covariance = Matrix(Symmetric(cov_full[rng_idx, rng_idx])))
    end

    if correction === :psis
        log_ratios = Vector{Float64}(undef, S)
        @inbounds for s in 1:S
            lp1 = _marginal_cluster_logistic_log_posterior(pf.layout_raw, pf.prior,
                view(d1.beta_draws, :, s), d1.log_σ2_b_draws[s])
            lp2 = _marginal_cluster_count_log_posterior(inf.layout_raw, inf.prior,
                inf.family, true, view(d2.beta_draws, :, s),
                d2.log_σ2_b_draws[s], d2.log_disp_draws[s])
            log_ratios[s] = (lp1 - d1.log_q[s]) + (lp2 - d2.log_q[s])
        end
        pr = _psis_weighted_summary(all_draws, log_ratios; level = Float64(level))
        psis_diagnostics = (; passed = pr.passed, failures = pr.failures,
            khat = pr.khat, ess = pr.ess, max_mcse_to_sd = pr.max_mcse_to_sd)
        ov = _slice_part(pr.mean, pr.sd, pr.lower, pr.upper, pr.mcse, pr.covariance, 1:m)
        ps = _slice_part(pr.mean, pr.sd, pr.lower, pr.upper, pr.mcse, pr.covariance, (m + 1):(2 * m))
        it = _slice_part(pr.mean, pr.sd, pr.lower, pr.upper, pr.mcse, pr.covariance, (2 * m + 1):(3 * m))
        metadata = (; fit.metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level = Float64(level), n_draws = S, correction,
            uncertainty = pr.passed ? :psis_refined_moments : :unavailable,
            intervals = pr.passed ? :pointwise : :unavailable,
            intervals_available = pr.passed, psis_diagnostics)
        return (; deltas, mean = ov.mean, sd = ov.sd, lower = ov.lower, upper = ov.upper,
            mcse = ov.mcse, covariance = ov.covariance,
            overall = ov, presence = ps, intensity = it, metadata)
    end

    α = (1.0 - Float64(level)) / 2.0
    μ = vec(mean(all_draws; dims = 2))
    sd = [std(view(all_draws, j, :)) for j in 1:(3 * m)]
    mc = sd ./ sqrt(S)
    lo = [quantile(view(all_draws, j, :), α) for j in 1:(3 * m)]
    hi = [quantile(view(all_draws, j, :), 1.0 - α) for j in 1:(3 * m)]
    cov_full = Matrix(Symmetric(cov(all_draws; dims = 2)))
    ov = _slice_part(μ, sd, lo, hi, mc, cov_full, 1:m)
    ps = _slice_part(μ, sd, lo, hi, mc, cov_full, (m + 1):(2 * m))
    it = _slice_part(μ, sd, lo, hi, mc, cov_full, (2 * m + 1):(3 * m))
    metadata = (; fit.metadata..., functional = :policy_vs_observed,
        factual_reference = :observed, target = target_kind, weighting, n_target,
        level = Float64(level), n_draws = S, correction,
        uncertainty = :structured_laplace_vi, intervals = :pointwise, intervals_available = true)
    return (; deltas, mean = ov.mean, sd = ov.sd, lower = ov.lower, upper = ov.upper,
        mcse = ov.mcse, covariance = ov.covariance,
        overall = ov, presence = ps, intensity = it, metadata)
end
