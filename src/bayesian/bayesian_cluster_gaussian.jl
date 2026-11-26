"""
    ClusterGaussianPrior(; mean, covariance, residual_shape, residual_scale, cluster_shape, cluster_scale)
    ClusterGaussianPrior(coefficient_prior::ConjugateGaussianPrior; cluster_shape, cluster_scale)

Explicit hierarchical Gaussian prior for a one-level random-intercept model:
`β | σ_ε² ~ Normal(mean, σ_ε² * covariance)`,
`σ_ε² ~ InverseGamma(residual_shape, residual_scale)`, and
`σ_b² ~ InverseGamma(cluster_shape, cluster_scale)`.

Requires symmetric positive-definite `covariance`, `residual_shape > 1`,
`residual_scale > 0`, `cluster_shape > 1`, and `cluster_scale > 0` so marginal
prior variances exist. Prior locations and scales must be supplied explicitly;
no outcome-dependent rescaling is performed.
"""
struct ClusterGaussianPrior
    coefficient_prior::ConjugateGaussianPrior
    cluster_shape::Float64
    cluster_scale::Float64

    function ClusterGaussianPrior(coefficient_prior::ConjugateGaussianPrior;
            cluster_shape::Real, cluster_scale::Real)
        ab, bb = Float64(cluster_shape), Float64(cluster_scale)
        isfinite(ab) && ab > 1 ||
            throw(ArgumentError("cluster variance prior shape must be finite and greater than one"))
        isfinite(bb) && bb > 0 ||
            throw(ArgumentError("cluster variance prior scale must be finite and positive"))
        new(coefficient_prior, ab, bb)
    end
end

function ClusterGaussianPrior(; mean::AbstractVector, covariance::AbstractMatrix,
        residual_shape::Real, residual_scale::Real,
        cluster_shape::Real, cluster_scale::Real)
    coef_prior = ConjugateGaussianPrior(; mean, covariance,
        shape = residual_shape, scale = residual_scale)
    return ClusterGaussianPrior(coef_prior; cluster_shape, cluster_scale)
end

Base.propertynames(::ClusterGaussianPrior) = (
    :coefficient_prior, :mean, :covariance,
    :residual_shape, :residual_scale, :cluster_shape, :cluster_scale,
)

function Base.getproperty(p::ClusterGaussianPrior, name::Symbol)
    cp = getfield(p, :coefficient_prior)
    if name === :mean
        return cp.mean
    elseif name === :covariance
        return cp.covariance
    elseif name === :residual_shape
        return cp.shape
    elseif name === :residual_scale
        return cp.scale
    else
        return getfield(p, name)
    end
end

function _transform_prior(prior::ConjugateGaussianPrior, t::FrozenAffineDesignTransform)
    length(prior.mean) == size(t.A, 1) ||
        throw(DimensionMismatch("prior dimension must match transform dimension"))
    μ_std = t.A_inv * prior.mean
    V_std = Matrix(Symmetric(t.A_inv * prior.covariance * t.A_inv'))
    return ConjugateGaussianPrior(; mean = μ_std, covariance = V_std,
        shape = prior.shape, scale = prior.scale)
end

_transform_prior(prior::ClusterGaussianPrior, t::FrozenAffineDesignTransform) =
    ClusterGaussianPrior(_transform_prior(prior.coefficient_prior, t);
        cluster_shape = prior.cluster_shape, cluster_scale = prior.cluster_scale)

"""
Cached cluster sufficient statistics for one-level random-intercept Gaussian
models. Avoids dense `N × N` or `K × K` allocations.
"""
struct HierarchicalGaussianSufficientStats{T}
    n::Int
    p::Int
    K::Int
    cluster_levels::Vector{T}
    cluster_index::Vector{Int}
    cluster_sizes::Vector{Int}
    cluster_x_sums::Matrix{Float64}
    cluster_y_sums::Vector{Float64}
    xtx::Matrix{Float64}
    xty::Vector{Float64}
    yty::Float64
end

function _cluster_sufficient_stats(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
        cluster_ids::AbstractVector)
    idx = _build_cluster_index(X, y, cluster_ids;
        validate_y = isfinite,
        invalid_msg = "hierarchical Gaussian inputs must be finite")
    n, p, K = idx.n, idx.p, idx.K
    Xf, yf = idx.X, idx.y
    x_sums = zeros(Float64, p, K)
    y_sums = zeros(Float64, K)
    @inbounds for i in 1:n
        k = idx.cluster_index[i]
        y_sums[k] += yf[i]
        for j in 1:p
            x_sums[j, k] += Xf[i, j]
        end
    end
    xtx = n == 0 ? zeros(Float64, p, p) : Matrix(Symmetric(Xf' * Xf))
    xty = n == 0 ? zeros(Float64, p) : Vector{Float64}(Xf' * yf)
    yty = n == 0 ? 0.0 : Float64(dot(yf, yf))
    T = eltype(idx.cluster_levels)
    return HierarchicalGaussianSufficientStats{T}(
        n, p, K, idx.cluster_levels, idx.cluster_index, idx.cluster_sizes,
        x_sums, y_sums, xtx, xty, yty)
end

function _solve_hierarchical_gaussian_block(stats::HierarchicalGaussianSufficientStats,
        prior::Union{ConjugateGaussianPrior, ClusterGaussianPrior};
        residual_precision::Real, cluster_precision::Real)
    cp = prior isa ClusterGaussianPrior ? prior.coefficient_prior : prior
    p, K = stats.p, stats.K
    length(cp.mean) == p || throw(DimensionMismatch("prior mean must match $p design columns"))
    τε = Float64(residual_precision)
    τb = Float64(cluster_precision)
    isfinite(τε) && τε > 0 || throw(ArgumentError("residual_precision must be finite and positive"))
    (isfinite(τb) && τb > 0) || τb == Inf ||
        throw(ArgumentError("cluster_precision must be positive (or Inf for zero cluster variance)"))

    prior_chol = cholesky(Symmetric(cp.covariance))
    V0_inv = prior_chol \ Matrix{Float64}(I, p, p)
    V0_inv_μ0 = prior_chol \ cp.mean

    # Q_ββ = τ_ε (V₀⁻¹ + X'X), h_β = τ_ε (V₀⁻¹ μ₀ + X'y)
    Λβ = τε .* (V0_inv .+ stats.xtx)
    hβ = τε .* (V0_inv_μ0 .+ stats.xty)

    inv_d = zeros(Float64, K)
    if isfinite(τb) && K > 0
        @inbounds for k in 1:K
            dk = τε * stats.cluster_sizes[k] + τb
            inv_dk = 1.0 / dk
            inv_d[k] = inv_dk
            sx = view(stats.cluster_x_sums, :, k)
            # Subtract (τ_ε² / d_k) s_{X,k} s_{X,k}' from Λ_β
            c_mat = (τε * τε) * inv_dk
            for j2 in 1:p
                sxj2 = sx[j2]
                for j1 in 1:p
                    Λβ[j1, j2] -= c_mat * sx[j1] * sxj2
                end
            end
            # Subtract (τ_ε² s_{y,k} / d_k) s_{X,k} from h_β
            c_vec = (τε * τε) * stats.cluster_y_sums[k] * inv_dk
            for j1 in 1:p
                hβ[j1] -= c_vec * sx[j1]
            end
        end
    end

    Λβ_sym = Symmetric(0.5 .* (Λβ .+ Λβ'))
    Λβ_chol = cholesky(Λβ_sym)
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
            w_k = τε * inv_dk
            sx = view(stats.cluster_x_sums, :, k)
            mb[k] = w_k * (stats.cluster_y_sums[k] - dot(sx, mβ))
            mul!(Σβ_sx, Σβ, sx)
            for j in 1:p
                cov_βb[j, k] = -w_k * Σβ_sx[j]
            end
            vb[k] = inv_dk + (w_k * w_k) * dot(sx, Σβ_sx)
            expected_cluster_ss += mb[k]^2 + vb[k]
        end
    end

    # Exact E[‖y - Xβ - Zb‖²] in O(K p + p²) without O(N) observation pass
    expected_residual_ss = stats.yty - 2.0 * dot(mβ, stats.xty) +
        dot(mβ, stats.xtx * mβ) + tr(stats.xtx * Σβ)
    if isfinite(τb) && K > 0
        @inbounds for k in 1:K
            sx = view(stats.cluster_x_sums, :, k)
            expected_residual_ss += -2.0 * mb[k] * stats.cluster_y_sums[k] +
                stats.cluster_sizes[k] * (mb[k]^2 + vb[k]) +
                2.0 * (dot(sx, mβ) * mb[k] + dot(sx, view(cov_βb, :, k)))
        end
    end
    expected_residual_ss = max(0.0, expected_residual_ss)

    return HierarchicalGaussianBlockState(
        mβ, Σβ, Λβ_chol, mb, vb, inv_d, cov_βb, τε, τb,
        expected_residual_ss, expected_cluster_ss)
end

"""
Generate joint draws `(beta_draws, cluster_draws)` (`p × n_draws`, `K × n_draws`)
from `q(β, b)` via the exact conditional factorisation `q(β) ∏_k q(b_k | β)`.
"""
function _sample_hierarchical_gaussian_block(state::HierarchicalGaussianBlockState,
        stats::HierarchicalGaussianSufficientStats;
        n_draws::Integer, rng::AbstractRNG = Random.default_rng())
    S = Int(n_draws)
    S > 0 || throw(ArgumentError("n_draws must be positive"))
    p, K = stats.p, stats.K
    # β = m_β + U⁻¹ ζ where Λ_β = U'U (Cholesky upper factor)
    Zβ = randn(rng, p, S)
    ldiv!(state.beta_precision_chol.U, Zβ)
    beta_draws = Zβ .+ state.beta_mean

    cluster_draws = zeros(Float64, K, S)
    if isfinite(state.cluster_precision) && K > 0
        τε = state.residual_precision
        # s_{X,k}' β^{(s)} for all k, s in one BLAS gemm
        sx_beta = stats.cluster_x_sums' * beta_draws
        @inbounds for k in 1:K
            inv_dk = state.cluster_cond_var[k]
            w_k = τε * inv_dk
            sd_k = sqrt(inv_dk)
            sy_k = stats.cluster_y_sums[k]
            for s in 1:S
                cluster_draws[k, s] = w_k * (sy_k - sx_beta[k, s]) + sd_k * randn(rng)
            end
        end
    end
    return (; beta_draws, cluster_draws)
end

function _hierarchical_gaussian_elbo(state::HierarchicalGaussianBlockState,
        stats::HierarchicalGaussianSufficientStats,
        prior::ClusterGaussianPrior,
        a_eps::Float64, b_eps::Float64, a_b::Float64, b_b::Float64)
    cp = prior.coefficient_prior
    p, K, n = stats.p, stats.K, stats.n
    prior_chol = cholesky(Symmetric(cp.covariance))
    V0_inv = prior_chol \ Matrix{Float64}(I, p, p)
    logdet_V0 = logdet(prior_chol)
    diff = state.beta_mean - cp.mean
    prior_quad = dot(diff, V0_inv * diff) + tr(V0_inv * state.beta_cov)

    τ_eps = a_eps / b_eps
    τ_b = a_b / b_b
    E_log_τ_eps = SpecialFunctions.digamma(a_eps) - log(b_eps)
    E_log_τ_b = SpecialFunctions.digamma(a_b) - log(b_b)

    # E_q[log p(y, β, b, τ_ε, τ_b)]
    e_log_p_y = -0.5 * n * log(2π) + 0.5 * n * E_log_τ_eps - 0.5 * τ_eps * state.expected_residual_ss
    e_log_p_β = -0.5 * p * log(2π) - 0.5 * logdet_V0 + 0.5 * p * E_log_τ_eps - 0.5 * τ_eps * prior_quad
    e_log_p_b = -0.5 * K * log(2π) + 0.5 * K * E_log_τ_b - 0.5 * τ_b * state.expected_cluster_ss
    e_log_p_τ_eps = cp.shape * log(cp.scale) - SpecialFunctions.loggamma(cp.shape) +
        (cp.shape - 1.0) * E_log_τ_eps - cp.scale * τ_eps
    e_log_p_τ_b = prior.cluster_shape * log(prior.cluster_scale) -
        SpecialFunctions.loggamma(prior.cluster_shape) +
        (prior.cluster_shape - 1.0) * E_log_τ_b - prior.cluster_scale * τ_b

    # H[q(β, b)] = (p+K)/2 * (1 + log(2π)) - 0.5 * (log|Λ_β| + Σ_k log d_k)
    logdet_Λβ = logdet(state.beta_precision_chol)
    sum_log_dk = K == 0 ? 0.0 : -sum(log, state.cluster_cond_var)
    h_q_βb = 0.5 * (p + K) * (1.0 + log(2π)) - 0.5 * (logdet_Λβ + sum_log_dk)
    h_q_τ_eps = a_eps - log(b_eps) + SpecialFunctions.loggamma(a_eps) +
        (1.0 - a_eps) * SpecialFunctions.digamma(a_eps)
    h_q_τ_b = a_b - log(b_b) + SpecialFunctions.loggamma(a_b) +
        (1.0 - a_b) * SpecialFunctions.digamma(a_b)

    return e_log_p_y + e_log_p_β + e_log_p_b + e_log_p_τ_eps + e_log_p_τ_b +
        h_q_βb + h_q_τ_eps + h_q_τ_b
end

"""
Structured coordinate-ascent variational inference for
`q(β, b) q(σ_ε⁻²) q(σ_b⁻²)` using exact Schur-complement block updates.
"""
function _structured_vi_hierarchical_gaussian(stats::HierarchicalGaussianSufficientStats,
        prior::ClusterGaussianPrior; maxiter::Integer = 1000, tol::Real = 1e-8)
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    cp = prior.coefficient_prior
    p, K, n = stats.p, stats.K, stats.n
    prior_chol = cholesky(Symmetric(cp.covariance))
    V0_inv = prior_chol \ Matrix{Float64}(I, p, p)

    a_eps = cp.shape + 0.5 * (n + p)
    a_b = prior.cluster_shape + 0.5 * K
    # Initialise at prior mean precisions
    b_eps = cp.scale * (a_eps / cp.shape)
    b_b = prior.cluster_scale * (a_b / prior.cluster_shape)

    elbo_trace = Float64[]
    sizehint!(elbo_trace, min(Int(maxiter), 128))
    prev_elbo = -Inf
    prev_mβ = fill(NaN, p)
    prev_b_eps = NaN
    prev_b_b = NaN
    converged = false
    iters = 0
    local state::HierarchicalGaussianBlockState

    for iter in 1:Int(maxiter)
        iters = iter
        τ_eps = a_eps / b_eps
        τ_b = a_b / b_b
        state = _solve_hierarchical_gaussian_block(stats, prior;
            residual_precision = τ_eps, cluster_precision = τ_b)

        diff = state.beta_mean - cp.mean
        prior_quad = dot(diff, V0_inv * diff) + tr(V0_inv * state.beta_cov)
        b_eps = cp.scale + 0.5 * (state.expected_residual_ss + prior_quad)
        b_b = prior.cluster_scale + 0.5 * state.expected_cluster_ss

        elbo = _hierarchical_gaussian_elbo(state, stats, prior, a_eps, b_eps, a_b, b_b)
        push!(elbo_trace, elbo)

        if iter > 1
            rel_elbo = abs(elbo - prev_elbo) / max(1.0, abs(elbo))
            rel_mβ = norm(state.beta_mean - prev_mβ) / max(1.0, norm(state.beta_mean))
            rel_eps = abs(b_eps - prev_b_eps) / max(1.0, abs(b_eps))
            rel_b = abs(b_b - prev_b_b) / max(1.0, abs(b_b))
            if max(rel_elbo, rel_mβ, rel_eps, rel_b) <= tol
                # Refresh Gaussian block at final converged precisions
                state = _solve_hierarchical_gaussian_block(stats, prior;
                    residual_precision = a_eps / b_eps, cluster_precision = a_b / b_b)
                converged = true
                break
            end
        end
        prev_elbo = elbo
        prev_mβ = copy(state.beta_mean)
        prev_b_eps = b_eps
        prev_b_b = b_b
    end

    if !converged
        state = _solve_hierarchical_gaussian_block(stats, prior;
            residual_precision = a_eps / b_eps, cluster_precision = a_b / b_b)
    end

    status = converged ? :converged : :maxiter_reached
    return (; state, residual_shape = a_eps, residual_scale = b_eps,
        cluster_shape = a_b, cluster_scale = b_b,
        elbo = isempty(elbo_trace) ? NaN : elbo_trace[end],
        elbo_trace, iterations = iters, converged, status)
end

"""
Exact analytically marginalised log-posterior
`log p(β, log σ_ε², log σ_b² | y)` (up to normalising constant) after
integrating out all cluster random intercepts `b_1, …, b_K` in `O(K p + p²)`.
Includes the change-of-variables Jacobians `+log σ_ε² + log σ_b²`.
"""
function _marginal_cluster_log_posterior(stats::HierarchicalGaussianSufficientStats,
        prior::ClusterGaussianPrior, β::AbstractVector{<:Real},
        log_sigma2_eps::Real, log_sigma2_b::Real)
    cp = prior.coefficient_prior
    p, K, n = stats.p, stats.K, stats.n
    length(β) == p || throw(DimensionMismatch("β length must match $p"))
    all(isfinite, β) && isfinite(log_sigma2_eps) && isfinite(log_sigma2_b) || return -Inf

    σ2_eps = exp(Float64(log_sigma2_eps))
    σ2_b = exp(Float64(log_sigma2_b))
    (σ2_eps > 0 && isfinite(σ2_eps) && σ2_b > 0 && isfinite(σ2_b)) || return -Inf
    τ_eps = 1.0 / σ2_eps
    τ_b = 1.0 / σ2_b

    # Marginal likelihood: Σ_k = σ²_ε I_{N_k} + σ²_b 1 1'
    quad_total = stats.yty - 2.0 * dot(β, stats.xty) + dot(β, stats.xtx * β)
    logdet_sum = 0.0
    cluster_corr = 0.0
    @inbounds for k in 1:K
        nk = stats.cluster_sizes[k]
        dk = τ_eps * nk + τ_b
        logdet_sum += (nk - 1) * log(σ2_eps) + log(σ2_eps + nk * σ2_b)
        sk = stats.cluster_y_sums[k] - dot(view(stats.cluster_x_sums, :, k), β)
        cluster_corr += (sk * sk) / dk
    end
    mahal = max(0.0, τ_eps * quad_total - (τ_eps * τ_eps) * cluster_corr)
    log_lik = -0.5 * n * log(2π) - 0.5 * logdet_sum - 0.5 * mahal

    # Prior on β | σ²_ε ~ N(μ₀, σ²_ε V₀)
    prior_chol = cholesky(Symmetric(cp.covariance))
    diff = Vector{Float64}(β) - cp.mean
    log_p_β = -0.5 * p * log(2π * σ2_eps) - 0.5 * logdet(prior_chol) -
        0.5 * τ_eps * dot(diff, prior_chol \ diff)

    # Priors on σ²_ε ~ InvGamma(a_ε, b_ε), σ²_b ~ InvGamma(a_b, b_b) with Jacobians σ²_ε, σ²_b
    # log p(log σ²) = a log b - logΓ(a) - a log σ² - b / σ²
    log_p_σ2_eps = cp.shape * log(cp.scale) - SpecialFunctions.loggamma(cp.shape) -
        cp.shape * log_sigma2_eps - cp.scale * τ_eps
    log_p_σ2_b = prior.cluster_shape * log(prior.cluster_scale) -
        SpecialFunctions.loggamma(prior.cluster_shape) -
        prior.cluster_shape * log_sigma2_b - prior.cluster_scale * τ_b

    return log_lik + log_p_β + log_p_σ2_eps + log_p_σ2_b
end

"""
Typed hierarchical Gaussian fit from structured variational inference
`q(β, b) q(σ_ε⁻²) q(σ_b⁻²)`.

Retains the frozen design, affine reparameterisation, cached cluster sufficient
statistics, prior, variational posterior state, marginal log-posterior
evaluator, and convergence diagnostics. Intervals and covariance are withheld
by downstream contrast calls whenever `status !== :converged`.
"""
struct BayesianClusterGCompFit
    posterior_mean::Vector{Float64}
    posterior_covariance::Matrix{Float64}
    cluster_mean::Vector{Float64}
    cluster_var::Vector{Float64}
    beta_cluster_cov::Matrix{Float64}
    residual_shape::Float64
    residual_scale::Float64
    cluster_shape::Float64
    cluster_scale::Float64
    status::Symbol
    converged::Bool
    iterations::Int
    elbo::Float64
    elbo_trace::Vector{Float64}
    prior::ClusterGaussianPrior
    transform::FrozenAffineDesignTransform
    stats_raw::HierarchicalGaussianSufficientStats
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
    fit_bayesian_cluster_gcomp(data, outcome, treatment, covariates; cluster, prior,
        basis=nothing, identification=nothing, handle_missing=:error,
        maxiter=1000, tol=1e-8, standardize=true)

Fit a one-level random-intercept hierarchical Gaussian outcome model using
structured variational inference `q(β, b) q(σ_ε⁻²) q(σ_b⁻²)` with exact
Schur-complement elimination and exact affine prior reparameterisation.

Returns a `BayesianClusterGCompFit`. Reaching `maxiter` without convergence
records `status = :maxiter_reached` and withholds downstream contrast intervals.
"""
function fit_bayesian_cluster_gcomp(data::AbstractDataFrame, outcome::Symbol,
        treatment::Symbol, covariates::AbstractVector{Symbol};
        cluster::Symbol, prior::ClusterGaussianPrior, basis = nothing,
        identification = nothing, handle_missing::Symbol = :error,
        maxiter::Integer = 1000, tol::Real = 1e-8, standardize::Bool = true)
    prep = _prepare_cluster_fit_frame(data, outcome, treatment, covariates, cluster, handle_missing)
    y = try
        Vector{Float64}(data[prep.observed, outcome])
    catch
        throw(ArgumentError("Gaussian outcome must be real numeric"))
    end
    all(isfinite, y) || throw(ArgumentError("observed outcomes must be finite"))

    graph_status = _bayesian_identification(identification, treatment, outcome, prep.covariates)
    id_status = basis === nothing ? graph_status : :user_assumed
    des = _build_cluster_fit_design(prep.frame, prep.observed, treatment, prep.covariates,
        basis, prior, standardize)

    stats_std = _cluster_sufficient_stats(des.X_std, y, prep.cids)
    stats_raw = _cluster_sufficient_stats(des.X_obs, y, prep.cids)
    vi = _structured_vi_hierarchical_gaussian(stats_std, des.prior_std; maxiter, tol)
    state_orig = _untransform_block_state(vi.state, des.transform)

    metadata = (; identification_status = id_status,
        graph_identification_status = graph_status,
        outcome_model_status = :user_assumed, identification,
        missingness_status = handle_missing == :mar ? :user_assumed_mar : :complete,
        n_observed = length(prep.observed), n_missing = nrow(data) - length(prep.observed),
        n_clusters = stats_raw.K, cluster_sizes = copy(stats_raw.cluster_sizes),
        likelihood = :hierarchical_gaussian, approximation = :structured_variational,
        convergence_status = vi.status,
        residual_variance_mean = vi.residual_scale / (vi.residual_shape - 1.0),
        cluster_variance_mean = vi.cluster_scale / (vi.cluster_shape - 1.0))

    return BayesianClusterGCompFit(
        state_orig.beta_mean, state_orig.beta_cov,
        state_orig.cluster_mean, state_orig.cluster_var, state_orig.beta_cluster_cov,
        vi.residual_shape, vi.residual_scale, vi.cluster_shape, vi.cluster_scale,
        vi.status, vi.converged, vi.iterations, vi.elbo, vi.elbo_trace,
        prior, des.transform, stats_raw, state_orig,
        outcome, treatment, prep.covariates, cluster, des.schema, des.covariate_schema, basis,
        des.design_metadata, metadata)
end

"""
Exact variational proposal log-density `log q(β, log σ_ε², log σ_b²)`
(including Jacobians `+log σ_ε² + log σ_b²`) matching `_marginal_cluster_log_posterior`.
"""
function _variational_global_log_density(block_state::HierarchicalGaussianBlockState,
        a_eps::Float64, b_eps::Float64, a_b::Float64, b_b::Float64,
        β::AbstractVector{<:Real}, log_sigma2_eps::Real, log_sigma2_b::Real)
    p = length(block_state.beta_mean)
    diff = Vector{Float64}(β) - block_state.beta_mean
    # Λ_β = U'U => diff' Λ_β diff = ‖U * diff‖²
    u_diff = block_state.beta_precision_chol.U * diff
    log_q_β = -0.5 * p * log(2π) + 0.5 * logdet(block_state.beta_precision_chol) -
        0.5 * sum(abs2, u_diff)
    # τ = exp(-log σ²) ~ Gamma(a, rate=b), Jacobian |dτ/d(log σ²)| = τ
    log_q_eps = a_eps * log(b_eps) - SpecialFunctions.loggamma(a_eps) -
        a_eps * log_sigma2_eps - b_eps * exp(-log_sigma2_eps)
    log_q_b = a_b * log(b_b) - SpecialFunctions.loggamma(a_b) -
        a_b * log_sigma2_b - b_b * exp(-log_sigma2_b)
    return log_q_β + log_q_eps + log_q_b
end

"""
Unnormalised log importance ratios `log p(θ^(s) | y) - log q(θ^(s))` on global
parameters `(β, log σ_ε², log σ_b²)` after analytically integrating cluster
random intercepts `b_1, …, b_K`.
"""
function _cluster_log_importance_ratios(stats::HierarchicalGaussianSufficientStats,
        prior::ClusterGaussianPrior, block_state::HierarchicalGaussianBlockState,
        a_eps::Float64, b_eps::Float64, a_b::Float64, b_b::Float64,
        beta_draws::AbstractMatrix{Float64}, σ2_eps::AbstractVector{Float64},
        σ2_b::AbstractVector{Float64})
    S = size(beta_draws, 2)
    log_ratios = Vector{Float64}(undef, S)
    @inbounds for s in 1:S
        βs = view(beta_draws, :, s)
        l_eps = log(σ2_eps[s])
        l_b = log(σ2_b[s])
        lp = _marginal_cluster_log_posterior(stats, prior, βs, l_eps, l_b)
        lq = _variational_global_log_density(block_state, a_eps, b_eps, a_b, b_b, βs, l_eps, l_b)
        log_ratios[s] = lp - lq
    end
    return log_ratios
end

"""
    bayesian_policy_contrast(fit::BayesianClusterGCompFit, target, policy;
        deltas, n_draws=2000, rng=Random.default_rng(), level=0.95,
        mc_level=0.95, weighting=:individual, cluster=fit.cluster,
        correction=:none)

Marginal `:policy_vs_observed` contrasts from a fitted hierarchical Gaussian
model, integrating over the cluster-effect distribution (`E[b_k] = 0`).
Supports `weighting = :individual` (equal row weight) and `weighting = :cluster`
(equal cluster weight; requires `cluster` column in `target`).

All requested `deltas` are evaluated from one shared set of `n_draws` posterior
draws, preserving cross-policy covariance. If `fit.status !== :converged`,
intervals and covariance are withheld (`NaN` / unavailable).
"""
function bayesian_policy_contrast(fit::BayesianClusterGCompFit, target::AbstractDataFrame, policy;
        deltas, n_draws::Integer = 2000, rng::AbstractRNG = Random.default_rng(),
        level::Real = 0.95, mc_level::Real = 0.95, weighting::Symbol = :individual,
        cluster::Union{Nothing, Symbol} = fit.cluster, correction::Symbol = :none)
    inp = _prepare_bayesian_contrast_inputs(fit, target, policy;
        deltas, n_draws, level, mc_level, weighting, cluster, correction)
    m, p, w = inp.m, inp.p, inp.w
    C = Matrix{Float64}(undef, m, p)
    for j in 1:m
        C[j, :] = vec(w' * (inp.X_policies[j] - inp.X_obs))
    end
    extra = (; contrast_matrix = C)

    if fit.status !== :converged
        return _unavailable_bayesian_contrast(C * fit.posterior_mean;
            deltas = inp.deltas, fit_metadata = fit.metadata,
            design_metadata = fit.design_metadata, target_kind = :fixed_cohort_marginal,
            weighting, n_target = inp.n_target, level = Float64(level),
            mc_level = Float64(mc_level), n_draws = Int(n_draws),
            correction, policy_constants = inp.policy_constants, extra_fields = extra)
    end

    # Draw global parameters (β, σ²_ε, σ²_b) from q(β) q(σ_ε⁻²) q(σ_b⁻²)
    S = Int(n_draws)
    Zβ = randn(rng, p, S)
    ldiv!(fit.block_state.beta_precision_chol.U, Zβ)
    beta_draws = Zβ .+ fit.posterior_mean
    contrast_draws = C * beta_draws # m × S

    log_ratios = if correction === :psis
        σ2_eps = [rand(rng, InverseGamma(fit.residual_shape, fit.residual_scale)) for _ in 1:S]
        σ2_b = [rand(rng, InverseGamma(fit.cluster_shape, fit.cluster_scale)) for _ in 1:S]
        _cluster_log_importance_ratios(fit.stats_raw, fit.prior, fit.block_state,
            fit.residual_shape, fit.residual_scale, fit.cluster_shape, fit.cluster_scale,
            beta_draws, σ2_eps, σ2_b)
    else
        nothing
    end

    return _summarise_bayesian_contrast_draws(contrast_draws;
        deltas = inp.deltas, level = Float64(level), mc_level = Float64(mc_level),
        S, correction, log_ratios, fit_metadata = fit.metadata,
        design_metadata = fit.design_metadata, target_kind = :fixed_cohort_marginal,
        weighting, n_target = inp.n_target, policy_constants = inp.policy_constants,
        vi_uncertainty = :structured_variational, extra_fields = extra)
end
