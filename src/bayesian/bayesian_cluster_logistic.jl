# One-level random-intercept hierarchical Bayesian logistic regression
# (Jaakkola-Jordan weighted Schur-complement structured VI, 1D Gauss-Hermite
# marginalisation, and global-parameter PSIS importance refinement).

"""
    ClusterLogisticPrior(; mean, covariance, cluster_shape, cluster_scale)
    ClusterLogisticPrior(coefficient_prior::GaussianCoefficientPrior; cluster_shape, cluster_scale)

Explicit hierarchical logistic prior:
`β ~ Normal(mean, covariance)` and `σ_b² ~ InverseGamma(cluster_shape, cluster_scale)`
for cluster random intercepts `b_k | σ_b² ~ Normal(0, σ_b²)`.

Requires symmetric positive-definite `covariance`, `cluster_shape > 1`, and
`cluster_scale > 0`. No outcome-dependent rescaling is performed.
"""
struct ClusterLogisticPrior
    coefficient_prior::GaussianCoefficientPrior
    cluster_shape::Float64
    cluster_scale::Float64

    function ClusterLogisticPrior(coefficient_prior::GaussianCoefficientPrior;
            cluster_shape::Real, cluster_scale::Real)
        ab, bb = Float64(cluster_shape), Float64(cluster_scale)
        isfinite(ab) && ab > 1 ||
            throw(ArgumentError("cluster variance prior shape must be finite and greater than one"))
        isfinite(bb) && bb > 0 ||
            throw(ArgumentError("cluster variance prior scale must be finite and positive"))
        new(coefficient_prior, ab, bb)
    end
end

function ClusterLogisticPrior(; mean::AbstractVector, covariance::AbstractMatrix,
        cluster_shape::Real, cluster_scale::Real)
    return ClusterLogisticPrior(GaussianCoefficientPrior(; mean, covariance);
        cluster_shape, cluster_scale)
end

Base.propertynames(::ClusterLogisticPrior) = (
    :coefficient_prior, :mean, :covariance, :cluster_shape, :cluster_scale,
)

function Base.getproperty(p::ClusterLogisticPrior, name::Symbol)
    cp = getfield(p, :coefficient_prior)
    if name === :mean
        return cp.mean
    elseif name === :covariance
        return cp.covariance
    else
        return getfield(p, name)
    end
end

function _transform_prior(prior::GaussianCoefficientPrior, t::FrozenAffineDesignTransform)
    length(prior.mean) == size(t.A, 1) ||
        throw(DimensionMismatch("prior dimension must match transform dimension"))
    μ_std = t.A_inv * prior.mean
    V_std = Matrix(Symmetric(t.A_inv * prior.covariance * t.A_inv'))
    return GaussianCoefficientPrior(; mean = μ_std, covariance = V_std)
end

_transform_prior(prior::ClusterLogisticPrior, t::FrozenAffineDesignTransform) =
    ClusterLogisticPrior(_transform_prior(prior.coefficient_prior, t);
        cluster_shape = prior.cluster_shape, cluster_scale = prior.cluster_scale)

@inline function _jj_lambda(ξ::Float64)
    a = abs(ξ)
    a < 1e-4 && return 0.125 - (a * a) / 192.0
    return tanh(0.5 * a) / (4.0 * a)
end

struct _LogisticClusterLayout{T}
    n::Int
    p::Int
    K::Int
    X::Matrix{Float64}
    y::Vector{Float64}
    kappa::Vector{Float64}       # y - 0.5
    cluster_levels::Vector{T}
    cluster_index::Vector{Int}
    cluster_rows::Vector{Vector{Int}}
    cluster_sizes::Vector{Int}
    cluster_kappa_sum::Vector{Float64}
    xt_kappa::Vector{Float64}
end

function _logistic_cluster_layout(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
        cluster_ids::AbstractVector)
    idx = _build_cluster_index(X, y, cluster_ids;
        validate_y = yi -> (yi == 0 || yi == 1),
        invalid_msg = "hierarchical logistic inputs must have finite X and binary 0/1 outcomes")
    kappa = idx.y .- 0.5
    ck_sum = [sum(i -> kappa[i], r; init = 0.0) for r in idx.cluster_rows]
    xt_kappa = idx.n == 0 ? zeros(Float64, idx.p) : Vector{Float64}(idx.X' * kappa)
    T = eltype(idx.cluster_levels)
    return _LogisticClusterLayout{T}(idx.n, idx.p, idx.K, idx.X, idx.y, kappa,
        idx.cluster_levels, idx.cluster_index, idx.cluster_rows, idx.cluster_sizes,
        ck_sum, xt_kappa)
end

"""
Exact weighted Schur-complement Gaussian block solve `q(β, b)` for hierarchical
logistic VI given observation weights `W = 2λ(ξ)` and cluster precision `τ_b`
(including `τ_b = Inf` for the non-hierarchical `σ_b² = 0` limit).
"""
function _solve_weighted_logistic_block(layout::_LogisticClusterLayout,
        prior::Union{GaussianCoefficientPrior, ClusterLogisticPrior},
        W::AbstractVector{Float64}; cluster_precision::Real)
    cp = prior isa ClusterLogisticPrior ? prior.coefficient_prior : prior
    return _solve_weighted_cluster_block(
        layout.X, layout.cluster_rows, layout.p, layout.K, layout.n,
        cp.mean, cp.covariance, W, layout.kappa;
        cluster_precision,
        xt_rhs = layout.xt_kappa,
        cluster_rhs_sum = layout.cluster_kappa_sum)
end

function _hierarchical_logistic_elbo(state::HierarchicalGaussianBlockState,
        layout::_LogisticClusterLayout, prior::ClusterLogisticPrior,
        ξ::Vector{Float64}, a_b::Float64, b_b::Float64)
    cp = prior.coefficient_prior
    p, K, n = layout.p, layout.K, layout.n
    prior_chol = cholesky(Symmetric(cp.covariance))
    V0_inv = prior_chol \ Matrix{Float64}(I, p, p)
    logdet_V0 = logdet(prior_chol)
    diff = state.beta_mean - cp.mean
    prior_quad = dot(diff, V0_inv * diff) + tr(V0_inv * state.beta_cov)

    # E_q[log p_JJ(y | β, b; ξ)]
    e_log_lik = 0.0
    mβ = state.beta_mean
    Σβ = state.beta_cov
    x_tmp = Vector{Float64}(undef, p)
    Σβ_x = Vector{Float64}(undef, p)
    @inbounds for i in 1:n
        k = layout.cluster_index[i]
        for j in 1:p
            x_tmp[j] = layout.X[i, j]
        end
        η_mean = dot(x_tmp, mβ) + state.cluster_mean[k]
        mul!(Σβ_x, Σβ, x_tmp)
        η_var = dot(x_tmp, Σβ_x) + 2.0 * dot(x_tmp, view(state.beta_cluster_cov, :, k)) +
            state.cluster_var[k]
        η_sq = η_mean * η_mean + max(0.0, η_var)
        ξi = ξ[i]
        λi = _jj_lambda(ξi)
        e_log_lik += _cl_log_logistic(ξi) - 0.5 * ξi + λi * (ξi * ξi) +
            layout.kappa[i] * η_mean - λi * η_sq
    end

    τ_b = a_b / b_b
    E_log_τ_b = SpecialFunctions.digamma(a_b) - log(b_b)
    e_log_p_β = -0.5 * p * log(2π) - 0.5 * logdet_V0 - 0.5 * prior_quad
    e_log_p_b = -0.5 * K * log(2π) + 0.5 * K * E_log_τ_b - 0.5 * τ_b * state.expected_cluster_ss
    e_log_p_τ_b = prior.cluster_shape * log(prior.cluster_scale) -
        SpecialFunctions.loggamma(prior.cluster_shape) +
        (prior.cluster_shape - 1.0) * E_log_τ_b - prior.cluster_scale * τ_b

    logdet_Λβ = logdet(state.beta_precision_chol)
    sum_log_dk = K == 0 ? 0.0 : -sum(log, state.cluster_cond_var)
    h_q_βb = 0.5 * (p + K) * (1.0 + log(2π)) - 0.5 * (logdet_Λβ + sum_log_dk)
    h_q_τ_b = a_b - log(b_b) + SpecialFunctions.loggamma(a_b) +
        (1.0 - a_b) * SpecialFunctions.digamma(a_b)

    return e_log_lik + e_log_p_β + e_log_p_b + e_log_p_τ_b + h_q_βb + h_q_τ_b
end

function _structured_vi_hierarchical_logistic(layout::_LogisticClusterLayout,
        prior::ClusterLogisticPrior; maxiter::Integer = 1000, tol::Real = 1e-8)
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    p, K, n = layout.p, layout.K, layout.n

    a_b = prior.cluster_shape + 0.5 * K
    b_b = prior.cluster_scale * (a_b / prior.cluster_shape)
    ξ = fill(1.0, n)
    W = [2.0 * _jj_lambda(ξi) for ξi in ξ]

    elbo_trace = Float64[]
    sizehint!(elbo_trace, min(Int(maxiter), 128))
    prev_elbo = -Inf
    prev_mβ = fill(NaN, p)
    prev_b_b = NaN
    converged = false
    iters = 0
    local state::HierarchicalGaussianBlockState
    x_tmp = Vector{Float64}(undef, p)
    Σβ_x = Vector{Float64}(undef, p)

    for iter in 1:Int(maxiter)
        iters = iter
        τ_b = a_b / b_b
        state = _solve_weighted_logistic_block(layout, prior, W; cluster_precision = τ_b)

        # Update ξ_{ki} = sqrt(E_q[(x_{ki}'β + b_k)²]) and W_{ki} = 2 λ(ξ_{ki})
        mβ = state.beta_mean
        Σβ = state.beta_cov
        @inbounds for i in 1:n
            k = layout.cluster_index[i]
            for j in 1:p
                x_tmp[j] = layout.X[i, j]
            end
            η_mean = dot(x_tmp, mβ) + state.cluster_mean[k]
            mul!(Σβ_x, Σβ, x_tmp)
            η_var = dot(x_tmp, Σβ_x) + 2.0 * dot(x_tmp, view(state.beta_cluster_cov, :, k)) +
                state.cluster_var[k]
            ξi = sqrt(max(0.0, η_mean * η_mean + max(0.0, η_var)))
            ξ[i] = ξi
            W[i] = 2.0 * _jj_lambda(ξi)
        end

        b_b = prior.cluster_scale + 0.5 * state.expected_cluster_ss
        elbo = _hierarchical_logistic_elbo(state, layout, prior, ξ, a_b, b_b)
        push!(elbo_trace, elbo)

        if iter > 1
            rel_elbo = abs(elbo - prev_elbo) / max(1.0, abs(elbo))
            rel_mβ = norm(state.beta_mean - prev_mβ) / max(1.0, norm(state.beta_mean))
            rel_b = abs(b_b - prev_b_b) / max(1.0, abs(b_b))
            if max(rel_elbo, rel_mβ, rel_b) <= tol
                state = _solve_weighted_logistic_block(layout, prior, W; cluster_precision = a_b / b_b)
                converged = true
                break
            end
        end
        prev_elbo = elbo
        prev_mβ = copy(state.beta_mean)
        prev_b_b = b_b
    end

    if !converged
        state = _solve_weighted_logistic_block(layout, prior, W; cluster_precision = a_b / b_b)
    end

    status = converged ? :converged : :maxiter_reached
    return (; state, ξ, cluster_shape = a_b, cluster_scale = b_b,
        elbo = isempty(elbo_trace) ? NaN : elbo_trace[end],
        elbo_trace, iterations = iters, converged, status)
end

"""
Adaptive Laplace-centred 1D Gauss-Hermite log-likelihood for one cluster:
`log ∫ exp(Σ_{i∈k} [y_i (η_i + b) - log(1 + exp(η_i + b))]) N(b; 0, σ_b²) db`.
"""
function _cluster_logistic_marginal_loglik(η0::AbstractVector{Float64},
        y::AbstractVector{Float64}, rows::Vector{Int}, σ2_b::Float64)
    τ_b = 1.0 / σ2_b
    # Find 1D mode b̂ of log p(y_k | η0, b) - 0.5 τ_b b² via Newton
    b = 0.0
    neg_hess = τ_b
    for _ in 1:10
        grad = -τ_b * b
        neg_hess = τ_b
        @inbounds for i in rows
            p_i = _cl_logistic(η0[i] + b)
            grad += y[i] - p_i
            neg_hess += p_i * (1.0 - p_i)
        end
        step = grad / neg_hess
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
            η = η0[i] + bq
            ll += _cl_log_logistic(y[i] == 1.0 ? η : -η)
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

"""
Exact (adaptive 1D Gauss-Hermite) marginalised log-posterior
`log p(β, log σ_b² | y)` (including Jacobian `+log σ_b²`).
"""
function _marginal_cluster_logistic_log_posterior(layout::_LogisticClusterLayout,
        prior::ClusterLogisticPrior, β::AbstractVector{<:Real}, log_sigma2_b::Real)
    cp = prior.coefficient_prior
    p, K = layout.p, layout.K
    length(β) == p || throw(DimensionMismatch("β length must match $p"))
    all(isfinite, β) && isfinite(log_sigma2_b) || return -Inf
    σ2_b = exp(Float64(log_sigma2_b))
    (σ2_b > 0 && isfinite(σ2_b)) || return -Inf

    η0 = layout.X * Vector{Float64}(β)
    log_lik = 0.0
    for k in 1:K
        log_lik += _cluster_logistic_marginal_loglik(η0, layout.y, layout.cluster_rows[k], σ2_b)
    end

    prior_chol = cholesky(Symmetric(cp.covariance))
    diff = Vector{Float64}(β) - cp.mean
    log_p_β = -0.5 * p * log(2π) - 0.5 * logdet(prior_chol) - 0.5 * dot(diff, prior_chol \ diff)
    log_p_σ2_b = prior.cluster_shape * log(prior.cluster_scale) -
        SpecialFunctions.loggamma(prior.cluster_shape) -
        prior.cluster_shape * log_sigma2_b - prior.cluster_scale / σ2_b
    return log_lik + log_p_β + log_p_σ2_b
end

# Student-t proposal density on (β, log σ_b²) for PSIS importance sampling
# (heavier tails prevent infinite-variance importance weights when JJ-VI
# slightly underestimates posterior variance; target posterior is unchanged).
function _draw_student_t_logistic_proposal(block_state::HierarchicalGaussianBlockState,
        a_b::Float64, b_b::Float64, S::Int, rng::AbstractRNG;
        df::Float64 = 7.0, log_var_scale_mult::Float64 = 1.25)
    p = length(block_state.beta_mean)
    Zβ = randn(rng, p, S)
    ldiv!(block_state.beta_precision_chol.U, Zβ)
    chi_draws = [rand(rng, Chisq(df)) / df for _ in 1:S]
    beta_draws = Matrix{Float64}(undef, p, S)
    @inbounds for s in 1:S
        inv_sqrt_u = 1.0 / sqrt(chi_draws[s])
        for j in 1:p
            beta_draws[j, s] = block_state.beta_mean[j] + inv_sqrt_u * Zβ[j, s]
        end
    end

    # Match location/scale of log(σ_b²) = -log(τ_b) under τ_b ~ Gamma(a_b, rate=b_b)
    μ_lb = log(b_b) - SpecialFunctions.digamma(a_b)
    sd_lb = log_var_scale_mult * sqrt(SpecialFunctions.trigamma(a_b))
    t_dist = LocationScale(μ_lb, sd_lb, TDist(df))
    log_σ2_b_draws = [rand(rng, t_dist) for _ in 1:S]
    σ2_b_draws = exp.(log_σ2_b_draws)

    # Exact multivariate Student-t log-density for β + univariate Student-t for log(σ_b²)
    c_mvt = SpecialFunctions.loggamma(0.5 * (df + p)) -
        SpecialFunctions.loggamma(0.5 * df) -
        0.5 * p * log(df * π) + 0.5 * logdet(block_state.beta_precision_chol)
    log_q = Vector{Float64}(undef, S)
    u_diff = Vector{Float64}(undef, p)
    @inbounds for s in 1:S
        for j in 1:p
            u_diff[j] = beta_draws[j, s] - block_state.beta_mean[j]
        end
        lmul!(block_state.beta_precision_chol.U, u_diff)
        quad = sum(abs2, u_diff)
        lq_β = c_mvt - 0.5 * (df + p) * log1p(quad / df)
        lq_lb = logpdf(t_dist, log_σ2_b_draws[s])
        log_q[s] = lq_β + lq_lb
    end
    return (; beta_draws, σ2_b_draws, log_σ2_b_draws, log_q)
end

"""
Typed hierarchical logistic fit from Jaakkola-Jordan weighted Schur-complement
structured variational inference `q(β, b) q(σ_b⁻²)`.
"""
struct BayesianClusterLogisticFit
    posterior_mean::Vector{Float64}
    posterior_covariance::Matrix{Float64}
    cluster_mean::Vector{Float64}
    cluster_var::Vector{Float64}
    beta_cluster_cov::Matrix{Float64}
    cluster_shape::Float64
    cluster_scale::Float64
    status::Symbol
    converged::Bool
    iterations::Int
    elbo::Float64
    elbo_trace::Vector{Float64}
    prior::ClusterLogisticPrior
    transform::FrozenAffineDesignTransform
    layout_raw::_LogisticClusterLayout
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
    fit_bayesian_cluster_logistic(data, outcome, treatment, covariates; cluster, prior,
        basis=nothing, identification=nothing, handle_missing=:error,
        maxiter=1000, tol=1e-8, standardize=true)

Fit a one-level random-intercept hierarchical Bayesian logistic outcome model
using Jaakkola-Jordan weighted Schur-complement structured VI `q(β, b) q(σ_b⁻²)`
and exact affine prior reparameterisation.
"""
function fit_bayesian_cluster_logistic(data::AbstractDataFrame, outcome::Symbol,
        treatment::Symbol, covariates::AbstractVector{Symbol};
        cluster::Symbol, prior::ClusterLogisticPrior, basis = nothing,
        identification = nothing, handle_missing::Symbol = :error,
        maxiter::Integer = 1000, tol::Real = 1e-8, standardize::Bool = true)
    prep = _prepare_cluster_fit_frame(data, outcome, treatment, covariates, cluster, handle_missing)
    y = try
        Float64.(data[prep.observed, outcome])
    catch
        throw(ArgumentError("binary logistic outcome must be numeric 0/1"))
    end
    all(yi -> yi == 0.0 || yi == 1.0, y) ||
        throw(ArgumentError("binary logistic outcome must contain only 0 and 1"))

    graph_status = _bayesian_identification(identification, treatment, outcome, prep.covariates)
    id_status = basis === nothing ? graph_status : :user_assumed
    des = _build_cluster_fit_design(prep.frame, prep.observed, treatment, prep.covariates,
        basis, prior, standardize)

    layout_std = _logistic_cluster_layout(des.X_std, y, prep.cids)
    layout_raw = _logistic_cluster_layout(des.X_obs, y, prep.cids)
    vi = _structured_vi_hierarchical_logistic(layout_std, des.prior_std; maxiter, tol)
    state_orig = _untransform_block_state(vi.state, des.transform)

    metadata = (; identification_status = id_status,
        graph_identification_status = graph_status,
        outcome_model_status = :user_assumed, identification,
        missingness_status = handle_missing == :mar ? :user_assumed_mar : :complete,
        n_observed = length(prep.observed), n_missing = nrow(data) - length(prep.observed),
        n_clusters = layout_raw.K, cluster_sizes = copy(layout_raw.cluster_sizes),
        likelihood = :hierarchical_bernoulli_logit, approximation = :jaakkola_jordan_structured_vi,
        convergence_status = vi.status,
        cluster_variance_mean = vi.cluster_scale / (vi.cluster_shape - 1.0))

    return BayesianClusterLogisticFit(
        state_orig.beta_mean, state_orig.beta_cov,
        state_orig.cluster_mean, state_orig.cluster_var, state_orig.beta_cluster_cov,
        vi.cluster_shape, vi.cluster_scale,
        vi.status, vi.converged, vi.iterations, vi.elbo, vi.elbo_trace,
        prior, des.transform, layout_raw, state_orig,
        outcome, treatment, prep.covariates, cluster, des.schema, des.covariate_schema, basis,
        des.design_metadata, metadata)
end

# Marginalised probability E_{b ~ N(0, σ²_b)}[logistic(η + b)] via 25-node GH
@inline function _marginal_logistic_prob(η::Float64, σ_b::Float64)
    σ_b <= 0.0 && return _cl_logistic(η)
    gh = _GH_25
    acc = 0.0
    @inbounds for q in eachindex(gh.nodes)
        acc += gh.weights[q] * _cl_logistic(η + σ_b * gh.nodes[q])
    end
    return acc
end

"""
    bayesian_policy_contrast(fit::BayesianClusterLogisticFit, target, policy;
        deltas, n_draws=2000, rng=Random.default_rng(), level=0.95,
        mc_level=0.95, weighting=:individual, cluster=fit.cluster,
        correction=:none, marginalize_random_effects=true)

Marginal `:policy_vs_observed` risk-difference contrasts from a fitted
hierarchical Bayesian logistic model.

When `marginalize_random_effects=true` (default), integrates over
`b_k ~ Normal(0, σ_b²)` via 25-node Gauss-Hermite quadrature for every joint
draw `(β^(s), σ_b²^(s))` (distinct from setting `b_k = 0` under the nonlinear
logit link). Supports `weighting = :individual` or `:cluster`, and
`correction = :none` or `:psis` (requiring `using PSIS`).
"""
function bayesian_policy_contrast(fit::BayesianClusterLogisticFit, target::AbstractDataFrame, policy;
        deltas, n_draws::Integer = 2000, rng::AbstractRNG = Random.default_rng(),
        level::Real = 0.95, mc_level::Real = 0.95, weighting::Symbol = :individual,
        cluster::Union{Nothing, Symbol} = fit.cluster, correction::Symbol = :none,
        marginalize_random_effects::Bool = true)
    inp = _prepare_bayesian_contrast_inputs(fit, target, policy;
        deltas, n_draws, level, mc_level, weighting, cluster, correction)
    m, p, n_target, w = inp.m, inp.p, inp.n_target, inp.w
    target_kind = marginalize_random_effects ? :fixed_cohort_marginal : :fixed_cohort_conditional_b0

    if fit.status !== :converged
        σ_b_pt = marginalize_random_effects ? sqrt(fit.cluster_scale / (fit.cluster_shape - 1.0)) : 0.0
        η_obs = inp.X_obs * fit.posterior_mean
        μ_obs = sum(i -> w[i] * _marginal_logistic_prob(η_obs[i], σ_b_pt), 1:n_target)
        μ_pt = [begin
            η_p = inp.X_policies[j] * fit.posterior_mean
            sum(i -> w[i] * _marginal_logistic_prob(η_p[i], σ_b_pt), 1:n_target) - μ_obs
        end for j in 1:m]
        return _unavailable_bayesian_contrast(μ_pt;
            deltas = inp.deltas, fit_metadata = fit.metadata,
            design_metadata = fit.design_metadata, target_kind,
            weighting, n_target, level = Float64(level), mc_level = Float64(mc_level),
            n_draws = Int(n_draws), correction, policy_constants = inp.policy_constants)
    end

    S = Int(n_draws)
    if correction === :psis
        prop = _draw_student_t_logistic_proposal(fit.block_state, fit.cluster_shape, fit.cluster_scale, S, rng)
        beta_draws = prop.beta_draws
        σ2_b_draws = prop.σ2_b_draws
        log_σ2_b_draws = prop.log_σ2_b_draws
        log_q_prop = prop.log_q
    else
        Zβ = randn(rng, p, S)
        ldiv!(fit.block_state.beta_precision_chol.U, Zβ)
        beta_draws = Zβ .+ fit.posterior_mean
        σ2_b_draws = [rand(rng, InverseGamma(fit.cluster_shape, fit.cluster_scale)) for _ in 1:S]
        log_σ2_b_draws = Float64[]
        log_q_prop = Float64[]
    end

    # Precompute linear predictors X * beta_draws (n_target × S)
    η_obs_mat = inp.X_obs * beta_draws
    η_pol_mats = [inp.X_policies[j] * beta_draws for j in 1:m]
    contrast_draws = Matrix{Float64}(undef, m, S)

    @inbounds for s in 1:S
        σ_b = marginalize_random_effects ? sqrt(σ2_b_draws[s]) : 0.0
        obs_mean = 0.0
        for i in 1:n_target
            obs_mean += w[i] * _marginal_logistic_prob(η_obs_mat[i, s], σ_b)
        end
        for j in 1:m
            pol_mat = η_pol_mats[j]
            pol_mean = 0.0
            for i in 1:n_target
                pol_mean += w[i] * _marginal_logistic_prob(pol_mat[i, s], σ_b)
            end
            contrast_draws[j, s] = pol_mean - obs_mean
        end
    end

    log_ratios = if correction === :psis
        lr = Vector{Float64}(undef, S)
        @inbounds for s in 1:S
            βs = view(beta_draws, :, s)
            lp = _marginal_cluster_logistic_log_posterior(fit.layout_raw, fit.prior, βs, log_σ2_b_draws[s])
            lr[s] = lp - log_q_prop[s]
        end
        lr
    else
        nothing
    end

    return _summarise_bayesian_contrast_draws(contrast_draws;
        deltas = inp.deltas, level = Float64(level), mc_level = Float64(mc_level),
        S, correction, log_ratios, fit_metadata = fit.metadata,
        design_metadata = fit.design_metadata, target_kind,
        weighting, n_target, policy_constants = inp.policy_constants,
        vi_uncertainty = :jaakkola_jordan_structured_vi)
end
