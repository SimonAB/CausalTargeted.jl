# Shared primitives for hierarchical Bayesian cluster models
# (affine design reparameterisation, cluster indexing, weighted Schur-complement
# block solver, Gauss-Hermite quadrature, PSIS refinement, and contrast summaries).

"""
Exact invertible linear reparameterisation `X_std = X * A` (`β = A * β_std`).

Because `X * β == X_std * β_std` identically, transforming the prior via
`μ_std = A⁻¹ * μ₀` and `V_std = A⁻¹ * V₀ * A⁻ᵀ` preserves the exact Bayesian
model while improving numerical conditioning of the global system.
"""
struct FrozenAffineDesignTransform
    A::Matrix{Float64}
    A_inv::Matrix{Float64}
    intercept_col::Union{Nothing, Int}
    centers::Vector{Float64}
    scales::Vector{Float64}
end

function _detect_intercept_column(X::AbstractMatrix{<:Real})
    n, p = size(X)
    n == 0 && return nothing
    for j in 1:p
        all(i -> X[i, j] == 1.0, 1:n) && return j
    end
    return nothing
end

function _freeze_affine_transform(X::AbstractMatrix{<:Real};
        intercept_col::Union{Nothing, Int} = _detect_intercept_column(X))
    n, p = size(X)
    all(isfinite, X) || throw(ArgumentError("design matrix must be finite"))
    if intercept_col !== nothing
        1 <= intercept_col <= p || throw(ArgumentError("intercept_col must lie in 1:$p"))
        if n > 0 && !all(i -> X[i, intercept_col] == 1.0, 1:n)
            throw(ArgumentError("declared intercept_col must be identically 1.0"))
        end
    end
    centers = zeros(p)
    scales = ones(p)
    A = Matrix{Float64}(I, p, p)
    A_inv = Matrix{Float64}(I, p, p)
    if n > 1
        for j in 1:p
            j == intercept_col && continue
            col = view(X, :, j)
            s = std(col)
            if isfinite(s) && s > 0
                scales[j] = s
                A[j, j] = 1.0 / s
                A_inv[j, j] = s
                if intercept_col !== nothing
                    c = mean(col)
                    centers[j] = c
                    # X_std[:, j] = (X[:, j] - c * X[:, int]) / s
                    A[intercept_col, j] = -c / s
                    A_inv[intercept_col, j] = c
                end
            end
        end
    end
    return FrozenAffineDesignTransform(A, A_inv, intercept_col, centers, scales)
end

_apply_affine_transform(X::AbstractMatrix{<:Real}, t::FrozenAffineDesignTransform) =
    Matrix{Float64}(X) * t.A

function _build_cluster_affine_transform(X_obs::AbstractMatrix{<:Real}, p::Int, standardize::Bool)
    return standardize ? _freeze_affine_transform(X_obs) :
        FrozenAffineDesignTransform(Matrix{Float64}(I, p, p), Matrix{Float64}(I, p, p),
            _detect_intercept_column(X_obs), zeros(p), ones(p))
end

# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature & scalar logistic helpers
# ---------------------------------------------------------------------------

# Probabilist's Gauss-Hermite nodes and weights for E_{u ~ N(0, 1)}[g(u)]
# via Golub-Welsch on the symmetric tridiagonal Jacobi matrix.
function _standard_normal_gh(Q::Int)
    Q >= 2 || throw(ArgumentError("quadrature order Q must be at least 2"))
    offdiag = [sqrt(Float64(i)) for i in 1:(Q - 1)]
    F = eigen(SymTridiagonal(zeros(Float64, Q), offdiag))
    nodes = Vector{Float64}(F.values)
    weights = [F.vectors[1, q]^2 for q in 1:Q]
    weights ./= sum(weights)
    return (; nodes, weights, log_weights = log.(weights))
end

const _GH_21 = _standard_normal_gh(21)
const _GH_25 = _standard_normal_gh(25)

@inline _cl_logistic(x::Float64) = _expit_stable(x)
@inline _cl_log_logistic(x::Float64) = x >= 0.0 ? -log1p(exp(-x)) : x - log1p(exp(x))

# ---------------------------------------------------------------------------
# Unified cluster index & weighted Schur-complement block solver
# ---------------------------------------------------------------------------

struct _ClusterIndex{T}
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

function _build_cluster_index(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
        cluster_ids::AbstractVector;
        validate_y::Function = yi -> isfinite(yi),
        invalid_msg::String = "hierarchical model inputs must be finite")
    n, p = size(X)
    p > 0 || throw(ArgumentError("design matrix must contain at least one column"))
    length(y) == n || throw(DimensionMismatch("outcome length must match design rows"))
    length(cluster_ids) == n || throw(DimensionMismatch("cluster_ids length must match design rows"))
    all(isfinite, X) && all(validate_y, y) || throw(ArgumentError(invalid_msg))
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
    return _ClusterIndex{T}(n, p, K, Xf, yf, levels, cidx, rows_by_cluster, sizes)
end

"""
Exact Gaussian block posterior `q(β, b)` for fixed residual precision `τ_ε` and
cluster precision `τ_b` (including `τ_b = Inf` for the zero-cluster-variance
non-hierarchical limit).
"""
struct HierarchicalGaussianBlockState
    beta_mean::Vector{Float64}
    beta_cov::Matrix{Float64}
    beta_precision_chol::Cholesky{Float64, Matrix{Float64}}
    cluster_mean::Vector{Float64}
    cluster_var::Vector{Float64}
    cluster_cond_var::Vector{Float64}
    beta_cluster_cov::Matrix{Float64}
    residual_precision::Float64
    cluster_precision::Float64
    expected_residual_ss::Float64
    expected_cluster_ss::Float64
end

function _untransform_block_state(state::HierarchicalGaussianBlockState,
        t::FrozenAffineDesignTransform)
    mβ = t.A * state.beta_mean
    Σβ = Matrix(Symmetric(t.A * state.beta_cov * t.A'))
    cov_βb = t.A * state.beta_cluster_cov
    # Precision transforms as A⁻ᵀ Λ_std A⁻¹
    Λ_orig = Matrix(Symmetric(t.A_inv' * Matrix(state.beta_precision_chol) * t.A_inv))
    Λ_chol = cholesky(Symmetric(Λ_orig))
    return HierarchicalGaussianBlockState(
        mβ, Σβ, Λ_chol, copy(state.cluster_mean), copy(state.cluster_var),
        copy(state.cluster_cond_var), cov_βb,
        state.residual_precision, state.cluster_precision,
        state.expected_residual_ss, state.expected_cluster_ss)
end

"""
Unified weighted Schur-complement Gaussian block solve `q(β, b)` for
non-Gaussian hierarchical VI (Jaakkola-Jordan logistic and Laplace-IRLS count)
given observation weights `W`, linear RHS `z_rhs` (or precomputed `xt_rhs` and
`cluster_rhs_sum`), and cluster precision `τ_b` (`Inf` for the `σ_b² = 0` limit).
"""
function _solve_weighted_cluster_block(
        X::Matrix{Float64},
        cluster_rows::Vector{Vector{Int}},
        p::Int, K::Int, n::Int,
        prior_mean::Vector{Float64},
        prior_cov::Matrix{Float64},
        W::AbstractVector{Float64},
        z_rhs::Union{Nothing, AbstractVector{Float64}};
        cluster_precision::Real,
        xt_rhs::Union{Nothing, Vector{Float64}} = nothing,
        cluster_rhs_sum::Union{Nothing, Vector{Float64}} = nothing)
    τb = Float64(cluster_precision)
    (isfinite(τb) && τb > 0) || τb == Inf ||
        throw(ArgumentError("cluster_precision must be positive or Inf"))

    prior_chol = cholesky(Symmetric(prior_cov))
    V0_inv = prior_chol \ Matrix{Float64}(I, p, p)
    rhs_proj = xt_rhs !== nothing ? xt_rhs :
        (n == 0 ? zeros(Float64, p) : Vector{Float64}(X' * z_rhs))
    hβ = (prior_chol \ prior_mean) .+ rhs_proj

    # Q_ββ = V₀⁻¹ + X' diag(W) X
    Λβ = copy(V0_inv)
    if n > 0
        XtW = X' .* reshape(W, 1, n)
        mul!(Λβ, XtW, X, 1.0, 1.0)
    end

    inv_d = zeros(Float64, K)
    s_Wx = zeros(Float64, p, K)
    hk_vec = cluster_rhs_sum !== nothing ? cluster_rhs_sum : zeros(Float64, K)
    if isfinite(τb) && K > 0
        @inbounds for k in 1:K
            rows = cluster_rows[k]
            w_sum = 0.0
            hk = cluster_rhs_sum !== nothing ? cluster_rhs_sum[k] : 0.0
            for i in rows
                wi = W[i]
                w_sum += wi
                if cluster_rhs_sum === nothing
                    hk += z_rhs[i]
                end
                for j in 1:p
                    s_Wx[j, k] += wi * X[i, j]
                end
            end
            if cluster_rhs_sum === nothing
                hk_vec[k] = hk
            end
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

# ---------------------------------------------------------------------------
# Shared cluster-fit scaffolding & target weighting
# ---------------------------------------------------------------------------

function _prepare_cluster_fit_frame(data::AbstractDataFrame, outcome::Symbol,
        treatment::Symbol, covariates::AbstractVector{Symbol},
        cluster::Symbol, handle_missing::Symbol)
    handle_missing in (:error, :mar) || throw(ArgumentError("handle_missing must be :error or :mar"))
    covs = collect(covariates)
    length(unique(covs)) == length(covs) || throw(ArgumentError("covariates must be unique"))
    treatment ∉ covs && outcome ∉ [treatment; covs] ||
        throw(ArgumentError("treatment, outcome and covariates must be distinct"))
    cluster ∉ [outcome; treatment; covs] ||
        throw(ArgumentError("cluster column must be distinct from outcome, treatment and covariates"))
    _validate_requested_columns(data, [outcome, cluster])
    frame = _bayesian_predictor_frame(data, treatment, covs)
    nrow(frame) > 0 || throw(ArgumentError("fitting data must contain at least one row"))
    observed = findall(!ismissing, data[!, outcome])
    length(observed) == nrow(data) || handle_missing == :mar || throw(ArgumentError(
        "missing outcomes require explicit handle_missing=:mar"))
    cids = data[observed, cluster]
    any(ismissing, cids) && throw(ArgumentError("cluster column must not contain missing values"))
    return (; covariates = covs, frame, observed, cids)
end

function _build_cluster_fit_design(frame::DataFrame, observed::Vector{Int},
        treatment::Symbol, covariates::Vector{Symbol}, basis, prior, standardize::Bool)
    schema = fit_covariate_schema(frame, [treatment; covariates])
    covariate_schema = fit_covariate_schema(frame, covariates)
    p = length(prior.mean)
    X_all = _bayesian_design(schema, basis, frame, treatment, covariates, frame[!, treatment], p)
    X_obs = X_all[observed, :]
    transform = _build_cluster_affine_transform(X_obs, p, standardize)
    X_std = _apply_affine_transform(X_obs, transform)
    prior_std = _transform_prior(prior, transform)
    feature_names = basis === nothing ? ["(Intercept)"; schema.feature_names] : ["basis_$j" for j in 1:p]
    design_metadata = (; feature_names, n_columns = p,
        basis_source = basis === nothing ? :covariate_schema : :user_function,
        adjustment = copy(covariates), categorical_levels = _bayesian_categorical_levels(frame),
        treatment_levels = collect(unique(frame[!, treatment])),
        standardized = standardize, outcome_scaling = :none)
    return (; schema, covariate_schema, p, X_obs, X_std, transform, prior_std, design_metadata)
end

function _target_row_weights(target::AbstractDataFrame, weighting::Symbol,
        cluster::Union{Nothing, Symbol})
    n = nrow(target)
    if weighting === :individual
        return fill(1.0 / n, n)
    elseif weighting === :cluster
        cluster === nothing && throw(ArgumentError(
            "cluster column must be supplied (or present on fit) for weighting=:cluster"))
        _validate_requested_columns(target, [cluster])
        cids = target[!, cluster]
        any(ismissing, cids) && throw(ArgumentError("target cluster column must not contain missing values"))
        counts = Dict{Any, Int}()
        for c in cids
            counts[c] = get(counts, c, 0) + 1
        end
        K = length(counts)
        return [1.0 / (K * counts[cids[i]]) for i in 1:n]
    else
        throw(ArgumentError("weighting must be :individual or :cluster"))
    end
end

# ---------------------------------------------------------------------------
# PSIS smoothing & operational refinement gates
# ---------------------------------------------------------------------------

const _PSIS_SMOOTHER = Ref{Any}(nothing)

function _psis_smooth_weights(log_ratios::AbstractVector{Float64})
    fn = _PSIS_SMOOTHER[]
    fn === nothing && throw(ArgumentError(
        "correction=:psis requires loading PSIS (`using PSIS`) to activate CausalTargetedPSISExt"))
    return fn(log_ratios)
end

function _weighted_quantile(values::AbstractVector{Float64}, weights::AbstractVector{Float64}, q::Float64)
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

"""
Apply PSIS smoothing and operational refinement gates to a `m × S` matrix of
contrast draws using one shared vector of log importance ratios.

Operational gates (plans/cluster-effects-bayesian-inference.md):
1. Finite log ratios, smoothed weights, `khat`, and `ess`.
2. Pareto `khat < khat_max` (default `0.7`).
3. `ess >= ess_min` (default `400.0`).
4. For every non-degenerate contrast (`sd > 0`), `mcse / sd <= mcse_sd_max` (default `0.05`).

Failed gates withhold intervals and covariance (`NaN`, `passed = false`) rather
than silently falling back to raw VI.
"""
function _psis_weighted_summary(draws::AbstractMatrix{Float64},
        log_ratios::AbstractVector{Float64};
        level::Float64 = 0.95, khat_max::Float64 = 0.7,
        ess_min::Float64 = 400.0, mcse_sd_max::Float64 = 0.05)
    m, S = size(draws)
    failures = Symbol[]
    if !all(isfinite, log_ratios)
        push!(failures, :nonfinite_log_ratios)
        return (; passed = false, failures, khat = NaN, ess = NaN,
            mean = fill(NaN, m), sd = fill(NaN, m), mcse = fill(NaN, m),
            lower = fill(NaN, m), upper = fill(NaN, m), covariance = fill(NaN, m, m),
            max_mcse_to_sd = NaN)
    end
    smoothed = _psis_smooth_weights(log_ratios)
    w = smoothed.weights
    khat = smoothed.khat
    ess = smoothed.ess
    if !(all(isfinite, w) && isfinite(khat) && isfinite(ess) && sum(w) > 0)
        push!(failures, :nonfinite_psis_weights)
        return (; passed = false, failures, khat, ess,
            mean = fill(NaN, m), sd = fill(NaN, m), mcse = fill(NaN, m),
            lower = fill(NaN, m), upper = fill(NaN, m), covariance = fill(NaN, m, m),
            max_mcse_to_sd = NaN)
    end
    w = w ./ sum(w)
    sum_w2 = sum(abs2, w)
    denom = max(1.0 - sum_w2, eps(Float64))

    μ = Vector{Float64}(undef, m)
    sd = Vector{Float64}(undef, m)
    mcse = Vector{Float64}(undef, m)
    lower = Vector{Float64}(undef, m)
    upper = Vector{Float64}(undef, m)
    α = (1.0 - level) / 2.0
    max_ratio = 0.0

    for j in 1:m
        vals = view(draws, j, :)
        μj = dot(w, vals)
        μ[j] = μj
        varj = 0.0
        mc_varj = 0.0
        @inbounds for s in 1:S
            d = vals[s] - μj
            ws = w[s]
            varj += ws * d * d
            mc_varj += (ws * ws) * d * d
        end
        sdj = sqrt(max(0.0, varj / denom))
        mcsej = sqrt(max(0.0, mc_varj))
        sd[j] = sdj
        mcse[j] = mcsej
        if sdj > 1e-14
            max_ratio = max(max_ratio, mcsej / sdj)
        end
        lower[j] = _weighted_quantile(vals, w, α)
        upper[j] = _weighted_quantile(vals, w, 1.0 - α)
    end

    cov_mat = zeros(Float64, m, m)
    for j1 in 1:m, j2 in j1:m
        c = 0.0
        @inbounds for s in 1:S
            c += w[s] * (draws[j1, s] - μ[j1]) * (draws[j2, s] - μ[j2])
        end
        cov_mat[j1, j2] = c / denom
        cov_mat[j2, j1] = cov_mat[j1, j2]
    end

    khat < khat_max || push!(failures, :khat_exceeded)
    ess >= ess_min || push!(failures, :ess_below_minimum)
    max_ratio <= mcse_sd_max || push!(failures, :mcse_exceeded)
    passed = isempty(failures)

    if !passed
        return (; passed = false, failures, khat, ess,
            mean = μ, sd = fill(NaN, m), mcse = fill(NaN, m),
            lower = fill(NaN, m), upper = fill(NaN, m), covariance = fill(NaN, m, m),
            max_mcse_to_sd = max_ratio)
    end
    return (; passed = true, failures, khat, ess, mean = μ, sd, mcse,
        lower, upper, covariance = Matrix(Symmetric(cov_mat)), max_mcse_to_sd = max_ratio)
end

# ---------------------------------------------------------------------------
# Shared bayesian_policy_contrast input preparation & draw summarisation
# ---------------------------------------------------------------------------

function _build_bayesian_policy_designs(fit, frame::DataFrame, a, discrete_values,
        deltas::Vector{Float64}, policy)
    p = length(fit.posterior_mean)
    categorical_levels = fit.design_metadata.categorical_levels
    X_obs = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, a, p;
        categorical_levels)
    m = length(deltas)
    X_policies = Vector{Matrix{Float64}}(undef, m)
    for (j, δ) in enumerate(deltas)
        shifted = policy isa DiscreteTreatmentPolicy ? discrete_values :
            apply_policy_values(Float64.(a), δ, policy)
        X_policies[j] = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, shifted, p;
            categorical_levels)
    end
    return (; p, X_obs, X_policies)
end

function _prepare_bayesian_contrast_inputs(fit, target::AbstractDataFrame, policy;
        deltas, n_draws::Integer, level::Real, mc_level::Real,
        weighting::Symbol, cluster::Union{Nothing, Symbol},
        correction::Symbol, check_mc_level::Bool = true)
    policy isa Union{SmoothBoundedLogitShiftPolicy, DiscreteTreatmentPolicy} ||
        throw(ArgumentError("use a fixed-bound smooth policy or DiscreteTreatmentPolicy; clamped ShiftPolicy is unsupported"))
    correction in (:none, :psis) || throw(ArgumentError("correction must be :none or :psis"))
    n_draws >= 2 || throw(ArgumentError("n_draws must be at least 2"))
    isfinite(level) && 0 < level < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
    if check_mc_level
        isfinite(mc_level) && 0 < mc_level < 1 ||
            throw(ArgumentError("mc_level must lie strictly between zero and one"))
    end
    deltas_f = Float64.(collect(deltas))
    !isempty(deltas_f) && all(isfinite, deltas_f) ||
        throw(ArgumentError("deltas must be non-empty and finite"))

    frame = _bayesian_predictor_frame(target, fit.treatment, fit.covariates)
    n_target = nrow(frame)
    n_target > 0 || throw(ArgumentError("target cohort must contain at least one row"))
    w = _target_row_weights(target, weighting, cluster)

    a = frame[!, fit.treatment]
    W_cov = transform_covariates(fit.covariate_schema, frame)
    discrete_values = policy isa DiscreteTreatmentPolicy ?
        _bayesian_discrete_values(a, W_cov, policy, fit.design_metadata.treatment_levels) : nothing
    designs = _build_bayesian_policy_designs(fit, frame, a, discrete_values, deltas_f, policy)
    policy_constants = policy isa SmoothBoundedLogitShiftPolicy ?
        (; kind = :smooth_bounded_logit_shift, lower = policy.lower, upper = policy.upper, deltas = copy(deltas_f)) :
        (; kind = :discrete, policy = deepcopy(policy), deltas = copy(deltas_f))
    return (; deltas = deltas_f, m = length(deltas_f), frame, n_target, w, a,
        discrete_values, p = designs.p, X_obs = designs.X_obs, X_policies = designs.X_policies,
        policy_constants)
end

function _summarise_mc_draws(draws::AbstractMatrix{Float64}, level::Float64)
    m, S = size(draws)
    α = (1.0 - level) / 2.0
    μ = vec(mean(draws; dims = 2))
    sd = [std(view(draws, j, :)) for j in 1:m]
    mcse = sd ./ sqrt(S)
    cov_mat = Matrix(Symmetric(cov(draws; dims = 2)))
    lower = [quantile(view(draws, j, :), α) for j in 1:m]
    upper = [quantile(view(draws, j, :), 1.0 - α) for j in 1:m]
    return (; mean = μ, sd, covariance = cov_mat, lower, upper, mcse)
end

function _unavailable_bayesian_contrast(μ_pt::Vector{Float64};
        deltas::Vector{Float64}, fit_metadata::NamedTuple, design_metadata::NamedTuple,
        target_kind::Symbol, weighting::Symbol, n_target::Int,
        level::Float64, mc_level::Float64, n_draws::Int,
        correction::Symbol, policy_constants::NamedTuple,
        extra_fields::NamedTuple = NamedTuple())
    m = length(deltas)
    nan_vec = fill(NaN, m)
    nan_mat = fill(NaN, m, m)
    metadata = (; fit_metadata..., functional = :policy_vs_observed,
        factual_reference = :observed, target = target_kind, weighting, n_target,
        level, mc_level, n_draws, correction,
        uncertainty = :unavailable, intervals = :unavailable,
        intervals_available = false, policy_constants, design = design_metadata)
    return merge(
        (; deltas, mean = μ_pt, sd = nan_vec, covariance = nan_mat,
            lower = nan_vec, upper = nan_vec, mcse = nan_vec),
        extra_fields,
        (; metadata),
    )
end

function _summarise_bayesian_contrast_draws(contrast_draws::Matrix{Float64};
        deltas::Vector{Float64}, level::Float64, mc_level::Float64,
        S::Int, correction::Symbol, log_ratios::Union{Nothing, Vector{Float64}} = nothing,
        fit_metadata::NamedTuple, design_metadata::NamedTuple,
        target_kind::Symbol, weighting::Symbol, n_target::Int,
        policy_constants::NamedTuple, vi_uncertainty::Symbol,
        extra_fields::NamedTuple = NamedTuple())
    if correction === :psis
        psis_res = _psis_weighted_summary(contrast_draws, log_ratios; level)
        psis_diagnostics = (; passed = psis_res.passed, failures = psis_res.failures,
            khat = psis_res.khat, ess = psis_res.ess, max_mcse_to_sd = psis_res.max_mcse_to_sd)
        metadata = (; fit_metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level, mc_level, n_draws = S, correction,
            uncertainty = psis_res.passed ? :psis_refined_moments : :unavailable,
            intervals = psis_res.passed ? :pointwise : :unavailable,
            intervals_available = psis_res.passed, psis_diagnostics,
            policy_constants, design = design_metadata)
        return merge(
            (; deltas, mean = psis_res.mean, sd = psis_res.sd,
                covariance = psis_res.covariance, lower = psis_res.lower,
                upper = psis_res.upper, mcse = psis_res.mcse),
            extra_fields,
            (; metadata),
        )
    end
    mc = _summarise_mc_draws(contrast_draws, level)
    metadata = (; fit_metadata..., functional = :policy_vs_observed,
        factual_reference = :observed, target = target_kind, weighting, n_target,
        level, mc_level, n_draws = S, correction,
        uncertainty = vi_uncertainty, intervals = :pointwise,
        intervals_available = true, policy_constants, design = design_metadata)
    return merge(
        (; deltas, mean = mc.mean, sd = mc.sd, covariance = mc.covariance,
            lower = mc.lower, upper = mc.upper, mcse = mc.mcse),
        extra_fields,
        (; metadata),
    )
end
