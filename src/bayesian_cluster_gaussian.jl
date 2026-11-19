using LinearAlgebra
using Random
using Statistics

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
    n, p = size(X)
    p > 0 || throw(ArgumentError("design matrix must contain at least one column"))
    length(y) == n || throw(DimensionMismatch("outcome length must match design rows"))
    length(cluster_ids) == n || throw(DimensionMismatch("cluster_ids length must match design rows"))
    all(isfinite, X) && all(isfinite, y) ||
        throw(ArgumentError("hierarchical Gaussian inputs must be finite"))
    any(ismissing, cluster_ids) && throw(ArgumentError("cluster_ids must not contain missing values"))
    Xf = Matrix{Float64}(X)
    yf = Vector{Float64}(y)
    T = nonmissingtype(eltype(cluster_ids))
    levels = Vector{T}()
    level_to_idx = Dict{T, Int}()
    cidx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        cid = cluster_ids[i]
        idx = get!(level_to_idx, cid) do
            push!(levels, cid)
            length(levels)
        end
        cidx[i] = idx
    end
    K = length(levels)
    sizes = zeros(Int, K)
    x_sums = zeros(Float64, p, K)
    y_sums = zeros(Float64, K)
    @inbounds for i in 1:n
        k = cidx[i]
        sizes[k] += 1
        y_sums[k] += yf[i]
        for j in 1:p
            x_sums[j, k] += Xf[i, j]
        end
    end
    xtx = n == 0 ? zeros(Float64, p, p) : Matrix(Symmetric(Xf' * Xf))
    xty = n == 0 ? zeros(Float64, p) : Vector{Float64}(Xf' * yf)
    yty = n == 0 ? 0.0 : Float64(dot(yf, yf))
    return HierarchicalGaussianSufficientStats{T}(
        n, p, K, levels, cidx, sizes, x_sums, y_sums, xtx, xty, yty)
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
