using CausalTargeted
using LinearAlgebra
using StableRNGs
using Statistics
using Test

@testset "Hierarchical Gaussian foundation (Milestone 1)" begin
    # Independent dense joint-Gaussian oracle over θ = [β; b] ∈ ℝ^(p + K):
    # explicitly forms the N × K cluster indicator matrix Z and W = [X Z],
    # constructs the dense (p + K) × (p + K) precision matrix Q and RHS h,
    # and computes exact moments via dense matrix inversion (no Schur algebra).
    function dense_joint_gaussian_oracle(
        X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real},
        cluster_ids::AbstractVector,
        prior::Union{ConjugateGaussianPrior, ClusterGaussianPrior};
        residual_precision::Real,
        cluster_precision::Real,
    )
        cp = prior isa ClusterGaussianPrior ? prior.coefficient_prior : prior
        N, p = size(X)
        Xf = Matrix{Float64}(X)
        yf = Vector{Float64}(y)
        τε = Float64(residual_precision)
        τb = Float64(cluster_precision)

        levels = unique(cluster_ids)
        K = length(levels)
        level_to_idx = Dict(lvl => k for (k, lvl) in enumerate(levels))
        Z = zeros(Float64, N, K)
        for i in 1:N
            Z[i, level_to_idx[cluster_ids[i]]] = 1.0
        end

        W = hcat(Xf, Z)
        V0_inv = inv(Matrix{Float64}(cp.covariance))
        μ0 = Vector{Float64}(cp.mean)

        Q = zeros(Float64, p + K, p + K)
        Q[1:p, 1:p] .= τε .* V0_inv
        for k in 1:K
            Q[p + k, p + k] = τb
        end
        Q .+= τε .* (W' * W)

        h = zeros(Float64, p + K)
        h[1:p] .= τε .* (V0_inv * μ0)
        h .+= τε .* (W' * yf)

        Σ_joint = inv(Q)
        Σ_joint = 0.5 .* (Σ_joint .+ Σ_joint')
        m_joint = Σ_joint * h

        beta_mean = m_joint[1:p]
        beta_cov = Σ_joint[1:p, 1:p]
        cluster_mean = K == 0 ? Float64[] : m_joint[(p + 1):(p + K)]
        cluster_cov = K == 0 ? zeros(Float64, 0, 0) : Σ_joint[(p + 1):(p + K), (p + 1):(p + K)]
        cluster_var = K == 0 ? Float64[] : diag(cluster_cov)
        beta_cluster_cov = K == 0 ? zeros(Float64, p, 0) : Σ_joint[1:p, (p + 1):(p + K)]

        resid = yf - W * m_joint
        expected_residual_ss = N == 0 ? 0.0 : dot(resid, resid) + tr((W' * W) * Σ_joint)
        expected_cluster_ss = K == 0 ? 0.0 : dot(cluster_mean, cluster_mean) + tr(cluster_cov)

        return (;
            levels,
            Z,
            Q,
            h,
            beta_mean,
            beta_cov,
            cluster_mean,
            cluster_cov,
            cluster_var,
            beta_cluster_cov,
            expected_residual_ss,
            expected_cluster_ss,
        )
    end

    # Shared unbalanced-cluster fixture (includes singletons, non-zero μ₀, correlated V₀)
    rng_base = StableRNG(1001)
    cluster_ids = ["cB", "cA", "cB", "cSingleton1", "cC", "cB", "cC", "cA", "cSingleton2", "cB", "cC", "cC", "cB"]
    N, p = length(cluster_ids), 3
    X = hcat(ones(N), randn(rng_base, N, 2))
    μ0 = [1.25, -0.75, 0.4]
    V0 = [1.2 0.35 -0.2; 0.35 0.9 0.25; -0.2 0.25 1.4]
    prior = ClusterGaussianPrior(;
        mean = μ0,
        covariance = V0,
        residual_shape = 2.5,
        residual_scale = 1.2,
        cluster_shape = 3.0,
        cluster_scale = 0.6,
    )
    b_true = Dict("cB" => 0.4, "cA" => -0.3, "cSingleton1" => 0.9, "cC" => -0.5, "cSingleton2" => 0.1)
    y = X * [0.8, -1.1, 0.5] .+ [b_true[c] for c in cluster_ids] .+ 0.3 .* randn(rng_base, N)
    τε = 2.3
    τb = 1.7

    stats = CausalTargeted._cluster_sufficient_stats(X, y, cluster_ids)
    state = CausalTargeted._solve_hierarchical_gaussian_block(
        stats, prior; residual_precision = τε, cluster_precision = τb)
    oracle = dense_joint_gaussian_oracle(
        X, y, cluster_ids, prior; residual_precision = τε, cluster_precision = τb)

    @testset "(a) Sparse/block moments match dense joint-Gaussian oracle" begin
        @test stats.cluster_levels == oracle.levels
        @test stats.cluster_sizes == [5, 2, 1, 4, 1]
        @test state.beta_mean ≈ oracle.beta_mean atol = 1e-11
        @test state.beta_cov ≈ oracle.beta_cov atol = 1e-11
        @test state.cluster_mean ≈ oracle.cluster_mean atol = 1e-11
        @test state.cluster_var ≈ oracle.cluster_var atol = 1e-11
        @test state.beta_cluster_cov ≈ oracle.beta_cluster_cov atol = 1e-11
        @test state.expected_residual_ss ≈ oracle.expected_residual_ss atol = 1e-11
        @test state.expected_cluster_ss ≈ oracle.expected_cluster_ss atol = 1e-11

        # Passing ConjugateGaussianPrior directly yields identical block moments.
        state_cp = CausalTargeted._solve_hierarchical_gaussian_block(
            stats, prior.coefficient_prior; residual_precision = τε, cluster_precision = τb)
        @test state_cp.beta_mean ≈ state.beta_mean atol = 1e-13
        @test state_cp.beta_cov ≈ state.beta_cov atol = 1e-13
        @test state_cp.cluster_mean ≈ state.cluster_mean atol = 1e-13
        @test state_cp.beta_cluster_cov ≈ state.beta_cluster_cov atol = 1e-13
    end

    @testset "(b) Joint draws match analytic means, covariances, and cross-covariances" begin
        n_draws = 20_000
        draws = CausalTargeted._sample_hierarchical_gaussian_block(
            state, stats; n_draws = n_draws, rng = StableRNG(2026))
        @test size(draws.beta_draws) == (p, n_draws)
        @test size(draws.cluster_draws) == (stats.K, n_draws)

        sample_beta_mean = vec(mean(draws.beta_draws; dims = 2))
        sample_cluster_mean = vec(mean(draws.cluster_draws; dims = 2))
        sample_beta_cov = cov(draws.beta_draws')
        sample_cluster_var = vec(var(draws.cluster_draws; dims = 2))
        sample_cluster_cov = cov(draws.cluster_draws')
        sample_beta_cluster_cov = cov(draws.beta_draws', draws.cluster_draws')

        for j in 1:p
            se = sqrt(state.beta_cov[j, j] / n_draws)
            @test abs(sample_beta_mean[j] - state.beta_mean[j]) <= max(0.01, 6.0 * se)
        end
        for k in 1:stats.K
            se = sqrt(state.cluster_var[k] / n_draws)
            @test abs(sample_cluster_mean[k] - state.cluster_mean[k]) <= max(0.01, 6.0 * se)
        end
        for j1 in 1:p, j2 in 1:p
            se = sqrt((state.beta_cov[j1, j1] * state.beta_cov[j2, j2] + state.beta_cov[j1, j2]^2) / (n_draws - 1))
            @test abs(sample_beta_cov[j1, j2] - state.beta_cov[j1, j2]) <= max(0.01, 6.0 * se)
        end
        for k in 1:stats.K
            se = state.cluster_var[k] * sqrt(2.0 / (n_draws - 1))
            @test abs(sample_cluster_var[k] - state.cluster_var[k]) <= max(0.01, 6.0 * se)
        end
        for k1 in 1:stats.K, k2 in 1:stats.K
            se = sqrt((oracle.cluster_cov[k1, k1] * oracle.cluster_cov[k2, k2] + oracle.cluster_cov[k1, k2]^2) / (n_draws - 1))
            @test abs(sample_cluster_cov[k1, k2] - oracle.cluster_cov[k1, k2]) <= max(0.01, 6.0 * se)
        end
        for j in 1:p, k in 1:stats.K
            se = sqrt((state.beta_cov[j, j] * state.cluster_var[k] + state.beta_cluster_cov[j, k]^2) / (n_draws - 1))
            @test abs(sample_beta_cluster_cov[j, k] - state.beta_cluster_cov[j, k]) <= max(0.01, 6.0 * se)
        end
    end

    @testset "(c) Zero cluster variance (cluster_precision = Inf) reduces to non-hierarchical posterior" begin
        state_inf = CausalTargeted._solve_hierarchical_gaussian_block(
            stats, prior; residual_precision = τε, cluster_precision = Inf)
        ref = CausalTargeted._bayesian_gaussian_posterior(
            X, y, prior.coefficient_prior; observation_variance = 1.0 / τε)

        @test state_inf.beta_mean ≈ ref.posterior_mean atol = 1e-12
        @test state_inf.beta_cov ≈ ref.posterior_covariance atol = 1e-12
        @test state_inf.cluster_mean == zeros(stats.K)
        @test state_inf.cluster_var == zeros(stats.K)
        @test state_inf.cluster_cond_var == zeros(stats.K)
        @test state_inf.beta_cluster_cov == zeros(p, stats.K)
        @test state_inf.expected_cluster_ss == 0.0
        expected_rss = sum(abs2, y .- X * ref.posterior_mean) + tr((X' * X) * ref.posterior_covariance)
        @test state_inf.expected_residual_ss ≈ expected_rss atol = 1e-12

        draws_inf = CausalTargeted._sample_hierarchical_gaussian_block(
            state_inf, stats; n_draws = 100, rng = StableRNG(3003))
        @test size(draws_inf.beta_draws) == (p, 100)
        @test size(draws_inf.cluster_draws) == (stats.K, 100)
        @test all(iszero, draws_inf.cluster_draws)
    end

    @testset "(d) Prior-only (n == 0) case" begin
        X0 = zeros(Float64, 0, p)
        y0 = Float64[]
        c0 = String[]
        stats0 = CausalTargeted._cluster_sufficient_stats(X0, y0, c0)
        state0 = CausalTargeted._solve_hierarchical_gaussian_block(
            stats0, prior; residual_precision = τε, cluster_precision = τb)
        oracle0 = dense_joint_gaussian_oracle(
            X0, y0, c0, prior; residual_precision = τε, cluster_precision = τb)

        @test stats0.n == 0
        @test stats0.K == 0
        @test state0.beta_mean ≈ μ0 atol = 1e-12
        @test state0.beta_cov ≈ V0 ./ τε atol = 1e-12
        @test state0.beta_mean ≈ oracle0.beta_mean atol = 1e-12
        @test state0.beta_cov ≈ oracle0.beta_cov atol = 1e-12
        @test isempty(state0.cluster_mean)
        @test isempty(state0.cluster_var)
        @test isempty(state0.cluster_cond_var)
        @test size(state0.beta_cluster_cov) == (p, 0)
        @test state0.expected_residual_ss == 0.0
        @test state0.expected_cluster_ss == 0.0

        draws0 = CausalTargeted._sample_hierarchical_gaussian_block(
            state0, stats0; n_draws = 10_000, rng = StableRNG(4004))
        @test size(draws0.beta_draws) == (p, 10_000)
        @test size(draws0.cluster_draws) == (0, 10_000)
        @test vec(mean(draws0.beta_draws; dims = 2)) ≈ μ0 atol = 0.05
        @test cov(draws0.beta_draws') ≈ V0 ./ τε atol = 0.05
    end

    @testset "(e) Rank-deficient designs remain SPD and match dense oracle" begin
        rng_e = StableRNG(5005)

        # Collinear columns (col 4 = 2 * col 2 - 0.5 * col 3)
        x1 = randn(rng_e, 8)
        x2 = randn(rng_e, 8)
        X_col = hcat(ones(8), x1, x2, 2.0 .* x1 .- 0.5 .* x2)
        c_col = [1, 1, 2, 2, 2, 3, 4, 4]
        y_col = randn(rng_e, 8)
        prior_col = ClusterGaussianPrior(;
            mean = [0.2, -0.3, 0.5, -0.1],
            covariance = [1.5 0.2 0.0 -0.1; 0.2 1.2 0.3 0.0; 0.0 0.3 1.0 0.2; -0.1 0.0 0.2 1.3],
            residual_shape = 2.0,
            residual_scale = 1.0,
            cluster_shape = 2.5,
            cluster_scale = 0.5,
        )
        stats_col = CausalTargeted._cluster_sufficient_stats(X_col, y_col, c_col)
        state_col = CausalTargeted._solve_hierarchical_gaussian_block(
            stats_col, prior_col; residual_precision = 1.8, cluster_precision = 1.1)
        oracle_col = dense_joint_gaussian_oracle(
            X_col, y_col, c_col, prior_col; residual_precision = 1.8, cluster_precision = 1.1)

        @test isposdef(Symmetric(state_col.beta_cov))
        @test isposdef(Symmetric(Matrix(state_col.beta_precision_chol)))
        @test state_col.beta_mean ≈ oracle_col.beta_mean atol = 1e-11
        @test state_col.beta_cov ≈ oracle_col.beta_cov atol = 1e-11
        @test state_col.cluster_mean ≈ oracle_col.cluster_mean atol = 1e-11
        @test state_col.cluster_var ≈ oracle_col.cluster_var atol = 1e-11
        @test state_col.beta_cluster_cov ≈ oracle_col.beta_cluster_cov atol = 1e-11
        @test state_col.expected_residual_ss ≈ oracle_col.expected_residual_ss atol = 1e-11
        @test state_col.expected_cluster_ss ≈ oracle_col.expected_cluster_ss atol = 1e-11

        # High-dimensional p > n (n = 4, p = 7)
        X_wide = randn(rng_e, 4, 7)
        c_wide = [:a, :a, :b, :c]
        y_wide = randn(rng_e, 4)
        V_wide = [0.5^abs(i - j) for i in 1:7, j in 1:7]
        prior_wide = ClusterGaussianPrior(;
            mean = collect(range(-0.6, 0.6; length = 7)),
            covariance = V_wide,
            residual_shape = 2.0,
            residual_scale = 1.0,
            cluster_shape = 2.0,
            cluster_scale = 0.5,
        )
        stats_wide = CausalTargeted._cluster_sufficient_stats(X_wide, y_wide, c_wide)
        state_wide = CausalTargeted._solve_hierarchical_gaussian_block(
            stats_wide, prior_wide; residual_precision = 1.4, cluster_precision = 2.2)
        oracle_wide = dense_joint_gaussian_oracle(
            X_wide, y_wide, c_wide, prior_wide; residual_precision = 1.4, cluster_precision = 2.2)

        @test isposdef(Symmetric(state_wide.beta_cov))
        @test isposdef(Symmetric(Matrix(state_wide.beta_precision_chol)))
        @test state_wide.beta_mean ≈ oracle_wide.beta_mean atol = 1e-11
        @test state_wide.beta_cov ≈ oracle_wide.beta_cov atol = 1e-11
        @test state_wide.cluster_mean ≈ oracle_wide.cluster_mean atol = 1e-11
        @test state_wide.cluster_var ≈ oracle_wide.cluster_var atol = 1e-11
        @test state_wide.beta_cluster_cov ≈ oracle_wide.beta_cluster_cov atol = 1e-11
        @test state_wide.expected_residual_ss ≈ oracle_wide.expected_residual_ss atol = 1e-11
        @test state_wide.expected_cluster_ss ≈ oracle_wide.expected_cluster_ss atol = 1e-11
    end

    @testset "(f) Exact affine reparameterisation preserves posterior moments" begin
        rng_f = StableRNG(6006)
        c_f = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4]
        n_f = length(c_f)

        function check_affine_equivalence(X_raw, y_raw, prior_raw, expected_intercept_col)
            t = CausalTargeted._freeze_affine_transform(X_raw)
            @test t.intercept_col == expected_intercept_col
            @test t.A * t.A_inv ≈ Matrix{Float64}(I, size(X_raw, 2), size(X_raw, 2)) atol = 1e-12

            X_std = CausalTargeted._apply_affine_transform(X_raw, t)
            @test X_std * t.A_inv ≈ X_raw atol = 1e-11

            prior_std = CausalTargeted._transform_prior(prior_raw, t)
            @test prior_std isa typeof(prior_raw)

            stats_raw = CausalTargeted._cluster_sufficient_stats(X_raw, y_raw, c_f)
            stats_std = CausalTargeted._cluster_sufficient_stats(X_std, y_raw, c_f)
            state_raw = CausalTargeted._solve_hierarchical_gaussian_block(
                stats_raw, prior_raw; residual_precision = 2.1, cluster_precision = 1.6)
            state_std = CausalTargeted._solve_hierarchical_gaussian_block(
                stats_std, prior_std; residual_precision = 2.1, cluster_precision = 1.6)
            state_un = CausalTargeted._untransform_block_state(state_std, t)

            @test state_un.beta_mean ≈ state_raw.beta_mean atol = 1e-10
            @test state_un.beta_cov ≈ state_raw.beta_cov atol = 1e-10
            @test Matrix(state_un.beta_precision_chol) ≈ Matrix(state_raw.beta_precision_chol) atol = 1e-10
            @test state_un.cluster_mean ≈ state_raw.cluster_mean atol = 1e-10
            @test state_un.cluster_var ≈ state_raw.cluster_var atol = 1e-10
            @test state_un.cluster_cond_var ≈ state_raw.cluster_cond_var atol = 1e-10
            @test state_un.beta_cluster_cov ≈ state_raw.beta_cluster_cov atol = 1e-10
            @test state_un.expected_residual_ss ≈ state_raw.expected_residual_ss atol = 1e-10
            @test state_un.expected_cluster_ss ≈ state_raw.expected_cluster_ss atol = 1e-10
            return (; t, X_std)
        end

        # 1. With intercept: origin shift + rescaling of covariates
        X_int = hcat(ones(n_f), 10.0 .+ 0.2 .* randn(rng_f, n_f), -5.0 .+ 4.0 .* randn(rng_f, n_f))
        y_int = 1.5 .+ 0.8 .* (X_int[:, 2] .- 10.0) .- 0.2 .* X_int[:, 3] .+ 0.3 .* randn(rng_f, n_f)
        res_int = check_affine_equivalence(X_int, y_int, prior, 1)
        @test res_int.X_std[:, 1] == ones(n_f)
        @test abs(mean(res_int.X_std[:, 2])) < 1e-12
        @test std(res_int.X_std[:, 2]) ≈ 1.0 atol = 1e-12
        @test abs(mean(res_int.X_std[:, 3])) < 1e-12
        @test std(res_int.X_std[:, 3]) ≈ 1.0 atol = 1e-12

        # 2. Without intercept: pure scaling (no centering)
        X_noint = hcat(10.0 .+ 0.5 .* randn(rng_f, n_f), -3.0 .+ 3.0 .* randn(rng_f, n_f))
        y_noint = 0.3 .* X_noint[:, 1] .- 0.4 .* X_noint[:, 2] .+ 0.3 .* randn(rng_f, n_f)
        prior2 = ClusterGaussianPrior(;
            mean = [0.4, -0.6],
            covariance = [1.3 0.25; 0.25 0.8],
            residual_shape = 2.5,
            residual_scale = 1.0,
            cluster_shape = 2.5,
            cluster_scale = 0.7,
        )
        res_noint = check_affine_equivalence(X_noint, y_noint, prior2, nothing)
        @test res_noint.t.centers == zeros(2)
        @test std(res_noint.X_std[:, 1]) ≈ 1.0 atol = 1e-12
        @test std(res_noint.X_std[:, 2]) ≈ 1.0 atol = 1e-12

        # 3. With a constant non-intercept column (both with and without intercept)
        X_const_int = hcat(ones(n_f), 8.0 .+ 0.5 .* randn(rng_f, n_f), fill(5.0, n_f))
        y_const = 0.7 .+ 0.4 .* X_const_int[:, 2] .+ 0.1 .* X_const_int[:, 3] .+ 0.2 .* randn(rng_f, n_f)
        res_const_int = check_affine_equivalence(X_const_int, y_const, prior, 1)
        @test res_const_int.t.scales[3] == 1.0
        @test res_const_int.t.centers[3] == 0.0
        @test res_const_int.X_std[:, 3] == fill(5.0, n_f)

        X_const_noint = hcat(2.0 .+ 0.4 .* randn(rng_f, n_f), fill(-3.5, n_f))
        res_const_noint = check_affine_equivalence(X_const_noint, y_noint, prior2, nothing)
        @test res_const_noint.t.scales[2] == 1.0
        @test res_const_noint.X_std[:, 2] == fill(-3.5, n_f)
    end

    @testset "(g) ClusterGaussianPrior ergonomics and argument validation" begin
        μ = [0.5, -0.2]
        V = [1.0 0.2; 0.2 1.5]
        cp = ConjugateGaussianPrior(; mean = μ, covariance = V, shape = 2.5, scale = 1.2)
        p_kw = ClusterGaussianPrior(;
            mean = μ,
            covariance = V,
            residual_shape = 2.5,
            residual_scale = 1.2,
            cluster_shape = 3.0,
            cluster_scale = 0.7,
        )
        p_cp = ClusterGaussianPrior(cp; cluster_shape = 3.0, cluster_scale = 0.7)

        @test Set(propertynames(p_kw)) == Set((
            :coefficient_prior, :mean, :covariance,
            :residual_shape, :residual_scale, :cluster_shape, :cluster_scale,
        ))
        for p_test in (p_kw, p_cp)
            @test p_test.coefficient_prior.mean == μ
            @test p_test.coefficient_prior.covariance == V
            @test p_test.mean == μ
            @test p_test.covariance == V
            @test p_test.residual_shape == 2.5
            @test p_test.residual_scale == 1.2
            @test p_test.cluster_shape == 3.0
            @test p_test.cluster_scale == 0.7
        end

        # Shape <= 1 or non-finite
        for bad_shape in (1.0, 0.5, -1.0, Inf, NaN)
            @test_throws ArgumentError ClusterGaussianPrior(;
                mean = μ, covariance = V,
                residual_shape = bad_shape, residual_scale = 1.0,
                cluster_shape = 2.0, cluster_scale = 1.0)
            @test_throws ArgumentError ClusterGaussianPrior(;
                mean = μ, covariance = V,
                residual_shape = 2.0, residual_scale = 1.0,
                cluster_shape = bad_shape, cluster_scale = 1.0)
            @test_throws ArgumentError ClusterGaussianPrior(cp; cluster_shape = bad_shape, cluster_scale = 1.0)
        end

        # Scale <= 0 or non-finite
        for bad_scale in (0.0, -0.5, Inf, NaN)
            @test_throws ArgumentError ClusterGaussianPrior(;
                mean = μ, covariance = V,
                residual_shape = 2.0, residual_scale = bad_scale,
                cluster_shape = 2.0, cluster_scale = 1.0)
            @test_throws ArgumentError ClusterGaussianPrior(;
                mean = μ, covariance = V,
                residual_shape = 2.0, residual_scale = 1.0,
                cluster_shape = 2.0, cluster_scale = bad_scale)
            @test_throws ArgumentError ClusterGaussianPrior(cp; cluster_shape = 2.0, cluster_scale = bad_scale)
        end

        # Non-PD or asymmetric covariances
        for bad_V in ([1.0 1.0; 1.0 1.0], [1.0 2.0; 2.0 1.0], [1.0 0.5; 0.0 1.0])
            @test_throws ArgumentError ClusterGaussianPrior(;
                mean = μ, covariance = bad_V,
                residual_shape = 2.0, residual_scale = 1.0,
                cluster_shape = 2.0, cluster_scale = 1.0)
        end

        # Non-finite prior parameters
        @test_throws ArgumentError ClusterGaussianPrior(;
            mean = [NaN, 0.0], covariance = V,
            residual_shape = 2.0, residual_scale = 1.0,
            cluster_shape = 2.0, cluster_scale = 1.0)
        @test_throws ArgumentError ClusterGaussianPrior(;
            mean = μ, covariance = [Inf 0.0; 0.0 1.0],
            residual_shape = 2.0, residual_scale = 1.0,
            cluster_shape = 2.0, cluster_scale = 1.0)

        # Non-finite data and missing cluster IDs
        X_ok = [1.0 0.0; 1.0 1.0; 1.0 2.0]
        y_ok = [0.1, 0.2, 0.3]
        c_ok = [1, 1, 2]
        @test_throws ArgumentError CausalTargeted._cluster_sufficient_stats([1.0 NaN; 1.0 1.0; 1.0 2.0], y_ok, c_ok)
        @test_throws ArgumentError CausalTargeted._cluster_sufficient_stats([1.0 Inf; 1.0 1.0; 1.0 2.0], y_ok, c_ok)
        @test_throws ArgumentError CausalTargeted._cluster_sufficient_stats(X_ok, [0.1, NaN, 0.3], c_ok)
        @test_throws ArgumentError CausalTargeted._cluster_sufficient_stats(X_ok, [0.1, -Inf, 0.3], c_ok)
        @test_throws ArgumentError CausalTargeted._cluster_sufficient_stats(X_ok, y_ok, Union{Missing, Int}[1, missing, 2])
        @test_throws ArgumentError CausalTargeted._freeze_affine_transform([1.0 NaN; 1.0 1.0])

        # Invalid precisions and dimension mismatches
        stats_ok = CausalTargeted._cluster_sufficient_stats(X_ok, y_ok, c_ok)
        for bad_τε in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError CausalTargeted._solve_hierarchical_gaussian_block(
                stats_ok, p_kw; residual_precision = bad_τε, cluster_precision = 1.0)
        end
        for bad_τb in (0.0, -1.0, NaN)
            @test_throws ArgumentError CausalTargeted._solve_hierarchical_gaussian_block(
                stats_ok, p_kw; residual_precision = 1.0, cluster_precision = bad_τb)
        end
        @test_throws DimensionMismatch CausalTargeted._cluster_sufficient_stats(X_ok, [0.1, 0.2], c_ok)
        @test_throws DimensionMismatch CausalTargeted._cluster_sufficient_stats(X_ok, y_ok, [1, 1])
        @test_throws DimensionMismatch CausalTargeted._solve_hierarchical_gaussian_block(
            stats_ok, prior; residual_precision = 1.0, cluster_precision = 1.0)
    end
end
