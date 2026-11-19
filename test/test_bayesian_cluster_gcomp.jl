using CausalTargeted
using DataFrames
using Distributions
using LinearAlgebra
using StableRNGs
using Statistics
using Test

@testset "Hierarchical Bayesian cluster g-computation (Milestone 2)" begin
    function simulate_cluster_data(
        rng,
        cluster_sizes::AbstractVector{Int};
        σ_b::Real = 0.6,
        σ_ε::Real = 0.8,
        β::AbstractVector{<:Real} = [1.0, 0.9, -0.5],
        binary_treatment::Bool = false,
    )
        K = length(cluster_sizes)
        N = sum(cluster_sizes)
        cluster = Vector{String}(undef, N)
        W = Vector{Float64}(undef, N)
        A = Vector{Float64}(undef, N)
        Y = Vector{Float64}(undef, N)
        idx = 0
        for k in 1:K
            b_k = Float64(σ_b) * randn(rng)
            cid = "c$k"
            for _ in 1:cluster_sizes[k]
                idx += 1
                w = randn(rng)
                a = binary_treatment ? Float64(rand(rng) < 1 / (1 + exp(-0.5 * w))) : (0.5 * w + 0.8 * randn(rng))
                y = β[1] + β[2] * a + β[3] * w + b_k + Float64(σ_ε) * randn(rng)
                cluster[idx] = cid
                W[idx] = w
                A[idx] = a
                Y[idx] = y
            end
        end
        return DataFrame(; cluster, A, W, Y)
    end

    default_prior(p::Int = 3) = ClusterGaussianPrior(;
        mean = zeros(p),
        covariance = 4.0 .* Matrix{Float64}(I, p, p),
        residual_shape = 2.5,
        residual_scale = 1.0,
        cluster_shape = 2.5,
        cluster_scale = 0.5,
    )

    @testset "1. ELBO monotonicity and convergence across cluster regimes" begin
        regimes = [
            (label = "balanced", sizes = fill(6, 10), σ_b = 0.6, σ_ε = 0.8, seed = 101),
            (label = "unequal_with_singletons", sizes = [1, 1, 2, 3, 5, 8, 1, 4, 6, 3], σ_b = 0.7, σ_ε = 0.8, seed = 102),
            (label = "weak_icc", sizes = [4, 5, 6, 3, 5, 4, 6, 5], σ_b = 0.05, σ_ε = 1.0, seed = 103),
            (label = "strong_icc", sizes = [4, 5, 6, 3, 5, 4, 6, 5], σ_b = 2.0, σ_ε = 0.4, seed = 104),
            (label = "small_K4", sizes = [3, 5, 4, 2], σ_b = 0.8, σ_ε = 0.7, seed = 105),
        ]
        prior = default_prior(3)
        for r in regimes
            df = simulate_cluster_data(StableRNG(r.seed), r.sizes; σ_b = r.σ_b, σ_ε = r.σ_ε)
            fit = fit_bayesian_cluster_gcomp(df, :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit.status === :converged
            @test fit.converged == true
            @test length(fit.elbo_trace) >= 2
            @test isfinite(fit.elbo)
            @test all(diff(fit.elbo_trace) .>= -1e-9)
        end
    end

    @testset "2. Exact marginal log-posterior vs dense MvNormal oracle" begin
        rng = StableRNG(2026)
        cluster_ids = ["cB", "cA", "cB", "cS1", "cC", "cB", "cC", "cA", "cS2", "cB", "cC", "cC", "cB"]
        N, p = length(cluster_ids), 3
        X = hcat(ones(N), randn(rng, N, 2))
        μ0 = [1.25, -0.75, 0.4]
        V0 = [1.2 0.35 -0.2; 0.35 0.9 0.25; -0.2 0.25 1.4]
        a_ε, b_ε = 2.5, 1.2
        a_b, b_b = 3.0, 0.6
        prior = ClusterGaussianPrior(;
            mean = μ0,
            covariance = V0,
            residual_shape = a_ε,
            residual_scale = b_ε,
            cluster_shape = a_b,
            cluster_scale = b_b,
        )
        b_true = Dict("cB" => 0.4, "cA" => -0.3, "cS1" => 0.9, "cC" => -0.5, "cS2" => 0.1)
        y = X * [0.8, -1.1, 0.5] .+ [b_true[c] for c in cluster_ids] .+ 0.3 .* randn(rng, N)
        stats = CausalTargeted._cluster_sufficient_stats(X, y, cluster_ids)

        levels = unique(cluster_ids)
        K = length(levels)
        lvl_idx = Dict(lvl => k for (k, lvl) in enumerate(levels))
        Z = zeros(Float64, N, K)
        for i in 1:N
            Z[i, lvl_idx[cluster_ids[i]]] = 1.0
        end
        ZZt = Z * Z'

        function dense_marginal_log_posterior(β, σ2_ε, σ2_b)
            Σ_y = σ2_ε * Matrix{Float64}(I, N, N) + σ2_b * ZZt
            return logpdf(MvNormal(X * β, Symmetric(Σ_y)), y) +
                   logpdf(MvNormal(μ0, Symmetric(σ2_ε * V0)), β) +
                   logpdf(InverseGamma(a_ε, b_ε), σ2_ε) + log(σ2_ε) +
                   logpdf(InverseGamma(a_b, b_b), σ2_b) + log(σ2_b)
        end

        test_points = [
            ([0.0, 0.0, 0.0], 1.0, 0.5),
            ([0.8, -1.1, 0.5], 0.09, 0.25),
            ([-1.3, 0.4, 1.7], 2.4, 0.04),
            ([1.5, -0.2, -0.9], 0.35, 3.2),
            ([0.25, 0.75, -0.15], 1.7, 1.1),
        ]
        for (β, σ2_ε, σ2_b) in test_points
            lp_fast = CausalTargeted._marginal_cluster_log_posterior(
                stats, prior, β, log(σ2_ε), log(σ2_b))
            lp_oracle = dense_marginal_log_posterior(β, σ2_ε, σ2_b)
            @test lp_fast ≈ lp_oracle atol = 1e-10
        end
    end

    @testset "3. Standardisation invariance" begin
        rng = StableRNG(303)
        sizes = [3, 6, 4, 1, 5, 7, 2, 5]
        df = simulate_cluster_data(rng, sizes; σ_b = 0.7, σ_ε = 0.6)
        # Shift and rescale predictors to exercise non-trivial affine standardisation
        df.A = -5.0 .+ 3.0 .* df.A
        df.W = 100.0 .+ 25.0 .* df.W
        prior = ClusterGaussianPrior(;
            mean = [0.5, -0.2, 0.1],
            covariance = [2.0 0.2 -0.1; 0.2 1.5 0.15; -0.1 0.15 1.2],
            residual_shape = 2.5,
            residual_scale = 1.1,
            cluster_shape = 2.5,
            cluster_scale = 0.6,
        )

        fit_std = fit_bayesian_cluster_gcomp(
            df, :Y, :A, [:W]; cluster = :cluster, prior, standardize = true, tol = 1e-11)
        fit_raw = fit_bayesian_cluster_gcomp(
            df, :Y, :A, [:W]; cluster = :cluster, prior, standardize = false, tol = 1e-11)

        @test fit_std.converged && fit_raw.converged
        @test fit_std.design_metadata.standardized == true
        @test fit_raw.design_metadata.standardized == false
        @test fit_std.posterior_mean ≈ fit_raw.posterior_mean atol = 1e-6
        @test fit_std.posterior_covariance ≈ fit_raw.posterior_covariance atol = 1e-6
        @test fit_std.cluster_mean ≈ fit_raw.cluster_mean atol = 1e-6
        @test fit_std.cluster_var ≈ fit_raw.cluster_var atol = 1e-6
        @test fit_std.residual_scale ≈ fit_raw.residual_scale atol = 1e-6
        @test fit_std.cluster_scale ≈ fit_raw.cluster_scale atol = 1e-6
    end

    @testset "4. Marginal policy contrasts (bayesian_policy_contrast)" begin
        @testset "Discrete static and continuous multi-delta policies" begin
            prior = default_prior(3)

            df_bin = simulate_cluster_data(
                StableRNG(401), fill(4, 12); σ_b = 0.5, σ_ε = 0.7, binary_treatment = true)
            fit_bin = fit_bayesian_cluster_gcomp(df_bin, :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit_bin.converged
            res_disc = bayesian_policy_contrast(
                fit_bin, df_bin, discrete_static_policy(1.0);
                deltas = [0.0], n_draws = 1000, rng = StableRNG(402))
            @test all(isfinite, res_disc.mean)
            @test all(>(0), res_disc.sd)
            @test issymmetric(res_disc.covariance)
            @test all(>=(-1e-12), eigvals(Symmetric(res_disc.covariance)))
            @test all(res_disc.lower .< res_disc.mean .< res_disc.upper)
            @test res_disc.metadata.uncertainty === :structured_variational

            df_cont = simulate_cluster_data(
                StableRNG(403), [4, 6, 3, 5, 7, 4, 5, 6]; σ_b = 0.6, σ_ε = 0.7)
            df_cont.A = clamp.(df_cont.A, -2.5, 2.5)
            fit_cont = fit_bayesian_cluster_gcomp(df_cont, :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit_cont.converged
            policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)
            res_cont = bayesian_policy_contrast(
                fit_cont, df_cont, policy;
                deltas = [-0.5, 0.5], n_draws = 1000, rng = StableRNG(404))
            @test length(res_cont.mean) == 2
            @test all(isfinite, res_cont.mean)
            @test all(>(0), res_cont.sd)
            @test size(res_cont.covariance) == (2, 2)
            @test issymmetric(res_cont.covariance)
            @test all(>=(-1e-12), eigvals(Symmetric(res_cont.covariance)))
            @test all(res_cont.lower .< res_cont.mean .< res_cont.upper)
            @test res_cont.metadata.uncertainty === :structured_variational
        end

        @testset "Individual vs cluster weighting" begin
            prior3 = default_prior(3)
            df_eq = simulate_cluster_data(StableRNG(405), fill(5, 8); σ_b = 0.6, σ_ε = 0.7)
            df_eq.A = clamp.(df_eq.A, -2.5, 2.5)
            fit_eq = fit_bayesian_cluster_gcomp(df_eq, :Y, :A, [:W]; cluster = :cluster, prior = prior3)
            policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)

            res_eq_ind = bayesian_policy_contrast(
                fit_eq, df_eq, policy;
                deltas = [-0.5, 0.5], n_draws = 800, rng = StableRNG(406), weighting = :individual)
            res_eq_clu = bayesian_policy_contrast(
                fit_eq, df_eq, policy;
                deltas = [-0.5, 0.5], n_draws = 800, rng = StableRNG(406), weighting = :cluster)
            @test res_eq_ind.contrast_matrix ≈ res_eq_clu.contrast_matrix atol = 1e-12
            @test res_eq_ind.mean ≈ res_eq_clu.mean atol = 1e-12
            @test res_eq_ind.sd ≈ res_eq_clu.sd atol = 1e-12
            @test res_eq_ind.covariance ≈ res_eq_clu.covariance atol = 1e-12
            @test res_eq_ind.lower ≈ res_eq_clu.lower atol = 1e-12
            @test res_eq_ind.upper ≈ res_eq_clu.upper atol = 1e-12

            # Unequal cluster sizes where covariate W (and hence A×W interaction) depends on cluster size
            rng_u = StableRNG(407)
            sizes_u = [2, 2, 2, 10, 10, 10]
            cluster_u = String[]
            W_u = Float64[]
            A_u = Float64[]
            Y_u = Float64[]
            for (k, nk) in enumerate(sizes_u)
                b_k = 0.5 * randn(rng_u)
                w_base = nk <= 2 ? 0.0 : 1.0
                for _ in 1:nk
                    w = w_base + 0.1 * randn(rng_u)
                    a = clamp(0.4 * w + 0.6 * randn(rng_u), -2.5, 2.5)
                    y = 0.5 + 1.1 * a - 0.3 * w + 1.5 * a * w + b_k + 0.5 * randn(rng_u)
                    push!(cluster_u, "u$k")
                    push!(W_u, w)
                    push!(A_u, a)
                    push!(Y_u, y)
                end
            end
            df_un = DataFrame(cluster = cluster_u, A = A_u, W = W_u, Y = Y_u)
            basis_int = (df, a) -> hcat(ones(nrow(df)), a, df.W, a .* df.W)
            prior4 = default_prior(4)
            fit_un = fit_bayesian_cluster_gcomp(
                df_un, :Y, :A, [:W]; cluster = :cluster, prior = prior4, basis = basis_int)
            @test fit_un.converged

            deltas_u = [-0.5, 0.5]
            res_un_ind = bayesian_policy_contrast(
                fit_un, df_un, policy;
                deltas = deltas_u, n_draws = 800, rng = StableRNG(408), weighting = :individual)
            res_un_clu = bayesian_policy_contrast(
                fit_un, df_un, policy;
                deltas = deltas_u, n_draws = 800, rng = StableRNG(408), weighting = :cluster)

            N_u = nrow(df_un)
            K_u = length(sizes_u)
            w_ind = fill(1.0 / N_u, N_u)
            size_map = Dict("u$k" => sizes_u[k] for k in 1:K_u)
            w_clu = [1.0 / (K_u * size_map[c]) for c in df_un.cluster]
            X_obs = basis_int(df_un, df_un.A)
            C_ind_expected = Matrix{Float64}(undef, length(deltas_u), 4)
            C_clu_expected = Matrix{Float64}(undef, length(deltas_u), 4)
            for (j, δ) in enumerate(deltas_u)
                a_shift = apply_policy_values(df_un.A, δ, policy)
                dX = basis_int(df_un, a_shift) - X_obs
                C_ind_expected[j, :] = vec(w_ind' * dX)
                C_clu_expected[j, :] = vec(w_clu' * dX)
            end

            expected_diff = C_clu_expected - C_ind_expected
            @test norm(expected_diff) > 1e-3
            @test (res_un_clu.contrast_matrix - res_un_ind.contrast_matrix) ≈ expected_diff atol = 1e-12
            @test !(res_un_clu.mean ≈ res_un_ind.mean)
        end

        @testset "Non-converged fit withholds intervals and covariance" begin
            prior = default_prior(3)
            df = simulate_cluster_data(StableRNG(409), [4, 5, 6, 3, 5]; σ_b = 0.6, σ_ε = 0.8)
            df.A = clamp.(df.A, -2.5, 2.5)
            fit_nc = fit_bayesian_cluster_gcomp(
                df, :Y, :A, [:W]; cluster = :cluster, prior, maxiter = 1)
            @test fit_nc.status === :maxiter_reached
            @test fit_nc.converged == false

            policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)
            res_nc = bayesian_policy_contrast(
                fit_nc, df, policy; deltas = [-0.5, 0.5], n_draws = 200, rng = StableRNG(410))
            @test res_nc.metadata.intervals_available == false
            @test res_nc.metadata.uncertainty === :unavailable
            @test all(isnan, res_nc.lower)
            @test all(isnan, res_nc.upper)
            @test all(isnan, res_nc.covariance)
            @test all(isfinite, res_nc.mean)
        end

        @testset "Validation and rejection checks" begin
            prior = default_prior(3)
            df = simulate_cluster_data(StableRNG(411), [4, 5, 6, 4, 5]; σ_b = 0.6, σ_ε = 0.8)
            df.A = clamp.(df.A, -2.5, 2.5)
            fit = fit_bayesian_cluster_gcomp(df, :Y, :A, [:W]; cluster = :cluster, prior)
            policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)

            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df, ShiftPolicy("z", 0.01, 0.99); deltas = [0.5])
            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df, policy; deltas = [0.5], correction = :unknown)

            df_miss = copy(df)
            df_miss.Y = Vector{Union{Missing, Float64}}(df.Y)
            df_miss.Y[1] = missing
            @test_throws ArgumentError fit_bayesian_cluster_gcomp(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior)

            fit_mar = fit_bayesian_cluster_gcomp(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior, handle_missing = :mar)
            fit_complete = fit_bayesian_cluster_gcomp(
                df_miss[2:end, :], :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit_mar.converged
            @test fit_mar.metadata.missingness_status === :user_assumed_mar
            @test fit_mar.metadata.n_missing == 1
            @test fit_mar.metadata.n_observed == nrow(df_miss) - 1
            @test fit_mar.posterior_mean ≈ fit_complete.posterior_mean atol = 1e-12

            df_cmiss = copy(df)
            df_cmiss.cluster = Vector{Union{Missing, String}}(df.cluster)
            df_cmiss.cluster[1] = missing
            @test_throws ArgumentError fit_bayesian_cluster_gcomp(
                df_cmiss, :Y, :A, [:W]; cluster = :cluster, prior)
            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df_cmiss, policy; deltas = [0.5], weighting = :cluster)
        end
    end

    @testset "5. 100-cluster warmed runtime smoke" begin
        sizes_100 = vcat(repeat([5, 6, 7], 33), [6]) # 100 clusters, 600 rows
        df_100 = simulate_cluster_data(StableRNG(501), sizes_100; σ_b = 0.6, σ_ε = 0.8)
        df_100.A = clamp.(df_100.A, -3.5, 3.5)
        prior = default_prior(3)

        fit_100 = fit_bayesian_cluster_gcomp(df_100, :Y, :A, [:W]; cluster = :cluster, prior)
        @test fit_100.converged == true
        @test fit_100.status === :converged
        @test fit_100.metadata.n_clusters == 100

        policy = smooth_bounded_logit_shift_policy(-4.0, 4.0)
        res_100 = bayesian_policy_contrast(
            fit_100, df_100, policy;
            deltas = [-0.5, 0.5], n_draws = 2000, rng = StableRNG(502))
        @test all(isfinite, res_100.mean)
        @test all(isfinite, res_100.sd)
        @test all(>(0), res_100.sd)
        @test all(isfinite, res_100.covariance)
        @test all(isfinite, res_100.lower)
        @test all(isfinite, res_100.upper)
        @test all(res_100.lower .< res_100.mean .< res_100.upper)
        @test res_100.metadata.uncertainty === :structured_variational
    end
end
