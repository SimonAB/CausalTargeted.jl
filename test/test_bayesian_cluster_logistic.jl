using CausalTargeted
using DataFrames
using Distributions
using LinearAlgebra
using PSIS
using QuadGK
using StableRNGs
using Statistics
using Test

@testset "Hierarchical Bayesian logistic regression (Option A)" begin
    function simulate_cluster_logistic(
        rng,
        cluster_sizes::AbstractVector{Int};
        σ_b::Real = 0.6,
        β::AbstractVector{<:Real} = [-0.2, 0.9, -0.5],
        binary_treatment::Bool = false,
    )
        K = length(cluster_sizes)
        N = sum(cluster_sizes)
        cluster = Vector{String}(undef, N)
        W = Vector{Float64}(undef, N)
        A = Vector{Float64}(undef, N)
        Y = Vector{Int}(undef, N)
        idx = 0
        for k in 1:K
            b_k = Float64(σ_b) * randn(rng)
            cid = "c$k"
            for _ in 1:cluster_sizes[k]
                idx += 1
                w = randn(rng)
                a = binary_treatment ? Float64(rand(rng) < 1 / (1 + exp(-0.5 * w))) : (0.5 * w + 0.8 * randn(rng))
                η = β[1] + β[2] * a + β[3] * w + b_k
                y = Int(rand(rng) < CausalTargeted._cl_logistic(η))
                cluster[idx] = cid
                W[idx] = w
                A[idx] = a
                Y[idx] = y
            end
        end
        return DataFrame(; cluster, A, W, Y)
    end

    default_logistic_prior(p::Int = 3) = ClusterLogisticPrior(;
        mean = zeros(p),
        covariance = 4.0 .* Matrix{Float64}(I, p, p),
        cluster_shape = 2.5,
        cluster_scale = 0.5,
    )

    @testset "1. Weighted Schur-complement block vs dense (p + K) oracle" begin
        function dense_weighted_logistic_oracle(
            X::AbstractMatrix{<:Real},
            y::AbstractVector{<:Real},
            cluster_ids::AbstractVector,
            prior::Union{GaussianCoefficientPrior, ClusterLogisticPrior},
            W::AbstractVector{Float64};
            cluster_precision::Real,
        )
            cp = prior isa ClusterLogisticPrior ? prior.coefficient_prior : prior
            N, p = size(X)
            Xf = Matrix{Float64}(X)
            κ = Float64.(y) .- 0.5
            τb = Float64(cluster_precision)
            V0_inv = inv(Matrix{Float64}(cp.covariance))
            μ0 = Vector{Float64}(cp.mean)

            levels = unique(cluster_ids)
            K = length(levels)
            lvl_idx = Dict(lvl => k for (k, lvl) in enumerate(levels))
            Z = zeros(Float64, N, K)
            for i in 1:N
                Z[i, lvl_idx[cluster_ids[i]]] = 1.0
            end

            if !isfinite(τb)
                Q_fe = V0_inv + Xf' * Diagonal(W) * Xf
                h_fe = V0_inv * μ0 + Xf' * κ
                Σ_fe = 0.5 .* (inv(Q_fe) .+ inv(Q_fe)')
                m_fe = Q_fe \ h_fe
                return (;
                    levels,
                    beta_mean = m_fe,
                    beta_cov = Σ_fe,
                    cluster_mean = zeros(Float64, K),
                    cluster_var = zeros(Float64, K),
                    beta_cluster_cov = zeros(Float64, p, K),
                    expected_cluster_ss = 0.0,
                )
            end

            XZ = hcat(Xf, Z)
            Q = zeros(Float64, p + K, p + K)
            Q[1:p, 1:p] .= V0_inv
            for k in 1:K
                Q[p + k, p + k] = τb
            end
            Q .+= XZ' * Diagonal(W) * XZ

            h = zeros(Float64, p + K)
            h[1:p] .= V0_inv * μ0
            h .+= XZ' * κ

            Σ_joint = inv(Q)
            Σ_joint = 0.5 .* (Σ_joint .+ Σ_joint')
            m_joint = Q \ h

            beta_mean = m_joint[1:p]
            beta_cov = Σ_joint[1:p, 1:p]
            cluster_mean = m_joint[(p + 1):(p + K)]
            cluster_cov = Σ_joint[(p + 1):(p + K), (p + 1):(p + K)]
            cluster_var = diag(cluster_cov)
            beta_cluster_cov = Σ_joint[1:p, (p + 1):(p + K)]
            expected_cluster_ss = dot(cluster_mean, cluster_mean) + tr(cluster_cov)

            return (;
                levels,
                beta_mean,
                beta_cov,
                cluster_mean,
                cluster_var,
                beta_cluster_cov,
                expected_cluster_ss,
            )
        end

        rng = StableRNG(1001)
        cluster_ids = ["cB", "cA", "cB", "cS1", "cC", "cB", "cC", "cA", "cS2", "cB", "cC", "cC", "cB"]
        N, p = length(cluster_ids), 3
        X = hcat(ones(N), randn(rng, N, 2))
        y = Float64.(rand(rng, N) .< 0.45)
        W = 0.05 .+ 0.2 .* rand(rng, N)
        μ0 = [1.25, -0.75, 0.4]
        V0 = [1.2 0.35 -0.2; 0.35 0.9 0.25; -0.2 0.25 1.4]
        prior = ClusterLogisticPrior(;
            mean = μ0,
            covariance = V0,
            cluster_shape = 3.0,
            cluster_scale = 0.6,
        )
        τb = 1.7

        layout = CausalTargeted._logistic_cluster_layout(X, y, cluster_ids)
        state = CausalTargeted._solve_weighted_logistic_block(
            layout, prior, W; cluster_precision = τb)
        oracle = dense_weighted_logistic_oracle(
            X, y, cluster_ids, prior, W; cluster_precision = τb)

        @test layout.cluster_levels == oracle.levels
        @test layout.cluster_sizes == [5, 2, 1, 4, 1]
        @test state.beta_mean ≈ oracle.beta_mean atol = 1e-11
        @test state.beta_cov ≈ oracle.beta_cov atol = 1e-11
        @test state.cluster_mean ≈ oracle.cluster_mean atol = 1e-11
        @test state.cluster_var ≈ oracle.cluster_var atol = 1e-11
        @test state.beta_cluster_cov ≈ oracle.beta_cluster_cov atol = 1e-11
        @test state.expected_cluster_ss ≈ oracle.expected_cluster_ss atol = 1e-11

        # Infinite cluster precision (σ_b² = 0) zeroes out all cluster moments
        state_inf = CausalTargeted._solve_weighted_logistic_block(
            layout, prior, W; cluster_precision = Inf)
        oracle_inf = dense_weighted_logistic_oracle(
            X, y, cluster_ids, prior, W; cluster_precision = Inf)
        @test state_inf.beta_mean ≈ oracle_inf.beta_mean atol = 1e-11
        @test state_inf.beta_cov ≈ oracle_inf.beta_cov atol = 1e-11
        @test state_inf.cluster_mean == zeros(layout.K)
        @test state_inf.cluster_var == zeros(layout.K)
        @test state_inf.beta_cluster_cov == zeros(p, layout.K)
        @test state_inf.expected_cluster_ss == 0.0
    end

    @testset "2. 1D Gauss-Hermite vs independent QuadGK oracle" begin
        for η in (-3.0, -0.7, 0.0, 1.2, 3.0), σ_b in (0.0, 0.2, 0.5, 0.8, 1.0)
            gh_prob = CausalTargeted._marginal_logistic_prob(η, σ_b)
            gk_prob, _ = quadgk(
                u -> CausalTargeted._cl_logistic(η + σ_b * u) * pdf(Normal(), u),
                -Inf,
                Inf;
                rtol = 1e-12,
            )
            @test gh_prob ≈ gk_prob atol = 1e-10
        end

        rng = StableRNG(2026)
        for N_k in (1, 6, 20)
            η0 = randn(rng, N_k) .* 1.2 .+ 0.3
            y_k = Float64.(rand(rng, N_k) .< 0.55)
            rows = collect(1:N_k)
            for σ2_b in (0.1, 1.5)
                ll_gh = CausalTargeted._cluster_logistic_marginal_loglik(η0, y_k, rows, σ2_b)
                integral, _ = quadgk(
                    b -> exp(sum(i -> CausalTargeted._cl_log_logistic(
                        y_k[i] == 1 ? η0[i] + b : -(η0[i] + b)), rows)) *
                         pdf(Normal(0.0, sqrt(σ2_b)), b),
                    -Inf,
                    Inf;
                    rtol = 1e-12,
                )
                @test ll_gh ≈ log(integral) atol = 1e-9
            end
        end
    end

    @testset "3. ELBO monotonicity and convergence across cluster regimes" begin
        regimes = [
            (label = "balanced", sizes = fill(6, 10), σ_b = 0.6, seed = 101),
            (label = "unequal_with_singletons", sizes = [1, 1, 2, 3, 5, 8, 1, 4, 6, 3], σ_b = 0.7, seed = 102),
            (label = "weak_icc", sizes = [4, 5, 6, 3, 5, 4, 6, 5], σ_b = 0.05, seed = 103),
            (label = "strong_icc", sizes = [4, 5, 6, 3, 5, 4, 6, 5], σ_b = 2.0, seed = 104),
            (label = "small_K5", sizes = [3, 5, 4, 2, 6], σ_b = 0.8, seed = 105),
        ]
        prior = default_logistic_prior(3)
        for r in regimes
            df = simulate_cluster_logistic(StableRNG(r.seed), r.sizes; σ_b = r.σ_b)
            fit = fit_bayesian_cluster_logistic(df, :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit.status === :converged
            @test fit.converged == true
            @test length(fit.elbo_trace) >= 2
            @test isfinite(fit.elbo)
            @test all(diff(fit.elbo_trace) .>= -1e-9)
        end
    end

    @testset "4. Standardisation invariance" begin
        rng = StableRNG(303)
        sizes = [3, 6, 4, 1, 5, 7, 2, 5]
        df = simulate_cluster_logistic(rng, sizes; σ_b = 0.7)
        df.A = -5.0 .+ 3.0 .* df.A
        df.W = 100.0 .+ 25.0 .* df.W
        prior = ClusterLogisticPrior(;
            mean = [0.2, -0.15, 0.05],
            covariance = [1.5 0.2 -0.1; 0.2 1.2 0.15; -0.1 0.15 1.0],
            cluster_shape = 2.5,
            cluster_scale = 0.6,
        )

        fit_std = fit_bayesian_cluster_logistic(
            df, :Y, :A, [:W]; cluster = :cluster, prior, standardize = true, tol = 1e-11)
        fit_raw = fit_bayesian_cluster_logistic(
            df, :Y, :A, [:W]; cluster = :cluster, prior, standardize = false, tol = 1e-11)

        @test fit_std.converged && fit_raw.converged
        @test fit_std.design_metadata.standardized == true
        @test fit_raw.design_metadata.standardized == false
        @test fit_std.posterior_mean ≈ fit_raw.posterior_mean atol = 1e-6
        @test fit_std.posterior_covariance ≈ fit_raw.posterior_covariance atol = 1e-6
        @test fit_std.cluster_mean ≈ fit_raw.cluster_mean atol = 1e-6
        @test fit_std.cluster_var ≈ fit_raw.cluster_var atol = 1e-6
        @test fit_std.cluster_scale ≈ fit_raw.cluster_scale atol = 1e-6
    end

    @testset "5. Marginal vs conditional policy contrasts and weighting" begin
        @testset "Jensen attenuation: marginalize_random_effects = true vs false" begin
            df_j = simulate_cluster_logistic(
                StableRNG(501), fill(8, 20); σ_b = 1.6, β = [-0.2, 1.3, -0.4])
            df_j.A = clamp.(df_j.A, -2.5, 2.5)
            prior = ClusterLogisticPrior(;
                mean = zeros(3),
                covariance = 4.0 .* Matrix{Float64}(I, 3, 3),
                cluster_shape = 2.5,
                cluster_scale = 1.5,
            )
            fit_j = fit_bayesian_cluster_logistic(df_j, :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit_j.converged

            policy = smooth_bounded_logit_shift_policy(-3.5, 3.5)
            res_marg = bayesian_policy_contrast(
                fit_j, df_j, policy;
                deltas = [0.8], n_draws = 1500, rng = StableRNG(502),
                marginalize_random_effects = true)
            res_cond = bayesian_policy_contrast(
                fit_j, df_j, policy;
                deltas = [0.8], n_draws = 1500, rng = StableRNG(502),
                marginalize_random_effects = false)

            @test res_marg.metadata.target === :fixed_cohort_marginal
            @test res_cond.metadata.target === :fixed_cohort_conditional_b0
            @test !(res_marg.mean ≈ res_cond.mean)
            @test abs(res_marg.mean[1]) < abs(res_cond.mean[1])
        end

        @testset "Individual vs cluster weighting and multi-delta grid" begin
            prior3 = default_logistic_prior(3)
            df_eq = simulate_cluster_logistic(StableRNG(503), fill(5, 8); σ_b = 0.6)
            df_eq.A = clamp.(df_eq.A, -2.5, 2.5)
            fit_eq = fit_bayesian_cluster_logistic(df_eq, :Y, :A, [:W]; cluster = :cluster, prior = prior3)
            @test fit_eq.converged

            policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)
            res_eq_ind = bayesian_policy_contrast(
                fit_eq, df_eq, policy;
                deltas = [-0.6, 0.3, 0.8], n_draws = 800, rng = StableRNG(504), weighting = :individual)
            res_eq_clu = bayesian_policy_contrast(
                fit_eq, df_eq, policy;
                deltas = [-0.6, 0.3, 0.8], n_draws = 800, rng = StableRNG(504), weighting = :cluster)

            @test res_eq_ind.mean ≈ res_eq_clu.mean atol = 1e-12
            @test res_eq_ind.sd ≈ res_eq_clu.sd atol = 1e-12
            @test res_eq_ind.covariance ≈ res_eq_clu.covariance atol = 1e-12
            @test res_eq_ind.lower ≈ res_eq_clu.lower atol = 1e-12
            @test res_eq_ind.upper ≈ res_eq_clu.upper atol = 1e-12

            # Multi-delta grid covariance and interval properties
            @test length(res_eq_ind.mean) == 3
            @test all(isfinite, res_eq_ind.mean)
            @test all(>(0), res_eq_ind.sd)
            @test size(res_eq_ind.covariance) == (3, 3)
            @test issymmetric(res_eq_ind.covariance)
            @test all(>=(-1e-12), eigvals(Symmetric(res_eq_ind.covariance)))
            @test all(isfinite, res_eq_ind.lower)
            @test all(isfinite, res_eq_ind.upper)
            @test all(res_eq_ind.lower .< res_eq_ind.mean .< res_eq_ind.upper)
            @test res_eq_ind.metadata.uncertainty === :jaakkola_jordan_structured_vi

            # Unequal cluster sizes where covariate W and treatment effect vary with cluster size
            rng_u = StableRNG(505)
            sizes_u = [2, 2, 2, 10, 10, 10]
            cluster_u = String[]
            W_u = Float64[]
            A_u = Float64[]
            Y_u = Int[]
            for (k, nk) in enumerate(sizes_u)
                b_k = 0.4 * randn(rng_u)
                w_base = nk <= 2 ? -0.8 : 1.0
                for _ in 1:nk
                    w = w_base + 0.15 * randn(rng_u)
                    a = clamp(0.4 * w + 0.6 * randn(rng_u), -2.5, 2.5)
                    η = -0.2 + 0.8 * a - 0.3 * w + 1.1 * a * w + b_k
                    y = Int(rand(rng_u) < CausalTargeted._cl_logistic(η))
                    push!(cluster_u, "u$k")
                    push!(W_u, w)
                    push!(A_u, a)
                    push!(Y_u, y)
                end
            end
            df_un = DataFrame(cluster = cluster_u, A = A_u, W = W_u, Y = Y_u)
            basis_int = (df, a) -> hcat(ones(nrow(df)), a, df.W, a .* df.W)
            prior4 = default_logistic_prior(4)
            fit_un = fit_bayesian_cluster_logistic(
                df_un, :Y, :A, [:W]; cluster = :cluster, prior = prior4, basis = basis_int)
            @test fit_un.converged

            res_un_ind = bayesian_policy_contrast(
                fit_un, df_un, policy;
                deltas = [-0.5, 0.5], n_draws = 800, rng = StableRNG(506), weighting = :individual)
            res_un_clu = bayesian_policy_contrast(
                fit_un, df_un, policy;
                deltas = [-0.5, 0.5], n_draws = 800, rng = StableRNG(506), weighting = :cluster)
            @test !(res_un_ind.mean ≈ res_un_clu.mean)
        end
    end

    @testset "6. Failure, PSIS, and validation contracts" begin
        prior = default_logistic_prior(3)
        df = simulate_cluster_logistic(StableRNG(601), fill(6, 10); σ_b = 0.5)
        df.A = clamp.(df.A, -2.5, 2.5)
        policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)

        @testset "Non-convergence withholds intervals and covariance" begin
            fit_nc = fit_bayesian_cluster_logistic(
                df, :Y, :A, [:W]; cluster = :cluster, prior, maxiter = 1)
            @test fit_nc.status === :maxiter_reached
            @test fit_nc.converged == false

            res_nc = bayesian_policy_contrast(
                fit_nc, df, policy; deltas = [-0.5, 0.5], n_draws = 200, rng = StableRNG(602))
            @test res_nc.metadata.intervals_available == false
            @test res_nc.metadata.uncertainty === :unavailable
            @test all(isnan, res_nc.lower)
            @test all(isnan, res_nc.upper)
            @test all(isnan, res_nc.covariance)
            @test all(isfinite, res_nc.mean)
        end

        @testset "PSIS global importance refinement gates" begin
            fit = fit_bayesian_cluster_logistic(df, :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit.converged

            res_ok = bayesian_policy_contrast(
                fit, df, policy;
                deltas = [-0.5, 0.5], n_draws = 2000, rng = StableRNG(603), correction = :psis)
            @test res_ok.metadata.intervals_available == true
            @test res_ok.metadata.uncertainty === :psis_refined_moments
            @test res_ok.metadata.psis_diagnostics.passed == true
            @test res_ok.metadata.psis_diagnostics.khat < 0.7
            @test res_ok.metadata.psis_diagnostics.ess >= 400
            @test all(isfinite, res_ok.lower)
            @test all(isfinite, res_ok.upper)
            @test all(isfinite, res_ok.covariance)
            @test issymmetric(res_ok.covariance)
            @test all(res_ok.lower .< res_ok.mean .< res_ok.upper)

            res_low = bayesian_policy_contrast(
                fit, df, policy;
                deltas = [-0.5, 0.5], n_draws = 80, rng = StableRNG(609), correction = :psis)
            @test res_low.metadata.intervals_available == false
            @test res_low.metadata.uncertainty === :unavailable
            @test :ess_below_minimum in res_low.metadata.psis_diagnostics.failures
            @test all(isnan, res_low.lower)
            @test all(isnan, res_low.upper)
            @test all(isnan, res_low.covariance)
        end

        @testset "Input and prior validation" begin
            fit = fit_bayesian_cluster_logistic(df, :Y, :A, [:W]; cluster = :cluster, prior)

            # Non-binary outcomes
            df_nonbin = copy(df)
            df_nonbin.Y = Float64.(df.Y)
            df_nonbin.Y[1] = 2.0
            @test_throws ArgumentError fit_bayesian_cluster_logistic(
                df_nonbin, :Y, :A, [:W]; cluster = :cluster, prior)
            df_nonbin.Y[1] = 0.5
            @test_throws ArgumentError fit_bayesian_cluster_logistic(
                df_nonbin, :Y, :A, [:W]; cluster = :cluster, prior)

            # Missing outcomes rejected without handle_missing = :mar, accepted with :mar
            df_miss = copy(df)
            df_miss.Y = Vector{Union{Missing, Int}}(df.Y)
            df_miss.Y[1] = missing
            @test_throws ArgumentError fit_bayesian_cluster_logistic(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior)

            fit_mar = fit_bayesian_cluster_logistic(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior, handle_missing = :mar)
            fit_complete = fit_bayesian_cluster_logistic(
                df_miss[2:end, :], :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit_mar.converged
            @test fit_mar.metadata.missingness_status === :user_assumed_mar
            @test fit_mar.metadata.n_missing == 1
            @test fit_mar.metadata.n_observed == nrow(df_miss) - 1
            @test fit_mar.posterior_mean ≈ fit_complete.posterior_mean atol = 1e-12

            # Missing cluster IDs
            df_cmiss = copy(df)
            df_cmiss.cluster = Vector{Union{Missing, String}}(df.cluster)
            df_cmiss.cluster[1] = missing
            @test_throws ArgumentError fit_bayesian_cluster_logistic(
                df_cmiss, :Y, :A, [:W]; cluster = :cluster, prior)
            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df_cmiss, policy; deltas = [0.5], weighting = :cluster)

            # Clamped ShiftPolicy rejected
            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df, ShiftPolicy("z", 0.01, 0.99); deltas = [0.5])

            # Invalid ClusterLogisticPrior parameters
            μ = zeros(3)
            V = Matrix{Float64}(I, 3, 3)
            for bad_shape in (1.0, 0.5, -1.0, Inf, NaN)
                @test_throws ArgumentError ClusterLogisticPrior(;
                    mean = μ, covariance = V, cluster_shape = bad_shape, cluster_scale = 1.0)
            end
            for bad_scale in (0.0, -0.5, Inf, NaN)
                @test_throws ArgumentError ClusterLogisticPrior(;
                    mean = μ, covariance = V, cluster_shape = 2.0, cluster_scale = bad_scale)
            end
        end
    end
end
