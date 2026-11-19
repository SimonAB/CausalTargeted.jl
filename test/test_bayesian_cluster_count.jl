using CausalTargeted
using DataFrames
using Distributions
using LinearAlgebra
using PSIS
using QuadGK
using StableRNGs
using Statistics
using Test

@testset "Hierarchical Bayesian count and hurdle models" begin
    function simulate_cluster_count(
        rng,
        cluster_sizes::AbstractVector{Int};
        family::Symbol = :poisson,
        θ::Real = 2.5,
        σ_b::Real = 0.6,
        β::AbstractVector{<:Real} = [0.2, 0.55, -0.35],
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
                a = clamp(0.4 * w + 0.7 * randn(rng), -2.5, 2.5)
                μ = exp(clamp(β[1] + β[2] * a + β[3] * w + b_k, -6.0, 5.0))
                y = if family === :poisson
                    rand(rng, Poisson(μ))
                else
                    p_nb = Float64(θ) / (Float64(θ) + μ)
                    rand(rng, NegativeBinomial(Float64(θ), p_nb))
                end
                cluster[idx] = cid
                W[idx] = w
                A[idx] = a
                Y[idx] = y
            end
        end
        return DataFrame(; cluster, A, W, Y)
    end

    function simulate_cluster_hurdle(
        rng,
        cluster_sizes::AbstractVector{Int};
        family::Symbol = :nb2,
        θ::Real = 2.0,
        σ_b1::Real = 0.5,
        σ_b2::Real = 0.45,
        β1::AbstractVector{<:Real} = [0.45, 0.7, -0.4],
        β2::AbstractVector{<:Real} = [0.5, 0.55, -0.25],
    )
        K = length(cluster_sizes)
        N = sum(cluster_sizes)
        cluster = Vector{String}(undef, N)
        W = Vector{Float64}(undef, N)
        A = Vector{Float64}(undef, N)
        Y = Vector{Int}(undef, N)
        idx = 0
        for k in 1:K
            b1 = Float64(σ_b1) * randn(rng)
            b2 = Float64(σ_b2) * randn(rng)
            cid = "h$k"
            for _ in 1:cluster_sizes[k]
                idx += 1
                w = randn(rng)
                a = clamp(0.35 * w + 0.7 * randn(rng), -2.5, 2.5)
                η1 = β1[1] + β1[2] * a + β1[3] * w + b1
                present = rand(rng) < (1.0 / (1.0 + exp(-η1)))
                y = 0
                if present
                    μ = exp(clamp(β2[1] + β2[2] * a + β2[3] * w + b2, -4.0, 4.0))
                    # Rejection sample zero-truncated count
                    while y == 0
                        y = if family === :poisson
                            rand(rng, Poisson(μ))
                        else
                            p_nb = Float64(θ) / (Float64(θ) + μ)
                            rand(rng, NegativeBinomial(Float64(θ), p_nb))
                        end
                    end
                end
                cluster[idx] = cid
                W[idx] = w
                A[idx] = a
                Y[idx] = y
            end
        end
        return DataFrame(; cluster, A, W, Y)
    end

    default_count_prior(p::Int = 3) = ClusterCountPrior(;
        mean = zeros(p),
        covariance = 4.0 .* Matrix{Float64}(I, p, p),
        cluster_shape = 2.5,
        cluster_scale = 0.5,
        log_dispersion_mean = 0.0,
        log_dispersion_sd = 1.5,
    )

    default_hurdle_prior(p::Int = 3) = ClusterHurdlePrior(;
        presence = ClusterLogisticPrior(;
            mean = zeros(p),
            covariance = 4.0 .* Matrix{Float64}(I, p, p),
            cluster_shape = 2.5,
            cluster_scale = 0.5,
        ),
        intensity = default_count_prior(p),
    )

    @testset "1. Weighted GLM Schur-complement block vs dense (p + K) oracle" begin
        function dense_weighted_glm_oracle(
            X::AbstractMatrix{<:Real},
            cluster_ids::AbstractVector,
            cp::GaussianCoefficientPrior,
            W::AbstractVector{Float64},
            z_rhs::AbstractVector{Float64};
            cluster_precision::Real,
        )
            N, p = size(X)
            Xf = Matrix{Float64}(X)
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
                h_fe = V0_inv * μ0 + Xf' * z_rhs
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
            h .+= XZ' * z_rhs

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

        rng = StableRNG(1101)
        cluster_ids = ["cB", "cA", "cB", "cS1", "cC", "cB", "cC", "cA", "cS2", "cB", "cC", "cC", "cB"]
        N, p = length(cluster_ids), 3
        X = hcat(ones(N), randn(rng, N, 2))
        y = Float64.(rand(rng, 0:8, N))
        W = 0.2 .+ 3.5 .* rand(rng, N)
        z_rhs = randn(rng, N) .* 2.0 .+ 0.5
        μ0 = [0.8, -0.45, 0.3]
        V0 = [1.4 0.3 -0.15; 0.3 0.85 0.2; -0.15 0.2 1.1]
        cp = GaussianCoefficientPrior(; mean = μ0, covariance = V0)
        τb = 1.9

        layout = CausalTargeted._count_cluster_layout(X, y, cluster_ids)
        state = CausalTargeted._solve_weighted_glm_block(
            layout, cp, W, z_rhs; cluster_precision = τb)
        oracle = dense_weighted_glm_oracle(
            X, cluster_ids, cp, W, z_rhs; cluster_precision = τb)

        @test layout.cluster_levels == oracle.levels
        @test layout.cluster_sizes == [5, 2, 1, 4, 1]
        @test state.beta_mean ≈ oracle.beta_mean atol = 1e-11
        @test state.beta_cov ≈ oracle.beta_cov atol = 1e-11
        @test state.cluster_mean ≈ oracle.cluster_mean atol = 1e-11
        @test state.cluster_var ≈ oracle.cluster_var atol = 1e-11
        @test state.beta_cluster_cov ≈ oracle.beta_cluster_cov atol = 1e-11
        @test state.expected_cluster_ss ≈ oracle.expected_cluster_ss atol = 1e-11

        state_inf = CausalTargeted._solve_weighted_glm_block(
            layout, cp, W, z_rhs; cluster_precision = Inf)
        oracle_inf = dense_weighted_glm_oracle(
            X, cluster_ids, cp, W, z_rhs; cluster_precision = Inf)
        @test state_inf.beta_mean ≈ oracle_inf.beta_mean atol = 1e-11
        @test state_inf.beta_cov ≈ oracle_inf.beta_cov atol = 1e-11
        @test state_inf.cluster_mean == zeros(layout.K)
        @test state_inf.cluster_var == zeros(layout.K)
        @test state_inf.beta_cluster_cov == zeros(p, layout.K)
        @test state_inf.expected_cluster_ss == 0.0
    end

    @testset "2. Observation derivatives & 1D Gauss-Hermite vs finite-difference / QuadGK oracles" begin
        @testset "Finite-difference check of _count_obs_loglik_grad_weight" begin
            h = 1e-5
            h2 = 1e-4
            modes = (
                (:poisson, false, [0.0, 1.0, 4.0]),
                (:poisson, true, [1.0, 2.0, 6.0]),
                (:nb2, false, [0.0, 2.0, 7.0]),
                (:nb2, true, [1.0, 3.0, 8.0]),
            )
            for (family, zt, ys) in modes, y in ys, η in (-1.2, 0.1, 1.4), θ in (1.2, 3.5)
                _, g, w = CausalTargeted._count_obs_loglik_grad_weight(y, η, family, θ, zt)
                ll_p, g_p, _ = CausalTargeted._count_obs_loglik_grad_weight(y, η + h, family, θ, zt)
                ll_m, g_m, _ = CausalTargeted._count_obs_loglik_grad_weight(y, η - h, family, θ, zt)
                ll_p2, _, _ = CausalTargeted._count_obs_loglik_grad_weight(y, η + h2, family, θ, zt)
                ll_0, _, _ = CausalTargeted._count_obs_loglik_grad_weight(y, η, family, θ, zt)
                ll_m2, _, _ = CausalTargeted._count_obs_loglik_grad_weight(y, η - h2, family, θ, zt)

                fd_g = (ll_p - ll_m) / (2 * h)
                fd_w_from_g = -(g_p - g_m) / (2 * h)
                fd_w_from_ll = -(ll_p2 - 2 * ll_0 + ll_m2) / (h2 * h2)

                @test g ≈ fd_g atol = 1e-5
                @test w ≈ fd_w_from_g atol = 1e-5
                @test w ≈ fd_w_from_ll atol = 1e-5
            end
        end

        @testset "QuadGK oracle for _cluster_count_marginal_loglik" begin
            rng = StableRNG(2201)
            for (family, zt) in ((:poisson, false), (:poisson, true), (:nb2, false), (:nb2, true))
                θ = family === :nb2 ? 2.2 : 1.0
                for N_k in (1, 5, 12)
                    η0 = randn(rng, N_k) .* 0.6 .+ 0.2
                    y_k = Float64.(rand(rng, (zt ? 1 : 0):6, N_k))
                    rows = collect(1:N_k)
                    for σ2_b in (0.15, 0.85)
                        ll_gh = CausalTargeted._cluster_count_marginal_loglik(
                            η0, y_k, rows, σ2_b, family, θ, zt)
                        integral, _ = quadgk(
                            b -> exp(sum(
                                i -> CausalTargeted._count_obs_loglik_grad_weight(
                                    y_k[i], η0[i] + b, family, θ, zt)[1],
                                rows,
                            )) * pdf(Normal(0.0, sqrt(σ2_b)), b),
                            -Inf,
                            Inf;
                            rtol = 1e-11,
                        )
                        @test ll_gh ≈ log(integral) atol = 1e-8
                    end
                end
            end
        end

        @testset "QuadGK oracle for zero-truncated _marginal_count_mean & exact log-normal check" begin
            for η in (-1.0, 0.0, 0.8), σ_b in (0.0, 0.25, 0.5)
                # Exact closed-form log-normal expectation for standard counts
                @test CausalTargeted._marginal_count_mean(η, σ_b, :poisson, 1.0, false) ≈
                    exp(η + 0.5 * σ_b^2) atol = 1e-13
                @test CausalTargeted._marginal_count_mean(η, σ_b, :nb2, 2.0, false) ≈
                    exp(η + 0.5 * σ_b^2) atol = 1e-13

                # Zero-truncated Poisson & NB2 against QuadGK
                for (family, θ) in ((:poisson, 1.0), (:nb2, 2.3))
                    gh_mean = CausalTargeted._marginal_count_mean(η, σ_b, family, θ, true)
                    if σ_b == 0.0
                        μ = exp(η)
                        p0 = family === :poisson ? exp(-μ) : exp(θ * (log(θ) - log(θ + μ)))
                        @test gh_mean ≈ μ / (1.0 - p0) atol = 1e-13
                    else
                        gk_mean, _ = quadgk(
                            u -> begin
                                ηu = η + σ_b * u
                                if abs(u) > 25.0
                                    return 0.0
                                end
                                μ = exp(ηu)
                                one_minus_p0 = family === :poisson ? -expm1(-μ) : -expm1(θ * (log(θ) - log(θ + μ)))
                                (one_minus_p0 <= 0.0 ? 1.0 : μ / one_minus_p0) * pdf(Normal(), u)
                            end,
                            -Inf,
                            Inf;
                            rtol = 1e-11,
                        )
                        @test gh_mean ≈ gk_mean atol = 1e-9
                    end
                end
            end
        end
    end

    @testset "3. Hierarchical Poisson & NB2 fits & marginal contrasts" begin
        @testset "Convergence across balanced & unbalanced (with singletons) regimes" begin
            prior = default_count_prior(3)
            regimes = [
                (label = "balanced", sizes = fill(6, 10), σ_b = 0.55, seed = 3101),
                (label = "unbalanced_singletons", sizes = [1, 1, 2, 4, 6, 8, 1, 3, 5, 7], σ_b = 0.65, seed = 3102),
            ]
            for family in (:poisson, :nb2), r in regimes
                df = simulate_cluster_count(StableRNG(r.seed), r.sizes; family, σ_b = r.σ_b)
                fit = fit_bayesian_cluster_count(df, :Y, :A, [:W]; cluster = :cluster, prior, family)
                @test fit.status === :converged
                @test fit.converged == true
                @test fit.family === family
                @test all(isfinite, fit.posterior_mean)
                @test all(isfinite, fit.posterior_covariance)
                @test all(isfinite, fit.cluster_mean)
                @test all(isfinite, fit.cluster_var)
                @test isfinite(fit.cluster_scale) && fit.cluster_scale > 0
                @test isfinite(fit.metadata.cluster_variance_mean) && fit.metadata.cluster_variance_mean > 0
                if family === :nb2
                    @test isfinite(fit.log_dispersion_mean)
                    @test isfinite(fit.log_dispersion_var) && fit.log_dispersion_var > 0
                    @test isfinite(fit.metadata.dispersion_mode) && fit.metadata.dispersion_mode > 0
                    @test fit.metadata.dispersion_mode ≈ exp(fit.log_dispersion_mean)
                else
                    @test isnan(fit.log_dispersion_mean)
                    @test isnan(fit.metadata.dispersion_mode)
                end
            end
        end

        @testset "Standardisation invariance" begin
            prior = ClusterCountPrior(;
                mean = [0.15, -0.1, 0.05],
                covariance = [1.5 0.2 -0.1; 0.2 1.2 0.15; -0.1 0.15 1.0],
                cluster_shape = 2.5,
                cluster_scale = 0.5,
                log_dispersion_mean = 0.2,
                log_dispersion_sd = 1.2,
            )
            sizes = [3, 6, 4, 1, 5, 7, 2, 5]
            for family in (:poisson, :nb2)
                df = simulate_cluster_count(StableRNG(3201), sizes; family, σ_b = 0.5)
                df.A = -3.0 .+ 2.5 .* df.A
                df.W = 50.0 .+ 15.0 .* df.W

                fit_std = fit_bayesian_cluster_count(
                    df, :Y, :A, [:W]; cluster = :cluster, prior, family, standardize = true, tol = 1e-11)
                fit_raw = fit_bayesian_cluster_count(
                    df, :Y, :A, [:W]; cluster = :cluster, prior, family, standardize = false, tol = 1e-11)

                @test fit_std.converged && fit_raw.converged
                @test fit_std.design_metadata.standardized == true
                @test fit_raw.design_metadata.standardized == false
                @test fit_std.posterior_mean ≈ fit_raw.posterior_mean atol = 1e-5
                @test fit_std.posterior_covariance ≈ fit_raw.posterior_covariance atol = 1e-5
                @test fit_std.cluster_mean ≈ fit_raw.cluster_mean atol = 1e-5
                @test fit_std.cluster_var ≈ fit_raw.cluster_var atol = 1e-5
                @test fit_std.cluster_scale ≈ fit_raw.cluster_scale atol = 1e-5
                if family === :nb2
                    @test fit_std.log_dispersion_mean ≈ fit_raw.log_dispersion_mean atol = 1e-5
                end
            end
        end

        @testset "Marginal vs conditional expected count contrast (exp(σ_b² / 2) factor)" begin
            prior = ClusterCountPrior(;
                mean = zeros(3),
                covariance = 4.0 .* Matrix{Float64}(I, 3, 3),
                cluster_shape = 2.5,
                cluster_scale = 1.2,
            )
            policy = smooth_bounded_logit_shift_policy(-3.5, 3.5)
            for (idx, family) in enumerate((:poisson, :nb2))
                df = simulate_cluster_count(
                    StableRNG(3300 + idx), fill(8, 16);
                    family, θ = 3.0, σ_b = 0.85, β = [0.2, 0.7, -0.3])
                fit = fit_bayesian_cluster_count(df, :Y, :A, [:W]; cluster = :cluster, prior, family)
                @test fit.converged

                res_marg = bayesian_policy_contrast(
                    fit, df, policy;
                    deltas = [0.6], n_draws = 1200, rng = StableRNG(3310 + idx),
                    marginalize_random_effects = true)
                res_cond = bayesian_policy_contrast(
                    fit, df, policy;
                    deltas = [0.6], n_draws = 1200, rng = StableRNG(3310 + idx),
                    marginalize_random_effects = false)

                @test res_marg.metadata.target === :fixed_cohort_marginal
                @test res_cond.metadata.target === :fixed_cohort_conditional_b0
                @test res_marg.mean[1] > res_cond.mean[1] > 0.0
            end
        end

        @testset "Individual vs cluster weighting" begin
            prior3 = default_count_prior(3)
            df_eq = simulate_cluster_count(StableRNG(3401), fill(5, 8); family = :poisson, σ_b = 0.5)
            fit_eq = fit_bayesian_cluster_count(df_eq, :Y, :A, [:W]; cluster = :cluster, prior = prior3)
            @test fit_eq.converged

            policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)
            res_eq_ind = bayesian_policy_contrast(
                fit_eq, df_eq, policy;
                deltas = [-0.5, 0.4], n_draws = 600, rng = StableRNG(3402), weighting = :individual)
            res_eq_clu = bayesian_policy_contrast(
                fit_eq, df_eq, policy;
                deltas = [-0.5, 0.4], n_draws = 600, rng = StableRNG(3402), weighting = :cluster)

            @test res_eq_ind.mean ≈ res_eq_clu.mean atol = 1e-12
            @test res_eq_ind.sd ≈ res_eq_clu.sd atol = 1e-12
            @test res_eq_ind.covariance ≈ res_eq_clu.covariance atol = 1e-12
            @test res_eq_ind.lower ≈ res_eq_clu.lower atol = 1e-12
            @test res_eq_ind.upper ≈ res_eq_clu.upper atol = 1e-12

            # Unequal cluster sizes with size-dependent treatment effect
            rng_u = StableRNG(3403)
            sizes_u = [2, 2, 2, 10, 10, 10]
            cluster_u = String[]
            W_u = Float64[]
            A_u = Float64[]
            Y_u = Int[]
            for (k, nk) in enumerate(sizes_u)
                b_k = 0.35 * randn(rng_u)
                w_base = nk <= 2 ? -0.9 : 0.9
                for _ in 1:nk
                    w = w_base + 0.15 * randn(rng_u)
                    a = clamp(0.35 * w + 0.6 * randn(rng_u), -2.5, 2.5)
                    μ = exp(clamp(0.1 + 0.5 * a - 0.2 * w + 0.7 * a * w + b_k, -5.0, 4.0))
                    y = rand(rng_u, Poisson(μ))
                    push!(cluster_u, "u$k")
                    push!(W_u, w)
                    push!(A_u, a)
                    push!(Y_u, y)
                end
            end
            df_un = DataFrame(cluster = cluster_u, A = A_u, W = W_u, Y = Y_u)
            basis_int = (df, a) -> hcat(ones(nrow(df)), a, df.W, a .* df.W)
            fit_un = fit_bayesian_cluster_count(
                df_un, :Y, :A, [:W]; cluster = :cluster, prior = default_count_prior(4), basis = basis_int)
            @test fit_un.converged

            res_un_ind = bayesian_policy_contrast(
                fit_un, df_un, policy;
                deltas = [-0.5, 0.5], n_draws = 600, rng = StableRNG(3404), weighting = :individual)
            res_un_clu = bayesian_policy_contrast(
                fit_un, df_un, policy;
                deltas = [-0.5, 0.5], n_draws = 600, rng = StableRNG(3404), weighting = :cluster)
            @test !(res_un_ind.mean ≈ res_un_clu.mean)
        end

        @testset "NB2 dispersion recovery & response to overdispersion" begin
            prior = default_count_prior(3)
            df_over = simulate_cluster_count(
                StableRNG(3501), fill(15, 16);
                family = :nb2, θ = 0.8, σ_b = 0.35, β = [1.0, 0.5, -0.3])
            df_mild = simulate_cluster_count(
                StableRNG(3502), fill(15, 16);
                family = :nb2, θ = 25.0, σ_b = 0.35, β = [1.0, 0.5, -0.3])

            fit_over = fit_bayesian_cluster_count(
                df_over, :Y, :A, [:W]; cluster = :cluster, prior, family = :nb2)
            fit_mild = fit_bayesian_cluster_count(
                df_mild, :Y, :A, [:W]; cluster = :cluster, prior, family = :nb2)

            @test fit_over.converged && fit_mild.converged
            @test isfinite(fit_over.metadata.dispersion_mode) && fit_over.metadata.dispersion_mode > 0
            @test isfinite(fit_mild.metadata.dispersion_mode) && fit_mild.metadata.dispersion_mode > 0
            @test fit_over.metadata.dispersion_mode ≈ exp(fit_over.log_dispersion_mean)
            @test fit_over.metadata.dispersion_mode < fit_mild.metadata.dispersion_mode
            @test 0.3 < fit_over.metadata.dispersion_mode < 2.5
        end
    end

    @testset "4. Two-part hierarchical hurdle (fit_bayesian_cluster_hurdle)" begin
        prior_h = default_hurdle_prior(3)
        policy = smooth_bounded_logit_shift_policy(-3.5, 3.5)

        for (idx, family) in enumerate((:poisson, :nb2))
            df_h = simulate_cluster_hurdle(
                StableRNG(4100 + idx), fill(10, 12);
                family, θ = 1.8, β1 = [0.3, 0.8, -0.4], β2 = [0.6, 0.5, -0.3])
            zf = mean(df_h.Y .== 0)
            @test 0.15 < zf < 0.7

            fit_h = fit_bayesian_cluster_hurdle(
                df_h, :Y, :A, [:W]; cluster = :cluster, prior = prior_h, family)
            @test fit_h.converged == true
            @test fit_h.status === :converged
            @test fit_h.presence_fit.converged
            @test fit_h.intensity_fit.converged
            @test fit_h.intensity_fit.zero_truncated == true
            @test fit_h.metadata.zero_fraction ≈ zf

            res_h = bayesian_policy_contrast(
                fit_h, df_h, policy;
                deltas = [-0.5, 0.6], n_draws = 1000, rng = StableRNG(4200 + idx))
            @test res_h.metadata.intervals_available == true
            @test res_h.metadata.uncertainty === :structured_laplace_vi
            @test res_h.mean ≈ res_h.overall.mean

            for part in (res_h.overall, res_h.presence, res_h.intensity)
                @test all(isfinite, part.mean)
                @test all(>(0), part.sd)
                @test all(isfinite, part.lower)
                @test all(isfinite, part.upper)
                @test all(part.lower .< part.mean .< part.upper)
                @test size(part.covariance) == (2, 2)
                @test issymmetric(part.covariance)
                @test all(>=(-1e-12), eigvals(Symmetric(part.covariance)))
            end
        end

        @testset "Non-converged hurdle fit withholds intervals and covariance" begin
            df_h = simulate_cluster_hurdle(StableRNG(4301), fill(8, 10); family = :nb2)
            fit_nc = fit_bayesian_cluster_hurdle(
                df_h, :Y, :A, [:W]; cluster = :cluster, prior = prior_h, family = :nb2, maxiter = 1)
            @test fit_nc.converged == false
            @test fit_nc.status === :maxiter_reached

            res_nc = bayesian_policy_contrast(
                fit_nc, df_h, policy; deltas = [-0.4, 0.4], n_draws = 200, rng = StableRNG(4302))
            @test res_nc.metadata.intervals_available == false
            @test res_nc.metadata.uncertainty === :unavailable
            for part in (res_nc.overall, res_nc.presence, res_nc.intensity)
                @test all(isnan, part.lower)
                @test all(isnan, part.upper)
                @test all(isnan, part.sd)
                @test all(isnan, part.covariance)
            end
        end
    end

    @testset "5. PSIS refinement & input validation" begin
        prior = default_count_prior(3)
        prior_h = default_hurdle_prior(3)
        df = simulate_cluster_count(StableRNG(5101), fill(6, 10); family = :poisson, σ_b = 0.5)
        policy = smooth_bounded_logit_shift_policy(-3.0, 3.0)

        @testset "PSIS global importance refinement gates" begin
            fit = fit_bayesian_cluster_count(
                df, :Y, :A, [:W]; cluster = :cluster, prior, family = :poisson)
            @test fit.converged

            res_ok = bayesian_policy_contrast(
                fit, df, policy;
                deltas = [-0.5, 0.5], n_draws = 2000, rng = StableRNG(5102), correction = :psis)
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
                deltas = [-0.5, 0.5], n_draws = 80, rng = StableRNG(5103), correction = :psis)
            @test res_low.metadata.intervals_available == false
            @test res_low.metadata.uncertainty === :unavailable
            @test :ess_below_minimum in res_low.metadata.psis_diagnostics.failures
            @test all(isnan, res_low.lower)
            @test all(isnan, res_low.upper)
            @test all(isnan, res_low.covariance)
        end

        @testset "Non-converged count fit withholds intervals and covariance" begin
            fit_nc = fit_bayesian_cluster_count(
                df, :Y, :A, [:W]; cluster = :cluster, prior, family = :poisson, maxiter = 1)
            @test fit_nc.status === :maxiter_reached
            @test fit_nc.converged == false

            res_nc = bayesian_policy_contrast(
                fit_nc, df, policy; deltas = [-0.5, 0.5], n_draws = 200, rng = StableRNG(5104))
            @test res_nc.metadata.intervals_available == false
            @test res_nc.metadata.uncertainty === :unavailable
            @test all(isnan, res_nc.lower)
            @test all(isnan, res_nc.upper)
            @test all(isnan, res_nc.covariance)
            @test all(isfinite, res_nc.mean)
        end

        @testset "Input and prior validation" begin
            fit = fit_bayesian_cluster_count(
                df, :Y, :A, [:W]; cluster = :cluster, prior, family = :poisson)

            # Negative and non-integer counts rejected
            df_neg = copy(df)
            df_neg.Y[1] = -1
            @test_throws ArgumentError fit_bayesian_cluster_count(
                df_neg, :Y, :A, [:W]; cluster = :cluster, prior)
            @test_throws ArgumentError fit_bayesian_cluster_hurdle(
                df_neg, :Y, :A, [:W]; cluster = :cluster, prior = prior_h)

            df_nonint = copy(df)
            df_nonint.Y = Float64.(df.Y)
            df_nonint.Y[1] = 1.5
            @test_throws ArgumentError fit_bayesian_cluster_count(
                df_nonint, :Y, :A, [:W]; cluster = :cluster, prior)
            @test_throws ArgumentError fit_bayesian_cluster_hurdle(
                df_nonint, :Y, :A, [:W]; cluster = :cluster, prior = prior_h)

            # Zero counts rejected when zero_truncated = true
            df_haszero = copy(df)
            df_haszero.Y[1] = 0
            @test_throws ArgumentError fit_bayesian_cluster_count(
                df_haszero, :Y, :A, [:W]; cluster = :cluster, prior, zero_truncated = true)

            # All-zero hurdle data rejected
            df_allzero = copy(df)
            df_allzero.Y .= 0
            @test_throws ArgumentError fit_bayesian_cluster_hurdle(
                df_allzero, :Y, :A, [:W]; cluster = :cluster, prior = prior_h)

            # Missing outcomes rejected without handle_missing = :mar, accepted with :mar
            df_miss = copy(df)
            df_miss.Y = Vector{Union{Missing, Int}}(df.Y)
            df_miss.Y[1] = missing
            @test_throws ArgumentError fit_bayesian_cluster_count(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior)
            @test_throws ArgumentError fit_bayesian_cluster_hurdle(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior = prior_h)

            fit_mar = fit_bayesian_cluster_count(
                df_miss, :Y, :A, [:W]; cluster = :cluster, prior, handle_missing = :mar)
            fit_complete = fit_bayesian_cluster_count(
                df_miss[2:end, :], :Y, :A, [:W]; cluster = :cluster, prior)
            @test fit_mar.converged
            @test fit_mar.metadata.missingness_status === :user_assumed_mar
            @test fit_mar.metadata.n_missing == 1
            @test fit_mar.metadata.n_observed == nrow(df_miss) - 1
            @test fit_mar.posterior_mean ≈ fit_complete.posterior_mean atol = 1e-12

            # Missing cluster IDs rejected
            df_cmiss = copy(df)
            df_cmiss.cluster = Vector{Union{Missing, String}}(df.cluster)
            df_cmiss.cluster[1] = missing
            @test_throws ArgumentError fit_bayesian_cluster_count(
                df_cmiss, :Y, :A, [:W]; cluster = :cluster, prior)
            @test_throws ArgumentError fit_bayesian_cluster_hurdle(
                df_cmiss, :Y, :A, [:W]; cluster = :cluster, prior = prior_h)
            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df_cmiss, policy; deltas = [0.5], weighting = :cluster)

            # Clamped ShiftPolicy rejected
            @test_throws ArgumentError bayesian_policy_contrast(
                fit, df, ShiftPolicy("z", 0.01, 0.99); deltas = [0.5])
            df_h = simulate_cluster_hurdle(StableRNG(5105), fill(6, 10); family = :poisson)
            fit_h = fit_bayesian_cluster_hurdle(
                df_h, :Y, :A, [:W]; cluster = :cluster, prior = prior_h, family = :poisson)
            @test_throws ArgumentError bayesian_policy_contrast(
                fit_h, df_h, ShiftPolicy("z", 0.01, 0.99); deltas = [0.5])

            # Invalid ClusterCountPrior parameters
            μ = zeros(3)
            V = Matrix{Float64}(I, 3, 3)
            for bad_shape in (1.0, 0.5, -1.0, Inf, NaN)
                @test_throws ArgumentError ClusterCountPrior(;
                    mean = μ, covariance = V, cluster_shape = bad_shape, cluster_scale = 1.0)
            end
            for bad_scale in (0.0, -0.5, Inf, NaN)
                @test_throws ArgumentError ClusterCountPrior(;
                    mean = μ, covariance = V, cluster_shape = 2.0, cluster_scale = bad_scale)
            end
            for bad_disp_mean in (Inf, -Inf, NaN)
                @test_throws ArgumentError ClusterCountPrior(;
                    mean = μ, covariance = V, cluster_shape = 2.0, cluster_scale = 1.0,
                    log_dispersion_mean = bad_disp_mean)
            end
            for bad_disp_sd in (0.0, -0.5, Inf, NaN)
                @test_throws ArgumentError ClusterCountPrior(;
                    mean = μ, covariance = V, cluster_shape = 2.0, cluster_scale = 1.0,
                    log_dispersion_sd = bad_disp_sd)
            end
        end
    end
end
