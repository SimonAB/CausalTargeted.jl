using Test
using CausalTargeted
using DataFrames
using Distributions
using LinearAlgebra
using StableRNGs
using Statistics

const CT = CausalTargeted

@testset "ExponentialTiltPolicy and calibration-balanced TMLE" begin
    @testset "Policy construction, validation, and tilt score ϕ(a)" begin
        p_lin = exponential_tilt_policy()
        @test p_lin isa ExponentialTiltPolicy
        @test p_lin.bound === nothing
        @test p_lin.center == 0.0

        p_lin_c = ExponentialTiltPolicy(bound = nothing, center = 1.5)
        @test tilt_score(3.5, p_lin_c) == 2.0
        @test tilt_score([1.5, 2.5, -0.5], p_lin_c) == [0.0, 1.0, -2.0]

        M = 2.0
        c = 0.75
        p_bnd = exponential_tilt_policy(bound = M, center = c)
        @test p_bnd.bound == M
        @test p_bnd.center == c
        @test tilt_score(c, p_bnd) == 0.0
        # Odd symmetry around center
        for x in (0.1, 0.8, 3.0, 12.0)
            @test isapprox(tilt_score(c + x, p_bnd), -tilt_score(c - x, p_bnd); atol = 1e-14)
        end
        # Unit slope at center: ϕ'(c) = 1
        h = 1e-6
        slope = (tilt_score(c + h, p_bnd) - tilt_score(c - h, p_bnd)) / (2h)
        @test isapprox(slope, 1.0; atol = 1e-9)
        # Strict bound |ϕ(a)| <= M even at extreme values
        extremes = [-1e6, -50.0, -5.0, 0.0, 5.0, 50.0, 1e6]
        scores = tilt_score(extremes, p_bnd)
        @test all(abs.(scores) .<= M)
        @test isapprox(scores[1], -M; atol = 1e-12)
        @test isapprox(scores[end], M; atol = 1e-12)

        # Validation guards
        @test_throws ArgumentError exponential_tilt_policy(bound = 0.0)
        @test_throws ArgumentError exponential_tilt_policy(bound = -1.0)
        @test_throws ArgumentError exponential_tilt_policy(bound = Inf)
        @test_throws ArgumentError exponential_tilt_policy(bound = NaN)
        @test_throws ArgumentError exponential_tilt_policy(center = NaN)
        @test_throws ArgumentError exponential_tilt_policy(center = Inf)
        @test_throws ArgumentError tilt_score(NaN, p_bnd)
        @test_throws ArgumentError tilt_score(Inf, p_lin)
        @test_throws ArgumentError CT.tilt_log_weight([1.0], NaN, p_bnd)
    end

    @testset "Bounded weight guarantees exp(-2|δ|M) <= H <= exp(2|δ|M)" begin
        rng = StableRNG(20261125)
        n = 120
        W = randn(rng, n, 3)
        # Inject extreme covariate leverage and extreme exposures
        W[1, 1] = 12.0
        W[2, 2] = -15.0
        A = 0.8 .* W[:, 1] .- 0.6 .* W[:, 2] .+ randn(rng, n)
        A[3] = 25.0
        A[4] = -25.0
        Y = 2.0 .* A .+ W[:, 1] .+ randn(rng, n)
        df = DataFrame(A = A, Y = Y, W1 = W[:, 1], W2 = W[:, 2], W3 = W[:, 3])

        for M in (1.0, 2.0), δ in (-2.0, -0.5, 0.75, 2.0)
            pol = exponential_tilt_policy(bound = M, center = 0.25)
            lo, hi = tilt_weight_bounds(δ, pol)
            @test isapprox(lo, exp(-2 * abs(δ) * M); rtol = 1e-14)
            @test isapprox(hi, exp(2 * abs(δ) * M); rtol = 1e-14)

            res = estimate_exponential_tilt(
                df, :A, :Y, [:W1, :W2, :W3], δ, pol;
                family_outcome = :gaussian, folds = 4, bootstrap_B = 99,
                rng = StableRNG(101),
            )
            @test all(h -> lo * (1 - 1e-12) <= h <= hi * (1 + 1e-12), res.H)
            @test res.weight_bounds == (lo, hi)
        end
    end

    @testset "Convex calibration-balanced Newton solver & exact covariate balance" begin
        rng = StableRNG(42)
        n = 250
        W = hcat(randn(rng, n), Float64.(rand(rng, Bool, n)), randn(rng, n) .* 3.0 .+ 5.0)
        A = 0.5 .* W[:, 1] .- 0.4 .* W[:, 2] .+ 0.2 .* W[:, 3] .+ 0.7 .* randn(rng, n)

        for pol in (exponential_tilt_policy(), exponential_tilt_policy(bound = 2.0, center = 0.5))
            for δ in (-1.5, 0.8, 1.5)
                fit = CT.fit_tilt_normalizer(W, A, δ, pol; lambda = 0.0)
                @test fit.converged
                @test fit.grad_inf_norm < 1e-10
                # Raw calibration weights satisfy exact finite-sample moment balance
                H_raw = CT.predict_tilt_weights(fit, W, A; enforce_bounds = false)
                @test isapprox(mean(H_raw), 1.0; atol = 1e-10)
                for j in 1:size(W, 2)
                    @test isapprox(mean(H_raw .* W[:, j]), mean(W[:, j]); atol = 1e-10)
                end
            end
        end

        # Ridge regularisation: intercept still enforces mean(H) == 1 and shrinks ||β_{-1}||
        pol = exponential_tilt_policy(bound = 2.0)
        fit0 = CT.fit_tilt_normalizer(W, A, 1.2, pol; lambda = 0.0)
        fit1 = CT.fit_tilt_normalizer(W, A, 1.2, pol; lambda = 0.1)
        fit2 = CT.fit_tilt_normalizer(W, A, 1.2, pol; lambda = 1.0)
        @test fit1.converged && fit2.converged
        @test isapprox(mean(CT.predict_tilt_weights(fit1, W, A; enforce_bounds = false)), 1.0; atol = 1e-10)
        @test isapprox(mean(CT.predict_tilt_weights(fit2, W, A; enforce_bounds = false)), 1.0; atol = 1e-10)
        @test norm(fit2.beta[2:end]) < norm(fit1.beta[2:end]) < norm(fit0.beta[2:end])

        # Intercept-only (0 columns) and constant-column matrices
        W_empty = zeros(n, 0)
        fit_empty = CT.fit_tilt_normalizer(W_empty, A, 1.0, pol)
        @test fit_empty.converged && fit_empty.iterations == 0
        @test isapprox(mean(CT.predict_tilt_weights(fit_empty, W_empty, A; enforce_bounds = false)), 1.0; atol = 1e-12)

        W_const = hcat(fill(3.0, n), fill(-2.0, n))
        fit_const = CT.fit_tilt_normalizer(W_const, A, 1.0, pol)
        @test fit_const.converged && fit_const.iterations == 0
        @test isempty(fit_const.active_cols)
        @test isapprox(mean(CT.predict_tilt_weights(fit_const, W_const, A; enforce_bounds = false)), 1.0; atol = 1e-12)

        # Collinear columns at λ = 0 still converge cleanly
        W_col = hcat(W[:, 1], 2.0 .* W[:, 1] .+ 1.0, W[:, 2])
        fit_col = CT.fit_tilt_normalizer(W_col, A, 0.9, pol; lambda = 0.0)
        @test fit_col.converged
        H_col = CT.predict_tilt_weights(fit_col, W_col, A; enforce_bounds = false)
        @test isapprox(mean(H_col), 1.0; atol = 1e-9)
        @test isapprox(mean(H_col .* W_col[:, 1]), mean(W_col[:, 1]); atol = 1e-9)

        # Zero tilt δ = 0
        fit_zero = CT.fit_tilt_normalizer(W, A, 0.0, pol)
        @test fit_zero.converged && fit_zero.iterations == 0
        @test all(iszero, fit_zero.beta)
        @test all(==(1.0), CT.predict_tilt_weights(fit_zero, W, A))
    end

    @testset "Exact Gaussian analytical recovery: Δ(δ) = δσ² and closed-form τ(δ)" begin
        rng = StableRNG(777)
        n = 4000
        σ_A = 0.75
        β_A = 1.6
        W1 = randn(rng, n)
        W2 = randn(rng, n)
        A = 0.4 .+ 0.6 .* W1 .- 0.5 .* W2 .+ σ_A .* randn(rng, n)
        Y = 2.0 .+ β_A .* A .+ 0.9 .* W1 .- 0.7 .* W2 .+ 0.8 .* randn(rng, n)
        df = DataFrame(A = A, Y = Y, W1 = W1, W2 = W2)

        δ = 0.8
        true_shift = δ * σ_A^2
        true_tau = β_A * true_shift

        # Linear tilt is invariant to `center` (including extreme center = 1e4 with log-weight shift)
        res1 = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], δ, exponential_tilt_policy(center = 0.0);
            family_outcome = :gaussian, folds = 5, bootstrap_B = 499, rng = StableRNG(11),
        )
        res2 = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], δ, exponential_tilt_policy(center = 2.5);
            family_outcome = :gaussian, folds = 5, bootstrap_B = 499, rng = StableRNG(11),
        )
        res_ext = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], δ, exponential_tilt_policy(center = 1e4);
            family_outcome = :gaussian, folds = 5, bootstrap_B = 499, rng = StableRNG(11),
        )
        @test isapprox(res1.estimate, res2.estimate; atol = 1e-9)
        @test isapprox(res1.shift_estimate, res2.shift_estimate; atol = 1e-9)
        @test isapprox(res1.estimate, res_ext.estimate; atol = 1e-7)
        @test isapprox(res1.shift_estimate, res_ext.shift_estimate; atol = 1e-7)

        @test abs(res1.shift_estimate - true_shift) < 0.05
        @test abs(res1.estimate - true_tau) < 0.10
        @test res1.shift_lower <= true_shift <= res1.shift_upper
        @test res1.lower <= true_tau <= res1.upper
        @test res1.boot_lower <= true_tau <= res1.boot_upper
        @test isapprox(mean(res1.ic), 0.0; atol = 1e-10)
        @test isapprox(mean(res1.shift_ic), 0.0; atol = 1e-10)

        # Trivial δ = 0
        res0 = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], 0.0, exponential_tilt_policy(bound = 2.0);
            family_outcome = :gaussian, folds = 5, rng = StableRNG(11),
        )
        @test res0.inference_status === :trivial
        @test res0.estimate == 0.0 && res0.se == 0.0
        @test res0.shift_estimate == 0.0 && res0.shift_se == 0.0
    end

    @testset "Canonical-link TMLE score equation across :gaussian, :binomial, :poisson, :negbin" begin
        rng = StableRNG(2026)
        n = 300
        W1 = randn(rng, n)
        W2 = Float64.(rand(rng, Bool, n))
        A = 0.4 .* W1 .- 0.3 .* W2 .+ 0.8 .* randn(rng, n)

        y_gauss = 1.0 .+ 0.7 .* A .+ 0.5 .* W1 .+ randn(rng, n)
        p_bin = [inv(1.0 + exp(-(0.2 + 0.6 * A[i] + 0.5 * W1[i] - 0.4 * W2[i]))) for i in 1:n]
        y_bin = Float64.(rand(rng, n) .< p_bin)
        μ_cnt = exp.(1.5 .+ 0.3 .* A .+ 0.35 .* W1 .- 0.25 .* W2)
        y_pois = Float64.([rand(rng, Poisson(m)) for m in μ_cnt])
        y_nb = Float64.([rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + m))) for m in μ_cnt])

        pol = exponential_tilt_policy(bound = 2.0)
        for (fam, ycol) in (
            (:gaussian, y_gauss),
            (:binomial, y_bin),
            (:poisson, y_pois),
            (:negbin, y_nb),
        )
            df = DataFrame(A = A, Y = ycol, W1 = W1, W2 = W2)
            for meth in (:plugin, :direct)
                res = estimate_exponential_tilt(
                    df, :A, :Y, [:W1, :W2], 1.0, pol;
                    family_outcome = fam, outcome_method = meth, folds = 4,
                    bootstrap_B = 199, rng = StableRNG(55),
                )
                @test res.inference_status === :available
                @test abs(res.tmle_score) <= 1e-7 * max(1.0, abs(res.psi))
                @test abs(res.shift_tmle_score) <= 1e-10
                @test abs(mean(res.ic)) <= 1e-7 * max(1.0, abs(res.psi))
                @test abs(mean(res.shift_ic)) <= 1e-10
                if fam === :binomial
                    @test all(x -> 0.0 <= x <= 1.0, res.theta_star)
                elseif fam in (:poisson, :negbin)
                    @test all(x -> x >= 0.0, res.theta_star)
                end
            end
        end
    end

    @testset "Double robustness: misspecified m_0 vs misspecified θ_δ" begin
        rng = StableRNG(909)
        n = 5000
        σ_A = 0.7
        β_A = 1.0
        W1 = randn(rng, n)
        W2 = randn(rng, n)
        # Strong confounding on both A and Y so omitting W from both creates large bias
        A = 1.2 .* W1 .- 1.0 .* W2 .+ σ_A .* randn(rng, n)
        Y = 1.0 .+ β_A .* A .+ 2.5 .* W1 .- 2.0 .* W2 .+ 0.7 .* randn(rng, n)
        df = DataFrame(A = A, Y = Y, W1 = W1, W2 = W2)

        δ = 0.6
        pol = exponential_tilt_policy(bound = nothing)
        true_tau = β_A * δ * σ_A^2

        # 1. Misspecified m_0 (empty covariates), well-specified θ_δ ([:W1, :W2])
        res_bad_m0 = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], δ, pol;
            normalizer_covariates = Symbol[],
            outcome_covariates = [:W1, :W2],
            outcome_method = :direct,
            folds = 5, bootstrap_B = 99, rng = StableRNG(12),
        )
        # 2. Well-specified m_0 ([:W1, :W2]), misspecified θ_δ (empty covariates)
        res_bad_theta = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], δ, pol;
            normalizer_covariates = [:W1, :W2],
            outcome_covariates = Symbol[],
            outcome_method = :direct,
            folds = 5, bootstrap_B = 99, rng = StableRNG(12),
        )
        # 3. Both misspecified: substantial confounding bias
        res_both_bad = estimate_exponential_tilt(
            df, :A, :Y, [:W1, :W2], δ, pol;
            normalizer_covariates = Symbol[],
            outcome_covariates = Symbol[],
            outcome_method = :direct,
            folds = 5, bootstrap_B = 99, rng = StableRNG(12),
        )

        @test abs(res_bad_m0.estimate - true_tau) < 0.15
        @test abs(res_bad_theta.estimate - true_tau) < 0.15
        @test abs(res_both_bad.estimate - true_tau) > 2.0
    end

    @testset "Studentised multiplier bootstrap CI sanity checks" begin
        # Reproducibility and agreement with Wald on large symmetric sample
        rng = StableRNG(314)
        ic_sym = randn(rng, 2000)
        ic_sym .-= mean(ic_sym)
        se_sym = std(ic_sym) / sqrt(length(ic_sym))
        w_lo, w_hi = CT.wald_ci(1.25, se_sym)

        b1 = studentised_multiplier_ci(1.25, ic_sym; B = 999, weights = :mammen, rng = StableRNG(7))
        b2 = studentised_multiplier_ci(1.25, ic_sym; B = 999, weights = :mammen, rng = StableRNG(7))
        br = studentised_multiplier_ci(1.25, ic_sym; B = 999, weights = :rademacher, rng = StableRNG(7))
        @test b1.lower == b2.lower && b1.upper == b2.upper
        @test isapprox(b1.lower, w_lo; atol = 0.02)
        @test isapprox(b1.upper, w_hi; atol = 0.02)
        @test isapprox(br.lower, w_lo; atol = 0.02)
        @test isapprox(br.upper, w_hi; atol = 0.02)

        # Small-n Rademacher studentised CI widens relative to Wald (t-like quantiles)
        rng_sm = StableRNG(88)
        ic_sm = randn(rng_sm, 25)
        ic_sm .-= mean(ic_sm)
        se_sm = std(ic_sm) / sqrt(length(ic_sm))
        wl_sm, wu_sm = CT.wald_ci(0.0, se_sm)
        br_sm = studentised_multiplier_ci(0.0, ic_sm; B = 1999, weights = :rademacher, rng = StableRNG(8))
        @test (br_sm.upper - br_sm.lower) > (wu_sm - wl_sm)

        # Small-n right-skewed EIF: Mammen studentised CI stretches the upper tail
        rng_sk = StableRNG(99)
        ic_skew = rand(rng_sk, Exponential(1.0), 30) .^ 2
        ic_skew .-= mean(ic_skew)
        bs = studentised_multiplier_ci(0.0, ic_skew; B = 1999, weights = :mammen, rng = StableRNG(9))
        @test bs.upper - 0.0 > 0.0 - bs.lower
        @test bs.q_lower < -1.96

        # Both independent_folds=true and independent_folds=false produce valid TMLE scores
        df_f = DataFrame(A = randn(StableRNG(3), 80), Y = randn(StableRNG(4), 80), W1 = randn(StableRNG(5), 80))
        rf1 = estimate_exponential_tilt(df_f, :A, :Y, [:W1], 0.5, exponential_tilt_policy(bound = 2.0); independent_folds = true, bootstrap_B = 99, rng = StableRNG(6))
        rf2 = estimate_exponential_tilt(df_f, :A, :Y, [:W1], 0.5, exponential_tilt_policy(bound = 2.0); independent_folds = false, bootstrap_B = 99, rng = StableRNG(6))
        @test abs(rf1.tmle_score) < 1e-10 && abs(rf2.tmle_score) < 1e-10

        # Degenerate zero IC
        bz = studentised_multiplier_ci(2.5, zeros(20); B = 99, rng = StableRNG(1))
        @test bz.lower == 2.5 && bz.upper == 2.5

        # Guard checks
        @test_throws ArgumentError studentised_multiplier_ci(0.0, [1.0])
        @test_throws ArgumentError studentised_multiplier_ci(0.0, randn(10); alpha = 0.0)
        @test_throws ArgumentError studentised_multiplier_ci(0.0, randn(10); B = 5)
        @test_throws ArgumentError studentised_multiplier_ci(0.0, randn(10); weights = :invalid)
    end

    @testset "End-to-end run_lmtp_grid integration" begin
        rng = StableRNG(20261125)
        n = 240
        W1 = randn(rng, n)
        group = repeat(["g1", "g2"], inner = n ÷ 2)
        A = 0.4 .* W1 .+ 0.3 .* (group .== "g2") .+ 0.8 .* randn(rng, n)
        Y = 1.0 .+ 1.2 .* A .+ 0.6 .* W1 .+ 0.7 .* randn(rng, n)
        df = DataFrame(A = A, Y = Y, W1 = W1, group = group)

        pol = exponential_tilt_policy(bound = 2.0)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W1],
            deltas = [-1.0, 0.0, 1.0],
            policy = pol,
            stratify_by = nothing,
            simultaneous = true,
            parallel = false,
            folds = 3,
            rng = StableRNG(71),
        )
        @test nrow(grid) == 3
        @test grid.inference_status == [:available, :trivial, :available]
        @test all(grid.support_status .== "exponential_tilt_policy")
        @test all(.!grid.weight_truncation)
        @test grid.est[2] == 0.0 && grid.se[2] == 0.0 && grid.effective_shift[2] == 0.0
        @test grid.effective_shift[1] < 0.0 && grid.effective_shift[3] > 0.0
        @test grid.est[1] < 0.0 && grid.est[3] > 0.0
        # Simultaneous bands widen pointwise Wald intervals on nonzero deltas
        @test isfinite(grid.crit_sim[1]) && grid.crit_sim[1] > 1.96
        @test grid.lwr_sim[1] <= grid.lwr[1] && grid.upr_sim[1] >= grid.upr[1]
        @test grid.lwr_sim[3] <= grid.lwr[3] && grid.upr_sim[3] >= grid.upr[3]

        # Stratified grid with simultaneous bands
        strat_grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W1],
            deltas = [-0.8, 0.0, 0.8],
            policy = pol,
            stratify_by = :group,
            simultaneous = true,
            parallel = false,
            folds = 3,
            rng = StableRNG(72),
        )
        @test nrow(strat_grid) == 6
        @test sort(unique(strat_grid.stratum)) == ["g1", "g2"]
        nz = strat_grid[strat_grid.delta .!= 0.0, :]
        @test all(nz.inference_status .== :available)
        @test all(isfinite, nz.se)
        @test all(nz.crit_sim .> 1.96)
        @test all(nz.lwr_sim .<= nz.lwr) && all(nz.upr_sim .>= nz.upr)

        # Count (:negbin) and :binomial families through run_lmtp_grid
        μ_nb = 50.0 .* exp.(0.25 .* A .+ 0.3 .* W1)
        df_nb = DataFrame(A = A, Y = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + m))) for m in μ_nb], W1 = W1)
        grid_nb = run_lmtp_grid(
            df_nb, :A, :Y;
            baseline = [:W1], deltas = [-1.0, 1.0],
            policy = pol, family_outcome = :negbin,
            stratify_by = nothing, simultaneous = false, parallel = false,
            folds = 3, rng = StableRNG(73),
        )
        @test all(grid_nb.inference_status .== :available)
        @test all(isfinite, grid_nb.se)
        @test grid_nb.est[1] < 0.0 && grid_nb.est[2] > 0.0

        p_bin = [inv(1.0 + exp(-(0.5 * A[i] + 0.4 * W1[i]))) for i in 1:n]
        df_bin = DataFrame(A = A, Y = Float64.(rand(rng, n) .< p_bin), W1 = W1)
        grid_bin = run_lmtp_grid(
            df_bin, :A, :Y;
            baseline = [:W1], deltas = [-1.0, 1.0],
            policy = exponential_tilt_policy(bound = nothing),
            family_outcome = :binomial,
            stratify_by = nothing, simultaneous = false, parallel = false,
            folds = 3, rng = StableRNG(74),
        )
        @test all(grid_bin.inference_status .== :available)
        @test all(isfinite, grid_bin.se)

        # Configuration error guards
        @test_throws ArgumentError run_lmtp_grid(
            df, :A, :Y; baseline = [:W1], deltas = [1.0], policy = pol,
            cv_trunc = true, stratify_by = nothing, simultaneous = false, parallel = false,
        )
        @test_throws ArgumentError run_lmtp_grid(
            df, :A, :Y; baseline = [:W1], deltas = [1.0], policy = pol,
            positivity = true, stratify_by = nothing, simultaneous = false, parallel = false,
        )
        @test_throws ArgumentError run_lmtp_grid(
            df, :A, :Y; baseline = [:W1], deltas = [1.0], policy = pol,
            estimator = :eif, stratify_by = nothing, simultaneous = false, parallel = false,
        )
        @test_throws ArgumentError run_lmtp_grid(
            df, :A, :Y; baseline = [:W1], deltas = [Inf], policy = pol,
            stratify_by = nothing, simultaneous = false, parallel = false,
        )
    end
end
