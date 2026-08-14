    @testset "nonlinear interaction recovery" begin
        df, truth = CausalTargeted.simulate_nonlinear_interaction_mtp(500; rng = StableRNG(20))
        sdA = std(df.A)
        eff = effective_raw_shift(df.A, 1.0 * sdA)
        t = truth.effects(eff)

        # GLM-only (misspecified for A×W1 interaction)
        grid_glm = run_lmtp_grid(
            df, :A, :Y; baseline = [:W1, :W2], deltas = [1.0 * sdA],
            folds = 3, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(20), shift_scale = "raw",
        )
        err_glm = abs(only(grid_glm.est) - t.te)

        # Interaction library (lean default no longer includes :glm_interact)
        grid_rich = run_lmtp_grid(
            df, :A, :Y; baseline = [:W1, :W2], deltas = [1.0 * sdA],
            folds = 3,
            learners_outcome = (:glm, :glm_interact, :mean),
            learners_trt = (:glm, :glm_interact, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(20), shift_scale = "raw",
        )
        err_rich = abs(only(grid_rich.est) - t.te)

        # Interaction terms should not inflate error vs GLM-only on this DGP
        @test err_rich <= err_glm + 0.05
        # Both should produce finite estimates
        @test isfinite(only(grid_glm.est))
        @test isfinite(only(grid_rich.est))
        # Direction should be correct
        @test only(grid_rich.est) * t.te > 0 || abs(t.te) < 0.05
    end

    @testset "small-n stability" begin
        for n_small in [20, 40, 80]
            r = CausalTargeted.run_julia_synthetic_once(:linear_mtp; n = n_small, delta = 1.0,
                folds = recommend_folds(n_small), rng = StableRNG(1),
                learners = recommend_learners(n_small))
            @test isfinite(only(r.estimate))
            @test only(r.se) > 0.001
        end
        # recommend_folds gives sensible values
        @test recommend_folds(20) == 2
        @test recommend_folds(50) == 3
        @test recommend_folds(200) == 5
        # recommend_learners drops risky / optional learners at small n
        @test :evotree ∉ recommend_learners(30)
        @test :glmnet ∉ recommend_learners(30)
        @test :glmnet ∈ adaptive_learners(30)  # MLJ loaded in test env
        @test :evotree ∈ adaptive_learners(50)  # EvoTrees loaded in test env
    end

    @testset "missing outcome IPCW" begin
        df, truth = CausalTargeted.simulate_missing_outcome_mtp(400; rng = StableRNG(30))
        sdA = std(df.A)
        eff = effective_raw_shift(Float64.(df.A), 1.0 * sdA)
        t = truth.effects(eff)

        # Complete-case (drop missing outcomes)
        grid_cc = run_lmtp_grid(
            df, :A, :Y; baseline = [:W], deltas = [1.0 * sdA],
            folds = 3, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(30), shift_scale = "raw", handle_missing = :drop,
        )
        err_cc = abs(only(grid_cc.est) - t.te)

        # IPCW
        grid_ipcw = run_lmtp_grid(
            df, :A, :Y; baseline = [:W], deltas = [1.0 * sdA],
            folds = 3, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(30), shift_scale = "raw", handle_missing = :ipcw,
        )
        err_ipcw = abs(only(grid_ipcw.est) - t.te)

        # Both should be finite
        @test isfinite(only(grid_cc.est))
        @test isfinite(only(grid_ipcw.est))
        # IPCW should differ from complete-case drop when MAR is informative
        @test !isapprox(only(grid_cc.est), only(grid_ipcw.est); atol = 1e-10)
        @test err_ipcw < err_cc || err_ipcw < 0.20
    end

    @testset "missing covariate imputation" begin
        df, truth = CausalTargeted.simulate_missing_covariate_mtp(400; rng = StableRNG(31))
        sdA = std(df.A)
        eff = effective_raw_shift(Float64.(df.A), 1.0 * sdA)
        t = truth.effects(eff)

        grid = run_lmtp_grid(
            df, :A, :Y; baseline = [:W], deltas = [1.0 * sdA],
            folds = 3, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(31), shift_scale = "raw", handle_missing = :impute,
        )
        @test isfinite(only(grid.est))
        @test abs(only(grid.est) - t.te) < 0.15
    end

    @testset "g-computation plug-in" begin
        df, truth = CausalTargeted.simulate_gcomp_nonlinear(500; rng = StableRNG(50))

        # Rich library: influence-function SE (n_boot=0); full refit bootstrap is costly
        res = run_gcomp(df, :A, :Y; covariates = [:W], folds = 3,
                        learners = RICH_SL_LEARNERS, rng = StableRNG(50), n_boot = 0)
        @test isfinite(res.estimate)
        @test abs(res.estimate - truth.ate) < 0.20
        @test res.ci_lower < truth.ate < res.ci_upper
        @test res.se > 0.01
    end

    @testset "DiD 2x2 recovery" begin
        df, truth = CausalTargeted.simulate_did_2x2(200; τ = 1.0, rng = StableRNG(40))
        res = run_did_2x2(df)
        @test abs(res.att - truth.att) < 0.20
        @test res.se > 0.01
        @test res.ci_lower < truth.att < res.ci_upper
    end

    @testset "DiD staggered recovery" begin
        df, truth = CausalTargeted.simulate_did_staggered(400; rng = StableRNG(41))
        res = run_did_staggered(df)
        agg = aggregate_did(res)
        # Aggregate ATT should be in the right ballpark
        @test abs(agg.att - truth.att_aggregate) < 0.25
        # Individual cohort ATTs should be recoverable
        early_rows = filter(r -> r.cohort == "early", eachrow(res))
        if !isempty(early_rows)
            mean_early = mean(r.att for r in early_rows)
            @test abs(mean_early - truth.att_early) < 0.30
        end
        @test nrow(res) >= 2
    end

    @testset "smooth nonlinear DGP + NN learner policy" begin
        df, truth = CausalTargeted.simulate_smooth_nonlinear_mtp(80; rng = StableRNG(21))
        @test nrow(df) == 80
        @test hasproperty(df, :W3)
        @test truth.name == "smooth_nonlinear_mtp"
        # Neural candidates must never enter small-n / adaptive presets
        @test !(:mlj_mlp in SMALL_N_SL_LEARNERS)
        @test !(:mlj_nn_binary in SMALL_N_SL_LEARNERS)
        @test !(:mlj_mlp in adaptive_learners(200; rich = true))
        @test !(:mlj_nn_binary in adaptive_learners(200; rich = true))
        @test !(:mlj_mlp in DEFAULT_SL_LEARNERS)
        @test !(:mlj_mlp in RICH_SL_LEARNERS)
    end
