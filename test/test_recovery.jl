    @testset "expanded synthetic recovery" begin
        # Core recovery at moderate n (lean learners)
        r1 = CausalTargeted.run_julia_synthetic_once(:linear_mtp; n = 300, delta = 1.0, folds = 3,
            rng = StableRNG(1), learners = (:glm, :mean))
        @test only(r1.abs_error) < 0.12

        if _HAS_CAUSAL_MEDIATION
            r2 = CausalTargeted.run_julia_synthetic_once(:binary_mediation; n = 400, folds = 3,
                rng = StableRNG(2), learners = (:glm, :mean), n_mc = 16)
            @test maximum(r2.abs_error) < 0.20

            r3 = CausalTargeted.run_julia_synthetic_once(:continuous_mtp_mediation; n = 500, delta = 1.0,
                folds = 3, rng = StableRNG(3), learners = (:glm, :mean), n_mc = 24)
            # TE can be noisier than NDE under nested MC; require TE direction + bound
            te = only(r3[r3.estimand .== "TE", :abs_error])
            @test te < 0.50
            @test only(r3[r3.estimand .== "TE", :sign_agree])
        end

        # Stress: misspecification still recovers A coefficient directionally
        r4 = CausalTargeted.run_julia_synthetic_once(:misspecified_nuisance_mtp; n = 400, delta = 1.0,
            folds = 3, rng = StableRNG(13), learners = (:glm, :mean))
        @test only(r4.sign_agree)

        # Intermediate confounding oracle is finite (DGP only; no estimator)
        df, truth = CausalTargeted.simulate_intermediate_confounding_mediation(200; rng = StableRNG(12))
        ora = truth.oracle(1.0)
        @test isfinite(ora.te) && isfinite(ora.nde) && isfinite(ora.nie)

        # Weak positivity: estimate finite (error may be large)
        r5 = CausalTargeted.run_julia_synthetic_once(:weak_positivity_mtp; n = 250, delta = 1.0,
            folds = 2, rng = StableRNG(11), learners = (:glm, :mean))
        @test isfinite(only(r5.estimate))
    end

    # =====================================================================
    # Improvement-driving tests (synthetic recovery)
    # =====================================================================

    @testset "learner richness" begin
        # GLM-only cannot capture Y = β_a A + β_w2 W² + ε
        r_glm = CausalTargeted.run_julia_synthetic_once(:misspecified_nuisance_mtp; n = 400, delta = 1.0,
            folds = 3, rng = StableRNG(13), learners = (:glm, :mean))
        r_rich = CausalTargeted.run_julia_synthetic_once(:misspecified_nuisance_mtp; n = 400, delta = 1.0,
            folds = 3, rng = StableRNG(13), learners = RICH_SL_LEARNERS)
        # Richer library should reduce absolute error
        @test only(r_rich.abs_error) < only(r_glm.abs_error)
        # Target: richer learners get |err| < 0.15
        @test only(r_rich.abs_error) < 0.15
        # Both should at least recover the sign
        @test only(r_glm.sign_agree)
        @test only(r_rich.sign_agree)
    end

    @testset "density ratio variants" begin
        df, truth = simulate_linear_mtp(500; rng = StableRNG(100))
        sdA = std(df.A)
        δ_raw = 1.0 * sdA
        eff = effective_raw_shift(df.A, δ_raw)
        truth_te = truth.effects(eff).te

        abs_errors = Dict{Symbol, Float64}()
        covers = Dict{Symbol, Bool}()
        for dr in (:gaussian, :classification, :hybrid)
            grid = run_lmtp_grid(
                df, :A, :Y;
                baseline = [:W],
                deltas = [δ_raw],
                folds = 3,
                density_ratio = dr,
                cv_trunc = false,
                parallel = false,
                simultaneous = false,
                cache_nuisances = false,
                rng = StableRNG(100),
                shift_scale = "raw",
            )
            est = only(grid.est)
            se = only(grid.se)
            ae = abs(est - truth_te)
            abs_errors[dr] = ae
            covers[dr] = isfinite(se) && se > 0 && ae <= 1.96 * se
            @test isfinite(est)
        end
        # At least one variant should cover
        @test any(values(covers))
        # cv_trunc should not substantially worsen error
        grid_cv = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W],
            deltas = [δ_raw],
            folds = 3,
            density_ratio = :gaussian,
            cv_trunc = true,
            parallel = false,
            simultaneous = false,
            cache_nuisances = false,
            rng = StableRNG(100),
            shift_scale = "raw",
        )
        ae_cv = abs(only(grid_cv.est) - truth_te)
        @test ae_cv < 1.5 * abs_errors[:gaussian] + 0.01
    end

    @testset "coverage calibration" begin
        n_seeds = 10
        # LMTP coverage across seeds
        lmtp_covers = Bool[]
        lmtp_ses = Float64[]
        for seed in 1:n_seeds
            r = CausalTargeted.run_julia_synthetic_once(:linear_mtp; n = 300, delta = 1.0,
                folds = 3, rng = StableRNG(seed), learners = (:glm, :mean))
            cov = only(r.cover_95)
            push!(lmtp_covers, cov === missing ? false : Bool(cov))
            push!(lmtp_ses, only(r.se))
        end
        # SEs should be non-degenerate
        @test all(se -> se > 0.01, lmtp_ses)
        # Coverage >= 0.6 catches gross undercoverage (true target: 0.95)
        @test mean(lmtp_covers) >= 0.6

        if _HAS_CAUSAL_MEDIATION
            # Binary mediation coverage across seeds
            med_covers = Bool[]
            med_ses = Float64[]
            for seed in 1:n_seeds
                r = CausalTargeted.run_julia_synthetic_once(:binary_mediation; n = 300, folds = 3,
                    rng = StableRNG(seed), learners = (:glm, :mean), n_mc = 16)
                te_row = only(eachrow(filter(row -> row.estimand == "TE", r)))
                cov = te_row.cover_95
                push!(med_covers, cov === missing ? false : Bool(cov))
                push!(med_ses, te_row.se)
            end
            @test all(se -> se > 0.005, med_ses)
            @test mean(med_covers) >= 0.6
        end
    end

    if _HAS_CAUSAL_MEDIATION
    @testset "mediator MC convergence" begin
        n_mc_values = [1, 16, 64, 128]
        te_errors = Dict{Int, Float64}()
        nde_errors = Dict{Int, Float64}()
        for nmc in n_mc_values
            r = CausalTargeted.run_julia_synthetic_once(:continuous_mtp_mediation; n = 500, delta = 1.0,
                folds = 3, rng = StableRNG(3), learners = (:glm, :mean), n_mc = nmc)
            te_errors[nmc] = only(r[r.estimand .== "TE", :abs_error])
            nde_errors[nmc] = only(r[r.estimand .== "NDE", :abs_error])
        end
        # MC (n_mc=128) should not be worse than pure plugin (n_mc=1) for TE
        # (allow small Monte Carlo noise; both use the same seed/DGP)
        @test te_errors[128] <= te_errors[1] + 0.05
        # Monotone improvement from 16→128 (on average)
        @test te_errors[128] <= te_errors[16] + 0.05
        # NDE should also improve separately (not just TE by cancellation)
        @test nde_errors[128] <= nde_errors[1] + 0.05
        # Larger sample with moderate MC should get TE |err| < 0.35 under full EIF
        r_big = CausalTargeted.run_julia_synthetic_once(:continuous_mtp_mediation; n = 800, delta = 1.0,
            folds = 3, rng = StableRNG(3), learners = (:glm, :mean), n_mc = 64)
        @test only(r_big[r_big.estimand .== "TE", :abs_error]) < 0.35
    end

    @testset "intermediate confounder adjustment" begin
        df, truth = CausalTargeted.simulate_intermediate_confounding_mediation(400; rng = StableRNG(12))
        ora = truth.oracle(1.0)

        # Baseline-only (ignores L as moc) — typically more biased for NDE
        r_w = run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [1.0], folds = 3, n_mc = 32,
            learners = (:glm, :mean), estimator = :onestep,
            parallel = false, cache_nuisances = false, rng = StableRNG(12),
        )
        nde_w = only(filter(row -> row.estimand == "NDE", eachrow(r_w))).est
        nde_err_w = abs(nde_w - ora.nde)

        # Proper moc handling
        r_moc = run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M], moc = [:L],
            deltas = [1.0], folds = 3, n_mc = 48,
            learners = (:glm, :mean), estimator = :onestep,
            parallel = false, cache_nuisances = false, rng = StableRNG(12),
        )
        nde_moc = only(filter(row -> row.estimand == "NDE", eachrow(r_moc))).est
        nde_err_moc = abs(nde_moc - ora.nde)

        @test nde_err_moc < nde_err_w + 0.05 || nde_err_moc < 0.40
        @test isfinite(nde_w)
        @test isfinite(nde_moc)
    end
    end # _HAS_CAUSAL_MEDIATION

    @testset "sqrt-n convergence" begin
        # Linear MTP: average |err| over 5 seeds at each n to smooth luck
        function mean_err_lmtp(n_obs, seeds)
            mean([begin
                r = CausalTargeted.run_julia_synthetic_once(:linear_mtp; n = n_obs, delta = 1.0,
                    folds = 3, rng = StableRNG(s), learners = (:glm, :mean))
                only(r.abs_error)
            end for s in seeds])
        end
        seeds = 1:5
        err_small = mean_err_lmtp(200, seeds)
        err_large = mean_err_lmtp(3200, seeds)
        # Larger n should have lower mean |err|
        @test err_large < err_small
        # Tight at large n (allow Monte Carlo / cross-fit noise across seeds)
        @test err_large < 0.06

        if _HAS_CAUSAL_MEDIATION
            # Binary mediation: TE should improve with n (averaged over seeds)
            function mean_err_med(n_obs, seeds)
                mean([begin
                    r = CausalTargeted.run_julia_synthetic_once(:binary_mediation; n = n_obs, folds = 3,
                        rng = StableRNG(s), learners = (:glm, :mean), n_mc = 16)
                    only(r[r.estimand .== "TE", :abs_error])
                end for s in seeds])
            end
            @test mean_err_med(800, seeds) < mean_err_med(200, seeds)
        end
    end
