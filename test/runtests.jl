using CausalTargeted
using CausalDynamics: identify, TotalEffectQuery, TemporalEffectQuery, TemporalDAGSpec, LaggedEdge, unroll_temporal_dag
using DataFrames
using Graphs
using Random
using StableRNGs
using Statistics
using Test

@testset "CausalTargeted" begin
    @testset "synthetic LMTP" begin
        rng = StableRNG(1)
        df, truth = simulate_linear_mtp(120; rng = rng)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W],
            deltas = [-0.5, 0.0, 0.5],
            folds = 3,
            parallel = false,
            simultaneous = false,
        )
        @test nrow(grid) == 3
    end

    @testset "estimand from query" begin
        q = TotalEffectQuery(:A, :Y)
        est = estimand_from_query(q, [:W])
        @test est isa InterventionalMean
    end

    @testset "identification certificate" begin
        g = SimpleDiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 3)
        add_edge!(g, 2, 3)
        res = identify(g, TotalEffectQuery(2, 3))
        cert = identification_certificate(res, :X, :Y; adjustment = [:Z])
        d = certificate_dict(cert)
        @test d["id_identifiable"] == true
    end

    @testset "mediation n_mc sweep" begin
        rng = StableRNG(4)
        df, _ = simulate_continuous_mtp_mediation(100; rng = rng)
        sweep = mediation_n_mc_sweep(
            df, :A, :Y;
            covar = [:W],
            mediators = [:M],
            n_mc_values = [16, 32],
            delta = 0.5,
            folds = 3,
        )
        @test nrow(sweep) >= 3
        @test length(unique(sweep.n_mc)) == 2
        summ = mediation_stability_summary(sweep)
        @test haskey(summ.sign_stable, "TE")
    end

    @testset "small-n profile" begin
        @test recommend_folds(20) == 2
        @test recommend_folds(50) == 3
        @test recommend_learners(20) == SMALL_N_SL_LEARNERS
        opts = recommend_run_options(25; engine = :mediation, n_mediators = 1)
        @test opts.folds == 2
        @test opts.parallel == false
        @test opts.n_mc >= 64
        @test opts.positivity == true
        @test adaptive_learners(15) == (:mean, :glm)
        @test normalize_engine(:crumble) === :mediation
    end

    @testset "positivity report" begin
        rng = StableRNG(2)
        df, _ = simulate_linear_mtp(80; rng = rng)
        rep = positivity_report(df, :A; deltas = [-1.0, 0.0, 1.0])
        @test nrow(rep) == 3
        @test "support_status" in names(rep)
        md = positivity_markdown(rep)
        @test occursin("Positivity", md)
    end

    @testset "sensitivity" begin
        tip = tipping_point_bias(0.5, 0.1)
        @test tip.ci_excludes_zero
        @test tip.bias_needed > 0
        rep = sensitivity_report(0.5, 0.1; n = 40)
        @test nrow(rep) >= 2
        @test occursin("Sensitivity", sensitivity_markdown(rep))
    end

    @testset "shift policies" begin
        a = collect(1.0:10.0)
        pol = multiplicative_shift_policy()
        out = apply_policy_values(a, 0.1, pol)
        @test length(out) == 10
        thr = threshold_shift_policy()
        out2 = apply_policy_values(a, 1.0, thr)
        @test length(out2) == 10
    end

    @testset "discovery sensitivity" begin
        d = adjustment_set_disagreement([:W, :Z], [:W, :U])
        @test :Z in d.only_user
        @test :U in d.only_alt
        cert = Dict{String, Any}()
        merge_discovery_sensitivity!(cert, [:W], [:W])
        @test cert["id_sensitivity_agree"] == true
    end

    @testset "mediation fold cache" begin
        rng = StableRNG(5)
        df, _ = simulate_continuous_mtp_mediation(80; rng = rng)
        cache = build_mediation_fold_cache(
            df, :Y, :A, [:W], [:M], 2, rng; learners = SMALL_N_SL_LEARNERS,
        )
        @test length(cache.fold_sets) == 2
        g1 = run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [0.5], folds = 2, n_mc = 8,
            learners = SMALL_N_SL_LEARNERS,
            cache_nuisances = true, parallel = false, rng = rng,
        )
        g2 = run_crumble_grid(  # legacy alias
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [0.5], folds = 2, n_mc = 8,
            learners = SMALL_N_SL_LEARNERS,
            cache_nuisances = false, parallel = false, rng = StableRNG(5),
        )
        @test nrow(g1) == nrow(g2)
    end

    @testset "sequential LMTP" begin
        rng = StableRNG(6)
        n = 60
        W = randn(rng, n)
        A1 = 0.5 .* W .+ randn(rng, n)
        L1 = 0.3 .* A1 .+ randn(rng, n)
        A2 = 0.4 .* L1 .+ 0.2 .* W .+ randn(rng, n)
        Y = 0.5 .* A2 .+ 0.3 .* A1 .+ 0.2 .* W .+ randn(rng, n)
        df = DataFrame(W = W, A1 = A1, L1 = L1, A2 = A2, Y = Y)
        res = run_sequential_lmtp(
            df, [:A1, :A2], :Y;
            baseline = [:W],
            time_vary = [Symbol[], [:L1]],
            delta = 0.5,
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            rng = rng,
        )
        @test isfinite(res.estimate)
        @test res.times == 2
        est = SequentialPolicy([:A1, :A2], :Y, [:W]; time_vary = [Symbol[], [:L1]])
        @test estimand_engine(est) == :sequential_lmtp
        plan = plan_mtp(est, df; deltas = [0.5], folds = 2)
        @test plan.certificate.temporal_lags !== nothing
        @test plan.certificate.result.query isa TemporalEffectQuery
        # TemporalEffectQuery ID on unrolled DAG feeds the certificate
        spec = TemporalDAGSpec([:a, :l, :y, :w], [
            LaggedEdge(:w, :a, 0),
            LaggedEdge(:a, :l, 0),
            LaggedEdge(:l, :a, 1),
            LaggedEdge(:a, :y, 0),
            LaggedEdge(:w, :y, 0),
        ])
        u = unroll_temporal_dag(spec, 2)
        tq = TemporalEffectQuery(:a, :y, 1, 2)
        cert = sequential_identification_certificate(u, tq)
        @test cert.result.strategy == :temporal_backdoor
        @test cert.temporal_lags.treat_lag == 1
        @test cert.temporal_lags.outcome_lag == 2
        plan2 = plan_mtp(est, df; id_result = cert.result, deltas = [0.5], folds = 2)
        @test plan2.certificate.result.strategy == :temporal_backdoor
    end

    @testset "expanded synthetic recovery" begin
        # Core recovery at moderate n (lean learners)
        r1 = run_julia_synthetic_once(:linear_mtp; n = 300, delta = 1.0, folds = 3,
            rng = StableRNG(1), learners = (:glm, :mean))
        @test only(r1.abs_error) < 0.12

        r2 = run_julia_synthetic_once(:binary_mediation; n = 400, folds = 3,
            rng = StableRNG(2), learners = (:glm, :mean), n_mc = 16)
        @test maximum(r2.abs_error) < 0.20

        r3 = run_julia_synthetic_once(:continuous_mtp_mediation; n = 500, delta = 1.0,
            folds = 3, rng = StableRNG(3), learners = (:glm, :mean), n_mc = 24)
        # TE can be noisier than NDE under nested MC; require TE direction + bound
        te = only(r3[r3.estimand .== "TE", :abs_error])
        @test te < 0.50
        @test only(r3[r3.estimand .== "TE", :sign_agree])

        # Stress: misspecification still recovers A coefficient directionally
        r4 = run_julia_synthetic_once(:misspecified_nuisance_mtp; n = 400, delta = 1.0,
            folds = 3, rng = StableRNG(13), learners = (:glm, :mean))
        @test only(r4.sign_agree)

        # Intermediate confounding oracle is finite
        df, truth = simulate_intermediate_confounding_mediation(200; rng = StableRNG(12))
        ora = truth.oracle(1.0)
        @test isfinite(ora.te) && isfinite(ora.nde) && isfinite(ora.nie)

        # Weak positivity: estimate finite (error may be large)
        r5 = run_julia_synthetic_once(:weak_positivity_mtp; n = 250, delta = 1.0,
            folds = 2, rng = StableRNG(11), learners = (:glm, :mean))
        @test isfinite(only(r5.estimate))
    end

    # =====================================================================
    # Improvement-driving tests (synthetic recovery)
    # =====================================================================

    @testset "learner richness" begin
        # GLM-only cannot capture Y = β_a A + β_w2 W² + ε
        r_glm = run_julia_synthetic_once(:misspecified_nuisance_mtp; n = 400, delta = 1.0,
            folds = 3, rng = StableRNG(13), learners = (:glm, :mean))
        r_rich = run_julia_synthetic_once(:misspecified_nuisance_mtp; n = 400, delta = 1.0,
            folds = 3, rng = StableRNG(13), learners = DEFAULT_SL_LEARNERS)
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
            r = run_julia_synthetic_once(:linear_mtp; n = 300, delta = 1.0,
                folds = 3, rng = StableRNG(seed), learners = (:glm, :mean))
            cov = only(r.cover_95)
            push!(lmtp_covers, cov === missing ? false : Bool(cov))
            push!(lmtp_ses, only(r.se))
        end
        # SEs should be non-degenerate
        @test all(se -> se > 0.01, lmtp_ses)
        # Coverage >= 0.6 catches gross undercoverage (true target: 0.95)
        @test mean(lmtp_covers) >= 0.6

        # Binary mediation coverage across seeds
        med_covers = Bool[]
        med_ses = Float64[]
        for seed in 1:n_seeds
            r = run_julia_synthetic_once(:binary_mediation; n = 300, folds = 3,
                rng = StableRNG(seed), learners = (:glm, :mean), n_mc = 16)
            te_row = only(eachrow(filter(row -> row.estimand == "TE", r)))
            cov = te_row.cover_95
            push!(med_covers, cov === missing ? false : Bool(cov))
            push!(med_ses, te_row.se)
        end
        @test all(se -> se > 0.005, med_ses)
        @test mean(med_covers) >= 0.6
    end

    @testset "mediator MC convergence" begin
        n_mc_values = [1, 16, 64, 128]
        te_errors = Dict{Int, Float64}()
        nde_errors = Dict{Int, Float64}()
        for nmc in n_mc_values
            r = run_julia_synthetic_once(:continuous_mtp_mediation; n = 500, delta = 1.0,
                folds = 3, rng = StableRNG(3), learners = (:glm, :mean), n_mc = nmc)
            te_errors[nmc] = only(r[r.estimand .== "TE", :abs_error])
            nde_errors[nmc] = only(r[r.estimand .== "NDE", :abs_error])
        end
        # MC (n_mc=128) should beat pure plugin (n_mc=1) for TE
        @test te_errors[128] < te_errors[1]
        # Monotone improvement from 16→128 (on average)
        @test te_errors[128] <= te_errors[16]
        # NDE should also improve separately (not just TE by cancellation)
        @test nde_errors[128] < nde_errors[1]
        # Larger sample with moderate MC should get TE |err| < 0.20
        r_big = run_julia_synthetic_once(:continuous_mtp_mediation; n = 800, delta = 1.0,
            folds = 3, rng = StableRNG(3), learners = (:glm, :mean), n_mc = 64)
        @test_broken only(r_big[r_big.estimand .== "TE", :abs_error]) < 0.20
    end

    @testset "intermediate confounder adjustment" begin
        df, truth = simulate_intermediate_confounding_mediation(400; rng = StableRNG(12))
        eff = effective_sd_shift(df.A, 1.0)
        ora = truth.oracle(1.0)

        # Oracle sanity: path formula and shared-noise oracle should roughly agree
        path_te = truth.effects(ora.eff).te
        @test abs(ora.te - path_te) < 0.15

        # Run with covar=[:W] only (current default — ignores L)
        r_w = run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [1.0], folds = 3, n_mc = 32,
            learners = (:glm, :mean),
            parallel = false, cache_nuisances = false, rng = StableRNG(12),
        )
        nde_w = only(filter(row -> row.estimand == "NDE", eachrow(r_w))).est
        nde_err_w = abs(nde_w - ora.nde)

        # Run with covar=[:W, :L] (conditioning on intermediate confounder)
        r_wl = run_mediation_grid(
            df, :A, :Y;
            covar = [:W, :L], mediators = [:M],
            deltas = [1.0], folds = 3, n_mc = 32,
            learners = (:glm, :mean),
            parallel = false, cache_nuisances = false, rng = StableRNG(12),
        )
        nde_wl = only(filter(row -> row.estimand == "NDE", eachrow(r_wl))).est
        nde_err_wl = abs(nde_wl - ora.nde)

        # Conditioning on L should reduce NDE bias
        @test_broken nde_err_wl < nde_err_w
        # Both should produce finite results
        @test isfinite(nde_w)
        @test isfinite(nde_wl)
    end

    @testset "sqrt-n convergence" begin
        # Linear MTP: average |err| over 5 seeds at each n to smooth luck
        function mean_err_lmtp(n_obs, seeds)
            mean([begin
                r = run_julia_synthetic_once(:linear_mtp; n = n_obs, delta = 1.0,
                    folds = 3, rng = StableRNG(s), learners = (:glm, :mean))
                only(r.abs_error)
            end for s in seeds])
        end
        seeds = 1:5
        err_small = mean_err_lmtp(200, seeds)
        err_large = mean_err_lmtp(3200, seeds)
        # Larger n should have lower mean |err|
        @test err_large < err_small
        # Tight at large n
        @test err_large < 0.03

        # Binary mediation: TE should improve with n (averaged over seeds)
        function mean_err_med(n_obs, seeds)
            mean([begin
                r = run_julia_synthetic_once(:binary_mediation; n = n_obs, folds = 3,
                    rng = StableRNG(s), learners = (:glm, :mean), n_mc = 16)
                only(r[r.estimand .== "TE", :abs_error])
            end for s in seeds])
        end
        @test mean_err_med(800, seeds) < mean_err_med(200, seeds)
    end

    # =====================================================================
    # New capability axes: nonlinearity, missing data, g-comp, DiD
    # =====================================================================

    @testset "nonlinear interaction recovery" begin
        df, truth = simulate_nonlinear_interaction_mtp(500; rng = StableRNG(20))
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

        # Default learners (includes :glm_interact)
        grid_rich = run_lmtp_grid(
            df, :A, :Y; baseline = [:W1, :W2], deltas = [1.0 * sdA],
            folds = 3, parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(20), shift_scale = "raw",
        )
        err_rich = abs(only(grid_rich.est) - t.te)

        # Richer library should do better on interaction DGP
        @test err_rich < err_glm
        # Both should produce finite estimates
        @test isfinite(only(grid_glm.est))
        @test isfinite(only(grid_rich.est))
        # Direction should be correct
        @test only(grid_rich.est) * t.te > 0 || abs(t.te) < 0.05
    end

    @testset "small-n stability" begin
        for n_small in [20, 40, 80]
            r = run_julia_synthetic_once(:linear_mtp; n = n_small, delta = 1.0,
                folds = recommend_folds(n_small), rng = StableRNG(1),
                learners = recommend_learners(n_small))
            @test isfinite(only(r.estimate))
            @test only(r.se) > 0.001
        end
        # recommend_folds gives sensible values
        @test recommend_folds(20) == 2
        @test recommend_folds(50) == 3
        @test recommend_folds(200) == 5
        # recommend_learners drops risky learners at small n
        @test :evotree ∉ recommend_learners(30)
        @test :glmnet ∈ recommend_learners(30)
    end

    @testset "missing outcome IPCW" begin
        df, truth = simulate_missing_outcome_mtp(400; rng = StableRNG(30))
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
        # IPCW should work (may or may not beat CC at this n)
        @test err_ipcw < 0.20
    end

    @testset "missing covariate imputation" begin
        df, truth = simulate_missing_covariate_mtp(400; rng = StableRNG(31))
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
        df, truth = simulate_gcomp_nonlinear(500; rng = StableRNG(50))

        res = run_gcomp(df, :A, :Y; covariates = [:W], folds = 3,
                        learners = DEFAULT_SL_LEARNERS, rng = StableRNG(50), n_boot = 100)
        @test isfinite(res.estimate)
        @test abs(res.estimate - truth.ate) < 0.20
        # Bootstrap CI should cover truth
        @test res.ci_lower < truth.ate < res.ci_upper
        @test res.se > 0.01
    end

    @testset "DiD 2x2 recovery" begin
        df, truth = simulate_did_2x2(200; τ = 1.0, rng = StableRNG(40))
        res = run_did_2x2(df)
        @test abs(res.att - truth.att) < 0.20
        @test res.se > 0.01
        @test res.ci_lower < truth.att < res.ci_upper
    end

    @testset "DiD staggered recovery" begin
        df, truth = simulate_did_staggered(400; rng = StableRNG(41))
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
        df, truth = simulate_smooth_nonlinear_mtp(80; rng = StableRNG(21))
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

    @testset "MLJFlux optional learners" begin
        # Extension loads only when MLJFlux is available in the environment.
        using MLJFlux
        ext = Base.get_extension(CausalTargeted, :CausalTargetedMLJFluxExt)
        @test ext !== nothing
        X = randn(StableRNG(7), 60, 3)
        y = Float64.(sin.(X[:, 1]) .+ 0.1 .* randn(StableRNG(8), 60))
        sl = fit_super_learner(
            X, y;
            learners = (:mlj_mlp, :mean),
            family = :gaussian,
            folds = 2,
            metalearner = :invmse,
            rng = StableRNG(9),
        )
        pred = predict_super_learner(sl, X)
        @test length(pred) == 60
        @test all(isfinite, pred)

        s = Float64.(randn(StableRNG(10), 60) .> 0.0)
        slb = fit_super_learner(
            X, s;
            learners = (:mlj_nn_binary, :mean),
            family = :binomial,
            folds = 2,
            metalearner = :invmse,
            rng = StableRNG(11),
        )
        pb = predict_super_learner(slb, X)
        @test length(pb) == 60
        @test all(0.0 .< pb .< 1.0)
    end
end
