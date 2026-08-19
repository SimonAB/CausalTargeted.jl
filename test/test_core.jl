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

    @testset "SuperLearnerFit and job RNG" begin
        X = randn(StableRNG(3), 40, 2)
        y = X[:, 1] .+ 0.1 .* randn(StableRNG(4), 40)
        sl = fit_super_learner(X, y; learners = (:glm, :mean), folds = 2, rng = StableRNG(5))
        @test sl isa SuperLearnerFit
        @test length(predict_super_learner(sl, X)) == 40
        @test !(:evotree in DEFAULT_SL_LEARNERS)
        @test :evotree in RICH_SL_LEARNERS
        s1 = CausalTargeted._job_rng(UInt(1), 1, "full_population", 0.5)
        s2 = CausalTargeted._job_rng(UInt(1), 2, "full_population", 0.5)
        @test rand(s1) != rand(s2)
        df, _ = simulate_linear_mtp(80; rng = StableRNG(11))
        g_a = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W], deltas = [0.5], folds = 2,
            learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(12),
        )
        g_b = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W], deltas = [0.5], folds = 2,
            learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(12),
        )
        @test g_a.est ≈ g_b.est
    end

    @testset "design_matrix preallocation" begin
        df = DataFrame(A = [0.0, 1.0], W = [2.0, 3.0], Z = [4.0, 5.0])
        X = design_matrix(df, [:W, :Z]; treatment = :A)
        @test size(X) == (2, 4)
        @test X[:, 1] == [1.0, 1.0]
        @test X[:, 2] == [0.0, 1.0]
        @test X[:, 3] == [2.0, 3.0]
        @test X[:, 4] == [4.0, 5.0]
        Xtv = design_matrix(df, [:W]; treatment = :A, treatment_values = [9.0, 8.0])
        @test Xtv[:, 2] == [9.0, 8.0]
        Xb = [1.0 2.0; 3.0 4.0]
        Xi = CausalTargeted._expand_interactions(Xb)
        @test size(Xi) == (2, 3)
        @test Xi[:, 1:2] == Xb
        @test Xi[:, 3] ≈ [2.0, 12.0]
        Xq = CausalTargeted._expand_quadratic(Xb)
        @test size(Xq) == (2, 4)
        @test Xq[:, 1:2] == Xb
        @test Xq[:, 3:4] ≈ [1.0 4.0; 9.0 16.0]
        @test CausalTargeted._expand_interactions(ones(2, 1)) == ones(2, 1)
        df2 = DataFrame(A = [0.0, 1.0, 0.5], W = [1.0, 2.0, 3.0])
        W = covariate_design_matrix(df2, [:W])
        X_asm = outcome_design_matrix(W, df2.A)
        X_df = design_matrix(df2, [:W]; treatment = :A)
        @test X_asm ≈ X_df
        X_cf = outcome_design_matrix(W, [1.0, 1.0, 1.0])
        @test X_cf ≈ design_matrix(df2, [:W]; treatment = :A, treatment_values = [1.0, 1.0, 1.0])
        df3 = DataFrame(A = [0.0, 1.0, 0.5], Y = [1.0, 2.0, 1.5], W = [1.0, 2.0, 3.0])
        m3 = fit_outcome_regression(df3, :Y, :A, [:W], 2, StableRNG(2); learners = (:glm, :mean))
        p_obs = predict_outcome(m3, df3)
        p_cf = predict_outcome(m3, df3; treatment_values = ones(3))
        @test length(p_obs) == 3
        @test length(p_cf) == 3
        @test size(m3.W) == (3, 2)
    end

    @testset "estimand from query" begin
        q = TotalEffectQuery(:A, :Y)
        est = estimand_from_query(q, [:W])
        @test est isa InterventionalMean

        recode = discrete_recode_policy(Dict("2" => "1"))
        q_pol = InterventionalPolicyQuery(:A, :Y; shift = recode)
        est_disc = estimand_from_query(q_pol, [:W])
        @test est_disc isa DiscreteInterventionalMean
        @test est_disc.policy === recode

        est_kw = estimand_from_query(InterventionalPolicyQuery(:A, :Y), [:W]; policies = recode)
        @test est_kw isa DiscreteInterventionalMean

        tq = TemporalEffectQuery(:a, :y, 1, 2)
        est_long = estimand_from_query(tq, [:W])
        @test est_long isa LongitudinalPolicy
        @test est_long.treat_lag == 1
        @test est_long.outcome_lag == 2

        est_seq = estimand_from_query(
            tq, [:W];
            treatments = [:A1, :A2],
            time_vary = [Symbol[], [:L1]],
            policies = recode,
        )
        @test est_seq isa SequentialPolicy
        @test est_seq.treatments == [:A1, :A2]
        @test length(est_seq.policies) == 2

        err = try
            estimand_from_query(tq, [:W]; policies = recode)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("treatments", sprint(showerror, err))
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
        if !_HAS_CAUSAL_MEDIATION
            @info "Skipping mediation n_mc sweep (CausalMediation not loaded)"
        else
            rng = StableRNG(4)
            df, _ = CausalTargeted.simulate_continuous_mtp_mediation(100; rng = rng)
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
    end

    @testset "mean-only contrast guard" begin
        df = DataFrame(A = rand(StableRNG(99), [0, 1], 40), Y = randn(StableRNG(99), 40))
        @test_throws ArgumentError run_gcomp(
            df, :A, :Y;
            covariates = Symbol[], folds = 2, learners = (:mean,), rng = StableRNG(99),
        )
        res = run_gcomp(
            df, :A, :Y;
            covariates = Symbol[], folds = 2, learners = (:glm, :mean),
            rng = StableRNG(99), n_boot = 0,
        )
        @test isfinite(res.estimate)
    end

    @testset "gcomp refitting bootstrap SE (CT#13)" begin
        df, truth = simulate_linear_mtp(200; rng = StableRNG(101))
        δ = 0.5
        r_if = run_gcomp(
            df, :A, :Y;
            covariates = [:W], delta = δ, folds = 2,
            learners = (:glm, :mean), rng = StableRNG(102), n_boot = 0,
        )
        r_boot = run_gcomp(
            df, :A, :Y;
            covariates = [:W], delta = δ, folds = 2,
            learners = (:glm, :mean), rng = StableRNG(102), n_boot = 40,
        )
        ψ = truth_shift_effect(truth, δ)
        @test isapprox(r_if.estimate, r_boot.estimate; atol = 1e-12)
        @test abs(r_boot.estimate - ψ) < 0.05
        # Refitting bootstrap SE must leave room for outcome-model uncertainty
        @test r_boot.se > 0.01
        @test r_boot.se > r_if.se
        @test r_boot.ci_lower < ψ < r_boot.ci_upper
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
