using CausalTargeted
using CausalDynamics: identify, TotalEffectQuery, TemporalEffectQuery, TemporalDAGSpec, LaggedEdge, unroll_temporal_dag
using DataFrames
using Graphs
using Random
using StableRNGs
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
end
