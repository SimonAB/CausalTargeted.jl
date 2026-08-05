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
