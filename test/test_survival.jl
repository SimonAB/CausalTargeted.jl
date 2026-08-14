@testset "survival / event-time LMTP" begin
    @testset "run_survival_lmtp finite + directional" begin
        rng = StableRNG(11)
        df, truth = CausalTargeted.simulate_discrete_survival_mtp(
            180; T = 3, β_a = -0.6, rng = rng,
        )
        shift = additive_shift_policy(; scale = "raw", lower_q = 0.0, upper_q = 1.0)
        res0 = run_survival_lmtp(
            df, truth.treatments, truth.surv;
            baseline = [:W],
            delta = 0.0,
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            shift = shift,
            rng = StableRNG(12),
        )
        res1 = run_survival_lmtp(
            df, truth.treatments, truth.surv;
            baseline = [:W],
            delta = 0.8,
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            shift = shift,
            rng = StableRNG(13),
        )
        @test isfinite(res0.estimate)
        @test isfinite(res1.estimate)
        @test res0.horizon == 3
        @test res0.estimand === :survival
        # β_a < 0 ⇒ raising A lowers hazard ⇒ higher event-free probability
        @test res1.estimate > res0.estimate - 0.05

        est = SurvivalPolicy(
            truth.treatments, truth.surv, [:W];
            shift = shift, horizon = 3,
        )
        @test estimand_engine(est) == :survival_lmtp
        plan = plan_mtp(est, df; deltas = [0.8], folds = 2)
        @test plan.certificate.temporal_lags.outcome_lag == 3
        @test plan.certificate.result.query isa TemporalEffectQuery

        grid = execute_estimand(
            est, df;
            deltas = [0.8],
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            metadata = false,
            rng = StableRNG(14),
        )
        @test nrow(grid) == 1
        @test only(grid.estimand) == "survival"
        @test isfinite(only(grid.est))
    end

    @testset "survival identification certificate" begin
        spec = TemporalDAGSpec([:a, :s, :w], [
            LaggedEdge(:w, :a, 0),
            LaggedEdge(:a, :s, 0),
            LaggedEdge(:w, :s, 0),
            LaggedEdge(:s, :s, 1),
        ])
        u = unroll_temporal_dag(spec, 3)
        tq = TemporalEffectQuery(:a, :s, 1, 3)
        cert = survival_identification_certificate(u, tq)
        @test cert.result.strategy == :temporal_backdoor
        @test cert.temporal_lags.outcome_lag == 3
    end

    @testset "soft oracle recovery (raw shift)" begin
        rng = StableRNG(21)
        df, truth = CausalTargeted.simulate_discrete_survival_mtp(
            400; T = 2, β_a = -0.7, α = -0.8, rng = rng,
        )
        δ = 0.75
        shift = additive_shift_policy(; scale = "raw", lower_q = 0.0, upper_q = 1.0)
        res = run_survival_lmtp(
            df, truth.treatments, truth.surv;
            baseline = [:W],
            delta = δ,
            folds = 3,
            learners = SMALL_N_SL_LEARNERS,
            shift = shift,
            rng = StableRNG(22),
        )
        ψ = truth.survival(δ)
        @test abs(res.estimate - ψ) < 0.20
    end

    @testset "survival LMTP with missing terminal S" begin
        rng = StableRNG(33)
        df, truth = CausalTargeted.simulate_discrete_survival_mtp(
            120; T = 2, rng = rng,
        )
        Sh = truth.surv[end]
        df[!, Sh] = Vector{Union{Float64, Missing}}(df[!, Sh])
        df[1:20, Sh] .= missing
        shift = additive_shift_policy(; scale = "raw", lower_q = 0.0, upper_q = 1.0)
        res_drop = run_survival_lmtp(
            df, truth.treatments, truth.surv;
            baseline = [:W], delta = 0.25, folds = 2,
            learners = SMALL_N_SL_LEARNERS, shift = shift,
            handle_missing = :drop, rng = StableRNG(34),
        )
        res_ipcw = run_survival_lmtp(
            df, truth.treatments, truth.surv;
            baseline = [:W], delta = 0.25, folds = 2,
            learners = SMALL_N_SL_LEARNERS, shift = shift,
            handle_missing = :ipcw, rng = StableRNG(34),
        )
        @test isfinite(res_drop.estimate)
        @test isfinite(res_ipcw.estimate)
        @test !isapprox(res_drop.estimate, res_ipcw.estimate; atol = 1e-10)
    end
end
