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

    @testset "Hajek IF SE matches returned IC (CT#15)" begin
        ψ = [1.0, 2.0, 3.0, 4.0]
        w = [1.0, 1.0, 2.0, 2.0]
        s = weighted_influence_summary(ψ, w)
        @test s.estimate ≈ sum(w .* ψ) / sum(w)
        w̄ = mean(w)
        ic_exp = (w ./ w̄) .* (ψ .- s.estimate)
        @test s.ic ≈ ic_exp
        @test s.se ≈ sqrt(mean(abs2, ic_exp) / length(ψ))
    end

    @testset "survival censoring IPCW not squared (CT#16)" begin
        # Document the algebra: Q already includes censor weights, so reweighting
        # by the same vector would square them.
        S = [1.0, 1.0, 0.0, 1.0]
        c = [2.0, 0.5, 0.0, 1.5]
        Q = S .* c
        @test mean(Q) ≉ CausalTargeted.transport_weighted_mean(Q, c)

        rng = StableRNG(60)
        df, truth = CausalTargeted.simulate_discrete_survival_mtp(
            220; T = 2, β_a = -0.5, rng = rng,
        )
        n = nrow(df)
        crng = StableRNG(61)
        df.C1 = Float64.(rand(crng, n) .< 0.12)
        df.C2 = Float64.((df.C1 .> 0.5) .| (rand(crng, n) .< 0.08))
        shift = additive_shift_policy(; scale = "raw", lower_q = 0.0, upper_q = 1.0)
        res = run_survival_lmtp(
            df, truth.treatments, truth.surv;
            baseline = [:W], censor = [:C1, :C2], delta = 0.25, folds = 2,
            learners = SMALL_N_SL_LEARNERS, shift = shift,
            handle_missing = :drop, rng = StableRNG(62),
        )
        @test isfinite(res.estimate)
        @test 0.0 <= res.estimate <= 1.0

        # Plugin reference: terminal S already multiplied by censor IPCW in Q.
        # Double-weighting that Q by the same censor vector must not be the estimand.
        ipcw_c = CausalTargeted._censor_ipcw(
            df, [:C1, :C2], [:W], 2;
            learners = SMALL_N_SL_LEARNERS, rng = StableRNG(62),
        )
        Q_term = Float64.(df[!, truth.surv[end]]) .* ipcw_c
        ψ_plugin = mean(Q_term)
        ψ_double = CausalTargeted.transport_weighted_mean(Q_term, ipcw_c)
        @test ψ_plugin ≉ ψ_double atol = 1e-8
        # With drop missingness weights, summary must not land on the squared path.
        @test abs(res.estimate - ψ_double) > abs(res.estimate - ψ_plugin) ||
              abs(res.estimate - ψ_plugin) < 0.35
    end
end
