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

        # Certificate → SequentialPolicy bridge (mirror mediation spec_from_identification)
        spec_auto = sequential_spec_from_identification(
            cert.result;
            treatments = [:A1, :A2],
            outcome = :Y,
            baseline = [:W],
            time_vary = [Symbol[], [:L1]],
        )
        @test spec_auto.treatments == [:A1, :A2]
        @test spec_auto.outcome === :Y
        merged = plan_sequential(
            SequentialPolicy([:A1, :A2], :Y, Symbol[]; time_vary = [Symbol[], [:L1]]),
            cert.result,
        )
        @test merged.baseline == Symbol.(cert.result.adjustment)
        grid = execute_estimand(
            spec_auto, df;
            id_result = cert.result,
            deltas = [0.5],
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            metadata = false,
            rng = StableRNG(8),
        )
        @test nrow(grid) >= 1
        @test isfinite(only(grid.est))
    end

    @testset "simulate_panel → sequential LMTP" begin
        _HAS_PANEL_API || begin
            @info "Skipping simulate_panel bridge (CausalDynamics panel API not loaded)"
            return
        end
        cdm = DiscreteTimeCDM(
            [:w, :a, :l, :y];
            initialise = (rng) -> (w = randn(rng), a = 0.0, l = 0.0, y = 0.0),
            sample_noise = (rng, state, t) -> (
                u_a = randn(rng), u_l = randn(rng), u_y = randn(rng),
            ),
            step = (state, t, noise, intervention) -> begin
                a = intervention_value(intervention, :a, t, 0.5 * state.w + noise.u_a)
                l = 0.3 * a + noise.u_l
                y = 0.4 * a + 0.2 * l + 0.1 * state.w + noise.u_y
                (w = state.w, a = a, l = l, y = y)
            end,
        )
        panel = CausalDynamics.simulate_panel(
            cdm, 80, 2;
            rng = StableRNG(9),
            baseline = [:w],
            timed = [:a, :l],
            terminal = [:y],
        )
        dfp = DataFrame(NamedTuple(panel))
        spec = TemporalDAGSpec([:a, :l, :y, :w], [
            LaggedEdge(:w, :a, 0),
            LaggedEdge(:a, :l, 0),
            LaggedEdge(:l, :a, 1),
            LaggedEdge(:a, :y, 0),
            LaggedEdge(:w, :y, 0),
        ])
        u = unroll_temporal_dag(spec, 2)
        tq = TemporalEffectQuery(:a, :y, 1, 2)
        id = identify(u, tq)
        est = sequential_spec_from_identification(
            id;
            baseline = [:w],
            time_vary = [Symbol[], [:l1]],
            shift = additive_shift_policy(; lower_q = 0.01, upper_q = 0.99),
        )
        @test est.treatments == [:a1, :a2]
        @test est.outcome === :y
        res = run_sequential_lmtp(
            dfp, est;
            delta = 0.5,
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            rng = StableRNG(10),
        )
        @test isfinite(res.estimate)
        @test res.times == 2
    end

    @testset "sequential LMTP with missing outcome" begin
        rng = StableRNG(26)
        n = 50
        W = randn(rng, n)
        A1 = 0.5 .* W .+ randn(rng, n)
        L1 = 0.3 .* A1 .+ randn(rng, n)
        A2 = 0.4 .* L1 .+ randn(rng, n)
        Y = Vector{Union{Float64, Missing}}(0.5 .* A2 .+ 0.3 .* A1 .+ 0.2 .* W .+ randn(rng, n))
        Y[1:10] .= missing
        df = DataFrame(W = W, A1 = A1, L1 = L1, A2 = A2, Y = Y)
        res = run_sequential_lmtp(
            df, [:A1, :A2], :Y;
            baseline = [:W],
            time_vary = [Symbol[], [:L1]],
            delta = 0.25,
            folds = 2,
            learners = SMALL_N_SL_LEARNERS,
            handle_missing = :drop,
            rng = StableRNG(27),
        )
        @test isfinite(res.estimate)
        @test res.se > 0
    end

    @testset "sequential factor LMTP" begin
        df, truth = simulate_sequential_factor_mtp(400; rng = StableRNG(71))
        policy = discrete_recode_policy(truth.recode)
        res = run_sequential_lmtp(
            df, [:A1, :A2], :Y;
            baseline = [:W],
            time_vary = [Symbol[], [:L1]],
            policies = [policy],
            folds = 3,
            learners = (:glm, :mean),
            rng = StableRNG(72),
        )
        @test isfinite(res.estimate)
        @test isfinite(res.se)
        @test res.times == 2
        @test res.density_ratio === :classification
        @test res.positivity.ok
        @test abs(res.estimate - truth.psi) < 0.25

        r = CausalTargeted.run_julia_synthetic_once(
            :sequential_factor_mtp; n = 400, folds = 3,
            rng = StableRNG(73), learners = (:glm, :mean),
        )
        @test only(r.abs_error) < 0.25

        # Time-specific recode: identity at t=1, 2→1 at t=2
        A2d = [a == "2" ? "1" : a for a in df.A2]
        psi_t2 = mean(CausalTargeted._sequential_factor_structural_mean(
            df.A1, A2d, df.W;
            β1 = truth.β1, β2 = truth.β2, α1 = truth.α1, α2 = truth.α2, β_w = truth.β_w,
        ))
        res_t2 = run_sequential_lmtp(
            df, [:A1, :A2], :Y;
            baseline = [:W],
            time_vary = [Symbol[], [:L1]],
            policies = [
                discrete_recode_policy(Dict{String, String}()),
                discrete_recode_policy(truth.recode),
            ],
            folds = 3,
            learners = (:glm, :mean),
            rng = StableRNG(74),
        )
        @test isfinite(res_t2.estimate)
        @test abs(res_t2.estimate - psi_t2) < 0.30

        mixed = DataFrame(W = df.W, A1 = df.A1, L1 = df.L1, A2 = randn(StableRNG(75), nrow(df)), Y = df.Y)
        err_mixed = try
            run_sequential_lmtp(
                mixed, [:A1, :A2], :Y;
                baseline = [:W],
                time_vary = [Symbol[], [:L1]],
                policies = [policy],
                folds = 2,
                learners = (:glm, :mean),
                rng = StableRNG(76),
            )
            nothing
        catch e
            e
        end
        @test err_mixed isa ArgumentError
        @test occursin("mix", sprint(showerror, err_mixed)) ||
            occursin("continuous", sprint(showerror, err_mixed))

        est = SequentialPolicy(
            [:A1, :A2], :Y, [:W];
            time_vary = [Symbol[], [:L1]],
            policies = policy,
        )
        @test length(est.policies) == 2
        grid = execute_estimand(est, df; folds = 2, rng = StableRNG(77), metadata = true)
        @test isfinite(only(grid.est))
        @test only(grid.meta_density_ratio) == "classification"
        @test only(grid.meta_engine) == "sequential_lmtp"
    end
