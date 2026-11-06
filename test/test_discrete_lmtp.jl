@testset "discrete treatment LMTP" begin
    df, truth = simulate_categorical_treatment_mtp(400; rng = StableRNG(51))
    policy = discrete_recode_policy(truth.recode)
    res = run_discrete_lmtp(
        df, :A, :Y;
        policy = policy,
        baseline = [:W],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(52),
    )
    @test isfinite(res.estimate)
    @test isfinite(res.se)
    @test abs(res.estimate - truth.te) < 0.20
    @test res.positivity.ok
    @test_throws ArgumentError run_discrete_lmtp(
        df, :A, :Y;
        policy = policy,
        baseline = [:W],
        density_ratio = :gaussian,
        folds = 2,
        rng = StableRNG(52),
    )

    r = CausalTargeted.run_julia_synthetic_once(
        :categorical_treatment_mtp; n = 400, folds = 3,
        rng = StableRNG(53), learners = (:glm, :mean),
    )
    @test only(r.abs_error) < 0.20

    empty_policy = discrete_recode_policy(Dict("0" => "3", "1" => "3", "2" => "3"))
    empty_res = run_discrete_lmtp(
        df, :A, :Y;
        policy = empty_policy,
        baseline = [:W],
        folds = 2,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(54),
    )
    @test isfinite(empty_res.estimate)
    @test !empty_res.positivity.ok
    @test "3" in empty_res.positivity.empty_support
    @test isnan(empty_res.se)
    @test empty_res.inference_status == :positivity_violation

    estimand = DiscreteInterventionalMean(:A, :Y, [:W], policy)
    grid = execute_estimand(estimand, df; folds = 2, rng = StableRNG(55), metadata = false)
    @test isfinite(only(grid.est))

    df_int = copy(df)
    df_int.A = parse.(Int, string.(df.A))
    res_int = run_discrete_lmtp(
        df_int, :A, :Y;
        policy = discrete_recode_policy(Dict(2 => 1)),
        baseline = [:W],
        folds = 2,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(56),
    )
    @test isfinite(res_int.estimate)
    @test res_int.n_changed == count(==(2), df_int.A)

    y_miss = Vector{Union{Float64, Missing}}(df.Y)
    y_miss[1:40] .= missing
    df_miss = copy(df)
    df_miss.Y = y_miss
    res_ipcw = run_discrete_lmtp(
        df_miss, :A, :Y;
        policy = policy,
        baseline = [:W],
        folds = 2,
        learners_outcome = (:glm, :mean),
        handle_missing = :ipcw,
        rng = StableRNG(57),
    )
    @test isfinite(res_ipcw.estimate)
    @test res_ipcw.se > 0

    static = discrete_static_policy("1"; levels = ["0", "1", "2"])
    @test static.mtp == false
    res_static = run_discrete_lmtp(
        df, :A, :Y;
        policy = static,
        baseline = [:W],
        folds = 2,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(58),
    )
    @test isfinite(res_static.estimate)
    @test res_static.positivity.ok

    grid_meta = execute_estimand(estimand, df; folds = 2, rng = StableRNG(59))
    @test only(grid_meta.meta_engine) == "discrete_lmtp"
    @test only(grid_meta.meta_density_ratio) == "classification"
    plan = plan_mtp(estimand, df; folds = 2)
    @test plan.estimated_fits == 4

    seq_df = DataFrame(
        W = df.W,
        A1 = df.A,
        A2 = df.A,
        Y = df.Y,
    )
    err = try
        run_sequential_lmtp(
            seq_df, [:A1, :A2], :Y;
            baseline = [:W],
            delta = 0.5,
            folds = 2,
            learners = (:glm, :mean),
            rng = StableRNG(60),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("run_discrete_lmtp", sprint(showerror, err))
    @test occursin("policies", sprint(showerror, err))
end

@testset "Apodemus-style three-arm discrete LMTP fixture" begin
    # Synthetic R / SS / SC panel: small n, imbalanced arms, cross-fitted folds.
    rng = StableRNG(61)
    n = 54
    arms = ["R", "SS", "SC"]
    A = vcat(fill("R", 26), fill("SS", 3), fill("SC", 25))
    Random.shuffle!(rng, A)
    df = DataFrame(
        mouse_id = string.(1:n),
        grid_type = A,
        fec2 = 0.3 .* (A .== "SS") .- 0.1 .* (A .== "SC") .+ 0.05 .* randn(rng, n),
    )
    opts = recommend_run_options(n; engine = :lmtp)
    contrast = run_discrete_lmtp_contrast(
        df, :grid_type, :fec2;
        arm_hi = "SS",
        arm_ref = "R",
        levels = arms,
        baseline = Symbol[],
        folds = opts.folds,
        learners_outcome = opts.learners_outcome,
        learners_trt = (:logistic, :mean),
        rng = StableRNG(62),
    )
    @test contrast.contrast == "SS_vs_R"
    @test isfinite(contrast.estimate)
    @test isfinite(contrast.se)
    @test contrast.se > 0
    @test contrast.positivity.ok
    @test contrast.hi.estimate - contrast.ref.estimate ≈ contrast.estimate
end

@testset "discrete LMTP cross-fold treatment levels" begin
    # Apodemus-style: three arms, small n; training folds may omit an arm seen at validation.
    rng = StableRNG(42)
    n = 60
    arms = ["R", "SS", "SC"]
    A = vcat(fill("R", 28), fill("SS", 4), fill("SC", 28))
    Random.shuffle!(rng, A)
    df = DataFrame(W = randn(rng, n), A = A, Y = randn(rng, n))
    policy = discrete_static_policy("SS"; levels = arms)
    est = run_discrete_lmtp(
        df, :A, :Y;
        policy = policy,
        baseline = [:W],
        folds = 5,
        learners_outcome = (:glm, :mean),
        learners_trt = (:logistic, :mean),
        rng = rng,
    )
    @test isfinite(est.estimate)
    @test isfinite(est.se)
end

@testset "DiscreteTimeCDM → discrete LMTP recovery" begin
    rng = StableRNG(99)
    T = 4
    truth_effect = 0.5
    cdm = DiscreteTimeCDM(
        [:grid_type, :fec];
        initialise = (rng) -> (grid_type = 0.0, fec = 0.0),
        sample_noise = (rng, state, t) -> (u_g = 0.0, u_f = randn(rng)),
        step = (state, t, noise, intervention) -> begin
            g = intervention_value(intervention, :grid_type, t, state.grid_type)
            fec = 0.2 * g + noise.u_f
            if t >= 2 && g == 1.0
                fec = fec + truth_effect
            end
            (grid_type = g, fec = fec)
        end,
    )
    n = 200
    arms = rand(rng, 0:1, n)
    rows = NamedTuple[]
    for i in 1:n
        gcode = Float64(arms[i])
        intervention = do_sequence(:grid_type, fill(gcode, T))
        traj = simulate(cdm, T; rng = rng, intervention = intervention)
        arm = gcode == 0.0 ? "R" : "SS"
        nt = (mouse_id = string(i), grid_type = arm)
        for t in 1:T
            c = panel_column_name(:fec, t)
            nt = merge(nt, NamedTuple{(c,)}((traj.series[:fec][t],)))
        end
        push!(rows, nt)
    end
    df = DataFrame(rows)
    col = panel_column_name(:fec, 2)
    res = run_discrete_lmtp_contrast(
        df, :grid_type, col;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        baseline = Symbol[],
        folds = 3,
        learners_outcome = SMALL_N_SL_LEARNERS,
        rng = StableRNG(100),
    )
    @test isfinite(res.estimate)
    @test abs(res.estimate - truth_effect) < 0.30
end

@testset "binary presence outcome family_outcome (#34)" begin
    rng = StableRNG(1)
    n = 300
    df = DataFrame(
        arm = rand(rng, ["R", "SS", "SC"], n),
        weight = randn(rng, n),
    )
    logit_p = @. -1.0 + 0.8 * (df.arm == "SS") + 0.1 * df.weight
    p = 1.0 ./ (1.0 .+ exp.(-logit_p))
    df.infected = Float64.(rand(rng, n) .< p)
    shared = (;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS", "SC"],
        baseline = [:weight],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(2),
    )
    res_bin = run_discrete_lmtp_contrast(
        df, :arm, :infected;
        family_outcome = :binomial,
        shared...,
    )
    res_gau = run_discrete_lmtp_contrast(
        df, :arm, :infected;
        family_outcome = :gaussian,
        shared...,
    )
    @test res_bin.estimate > 0.1
    @test res_bin.estimate > res_gau.estimate + 0.02
    @test abs(res_bin.estimate - res_gau.estimate) > 0.02

    df_bad = copy(df)
    df_bad.infected = rand(rng, n)
    @test_throws ArgumentError run_discrete_lmtp_contrast(
        df_bad, :arm, :infected;
        family_outcome = :binomial,
        shared...,
    )

    infected_miss = Vector{Union{Float64, Missing}}(df.infected)
    infected_miss[1:30] .= missing
    df_ipcw = copy(df)
    df_ipcw.infected = infected_miss
    res_ipcw = run_discrete_lmtp_contrast(
        df_ipcw, :arm, :infected;
        family_outcome = :binomial,
        handle_missing = :ipcw,
        shared...,
    )
    @test isfinite(res_ipcw.estimate)
    @test res_ipcw.estimate > res_gau.estimate
end

@testset "discrete LMTP cluster-robust SE" begin
    rng = StableRNG(71)
    n = 80
    cluster = repeat(1:20; inner=4)
    arms = rand(rng, ["R", "SS"], n)
    df = DataFrame(
        unit_id = cluster,
        grid_type = arms,
        Y = randn(rng, n) .+ 0.2 .* (arms .== "SS"),
    )
    res_unit = run_discrete_lmtp(
        df, :grid_type, :Y;
        policy = discrete_static_policy("SS"; levels = ["R", "SS"]),
        baseline = Symbol[],
        folds = 2,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(72),
    )
    res_cluster = run_discrete_lmtp(
        df, :grid_type, :Y;
        policy = discrete_static_policy("SS"; levels = ["R", "SS"]),
        baseline = Symbol[],
        folds = 2,
        learners_outcome = (:glm, :mean),
        cluster = :unit_id,
        rng = StableRNG(72),
    )
    @test res_cluster.covariance_kind == :cluster
    @test res_cluster.se > 0
    @test res_unit.se > 0
end
