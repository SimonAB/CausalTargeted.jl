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
