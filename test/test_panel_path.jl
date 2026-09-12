using CausalDynamics
using CausalDynamics:
    TemporalDAGSpec, LaggedEdge, unroll_temporal_dag, TemporalEffectQuery,
    plan_targeted_estimation, panel_column_name,
    NodeOutcomeSpec, hurdle, binary, count_outcome
using CausalTargeted
using DataFrames
using Distributions
using Random
using StableRNGs
using Test

@testset "panel path: CDM → planner → run_estimation_plan (#33)" begin
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
    spec = TemporalDAGSpec(
        [:grid_type, :fec],
        [LaggedEdge(:grid_type, :fec, 0)],
    )
    u = unroll_temporal_dag(spec, T)
    query = TemporalEffectQuery(:grid_type, :fec, 2, 2)

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
    wide_cols = propertynames(df)

    plan = plan_targeted_estimation(
        u, query, wide_cols;
        unit_level = [:grid_type],
        data = df,
        min_n = 20,
    )
    @test plan.engine === :discrete_lmtp
    @test plan.treatment === :grid_type
    @test plan.outcome === panel_column_name(:fec, 2)
    @test plan.identifiable
    @test plan.estimability === :ok
    @test plan.min_complete_n == n

    res = run_estimation_plan(
        df, plan;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        folds = 3,
        learners_outcome = SMALL_N_SL_LEARNERS,
        rng = StableRNG(100),
    )
    @test isfinite(res.estimate)
    @test abs(res.estimate - truth_effect) < 0.30
    @test res.identification_status === :identified
    @test res.identifiable

    @testset "non-identifiable plans are refused unless declared exploratory" begin
        unidentified = CausalDynamics.EstimationPlan(
            plan.engine, plan.treatment, plan.outcome, plan.baseline, plan.query,
            false, :none, plan.adjustment_columns, plan.missing_columns;
            min_complete_n = plan.min_complete_n,
            estimability = plan.estimability,
        )
        run_kwargs = (;
            arm_hi = "SS", arm_ref = "R", levels = ["R", "SS"], folds = 3,
            learners_outcome = SMALL_N_SL_LEARNERS, rng = StableRNG(100),
        )
        @test_throws ArgumentError run_estimation_plan(df, unidentified; run_kwargs...)
        @test_throws ArgumentError run_estimation_plan(
            df, unidentified; on_unidentified = :maybe, run_kwargs...,
        )
        explored = @test_logs (:warn, r"exploratory") match_mode = :any run_estimation_plan(
            df, unidentified; on_unidentified = :exploratory, run_kwargs...,
        )
        @test isfinite(explored.estimate)
        @test explored.identification_status === :not_identified_by_procedure
        @test !explored.identifiable
    end
end

@testset "panel path: three-arm planner + contrast" begin
    rng = StableRNG(20260828)
    T = 4
    n = 120
    arms = rand(rng, ["R", "SS", "SC"], n)
    df = DataFrame(
        mouse_id = string.(1:n),
        grid_type = arms,
    )
    for t in 1:T
        col = panel_column_name(:fec, t)
        effect = 0.25 .* (arms .== "SS") .- 0.08 .* (arms .== "SC")
        df[!, col] = effect .+ 0.05 .* randn(rng, n)
    end

    spec = TemporalDAGSpec([:grid_type, :fec], [LaggedEdge(:grid_type, :fec, 0)])
    u = unroll_temporal_dag(spec, T)
    query = TemporalEffectQuery(:grid_type, :fec, 2, 2)
    plan = plan_targeted_estimation(
        u, query, propertynames(df);
        unit_level = [:grid_type],
        data = df,
    )
    @test plan.engine === :discrete_lmtp
    res = run_estimation_plan(
        df, plan;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS", "SC"],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(1),
    )
    @test isfinite(res.estimate)
    @test res.positivity.ok
end

@testset "panel path: hurdle planner → two-part runner" begin
    rng = StableRNG(7)
    n = 150
    arms = rand(rng, ["R", "SS"], n)
    df = DataFrame(mouse_id = string.(1:n), grid_type = arms)
    for t in 1:2
        bin_col = panel_column_name(:fec_bin, t)
        int_col = panel_column_name(:fec_intensity, t)
        p = 0.3 .+ 0.2 .* (arms .== "SS")
        df[!, bin_col] = Float64.(rand(rng, n) .< p)
        df[!, int_col] = abs.(randn(rng, n)) .+ 0.5 .* (arms .== "SS")
    end

    spec = TemporalDAGSpec([:grid_type, :fec], [LaggedEdge(:grid_type, :fec, 0)])
    u = unroll_temporal_dag(spec, 2)
    query = TemporalEffectQuery(:grid_type, :fec, 2, 2)
    outcome_specs = Dict(:fec => NodeOutcomeSpec(hurdle, :fec_bin, :fec_intensity))
    plan = plan_targeted_estimation(
        u, query, propertynames(df);
        unit_level = [:grid_type],
        outcome_specs = outcome_specs,
        data = df,
    )
    @test plan.engine === :two_part_discrete_lmtp
    res = run_estimation_plan(
        df, plan;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(8),
    )
    @test res.presence.estimate > 0.05
    @test isfinite(res.intensity.estimate)
end

@testset "panel path: binary planner → binomial nuisances (#44)" begin
    rng = StableRNG(21)
    n = 200
    arms = rand(rng, ["R", "SS"], n)
    df = DataFrame(mouse_id = string.(1:n), grid_type = arms)
    logit_p = @. -0.5 + 0.7 * (arms .== "SS")
    p = 1.0 ./ (1.0 .+ exp.(-logit_p))
    df[!, :infected2] = Float64.(rand(rng, n) .< p)

    spec = TemporalDAGSpec([:grid_type, :infected], [LaggedEdge(:grid_type, :infected, 0)])
    u = unroll_temporal_dag(spec, 2)
    query = TemporalEffectQuery(:grid_type, :infected, 2, 2)
    outcome_specs = Dict(:infected => NodeOutcomeSpec(binary))
    plan = plan_targeted_estimation(
        u, query, propertynames(df);
        unit_level = [:grid_type],
        outcome_specs = outcome_specs,
        data = df,
    )
    @test plan.family_outcome === :binomial
    res = run_estimation_plan(
        df, plan;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        folds = 3,
        learners_outcome = (:glm, :mean),
        rng = StableRNG(22),
    )
    @test res.estimate > 0.05
end

@testset "panel path: count planner → NB nuisances (#24)" begin
    rng = StableRNG(31)
    n = 300
    arms = rand(rng, [0, 1], n)
    df = DataFrame(
        mouse_id = string.(1:n),
        grid_type = [a == 0 ? "R" : "SS" for a in arms],
    )
    μ = @. exp(0.4 + 0.35 * arms)
    df[!, :count2] = [rand(rng, NegativeBinomial(2.0, 2 / (2 + μᵢ))) for μᵢ in μ]

    spec = TemporalDAGSpec([:grid_type, :count], [LaggedEdge(:grid_type, :count, 0)])
    u = unroll_temporal_dag(spec, 2)
    query = TemporalEffectQuery(:grid_type, :count, 2, 2)
    outcome_specs = Dict(:count => NodeOutcomeSpec(count_outcome))
    plan = plan_targeted_estimation(
        u, query, propertynames(df);
        unit_level = [:grid_type],
        outcome_specs = outcome_specs,
        data = df,
    )
    @test plan.family_outcome === :negbin
    res = run_estimation_plan(
        df, plan;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        folds = 3,
        learners_outcome = (:glm_nb, :mean),
        rng = StableRNG(32),
    )
    @test res.estimate > 0
end

@testset "panel path: structural_skip (#26)" begin
    spec = TemporalDAGSpec([:grid_type, :fec], [LaggedEdge(:grid_type, :fec, 0)])
    u = unroll_temporal_dag(spec, 4)
    query = TemporalEffectQuery(:grid_type, :fec, 2, 4)
    df = DataFrame(
        mouse_id = string.(1:20),
        grid_type = fill("R", 20),
        fec2 = rand(20),
    )
    plan = plan_targeted_estimation(
        u, query, propertynames(df);
        unit_level = [:grid_type],
    )
    @test plan.estimability === :structural_skip
    @test_throws ArgumentError run_estimation_plan(
        df, plan;
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
    )
end
