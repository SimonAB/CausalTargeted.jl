using CausalDynamics
using CausalDynamics:
    TemporalDAGSpec, TemporalEffectQuery, unroll_temporal_dag,
    NodeOutcomeSpec, binary, count_outcome
using CausalTargeted
using DataFrames
using StableRNGs
using Test

@testset "estimation plan bridge opts (#24, #44)" begin
    spec = TemporalDAGSpec([:arm, :y], [(:arm, :y, 0)])
    u = unroll_temporal_dag(spec, 2)
    query = TemporalEffectQuery(:arm, :y, 1, 1)
    df = DataFrame(
        arm = rand(StableRNG(1), [0, 1], 80),
        y = rand(StableRNG(2), 0:12, 80),
    )

    plan_count = CausalTargeted.plan_targeted_estimation(
        u, query, propertynames(df);
        outcome_specs = Dict(:y => NodeOutcomeSpec(count_outcome)),
    )
    @test plan_count.family_outcome === :negbin
    opts_count = CausalTargeted._estimation_plan_runner_opts(df, plan_count)
    @test opts_count.family_outcome === :negbin
    @test :glm_nb in opts_count.learners_outcome

    @test_throws "count_family must be" CausalTargeted.plan_targeted_estimation(
        u, query, propertynames(df);
        outcome_specs = Dict(:y => NodeOutcomeSpec(count_outcome)),
        count_family = :gaussian,
        data = (arm = [0, 1],),
    )

    df_bin = DataFrame(
        arm = rand(StableRNG(3), [0, 1], 80),
        y = Float64.(rand(StableRNG(4), [0, 1], 80)),
    )
    plan_bin = CausalTargeted.plan_targeted_estimation(
        u, query, propertynames(df_bin);
        outcome_specs = Dict(:y => NodeOutcomeSpec(binary)),
    )
    @test plan_bin.family_outcome === :binomial
    opts_bin = CausalTargeted._estimation_plan_runner_opts(df_bin, plan_bin)
    @test opts_bin.family_outcome === :binomial
end
