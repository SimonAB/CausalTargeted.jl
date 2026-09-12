using CausalTargeted
using CausalDynamics: TemporalDAGSpec, LaggedEdge, unroll_temporal_dag
using DataFrames
using Distributions
using Random
using StableRNGs
using Test

const NODE_PARTS = Dict(:fec => (:fec_bin, :fec_intensity))

@testset "hurdle CI tests" begin
    function simulate_hurdle_ci_dgp(n::Int; rng, sex_effect::Real = 0.0)
        sex = rand(rng, Bernoulli(0.5), n)
        grid_type = rand(rng, ["R", "SS", "SC"], n)
        weight = randn(rng, n)
        logit_p = @. -1.0 + 0.6 * (grid_type == "SS") + 0.25 * weight + sex_effect * sex
        fec_bin = rand.(rng, Bernoulli.(1 ./ (1 .+ exp.(-logit_p))))
        fec_intensity = Vector{Union{Missing, Float64}}(missing, n)
        for i in 1:n
            fec_bin[i] || continue
            fec_intensity[i] = 1.0 + 0.2 * weight[i] + randn(rng) * 0.4
        end
        return DataFrame(
            sex = sex,
            grid_type = grid_type,
            weight = weight,
            fec_bin = Float64.(fec_bin),
            fec_intensity = fec_intensity,
        )
    end

    spec = TemporalDAGSpec(
        [:sex, :grid_type, :weight, :fec],
        [
            LaggedEdge(:grid_type, :fec, 0),
            LaggedEdge(:weight, :fec, 0),
        ],
    )
    unrolling = unroll_temporal_dag(spec, 1)
    statements = CausalTargeted.local_markov_statements(unrolling)

    @test any(st -> st.label_x == "sex[1]" && st.label_y == "fec[1]" && isempty(st.label_z), statements)
    @test any(st -> st.label_x == "sex[1]" && st.label_y == "fec[1]" && st.implied_by_dag, statements)

    @testset "faithful DGP accepts implied hurdle independences" begin
        df = simulate_hurdle_ci_dgp(2000; rng = StableRNG(11))
        results = CausalTargeted.test_implied_hurdle_independences(
            statements, df, NODE_PARTS; α = 0.05,
        )
        @test !isempty(results)
        sex_fec = filter(r -> r.x == "sex[1]" && r.y == "fec[1]" && isempty(r.z), results)
        @test length(sex_fec) == 2
        @test all(r -> r.independent && !r.skipped, sex_fec)
        implied_rows = filter(
            st -> st.implied_by_dag &&
                endswith(st.label_x, "[1]") &&
                endswith(st.label_y, "[1]"),
            statements,
        )
        hurdle_implied = CausalTargeted.test_implied_hurdle_independences(
            implied_rows, df, NODE_PARTS; α = 0.05,
        )
        tested = filter(r -> !r.skipped, hurdle_implied)
        @test !isempty(tested)
        @test all(r -> r.independent, tested)
    end

    @testset "categorical grid_type predictor uses dummy coding" begin
        df = simulate_hurdle_ci_dgp(800; rng = StableRNG(13))
        stmt = CausalTargeted.IndependenceStatement(
            1, 2, Int[], "grid_type[1]", "fec[1]", String[], true,
        )
        results = CausalTargeted.test_implied_hurdle_independences([stmt], df, NODE_PARTS; α = 0.05)
        @test length(results) == 2
        @test all(r -> r.n >= 10 && !r.skipped, results)
        @test any(r -> !r.independent, results)
    end

    @testset "planted sex effect rejects false independence" begin
        df = simulate_hurdle_ci_dgp(800; rng = StableRNG(17), sex_effect = 1.2)
        stmt = CausalTargeted.IndependenceStatement(
            1, 2, Int[], "sex[1]", "fec[1]", String[], true,
        )
        results = CausalTargeted.test_implied_hurdle_independences([stmt], df, NODE_PARTS; α = 0.05)
        pres = only(filter(r -> r.part == "presence", results))
        @test !pres.independent
        @test pres.p < 0.05
    end

    @testset "CausalTargeted.default_hurdle_label_to_col" begin
        @test CausalTargeted.default_hurdle_label_to_col("fec[1]") == :fec
        @test CausalTargeted.default_hurdle_label_to_col("fec[2]") == :fec_t2
    end

    @testset "hurdle colmap lag panel (#43)" begin
        df = DataFrame(
            sex_t1 = rand(120),
            sex_t2 = rand(120),
            grid_arm = rand(["R", "SS"], 120),
            fec_bin_t1 = Float64.(rand(120) .< 0.4),
            fec_intensity_t1 = abs.(randn(120)) .+ 0.5,
            fec_bin_t2 = Float64.(rand(120) .< 0.4),
            fec_intensity_t2 = abs.(randn(120)) .+ 0.5,
            weight_t2 = randn(120),
        )
        colmap = merge(
            CausalTargeted.hurdle_colmap_lag_panel([:fec]; occasions = (1, 2), unit_level = [:sex]),
            CausalTargeted.hurdle_colmap_grid_arm(time = 2, col = :grid_arm),
        )
        @test colmap["fec[2]"] == :fec_bin_t2
        @test colmap["sex[2]"] == :sex_t2
        @test colmap["grid_type[2]"] == :grid_arm

        stmt = CausalTargeted.IndependenceStatement(
            1, 2, Int[], "grid_type[2]", "fec[2]", String[], true,
        )
        lag_parts = Dict(:fec => (:fec_bin_t2, :fec_intensity_t2))
        results = CausalTargeted.test_implied_hurdle_independences(
            [stmt], df, lag_parts; colmap = colmap, α = 0.05,
        )
        @test length(results) == 2
        @test all(r -> r.n >= 10 && !r.skipped, results)
    end
end
