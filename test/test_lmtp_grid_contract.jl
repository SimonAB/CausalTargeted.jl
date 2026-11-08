using DataFrames
using CausalTargeted
using StableRNGs
using Statistics
using Test

function lmtp_grid_contract_data(n = 400)
    rng = StableRNG(7101)
    a = collect(range(0.0, 1.0; length = n))
    w = randn(rng, n)
    y = 0.4 .* a .+ 0.6 .* w .+ randn(rng, n)
    return DataFrame(A = a, W = w, Y = y)
end

function run_lmtp_grid_contract(data; deltas = [0.0, 0.1, 0.5, 0.95], kwargs...)
    return CausalTargeted.run_lmtp_grid(
        data, :A, :Y;
        baseline = [:W],
        deltas,
        lower_q = 0.05,
        upper_q = 0.95,
        shift_scale = "raw",
        folds = 2,
        learners_outcome = (:glm, :mean),
        learners_trt = (:glm, :mean),
        parallel = false,
        rng = StableRNG(7102),
        kwargs...,
    )
end

@testset "Quantile-clamped LMTP grids preserve points and withhold inference" begin
    data = lmtp_grid_contract_data()

    for cache_nuisances in (true, false)
        grid = run_lmtp_grid_contract(
            data;
            cache_nuisances,
            simultaneous = true,
            n_boot_sim = 19,
        )

        zero = grid[grid.delta .== 0.0, :]
        nonzero = grid[grid.delta .!= 0.0, :]
        @test nrow(zero) == 1
        @test only(zero.est) == 0.0
        @test only(zero.se) == 0.0
        @test only(zero.inference_status) === :trivial
        @test all(isfinite, nonzero.est)
        @test all(nonzero.inference_status .== :unsupported_clamped_policy)
        for column in (:se, :lwr, :upr, :lwr_sim, :upr_sim, :crit_sim)
            @test all(isnan, nonzero[!, column])
        end

        @test grid.targeting_kind[grid.delta .== 0.0] == [:trivial]
        @test all(grid.targeting_kind[grid.delta .!= 0.0] .== :weighted_intercept)
        @test all(grid.weight_truncation)
        @test only(grid.targeting_weight[grid.delta .== 0.1]) == 1.0
        @test 0.0 < only(grid.targeting_weight[grid.delta .== 0.5]) < 1.0
        @test only(grid.targeting_weight[grid.delta .== 0.95]) == 0.0
    end
end

@testset "Untruncated weights do not opt a clamped grid into regular inference" begin
    data = lmtp_grid_contract_data()
    @test_throws ArgumentError run_lmtp_grid_contract(
        data; deltas = [0.0, 0.1], weight_truncation = false,
    )

    zero_only = CausalTargeted.run_lmtp_grid(
        data, :A, :Y;
        baseline = [:W], deltas = [0.0], weight_truncation = false,
        lower_q = 0.05, upper_q = 0.95, folds = 2,
        learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
        parallel = false, rng = StableRNG(7103),
    )
    @test only(zero_only.est) == 0.0
    @test only(zero_only.inference_status) === :trivial
    @test_throws ArgumentError CausalTargeted.run_lmtp_grid(
        data, :A, :Y;
        baseline = [:W], deltas = [0.0], estimator = :unknown,
        lower_q = 0.05, upper_q = 0.95, folds = 2,
        learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
        parallel = false, rng = StableRNG(7105),
    )
end

@testset "IPCW reweights point scores without restoring unsupported intervals" begin
    point_score = [0.0, 1.0, 2.0, 4.0]
    weights = [1.0, 1.0, 2.0, 4.0]
    row = (
        est = mean(point_score), se = NaN, lwr = NaN, upr = NaN,
        inference_status = :unsupported_clamped_policy,
    )
    weighted_row, ic = CausalTargeted._apply_ipcw_to_lmtp_row(
        row, nothing, weights;
        point_score = point_score,
        inference_status = :unsupported_clamped_policy,
    )
    expected = CausalTargeted.weighted_influence_summary(point_score, weights)

    @test weighted_row.est ≈ expected.estimate
    @test weighted_row.inference_status === :unsupported_clamped_policy
    @test all(isnan, (weighted_row.se, weighted_row.lwr, weighted_row.upr))
    @test ic === nothing
end

@testset "Imputed covariate unavailability takes precedence over clamp status" begin
    data = lmtp_grid_contract_data()
    data.W = Vector{Union{Missing, Float64}}(data.W)
    data.W[[4, 37, 88, 163, 247, 319]] .= missing

    grid = CausalTargeted.run_lmtp_grid(
        data, :A, :Y;
        baseline = [:W], deltas = [0.0, 0.5],
        lower_q = 0.05, upper_q = 0.95, shift_scale = "raw",
        folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
        parallel = false, cache_nuisances = false, handle_missing = :impute,
        rng = StableRNG(7104),
    )
    nonzero = grid.delta .!= 0.0

    @test all(grid.inference_status[nonzero] .== :unsupported_covariate_imputation)
    for column in (:se, :lwr, :upr, :lwr_sim, :upr_sim, :crit_sim)
        @test all(isnan, grid[nonzero, column])
    end
end
