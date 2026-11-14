using Test, LinearAlgebra, Statistics

include(joinpath(@__DIR__, "stress", "policy_calibration_support.jl"))
using .PolicyCalibrationSupport

@testset "Prespecified policy calibration cells and simulations" begin
    screen = calibration_cells(:screen)
    cells = calibration_cells(:validation)
    @test length(screen) == 2
    @test length(cells) == 14
    @test length(unique(c.id for c in cells)) == 14
    @test all(c.n == 20 && c.scenario == :balanced for c in screen)
    @test count(c -> c.experiment == :fixed_parameter, cells) == 8
    @test count(c -> c.experiment == :prior_predictive, cells) == 6
    @test_throws ArgumentError calibration_cells(:unknown)
    seeds = UInt64[]
    for cell in cells
        sim = simulate_policy_replication(cell, 3)
        again = simulate_policy_replication(cell, 3)
        @test sim == again
        @test size(sim.X) == (cell.n, 2)
        @test sim.X == hcat(ones(cell.n), sim.A)
        @test Set(sim.A) == Set([0, 1])
        @test count(iszero, sim.A) == (cell.scenario == :rare_control ? 1 : cell.n ÷ 2)
        @test sim.policy_truth ≈ sim.policy_scale * sim.arm_truth
        @test sim.separation in (:none, :complete, :quasi, :single_class)
        expected_variance = cell.scenario == :strong_prior ? 0.25 : 2.25
        @test cell.prior_covariance == expected_variance * Matrix{Float64}(I, 2, 2)
        if cell.experiment == :fixed_parameter
            @test sim.beta == (cell.scenario == :separated ? [-3.0, 6.0] : [-0.3, 0.9])
        else
            @test sim.beta != simulate_policy_replication(cell, 4).beta
        end
        @test sim.draw_seed != simulate_policy_replication(cell, 4).draw_seed
        push!(seeds, sim.draw_seed)
    end
    @test length(unique(seeds)) == 14
    @test simulate_policy_replication(cells[1], 2) == simulate_policy_replication(reverse(cells)[end], 2)
    @test simulate_policy_replication(cells[1], 2; seed=9) != simulate_policy_replication(cells[1], 2; seed=10)
    @test_throws ArgumentError simulate_policy_replication(cells[1], 0)
    @test PolicyCalibrationSupport.classify_separation([0, 0, 1, 1], [0, 0, 1, 1]) == :complete
    @test PolicyCalibrationSupport.classify_separation([0, 0, 1, 1], [0, 0, 0, 1]) == :quasi
    @test PolicyCalibrationSupport.classify_separation([0, 0, 1, 1], [0, 0, 0, 0]) == :single_class
    @test PolicyCalibrationSupport.classify_separation([0, 0, 1, 1], [0, 1, 0, 1]) == :none
end

@testset "Frequency compression preserves the Bernoulli likelihood" begin
    A = [0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 0, 1, 1, 1]
    compressed = compress_binary_likelihood(A, y)
    @test size(compressed.X, 1) == 4
    @test sum(compressed.weights) == length(A)
    @test eltype(compressed.weights) <: Integer
    @test all(>(0), compressed.weights)
    for beta in ([-0.3, 0.9], [-3.0, 6.0], [2.0, -4.0])
        loglik(X, outcomes, weights) = sum(weights .* (outcomes .* (X * beta) .- log1p.(exp.(X * beta))))
        @test loglik(hcat(ones(length(A)), A), y, ones(length(A))) ≈ loglik(compressed.X, compressed.y, compressed.weights)
    end
    @test size(compress_binary_likelihood([0, 1], [0, 1]).X, 1) == 2
    for (badA, bady) in (([0, 2], [0, 1]), ([0, 1], [0, -1]), ([0, missing], [0, 1]), ([0, 1], [0, missing]), (Int[], Int[]), ([0], [0, 1]))
        @test_throws ArgumentError compress_binary_likelihood(badA, bady)
    end
end

policy_row(rep, method, available, covered; cell="cell", status=:ok, pit=0.5, width=2.0, error=0.1, experiment=:prior_predictive) =
    (; cell_id=cell, replication=rep, method, available, covered, pit, width, policy_error=error, status, n=20, scenario=:balanced, experiment)

@testset "Calibration summaries retain unavailable attempts" begin
    rows = [policy_row(1, :variational, true, true; pit=0.0, error=-1.0),
        policy_row(2, "variational", true, false; pit=1.0, width=Inf, error=1.0),
        policy_row(3, :variational, false, missing; status="fit_failure", pit=missing, width=missing, error=missing),
        policy_row(4, :variational, false, missing; status=:fit_failure, pit=missing, width=missing, error=missing)]
    s = only(summarise_policy_rows(rows))
    @test (s.attempted, s.available, s.failed) == (4, 2, 2)
    @test s.coverage_available == 0.5
    @test s.coverage_all_attempts == 0.25
    @test s.mcse_available ≈ sqrt(0.5 * 0.5 / 2)
    @test s.mcse_all_attempts ≈ sqrt(0.25 * 0.75 / 4)
    @test s.pit_deciles == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    @test s.pit_count == 2 && s.pit_label == :sbc
    @test s.width_count == 1 && s.width_mean == 2.0
    @test s.policy_error_count == 2 && s.policy_error_bias == 0.0 && s.policy_error_rmse == 1.0
    @test s.failure_reasons == "fit_failure=2"
    absent = only(summarise_policy_rows([policy_row(1, :reference, false, missing; status=:reference_failure, experiment=:fixed_parameter)]))
    @test isnan(absent.coverage_available)
    @test isnan(absent.mcse_available)
    @test absent.coverage_all_attempts == 0.0
    @test absent.pit_count == 0 && absent.width_count == 0 && absent.policy_error_count == 0
    @test absent.pit_label == :fixed_parameter_not_sbc
    @test absent.separation_unknown == 1
    @test (s.separation_none, s.separation_complete, s.separation_quasi,
        s.separation_single_class, s.separation_unknown) == (0, 0, 0, 0, 4)
    separated_rows = [merge(rows[1], (; separation=:none)),
        merge(rows[2], (; separation="complete")),
        merge(rows[3], (; separation=:quasi)),
        merge(rows[4], (; separation=:single_class))]
    separated = only(summarise_policy_rows(separated_rows))
    @test (separated.separation_none, separated.separation_complete,
        separated.separation_quasi, separated.separation_single_class,
        separated.separation_unknown) == (1, 1, 1, 1, 0)
    @test separated.failed == 2
    @test isempty(summarise_policy_rows(NamedTuple[]))
    @test_throws ArgumentError summarise_policy_rows([policy_row(1, :variational, true, missing)])
end

@testset "Paired coverage uncertainty uses within-replication differences" begin
    rows = [policy_row(1, :variational, true, false), policy_row(1, "linear_response", true, true),
        policy_row(2, :variational, true, true), policy_row(2, :linear_response, true, true),
        policy_row(3, :variational, true, true), policy_row(3, :linear_response, true, false),
        policy_row(4, :variational, true, false), policy_row(4, :linear_response, false, missing; status=:covariance_failure),
        policy_row(5, :variational, false, missing; status=:fit_failure), policy_row(5, :linear_response, false, missing; status=:fit_failure)]
    s = only(paired_policy_summary(rows))
    @test s.paired == 3
    @test s.attempts == 5
    @test s.coverage_delta == 0.0
    @test s.coverage_delta_mcse ≈ std([1.0, 0.0, -1.0]) / sqrt(3)
    @test (s.variational_available, s.linear_response_available, s.availability_difference) == (4, 3, -1)
    @test s.all_attempts_paired == 5
    @test s.all_attempts_inclusion_delta == 0.0
    @test s.all_attempts_inclusion_delta_mcse ≈ std([1.0, 0.0, -1.0, 0.0, 0.0]) / sqrt(5)
    @test_throws ArgumentError paired_policy_summary(vcat(rows, [rows[1]]))
    no_pairs = only(paired_policy_summary(rows[end-1:end]))
    @test no_pairs.paired == 0
    @test isnan(no_pairs.coverage_delta) && isnan(no_pairs.coverage_delta_mcse)
    @test no_pairs.all_attempts_inclusion_delta == 0.0
    single_pair = only(paired_policy_summary(rows[1:2]))
    @test isnan(single_pair.coverage_delta_mcse)
    @test isnan(single_pair.all_attempts_inclusion_delta_mcse)
    differing_availability = [policy_row(1, :variational, true, true),
        policy_row(1, :linear_response, false, missing; status=:covariance_failure),
        policy_row(2, :variational, false, missing; status=:fit_failure),
        policy_row(2, :linear_response, true, false),
        policy_row(3, :variational, true, false), policy_row(3, :linear_response, true, true)]
    differing = only(paired_policy_summary(differing_availability))
    @test differing.paired == 1 && differing.coverage_delta == 1.0
    @test differing.all_attempts_paired == 3
    @test differing.all_attempts_inclusion_delta == 0.0
    @test differing.all_attempts_inclusion_delta_mcse ≈ std([-1.0, 0.0, 1.0]) / sqrt(3)
    @test_throws ArgumentError paired_policy_summary(differing_availability[1:1])
end
