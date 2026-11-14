using Test
include(joinpath(@__DIR__, "stress", "multivariate_policy_calibration.jl"))
using .MultivariatePolicyCalibration
const Driver = MultivariatePolicyCalibration

@testset "policy calibration checkpoint contract" begin
    config = (; profile = "screen", seed = 71, draws = 16, cells = ["tiny"],
        source_hashes = Dict("fixture" => "abc"),
        environment_contract = Dict("julia" => "1.13.3", "RxInfer" => "6.0.0", "BLAS" => "fixture"))
    @test Driver.configuration_fingerprint(config) == Driver.configuration_fingerprint(config)
    @test Driver.configuration_fingerprint(config) != Driver.configuration_fingerprint(merge(config, (; seed = 72)))
    @test Driver.configuration_fingerprint(config) != Driver.configuration_fingerprint(merge(config, (; draws = 17)))
    @test Driver.configuration_fingerprint(config) != Driver.configuration_fingerprint(merge(config, (; source_hashes = Dict("fixture" => "def"))))
    rows = [Driver.empty_policy_row(; cell_id = "tiny", replication = 1, method,
        seed = 71, scenario = "balanced", experiment = "fixed_parameter", n = 20)
        for method in Driver.POLICY_METHODS]
    @test Driver.validate_checkpoint_rows(rows, ["tiny"]) == Set([("tiny", 1)])
    @test_throws ArgumentError Driver.validate_checkpoint_rows(rows[1:2], ["tiny"])
    @test_throws ArgumentError Driver.validate_checkpoint_rows(vcat(rows, rows[1:1]), ["tiny"])
    @test_throws ArgumentError Driver.validate_checkpoint_rows(rows, ["different"])
    @test_throws ArgumentError Driver.validate_checkpoint_rows([merge(rows[1], (; replication = 0))], ["tiny"])
    mktempdir() do directory
        checkpoint = Driver.open_checkpoint(directory, config; provenance = Dict("source_dirty" => false))
        Driver.append_dataset!(checkpoint, rows)
        resumed = Driver.open_checkpoint(directory, config; resume = true,
            provenance = Dict("source_dirty" => true))
        @test resumed.fingerprint == checkpoint.fingerprint
        changed_backend = merge(config, (; environment_contract = merge(config.environment_contract, Dict("RxInfer" => "6.0.1"))))
        @test_throws ArgumentError Driver.open_checkpoint(directory, changed_backend; resume = true)
        @test resumed.completed == Set([("tiny", 1)])
        @test length(resumed.rows) == 3
        @test_throws ArgumentError Driver.append_dataset!(resumed, rows)
        @test_throws ArgumentError Driver.open_checkpoint(directory, merge(config, (; seed = 72)); resume = true)
        @test_throws ArgumentError Driver.open_checkpoint(directory, config)
        open(joinpath(directory, "raw.tsv"), "a") do io
            write(io, "tampered\n")
        end
        @test_throws ArgumentError Driver.open_checkpoint(directory, config; resume = true)
    end
    mktempdir() do directory
        checkpoint = Driver.open_checkpoint(directory, config)
        Driver.append_dataset!(checkpoint, rows)
        raw = joinpath(directory, "raw.tsv")
        content = read(raw, String)
        write(raw, replace(content, "balanced" => "rare_control"))
        @test_throws ArgumentError Driver.open_checkpoint(directory, config; resume = true)
    end
    mktempdir() do directory
        checkpoint = Driver.open_checkpoint(directory, config)
        Driver.append_dataset!(checkpoint, rows)
        raw = joinpath(directory, "raw.tsv")
        lines = readlines(raw)
        write(raw, join(lines[1:3], '\n') * "\n")
        @test_throws ArgumentError Driver.open_checkpoint(directory, config; resume = true)
    end
end

@testset "policy driver CLI" begin
    @test Driver.parse_options(String[]).replications == 50
    @test Driver.parse_options(["--profile", "validation"]).replications == 500
    options = Driver.parse_options(["--replications", "2", "--seed", "5", "--draws", "64", "--cells", "a,b", "--resume"])
    @test options.replications == 2 && options.seed == 5 && options.draws == 64
    @test options.cells == ["a", "b"] && options.resume
    @test_throws ArgumentError Driver.parse_options(["--replications", "0"])
    @test_throws ArgumentError Driver.parse_options(["--profile", "unknown"])
    @test_throws ArgumentError Driver.parse_options(["--unexpected"])
end

# Use small synthetic seams, with no RxInfer fit or adaptive integration.
@eval Driver include(joinpath(@__DIR__, "stress", "policy_calibration_support.jl"))
@testset "policy driver failure denominators and common draws" begin
    cell = first(Driver.PolicyCalibrationSupport.calibration_cells(:screen))
    covariance = [1.0 0.0; 0.0 1.0]
    mock_fit = (X, y, μ, V) -> (; converged = true, mean = zeros(2),
        variational_covariance = covariance, corrected_covariance = covariance,
        n_iterations = 3, status = :converged)
    calls = Bool[]
    mock_cdf = function(X, y, μ, V, truth; gaussian = false, weights = nothing)
        push!(calls, gaussian)
        gaussian || error("deliberate posterior reference failure")
        (; available = true, covered = true, pit = 0.5, status = :ok,
            cdf_change = 0.0, tail_error = 0.0, mass_change = 0.0)
    end
    rows = Driver.run_policy_dataset(cell, 1; seed = 9, draws = 32,
        fit_engine = mock_fit, checked_cdf = mock_cdf)
    simulated = Driver.PolicyCalibrationSupport.simulate_policy_replication(cell, 1; seed = 9)
    for row in rows
        @test row.identification_status == "user_assumed"
        @test row.control_n == count(==(0), simulated.A)
        @test row.treated_n == count(==(1), simulated.A)
        @test row.control_successes == count(index -> simulated.A[index] == 0 && simulated.y[index] == 1, eachindex(simulated.A))
        @test row.treated_successes == count(index -> simulated.A[index] == 1 && simulated.y[index] == 1, eachindex(simulated.A))
        @test row.control_n + row.treated_n == row.n
    end
    @test calls == [false, true, true]
    @test rows[1].status == "reference_exception" && !rows[1].available
    @test ismissing(rows[1].covered)
    @test rows[2].available && rows[3].available
    @test rows[2].policy_mean == rows[3].policy_mean
    @test rows[2].mean_mcse == rows[3].mean_mcse
    @test rows[2].draws == 32 && rows[2].width > 0
    bad_corrected = (X, y, μ, V) -> merge(mock_fit(X, y, μ, V), (; corrected_covariance = nothing))
    rows = Driver.run_policy_dataset(cell, 1; seed = 9, draws = 32,
        fit_engine = bad_corrected, checked_cdf = mock_cdf)
    @test rows[2].available && !rows[3].available
    @test rows[3].status == "linear_response_covariance_failure"
    @test ismissing(rows[3].covered)
    all_cdf = (X, y, μ, V, truth; kwargs...) -> (; available = true,
        covered = false, pit = 0.99, status = :ok,
        cdf_change = 0.0, tail_error = 0.0, mass_change = 0.0)
    fit_failure = (args...) -> error("deliberate fit failure")
    rows = Driver.run_policy_dataset(cell, 2; seed = 9, draws = 32,
        fit_engine = fit_failure, checked_cdf = all_cdf)
    @test rows[1].available && !rows[1].covered
    @test all(row.status == "fit_exception" && !row.available for row in rows[2:3])
    @test Driver.validate_checkpoint_rows(rows, [string(cell.id)]) == Set([(string(cell.id), 2)])
    mktempdir() do directory
        config = (; profile = "screen", seed = 9, draws = 32, cells = [string(cell.id)], source_hashes = Dict("test" => "fixture"))
        checkpoint = Driver.open_checkpoint(directory, config)
        Driver.append_dataset!(checkpoint, rows)
        Driver.write_reports!(checkpoint)
        @test all(isfile(joinpath(directory, file)) for file in ("raw.tsv", "summary.tsv", "paired.tsv", "metadata.toml", "report.md"))
    end
end

@testset "resume extends the target without repeating datasets" begin
    cells = [(; id = "a"), (; id = "b")]
    completed = Set([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    @test isempty(Driver.pending_datasets(cells, 2, completed))
    remaining = Driver.pending_datasets(cells, 500, completed)
    @test length(remaining) == 996
    @test all(replication >= 3 for (_, replication) in remaining)
    @test_throws ArgumentError Driver.pending_datasets(cells, 1, completed)
end

@testset "invalid cohort cannot receive continuous policy PIT coverage" begin
    cell = first(Driver.PolicyCalibrationSupport.calibration_cells(:screen))
    dataset = Driver.PolicyCalibrationSupport.simulate_policy_replication(cell, 1; seed = 9)
    for invalid in (merge(dataset, (; A = ones(cell.n), policy_scale = 0.0, policy_truth = 0.0)),
                    merge(dataset, (; policy_scale = 0.25)),
                    merge(dataset, (; policy_truth = dataset.policy_truth + 0.1)))
        simulator = (args...; kwargs...) -> invalid
        rows = Driver.run_policy_dataset(cell, 1; seed = 9, simulator,
            checked_cdf = (args...; kwargs...) -> error("must not reach integration"))
        @test length(rows) == 3
        @test all(row.status == "dataset_exception" && !row.available && ismissing(row.covered) for row in rows)
    end
end

@testset "resume preserves first-call compilation timing" begin
    mktempdir() do directory
        config = (; profile = "screen", seed = 1, draws = 8, cells = ["tiny"], source_hashes = Dict("test" => "fixture"))
        checkpoint = Driver.open_checkpoint(directory, config)
        Driver.record_warmup_timing!(checkpoint, 19.0)
        @test checkpoint.metadata["first_call_including_compilation_seconds"] == 19.0
        resumed = Driver.open_checkpoint(directory, config; resume = true)
        Driver.record_warmup_timing!(resumed, 2.0; resume = true)
        @test resumed.metadata["first_call_including_compilation_seconds"] == 19.0
        @test Driver.open_checkpoint(directory, config; resume = true).metadata["resume_warmup_seconds"] == [2.0]
    end
end
