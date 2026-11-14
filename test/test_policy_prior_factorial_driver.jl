using Test, LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "stress", "policy_prior_factorial.jl"))
const FactorialDriver = PolicyPriorFactorial
const FD = FactorialDriver

@testset "v2 factorial CLI" begin
    @test FD.parse_options(String[]).replications == 50
    @test FD.parse_options(String[]).seed == 20261115
    @test FD.parse_options(String[]).fidelity_replications == 0
    @test FD.parse_options(["--profile", "validation"]).replications == 1000
    @test FD.parse_options(["--profile", "validation"]).fidelity_replications == 100
    @test FD.parse_options(["--experiment", "prior_predictive"]).experiment == "prior_predictive"
    @test FD.parse_options(["--cells", "a,b", "--resume"]).cells == ["a", "b"]
    @test_throws ArgumentError FD.parse_options(["--fidelity-replications", "-1"])
    @test_throws ArgumentError FD.parse_options(["--profile", "screen"])
end

function factorial_fixture()
    cell = (; id = "tiny", n = 20, control_n = 10, experiment = :fixed_parameter, beta_fixed = [-0.3, 0.9])
    priors = [(; id = "p$i", mean = zeros(2), covariance = Matrix{Float64}(I, 2, 2),
        scale = 1.0, centring = "zero", deployable = true) for i in 1:12]
    dataset = (; A = repeat([0, 1], 10), X = hcat(ones(20), repeat([0, 1], 10)),
        y = repeat([0, 1], 10), beta = [-0.3, 0.9], arm_truth = 0.2,
        policy_truth = 0.1, policy_scale = 0.5, separation = :complete,
        draw_seed = FD.Legacy.PolicyCalibrationSupport.replication_seed("tiny", 1, 1, 1), dataset_id = "tiny:1")
    (; cell, priors, dataset)
end

@testset "one simulation and fit per prior; reference failure stays local" begin
    f = factorial_fixture()
    simulations = Ref(0); fits = Ref(0)
    simulator = function(args...; kwargs...)
        simulations[] += 1; f.dataset
    end
    fit_engine = function(args...)
        fits[] += 1
        (; converged = true, mean = zeros(2), variational_covariance = Matrix{Float64}(I, 2, 2),
            corrected_covariance = Matrix{Float64}(I, 2, 2), n_iterations = 4, status = :converged)
    end
    checked_cdf = function(args...; gaussian = false, kwargs...)
        gaussian || error("reference deliberately unavailable")
        (; available = true, covered = true, pit = 0.5, status = :ok,
            cdf_change = 0.0, tail_error = 0.0, mass_change = 0.0)
    end
    rows = FD.evaluate_factorial_dataset(f.cell, 1; seed = 1, draws = 16, simulator,
        prior_builder = (args...) -> f.priors, fit_engine, checked_cdf)
    @test simulations[] == 1 && fits[] == 12
    @test length(rows) == 36
    @test all(row.dataset_id == "tiny:1" && row.draw_seed == f.dataset.draw_seed for row in rows)
    @test all(!row.available && ismissing(row.covered) for row in rows if row.method == "adaptive_reference")
    @test all(row.available && row.covered for row in rows if row.method != "adaptive_reference")
    @test length(unique(row.policy_mean for row in rows if row.method != "adaptive_reference")) == 1
    @test all(row.fidelity_status == "not_selected" && !row.fidelity_available for row in rows)
    config = (; cells = ["tiny"], prior_ids = Dict("tiny" => string.(getproperty.(f.priors, :id))),
        seed = 1, draws = 16, source_hashes = Dict("fixture" => "a"), environment_contract = Dict("backend" => "a"))
    @test FD.validate_checkpoint_rows(rows, config) == Set([("tiny", 1)])
    @test_throws ArgumentError FD.validate_checkpoint_rows(rows[1:end-1], config)
    @test_throws ArgumentError FD.validate_checkpoint_rows(vcat(rows, rows[1:1]), config)
    altered = copy(rows); altered[4] = merge(altered[4], (; draw_seed = UInt64(72)))
    @test_throws ArgumentError FD.validate_checkpoint_rows(altered, config)
    altered[4] = merge(rows[4], (; policy_truth = 0.12))
    @test_throws ArgumentError FD.validate_checkpoint_rows(altered, config)
    mktempdir() do directory
        checkpoint = FD.open_checkpoint(directory, config)
        FD.append_dataset!(checkpoint, rows)
        @test length(FD.open_checkpoint(directory, config; resume = true).rows) == 36
        @test_throws ArgumentError FD.append_dataset!(checkpoint, rows)
        @test_throws ArgumentError FD.open_checkpoint(directory, merge(config, (; draws = 17)); resume = true)
        changed = merge(config, (; environment_contract = Dict("backend" => "b")))
        @test_throws ArgumentError FD.open_checkpoint(directory, changed; resume = true)
        write(joinpath(directory, "raw.tsv"), replace(read(joinpath(directory, "raw.tsv"), String), "complete" => "changed"))
        @test_throws ArgumentError FD.open_checkpoint(directory, config; resume = true)
    end
    mktempdir() do directory
        checkpoint = FD.open_checkpoint(directory, config)
        FD.append_dataset!(checkpoint, rows)
        path = joinpath(directory, "raw.tsv"); lines = readlines(path)
        write(path, join(lines[1:end-1], '\n') * "\n")
        @test_throws ArgumentError FD.open_checkpoint(directory, config; resume = true)
    end
    mktempdir() do directory
        checkpoint = FD.open_checkpoint(directory, config)
        FD.append_dataset!(checkpoint, rows)
        path = joinpath(directory, "metadata.toml")
        write(path, replace(read(path, String), "schema_version = 2" => "schema_version = 1"))
        @test_throws ArgumentError FD.open_checkpoint(directory, config; resume = true)
    end
end

@testset "simulation failures retain all declared attempts" begin
    f = factorial_fixture()
    rows = FD.evaluate_factorial_dataset(f.cell, 1; seed = 1,
        simulator = (args...; kwargs...) -> error("simulation failure"),
        prior_builder = (args...) -> f.priors)
    @test length(rows) == 36
    @test all(row.status == "simulation_failure" && !row.available && ismissing(row.covered) for row in rows)
    @test length(unique(row.dataset_id for row in rows)) == 1
end

@testset "fidelity seam is selected prospectively and retains known truth" begin
    f = factorial_fixture()
    fit = (args...) -> (; converged = true, mean = zeros(2), variational_covariance = Matrix{Float64}(I, 2, 2),
        corrected_covariance = Matrix{Float64}(I, 2, 2), n_iterations = 4, status = :converged)
    cdf = (args...; kwargs...) -> (; available = true, covered = false, pit = 0.99, status = :ok,
        cdf_change = 0.0, tail_error = 0.0, mass_change = 0.0)
    count = Ref(0)
    fidelity = function(rows, args...)
        count[] += 1
        error("deliberate fidelity failure")
    end
    rows = FD.evaluate_factorial_dataset(f.cell, 1; seed = 1, draws = 8, fidelity_replications = 1,
        simulator = (args...; kwargs...) -> f.dataset, prior_builder = (args...) -> f.priors,
        fit_engine = fit, checked_cdf = cdf, fidelity_engine = fidelity)
    @test count[] == 12
    @test all(row.available && !row.covered && row.fidelity_selected && !row.fidelity_available for row in rows)
    @test all(startswith(row.fidelity_status, "fidelity_exception:") for row in rows)
    @test all(isnan(row.endpoint_error) && isnan(row.mean_error_sd) for row in rows)
end

@testset "CDF numerical decision bounds survive the v2 bridge" begin
    f = factorial_fixture()
    raw_covariance = Matrix{Float64}(I, 2, 2)
    lr_covariance = 2raw_covariance
    fit = (args...) -> (; converged = true, mean = zeros(2), variational_covariance = raw_covariance,
        corrected_covariance = lr_covariance, n_iterations = 4, status = :converged)
    cdf = function(X, y, mean, covariance, truth; gaussian = false, kwargs...)
        lower = gaussian ? (covariance == raw_covariance ? 0.11 : 0.22) : 0.33
        (; available = true, covered = true, pit = 0.5, status = :ok,
            cdf_change = 0.0, tail_error = 0.0, mass_change = 0.0,
            decision_lower = lower, decision_upper = lower + 0.01)
    end
    evaluate(fit_engine) = FD.evaluate_factorial_dataset(f.cell, 1; seed = 1, draws = 8,
        simulator = (args...; kwargs...) -> f.dataset, prior_builder = (args...) -> f.priors,
        fit_engine, checked_cdf = cdf)
    rows = evaluate(fit)
    for (method, lower) in (("adaptive_reference", 0.33), ("variational", 0.11), ("linear_response", 0.22))
        matching = filter(row -> row.method == method, rows)
        @test all(row.cdf_decision_lower == lower && row.cdf_decision_upper == lower + 0.01 for row in matching)
        @test all(row.available && row.covered for row in matching)
    end
    raw_failure = (args...) -> merge(fit(args...), (; variational_covariance = nothing))
    skipped = evaluate(raw_failure)
    @test all(isnan(row.cdf_decision_lower) && !row.available for row in skipped if row.method == "variational")
    @test all(row.cdf_decision_lower == 0.22 && row.available for row in skipped if row.method == "linear_response")
    @test all(isnan(row.cdf_decision_lower) && isnan(row.cdf_decision_upper)
        for row in FD.evaluate_factorial_dataset(f.cell, 1; seed = 1,
            simulator = (args...; kwargs...) -> error("failure"), prior_builder = (args...) -> f.priors))
end

@eval FD.Legacy include(joinpath(@__DIR__, "stress", "policy_calibration_reference.jl"))

@testset "A1 dimensionless moment-error propagation" begin
    # m=[1,E(u0),E(u1),E(u0²),E(u0u1),E(u1²)].
    # e/(mass-e)=.1, giving dmean=[.2,.3] and dsecond=[.4,.3,.6].
    result = FD.moment_error_bounds([1.0, 1.0, -2.0, 3.0, -2.0, 5.0], 11.0, 1.0)
    @test result.mean_bound ≈ 0.3
    @test result.covariance_bounds ≈ [0.84 1.06; 1.06 1.89]
    @test result.covariance_relative_bound ≈ norm([0.84 1.06; 1.06 1.89]) / norm([2.0 0.0; 0.0 1.0])
    @test_throws ArgumentError FD.moment_error_bounds(ones(6), 1.0, -0.1)
    @test_throws ArgumentError FD.moment_error_bounds(ones(6), 1.0, 1.0)
end

@testset "A1 Gaussian moment quadrature uses a common dimensionless scale" begin
    R = FD.Legacy.LogisticTailReference
    covariance = [0.25 0.1; 0.1 0.64]
    scales = sqrt.(diag(covariance))
    expected = covariance ./ (scales * transpose(scales))
    for centre in ([0.0, 0.0], [10000.0, -20000.0])
        rule = R.bivariate_policy_reference(zeros(0, 2), Float64[], centre, covariance;
            rtol = 1e-7, atol = 1e-10)
        result = FD.standardised_moments(rule, centre, scales)
        @test maximum(abs.(result.mean)) < 1e-7
        @test result.covariance ≈ expected atol = 1e-7
        bounds = FD.moment_error_bounds(result.normalised_raw, result.mass, result.integration_error)
        @test bounds.mean_bound <= 5e-4
        @test bounds.covariance_relative_bound <= 5e-4
    end
    rule = R.bivariate_policy_reference(zeros(0, 2), Float64[], zeros(2), covariance)
    for invalid in ([0.0, 1.0], [-1.0, 1.0], [NaN, 1.0], [Inf, 1.0], [1.0])
        @test_throws ArgumentError FD.standardised_moments(rule, zeros(2), invalid)
    end
end

@testset "A1 exact oracle fixture passes dimensionless gates" begin
    R = FD.Legacy.LogisticTailReference
    X = [1.0 0.0; 1.0 0.0; 1.0 1.0]
    y = [0.0, 1.0, 1.0]; weights = [6, 4, 10]
    μ = [-0.3, 3.0]; covariance = 0.25Matrix{Float64}(I, 2, 2)
    coarse = R.bivariate_policy_reference(X, y, μ, covariance; weights, rtol = 1e-5, atol = 1e-8)
    fine = R.bivariate_policy_reference(X, y, μ, covariance; weights, rtol = 1e-7, atol = 1e-10)
    refined = FD.refined_standardised_moments(coarse, fine)
    @test refined.available
    @test isempty(refined.failures)
    @test refined.coarse_bounds.mean_bound <= 5e-4
    @test refined.coarse_bounds.covariance_relative_bound <= 5e-4
    @test refined.fine_bounds.mean_bound <= 5e-4
    @test refined.fine_bounds.covariance_relative_bound <= 5e-4
    @test refined.mean_change <= 5e-4
    @test refined.covariance_change <= 5e-4
    @test refined.moments.mean ≈ [-0.25029761203958156, 3.1492947081338274] atol = 1e-6
    @test refined.moments.covariance ≈ [0.14584935017531098 -0.0173479850902859; -0.0173479850902859 0.222385911201485] atol = 1e-6
end
