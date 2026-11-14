using Test, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "stress", "policy_public_api_validation.jl"))
using .PolicyPublicAPIValidation

@testset "Public validation options and endpoint-reference gates" begin
    @test parse_options(String[]).replications == 2
    @test parse_options(["--profile", "main"]).replications == 100
    o = parse_options(["--replications", "3", "--streams", "2", "--draws", "10,20", "--cells", "a,b"])
    @test o.draws == [10, 20] && o.streams == 2 && o.cells == ["a", "b"]
    for args in (["--streams", "0"], ["--draws", "1"], ["--seed", "-1"], ["--unknown", "1"])
        @test_throws ArgumentError parse_options(args)
    end
    probe = (; lower=-0.2, upper=0.5, error=1e-9, residual=1e-9, closure=1e-9, log_mass=0.0)
    reference = assess_endpoint_rules(probe, probe; scale=0.1)
    @test reference.available && reference.lower ≈ -0.02 && reference.upper == 0.05
    @test !assess_endpoint_rules(probe, merge(probe, (; error=1e-3)); scale=0.1).available
    @test !assess_endpoint_rules(probe, merge(probe, (; lower=-0.21)); scale=0.1).available
end

@testset "Public fits and streams retain independent reference failures" begin
    calls, seeds = Ref(0), Float64[]
    fitter = (data, prior, method) -> begin
        calls[] += 1
        (; posterior_mean=zeros(2), posterior_covariance=Matrix{Float64}(I, 2, 2),
            metadata=(; uncertainty_available=true), status=:converged)
    end
    reference = (fit, scale) -> (; available=false, status="unresolved", lower=NaN, upper=NaN,
        error=1.0, endpoint_change=1.0, mass_change=0.0)
    contrast = (fit, data; draws, rng) -> begin
        push!(seeds, rand(rng))
        (; mean=[0.1], lower=[-0.1], upper=[0.3], mcse=[0.01],
            endpoint_mc_bounds=(; lower_endpoint=(; lower=[-0.2], upper=[0.0]),
                upper_endpoint=(; lower=[0.2], upper=[0.4])),
            metadata=(; uncertainty=:gaussian_moment_approximation,
                endpoint_mc_scope=:conditional_on_gaussian_moments, endpoint_mc_simultaneous=false))
    end
    cell = first(PolicyPublicAPIValidation.Support.factorial_cells())
    rows = evaluate_public_dataset(cell, 1; streams=2, draws=[10, 20], fitter, ideal_reference=reference, public_contrast=contrast)
    @test calls[] == 6 && length(rows) == 24
    @test all(r.available && !r.reference_available && isnan(r.lower_endpoint_error) for r in rows)
    @test all(ismissing(r.lower_mc_contains_ideal) for r in rows)
    @test all(r.fit_calls_per_dataset_prior == 2 for r in rows)
    @test seeds[1] == seeds[2] == seeds[5] == seeds[9]
    @test seeds[1] != seeds[3]
    @test length(unique(r.dataset_id for r in rows)) == 1
    @test Set(r.prior_id for r in rows) == Set(("zero_s0p5", "zero_s1p5", "zero_s3p0"))
    failing = (fit, data; kwargs...) -> error("public failure")
    failed = evaluate_public_dataset(cell, 1; streams=1, draws=[10], fitter,
        ideal_reference=reference, public_contrast=failing)
    @test length(failed) == 6 && all(!r.available && r.status == "public_exception" for r in failed)
    good_reference = (fit, scale) -> (; available=true, status="ok", lower=-0.05, upper=0.25,
        error=1e-9, endpoint_change=1e-9, mass_change=0.0)
    good = only(evaluate_public_dataset(cell, 1; streams=1, draws=[10], fitter,
        ideal_reference=good_reference, public_contrast=contrast, methods=(:variational,),
        priors=[first(PolicyPublicAPIValidation.Support.prior_specs(cell, [-0.3, 0.9]))]))
    @test good.lower_endpoint_error ≈ -0.05 && good.upper_endpoint_error ≈ 0.05
    @test good.lower_endpoint_squared_error ≈ 0.0025
    @test good.lower_mc_contains_ideal && good.upper_mc_contains_ideal
end

@testset "Public summaries use dataset rather than draw-stream uncertainty" begin
    row(id, stream, error; available=true, reference_available=true) =
        (; cell_id="cell", dataset_id=id, prior_id="prior", method="variational", draws=500,
            stream, available, reference_available, lower_endpoint_error=error,
            upper_endpoint_error=error, lower_mc_contains_ideal=true, upper_mc_contains_ideal=true,
            width=0.4, mean_mcse=0.01, draw_seconds=0.1)
    rows = [row("a", 1, 0.0), row("a", 2, 2.0), row("b", 1, 3.0), row("b", 2, 5.0)]
    s = only(public_api_summaries(rows))
    @test s.datasets == 2 && s.attempted_streams == 4 && s.available_streams == 4
    @test s.lower_endpoint_error_mean == 2.5
    @test s.lower_endpoint_error_mcse ≈ std([1.0, 4.0]) / sqrt(2)
    @test s.lower_endpoint_error_count == 2
    @test s.lower_mc_contains_ideal_mean == 1
    one = only(public_api_summaries(rows[1:2]))
    @test isnan(one.lower_endpoint_error_mcse)
    absent = only(public_api_summaries([row("a", 1, NaN; reference_available=false)]))
    @test absent.lower_endpoint_error_count == 0 && isnan(absent.lower_endpoint_error_mean)
    @test absent.lower_mc_contains_ideal_count == 0
    unavailable = only(public_api_summaries([row("a", 1, NaN; available=false)]))
    @test unavailable.availability_mean == 0 && unavailable.availability_count == 1
    @test_throws ArgumentError public_api_summaries(vcat(rows, [rows[1]]))
end
