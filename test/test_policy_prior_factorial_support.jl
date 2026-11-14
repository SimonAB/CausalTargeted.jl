using Test, LinearAlgebra, Statistics

include(joinpath(@__DIR__, "stress", "policy_prior_factorial_support.jl"))
using .PolicyPriorFactorialSupport

@testset "Factorial cohorts, prior diagnostics and deterministic data" begin
    cells = factorial_cells()
    @test length(cells) == 8
    @test length(unique(c.id for c in cells)) == 8
    @test Set((c.n, c.control_n, c.beta_fixed[2]) for c in cells) ==
        Set((n, controls, effect) for n in (20, 50) for controls in (n ÷ 2, n ÷ 10) for effect in (0.9, 3.0))
    for cell in cells
        sim = simulate_factorial_dataset(cell, 7)
        @test sim == simulate_factorial_dataset(cell, 7)
        @test sim.beta == [-0.3, cell.beta_fixed[2]]
        @test size(sim.X) == (cell.n, 2)
        @test sim.X == hcat(ones(cell.n), sim.A)
        @test count(iszero, sim.A) == cell.control_n
        @test Set(sim.A) == Set([0, 1])
        logistic(x) = inv(1 + exp(-x))
        @test sim.arm_truth ≈ logistic(sum(sim.beta)) - logistic(sim.beta[1])
        @test sim.policy_scale == cell.control_n / cell.n
        @test sim.policy_truth ≈ sim.policy_scale * sim.arm_truth
        @test sim.separation in (:none, :quasi, :complete, :single_class)
        specs = prior_specs(cell, sim.beta)
        @test length(specs) == length(unique(p.id for p in specs)) == 12
        @test count(p -> p.deployable, specs) == 3
        for p in specs
            @test p.covariance == p.scale^2 * Matrix{Float64}(I, 2, 2)
            @test p.deployable == (p.centring == :zero)
            expected = p.centring == :zero ? zeros(2) : sim.beta +
                (p.centring == :oracle ? 0 : p.centring == :oracle_plus ? 2p.scale : -2p.scale) * [0, 1]
            @test p.mean == expected
            @test p.centring == :zero || occursin("oracle", p.id)
            @test simulate_factorial_dataset(merge(cell, (; fitted_prior=p)), 7) == sim
        end
        @test sim.dataset_id != simulate_factorial_dataset(cell, 8).dataset_id
        @test sim.draw_seed != simulate_factorial_dataset(cell, 8).draw_seed
        @test sim.dataset_id != simulate_factorial_dataset(cell, 7; seed=11).dataset_id
    end
    predictive = factorial_cells(; experiment=:prior_predictive)
    @test length(predictive) == length(unique(c.id for c in predictive)) == 12
    @test Set(c.prior_scale for c in predictive) == Set((0.5, 1.5, 3.0))
    for cell in predictive
        sim = simulate_factorial_dataset(cell, 1)
        @test sim == simulate_factorial_dataset(cell, 1)
        @test sim.beta != simulate_factorial_dataset(cell, 2).beta
        p = only(prior_specs(cell, sim.beta))
        @test p.mean == zeros(2) && p.deployable && p.scale == cell.prior_scale
        @test count(iszero, sim.A) == cell.control_n
    end
    @test_throws ArgumentError factorial_cells(; experiment=:unknown)
    for bad in (merge(cells[1], (; control_n=0)), merge(cells[1], (; control_n=cells[1].n)),
        merge(cells[1], (; n=1)), merge(cells[1], (; beta_fixed=[0.0])),
        merge(cells[1], (; experiment=:unknown)), merge(predictive[1], (; prior_scale=-1.0)))
        @test_throws ArgumentError simulate_factorial_dataset(bad, 1)
    end
    @test_throws ArgumentError simulate_factorial_dataset(cells[1], 0)
    @test_throws ArgumentError simulate_factorial_dataset(cells[1], 1; seed=-1)
    @test_throws ArgumentError prior_specs(cells[1], [NaN, 1.0])
end

factorial_row(dataset, prior, method; available=true, covered=true, width=2.0,
    policy_mean=0.1, policy_truth=0.0, experiment=:fixed_parameter, pit=0.5,
    status=:ok, cell_id="cell", kwargs...) =
    (; dataset_id=String(dataset), cell_id, prior_id=String(prior), method, experiment,
        available, covered, width, policy_mean, policy_truth, pit, status, kwargs...)

@testset "Failure denominators and finite factorial metrics" begin
    rows = [factorial_row("1", "p", :variational; covered=true, policy_mean=-1.0, pit=0.0,
            reference_content=0.9, endpoint_error=0.1, mean_error_sd=0.2, covariance_error=0.3, seconds=1.0),
        factorial_row("2", "p", :variational; covered=false, width=Inf, policy_mean=1.0, pit=1.0),
        factorial_row("3", "p", :variational; available=false, covered=missing, width=missing,
            policy_mean=missing, status=:fit_failure)]
    s = only(factorial_summaries(rows).summaries)
    @test (s.attempted, s.available, s.failed, s.coverage_count) == (3, 2, 1, 2)
    @test s.coverage_available == 0.5 && s.coverage_all_attempts == 1/3
    @test s.mcse_available ≈ sqrt(0.25 / 2)
    @test s.mcse_all_attempts ≈ sqrt((1/3) * (2/3) / 3)
    @test s.width_count == 1 && s.width_mean == 2
    @test s.policy_error_count == 2 && s.policy_error_bias == 0 && s.policy_error_rmse == 1
    @test s.reference_content_count == 1 && s.reference_content_mean == 0.9
    @test s.endpoint_error_count == s.mean_error_sd_count == s.covariance_error_count == s.seconds_count == 1
    @test s.failure_reasons == "fit_failure=1"
    @test s.separation_unknown == 3
    @test s.pit_count == 0 && s.pit_label == :fixed_parameter_not_sbc
    predictive = only(factorial_summaries([merge(r, (; experiment=:prior_predictive)) for r in rows]).summaries)
    @test predictive.pit_label == :prior_predictive
    @test predictive.pit_deciles == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    partial = only(factorial_summaries([factorial_row("1", "p", :reference; covered=missing)]).summaries)
    @test partial.available == 1 && partial.coverage_count == 0 && isnan(partial.coverage_available)
    failed = only(factorial_summaries([rows[end]]).summaries)
    @test failed.width_count == 0 && isnan(failed.width_mean)
    @test failed.coverage_all_attempts == 0 && isnan(failed.coverage_available)
    @test isempty(factorial_summaries(NamedTuple[]).summaries)
    @test_throws ArgumentError factorial_summaries(vcat(rows, [rows[1]]))
    @test_throws ArgumentError factorial_summaries([factorial_row("1", "p", :reference; covered=2)])
    @test_throws ArgumentError factorial_summaries([rows[1],
        factorial_row("1", "other_prior", :reference; experiment=:prior_predictive)])
    separated = only(factorial_summaries([merge(rows[1], (; separation=:none)),
        merge(rows[2], (; separation=:complete)), merge(rows[3], (; separation=:quasi))]).summaries)
    @test (separated.separation_none, separated.separation_complete, separated.separation_quasi) == (1, 1, 1)
end

@testset "Paired common-dataset differences and MCSE" begin
    s = paired_metric_summary([0.0, 1.0, missing, Inf], [1.0, 0.0, 2.0, 3.0])
    @test s.count == 2 && s.delta == 0
    @test s.mcse ≈ std([1.0, -1.0]) / sqrt(2)
    @test isnan(paired_metric_summary([1.0], [2.0]).mcse)
    @test isnan(paired_metric_summary([missing], [1.0]).delta)
    @test_throws ArgumentError paired_metric_summary([1.0], [1.0, 2.0])
    rows = [factorial_row("1", "p", :variational; covered=false, width=1.0),
        factorial_row("1", "p", :linear_response; width=3.0),
        factorial_row("2", "p", :variational; width=2.0),
        factorial_row("2", "p", :linear_response; covered=false, width=1.0),
        factorial_row("3", "p", :variational; width=7.0),
        factorial_row("3", "p", :linear_response; available=false, covered=missing, width=missing),
        factorial_row("4", "p", :variational; width=100.0)]
    pair = only(factorial_summaries(rows).paired_methods)
    @test pair.attempts == 3 && pair.paired == 2
    @test (pair.attempted_a, pair.attempted_b, pair.unmatched_a, pair.unmatched_b) == (4, 3, 1, 0)
    @test pair.coverage_delta == 0 && pair.coverage_delta_mcse ≈ 1
    @test pair.width_count == 2 && pair.width_delta == 0.5
    @test pair.width_delta_mcse ≈ std([2.0, -1.0]) / sqrt(2)
    @test pair.availability_delta ≈ -1/3
    @test pair.all_attempts_inclusion_delta ≈ -1/3
    prior_rows = [merge(r, (; prior_id=r.method == :variational ? "a" : "b", method=:reference)) for r in rows]
    prior_pair = only(factorial_summaries(prior_rows).paired_priors)
    @test prior_pair.prior_a == "a" && prior_pair.prior_b == "b"
    @test prior_pair.width_count == 2 && prior_pair.width_delta == 0.5
    @test prior_pair.comparison_label == :exploratory
    none = only(factorial_summaries(rows[5:6]).paired_methods)
    @test none.paired == 0 && none.width_count == 0 && isnan(none.width_delta)
end

@testset "Central prior-predictive inclusion retains numerical ambiguity" begin
    rows = Any[merge(factorial_row(string(i), "p", :variational; experiment=:prior_predictive),
        (; cdf_decision_lower=lo, cdf_decision_upper=hi, status="ok"))
        for (i,(lo,hi)) in enumerate(((0.4,0.41),(0.12,0.13),(0.249,0.251)))]
    push!(rows,merge(factorial_row("4","p",:variational;experiment=:prior_predictive,
        available=false,covered=missing),(;cdf_decision_lower=0.974,cdf_decision_upper=0.976,status="boundary_ambiguous")))
    push!(rows,merge(factorial_row("5","p",:variational;experiment=:prior_predictive,
        available=false,covered=missing),(;cdf_decision_lower=NaN,cdf_decision_upper=NaN,status="fit_failure")))
    summary=only(factorial_summaries(rows).summaries)
    @test summary.central50_count == 3
    @test summary.central50_ambiguous == 1
    @test summary.central50_unavailable == 2
    @test summary.central50_coverage_available == 1/3
    @test summary.central50_coverage_all_attempts == 1/5
    @test summary.central80_count == 4
    @test summary.central80_coverage_available == 3/4
    @test summary.central80_coverage_all_attempts == 3/5
    fixed=only(factorial_summaries([merge(rows[1],(;experiment=:fixed_parameter))]).summaries)
    @test fixed.central50_count == 0 && isnan(fixed.central50_coverage_available)
end

@testset "Independent fidelity metrics survive ambiguous truth inclusion" begin
    raw = merge(factorial_row("1", "p", :variational; available=false,covered=missing),
        (; status="boundary_ambiguous",reference_content=0.94,reference_content_available=true,
        endpoint_error=0.1,endpoint_available=true,mean_error_sd=0.2,covariance_error=0.3,moments_available=true))
    corrected = merge(raw,(;method=:linear_response,reference_content=0.95,endpoint_error=0.05))
    summary = only(filter(r->r.method==:variational,factorial_summaries([raw,corrected]).summaries))
    @test summary.coverage_count == 0
    @test summary.reference_content_count == 1 && summary.reference_content_mean == 0.94
    @test summary.endpoint_error_count == 1 && summary.endpoint_error_mean == 0.1
    @test summary.mean_error_sd_count == 1
    pair = only(factorial_summaries([raw,corrected]).paired_methods)
    @test pair.reference_content_count == 1 && pair.reference_content_delta ≈ 0.01
    @test pair.endpoint_error_count == 1 && pair.endpoint_error_delta ≈ -0.05
end
