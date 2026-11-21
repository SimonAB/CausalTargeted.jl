using Test, LinearAlgebra, Statistics, DataFrames, CSV, TOML, Distributions
LinearAlgebra.BLAS.set_num_threads(1)

using CausalTargeted
include(joinpath(@__DIR__, "stress", "phase5_heldout_confirmation.jl"))
using .Phase5HeldoutConfirmation

@inline _test_logistic(x::Real) = x >= 0 ? inv(1.0 + exp(-x)) : (ex = exp(x); ex / (1.0 + ex))

@testset "Phase 5 held-out scenarios, priors, and exact cohort psi_true" begin
    cells = heldout_cells()
    @test length(cells) == 3
    @test [c.id for c in cells] == [
        "heldout_n80_c24_b1p6",
        "heldout_n150_c30_b4p0",
        "heldout_n30_c1_b0p9",
    ]

    expected_specs = Dict(
        "heldout_n80_c24_b1p6" => (n = 80, n0 = 24, n1 = 56, beta = [-0.6, 1.6]),
        "heldout_n150_c30_b4p0" => (n = 150, n0 = 30, n1 = 120, beta = [-2.0, 4.0]),
        "heldout_n30_c1_b0p9" => (n = 30, n0 = 1, n1 = 29, beta = [-0.3, 0.9]),
    )
    for cell in cells
        sp = expected_specs[cell.id]
        @test cell.n == sp.n
        @test cell.n0 == sp.n0
        @test cell.n1 == sp.n1
        @test cell.beta_true == sp.beta

        sim = simulate_heldout_dataset(cell, 1; seed = 20261121)
        @test sim.df isa DataFrame
        @test propertynames(sim.df) == [:A, :Y]
        @test nrow(sim.df) == sp.n
        @test count(==(0), sim.df.A) == sp.n0
        @test count(==(1), sim.df.A) == sp.n1
        @test all(y -> y in (0, 1), sim.df.Y)
        @test sim.X_obs == hcat(ones(sp.n), Float64.(sim.A))
        @test sim.X_alt == hcat(ones(sp.n), ones(sp.n))

        expected_psi = (sp.n0 / sp.n) * (_test_logistic(sp.beta[1] + sp.beta[2]) - _test_logistic(sp.beta[1]))
        @test sim.psi_true ≈ expected_psi atol = 1e-14

        schema = fit_covariate_schema(sim.df, [:A])
        @test design_matrix(schema, sim.df) == sim.X_obs
    end

    priors = prior_specs(2)
    @test [p.id for p in priors] == ["zero_s1p5", "zero_s3p0"]
    @test priors[1].mean == zeros(2) && priors[1].covariance == 2.25 * Matrix{Float64}(I, 2, 2)
    @test priors[2].mean == zeros(2) && priors[2].covariance == 9.0 * Matrix{Float64}(I, 2, 2)
    @test_throws ArgumentError prior_specs(0)
end

@testset "Laplace MAP baseline solver accuracy and failure handling" begin
    cell = first(heldout_cells())
    sim = simulate_heldout_dataset(cell, 1; seed = 20261121)
    prior = first(prior_specs(2))
    res = fit_laplace_map(sim.X, sim.y, prior.mean, prior.covariance)
    @test res.converged
    @test res.status == "converged"
    @test length(res.posterior_mean) == 2
    @test size(res.posterior_covariance) == (2, 2)
    @test issymmetric(res.posterior_covariance)
    @test isposdef(res.posterior_covariance)

    # Verify exact first-order optimality and observed negative Hessian inverse
    p_hat = _test_logistic.(sim.X * res.posterior_mean)
    grad = transpose(sim.X) * (sim.y .- p_hat) .- (prior.covariance \ (res.posterior_mean .- prior.mean))
    @test norm(grad, Inf) < 1e-8
    H_obs = transpose(sim.X) * (sim.X .* (p_hat .* (1.0 .- p_hat))) + inv(prior.covariance)
    @test res.posterior_covariance ≈ inv(H_obs) atol = 1e-10

    bad_prior = fit_laplace_map(sim.X, sim.y, zeros(2), [1.0 2.0; 2.0 1.0])
    @test !bad_prior.converged
    @test bad_prior.status == "hessian_not_pd"

    X_nan = copy(sim.X)
    X_nan[1, 1] = NaN
    bad_input = fit_laplace_map(X_nan, sim.y, prior.mean, prior.covariance)
    @test !bad_input.converged
    @test bad_input.status == "newton_failed"
end

@testset "CLI option parsing" begin
    dflt = parse_options(String[])
    @test dflt.replications == 1000
    @test dflt.seed == 20261121
    @test dflt.draws == 2000
    @test isempty(dflt.cells)

    opts = parse_options(["--replications", "5", "--seed", "77", "--out", "/tmp/p5_test", "--draws", "500"])
    @test opts.replications == 5 && opts.seed == 77 && opts.out == "/tmp/p5_test" && opts.draws == 500
    @test_throws ArgumentError parse_options(["--replications", "0"])
    @test_throws ArgumentError parse_options(["--seed", "-1"])
    @test_throws ArgumentError parse_options(["--draws", "1"])
    @test_throws ArgumentError parse_options(["--unknown", "1"])
end

@testset "Gate-failure isolation and duplicate row rejection" begin
    cell0 = first(heldout_cells())
    sim0 = simulate_heldout_dataset(cell0, 1; seed = 20261121)
    p0 = first(prior_specs(2))
    fail_ref = (X, y, mu0, Sigma0, X_alt, X_obs; rng = nothing) -> (;
        available = false,
        beta_mean = fill(NaN, size(X, 2)),
        beta_cov = fill(NaN, size(X, 2), size(X, 2)),
        policy_mean = NaN,
        policy_lower = NaN,
        policy_upper = NaN,
        mcse = NaN,
        khat = 0.85,
        ess = 120.0,
        failure_reason = "khat_gate",
    )
    rows_fail = evaluate_dataset_prior(cell0, sim0, p0, 1; seed = 20261121, n_draws = 200,
        reference_fn = fail_ref, interval_content_fn = (d, w, lo, hi) -> NaN)
    @test length(rows_fail) == 4
    @test [r.method for r in rows_fail] == ["tier1_psis", "variational", "linear_response", "laplace_map"]
    @test !rows_fail[1].available && ismissing(rows_fail[1].causal_covered) && rows_fail[1].pareto_k == 0.85
    for idx in 2:4
        @test rows_fail[idx].available
        @test !rows_fail[idx].reference_available
        @test !rows_fail[idx].fidelity_available
        @test rows_fail[idx].causal_covered isa Bool
        @test isnan(rows_fail[idx].endpoint_error)
    end

    @test_throws ArgumentError summarise_confirmation_rows(vcat(rows_fail, rows_fail[1:1]))
    @test_throws ArgumentError paired_method_summaries(vcat(rows_fail, rows_fail[2:2]))
end

@testset "End-to-end 2-rep smoke run, outputs, and deterministic reproducibility" begin
    mktempdir() do tmpdir
        out1 = joinpath(tmpdir, "run1")
        out2 = joinpath(tmpdir, "run2")

        res1 = run_confirmation((;
            replications = 2,
            seed = 20261121,
            draws = 2000,
            out = out1,
            cells = String[],
        ); quiet = true)

        res2 = run_confirmation((;
            replications = 2,
            seed = 20261121,
            draws = 2000,
            out = out2,
            cells = String[],
        ); quiet = true)

        # 3 cells × 2 reps × 2 priors × 4 methods = 48 rows
        @test length(res1.rows) == 48
        @test length(res2.rows) == 48
        # 3 cells × 2 priors × 4 methods = 24 summary rows
        @test length(res1.summaries) == 24
        # 3 cells × 2 priors × 2 pairs (LRVB-VI, LRVB-Laplace) = 12 paired rows
        @test length(res1.paired) == 12
        @test Set(p.contrast for p in res1.paired) == Set((
            "linear_response_minus_variational",
            "linear_response_minus_laplace_map",
        ))
        @test all(p -> p.attempts == 2 && p.method_b == "linear_response", res1.paired)

        # Check all 4 methods and all 3 cells appear and produce finite metrics
        @test Set(r.cell_id for r in res1.rows) == Set((
            "heldout_n80_c24_b1p6",
            "heldout_n150_c30_b4p0",
            "heldout_n30_c1_b0p9",
        ))
        @test Set(r.method for r in res1.rows) == Set((
            "tier1_psis",
            "variational",
            "linear_response",
            "laplace_map",
        ))
        @test any(r -> r.method == "tier1_psis" && r.available && r.pareto_k < 0.7 && r.psis_ess >= 400, res1.rows)
        for method in ("variational", "linear_response", "laplace_map")
            mrows = filter(r -> r.method == method, res1.rows)
            @test length(mrows) == 12
            @test all(r -> r.available && isfinite(r.causal_bias) && r.causal_covered isa Bool && r.width > 0, mrows)
            @test any(r -> r.fidelity_available, mrows)
            for r in mrows
                if r.fidelity_available
                    @test isfinite(r.mean_error_sd) && r.mean_error_sd >= 0
                    @test isfinite(r.covariance_error) && r.covariance_error >= 0
                    @test isfinite(r.policy_error_vs_ref)
                    @test 0.0 <= r.reference_content <= 1.0
                    @test isfinite(r.endpoint_error) && r.endpoint_error >= 0
                end
            end
        end

        # Verify deterministic reproducibility across repeated runs (all non-timing fields bit-for-bit identical)
        for (r1, r2) in zip(res1.rows, res2.rows)
            for k in propertynames(r1)
                k == :seconds && continue
                @test isequal(getproperty(r1, k), getproperty(r2, k))
            end
        end
        for (s1, s2) in zip(res1.summaries, res2.summaries)
            for k in propertynames(s1)
                k in (:seconds_mean, :seconds_total) && continue
                @test isequal(getproperty(s1, k), getproperty(s2, k))
            end
        end

        # Verify output files creation and schema
        for fname in ("raw.tsv", "summary.tsv", "paired_methods.tsv", "metadata.toml", "report.md")
            @test isfile(joinpath(out1, fname))
            @test filesize(joinpath(out1, fname)) > 0
        end
        meta = TOML.parsefile(joinpath(out1, "metadata.toml"))
        @test meta["status"] == "complete"
        @test meta["schema"] == "phase5-heldout-confirmation-v1"
        @test meta["replications"] == 2
        @test meta["seed"] == 20261121
        @test meta["draws"] == 2000
        @test meta["datasets"] == 6
        @test meta["rows"] == 48
        @test meta["blas_threads"] == 1

        raw_tbl = CSV.File(joinpath(out1, "raw.tsv"); delim = '\t', missingstring = "NA")
        sum_tbl = CSV.File(joinpath(out1, "summary.tsv"); delim = '\t', missingstring = "NA")
        pair_tbl = CSV.File(joinpath(out1, "paired_methods.tsv"); delim = '\t', missingstring = "NA")
        @test length(raw_tbl) == 48
        @test length(sum_tbl) == 24
        @test length(pair_tbl) == 12
    end
end
