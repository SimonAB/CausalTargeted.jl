using Test, LinearAlgebra, Statistics, DataFrames, CSV, TOML
LinearAlgebra.BLAS.set_num_threads(1)

using CausalTargeted
include(joinpath(@__DIR__, "stress", "causal_policy_validation.jl"))
using .CausalPolicyValidation

@inline _test_logistic(x::Real) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))

@testset "Phase 4 DGP cells, priors, and cohort psi_true" begin
    cells = causal_policy_cells()
    @test length(cells) == 12
    @test length(unique(c.id for c in cells)) == 12
    @test Set(c.mechanism for c in cells) == Set((
        "mc_baseline", "mc_corr", "mc_overlap",
        "mc_interact_spec", "mc_interact_omit", "mc_nonlinear_omit",
    ))
    @test Set(c.n for c in cells) == Set((80, 200))

    priors4 = prior_specs(4)
    priors5 = prior_specs(5)
    @test [p.id for p in priors4] == ["zero_s1p5", "zero_s3p0"]
    @test priors4[1].mean == zeros(4) && priors4[1].covariance == 2.25 * Matrix{Float64}(I, 4, 4)
    @test priors4[2].mean == zeros(4) && priors4[2].covariance == 9.0 * Matrix{Float64}(I, 4, 4)
    @test priors5[1].mean == zeros(5) && priors5[2].covariance == 9.0 * Matrix{Float64}(I, 5, 5)
    @test_throws ArgumentError prior_specs(0)

    for cell in cells
        sim = simulate_causal_policy_dataset(cell, 1; seed = 20261121)
        df = sim.df
        @test df isa DataFrame
        @test propertynames(df) == [:W1, :W2, :A, :Y]
        @test nrow(df) == cell.n
        @test eltype(df.W1) == Float64 && eltype(df.W2) == Float64
        @test eltype(df.A) == Int && eltype(df.Y) == Int
        @test all(isfinite, df.W1) && all(isfinite, df.W2)
        @test all(a -> a in (0, 1), df.A)
        @test all(y -> y in (0, 1), df.Y)
        @test isfinite(sim.psi_true) && -1.0 <= sim.psi_true <= 1.0

        # Independent recomputation of cohort psi_true
        eta_fn = (a, w1, w2) -> begin
            e = -0.5 + 1.0 * a + 0.6 * w1 - 0.5 * w2
            cell.interaction_truth && (e += 0.9 * a * w1)
            cell.nonlinear_truth && (e += 0.8 * (w1^2 - 1.0))
            e
        end
        expected_psi = mean(
            _test_logistic(eta_fn(1, df.W1[i], df.W2[i])) -
            _test_logistic(eta_fn(df.A[i], df.W1[i], df.W2[i]))
            for i in 1:cell.n
        )
        @test sim.psi_true ≈ expected_psi atol = 1e-14
    end

    # Confounder correlation checks: mc_corr (rho=0.7) vs mc_baseline (rho=0.0)
    corr_cell = only(filter(c -> c.mechanism == "mc_corr" && c.n == 200, cells))
    base_cell = only(filter(c -> c.mechanism == "mc_baseline" && c.n == 200, cells))
    corr_w1 = Float64[]; corr_w2 = Float64[]
    base_w1 = Float64[]; base_w2 = Float64[]
    for rep in 1:15
        sc = simulate_causal_policy_dataset(corr_cell, rep; seed = 20261121)
        sb = simulate_causal_policy_dataset(base_cell, rep; seed = 20261121)
        append!(corr_w1, sc.W1); append!(corr_w2, sc.W2)
        append!(base_w1, sb.W1); append!(base_w2, sb.W2)
    end
    @test cor(corr_w1, corr_w2) > 0.60
    @test abs(cor(base_w1, base_w2)) < 0.15
end

@testset "Additive (p=4) and interaction (p=5) design builders" begin
    cells = causal_policy_cells()
    base_cell = only(filter(c -> c.mechanism == "mc_baseline" && c.n == 80, cells))
    spec_cell = only(filter(c -> c.mechanism == "mc_interact_spec" && c.n == 80, cells))
    omit_cell = only(filter(c -> c.mechanism == "mc_interact_omit" && c.n == 80, cells))

    @test base_cell.p == 4 && !base_cell.interaction_basis && !base_cell.misspecified
    @test spec_cell.p == 5 && spec_cell.interaction_basis && !spec_cell.misspecified
    @test omit_cell.p == 4 && !omit_cell.interaction_basis && omit_cell.misspecified

    sim4 = simulate_causal_policy_dataset(base_cell, 3; seed = 20261121)
    @test size(sim4.X_obs) == (80, 4)
    @test size(sim4.X_alt) == (80, 4)
    @test sim4.X_obs == hcat(ones(80), Float64.(sim4.A), sim4.W1, sim4.W2)
    @test sim4.X_alt == hcat(ones(80), ones(80), sim4.W1, sim4.W2)
    @test additive_basis(sim4.df, sim4.A) == sim4.X_obs

    # Verify additive_basis matches CausalTargeted's CovariateSchema design matrix
    schema = fit_covariate_schema(sim4.df, [:A, :W1, :W2])
    @test design_matrix(schema, sim4.df) == sim4.X_obs

    sim5 = simulate_causal_policy_dataset(spec_cell, 3; seed = 20261121)
    @test size(sim5.X_obs) == (80, 5)
    @test size(sim5.X_alt) == (80, 5)
    @test sim5.X_obs == hcat(ones(80), Float64.(sim5.A), sim5.W1, sim5.W2, Float64.(sim5.A) .* sim5.W1)
    @test sim5.X_alt == hcat(ones(80), ones(80), sim5.W1, sim5.W2, sim5.W1)
    @test interaction_basis(sim5.df, sim5.A) == sim5.X_obs
end

@testset "Deterministic reproducibility" begin
    cells = causal_policy_cells()
    for cell in (cells[1], cells[7])
        s1 = simulate_causal_policy_dataset(cell, 4; seed = 12345)
        s2 = simulate_causal_policy_dataset(cell, 4; seed = 12345)
        @test s1.df == s2.df
        @test s1.psi_true === s2.psi_true
        @test s1.dataset_id == s2.dataset_id
        @test s1.draw_seed === s2.draw_seed
        @test s1.psis_seed === s2.psis_seed
        @test s1.X_obs == s2.X_obs
        @test s1.X_alt == s2.X_alt

        s_other_rep = simulate_causal_policy_dataset(cell, 5; seed = 12345)
        s_other_seed = simulate_causal_policy_dataset(cell, 4; seed = 99999)
        @test s1.df != s_other_rep.df
        @test s1.dataset_id != s_other_rep.dataset_id
        @test s1.draw_seed != s_other_rep.draw_seed
        @test s1.df != s_other_seed.df
        @test s1.dataset_id != s_other_seed.dataset_id
    end
end

@testset "CLI options, real CausalTargeted evaluation, and summary aggregation" begin
    @test parse_options(String[]).profile == "pilot"
    @test parse_options(String[]).replications == 50
    @test parse_options(["--profile", "main"]).replications == 500
    opts = parse_options(["--profile", "pilot", "--replications", "3", "--seed", "77", "--out", "/tmp/cpv_test"])
    @test opts.profile == "pilot" && opts.replications == 3 && opts.seed == 77 && opts.out == "/tmp/cpv_test"
    @test_throws ArgumentError parse_options(["--profile", "invalid"])
    @test_throws ArgumentError parse_options(["--replications", "0"])
    @test_throws ArgumentError parse_options(["--seed", "-1"])
    @test_throws ArgumentError parse_options(["--unknown", "1"])

    # Evaluate real fit_bayesian_logistic + bayesian_policy_contrast with a mock Tier 1 PSIS reference
    cells = causal_policy_cells()
    test_cells = [
        only(filter(c -> c.id == "mc_baseline_n80", cells)),
        only(filter(c -> c.id == "mc_interact_spec_n80", cells)),
    ]
    mock_ref = (X, y, mu0, Sigma0, X_alt, X_obs; rng = nothing) -> begin
        p = size(X, 2)
        available = length(y) > 0 && p >= 4
        # Deterministic synthetic PSIS draws centered near 0.2
        draws = fill(0.20, 200) .+ collect(range(-0.10, 0.10; length = 200))
        weights = fill(1.0 / 200, 200)
        (;
            available,
            posterior_mean = copy(mu0),
            posterior_covariance = 0.25 * Matrix{Float64}(I, p, p),
            policy_mean = 0.20,
            policy_lower = quantile(draws, 0.025),
            policy_upper = quantile(draws, 0.975),
            policy_mcse = 0.005,
            pareto_k = 0.25,
            ess = 1500.0,
            policy_draws = draws,
            weights,
            failure_reason = "ok",
        )
    end
    mock_wic = (draws, weights, lo, hi) -> sum(w for (d, w) in zip(draws, weights) if lo <= d <= hi; init = 0.0)

    mktempdir() do tmpdir
        outdir = joinpath(tmpdir, "run_mock")
        res = run_validation(
            (; profile = "pilot", replications = 2, seed = 20261121, draws = 200,
                out = outdir, cells = [c.id for c in test_cells]);
            reference_fn = mock_ref, interval_content_fn = mock_wic, quiet = true)

        # 2 cells × 2 reps × 2 priors × 3 methods = 24 rows
        @test length(res.rows) == 24
        @test all(r -> r.available && r.reference_available && r.fidelity_available, res.rows)
        @test all(r -> isfinite(r.mean_error_sd) && isfinite(r.covariance_error), res.rows)
        @test all(r -> isfinite(r.policy_error_vs_ref) && isfinite(r.endpoint_error), res.rows)
        @test all(r -> 0.0 <= r.reference_content <= 1.0, res.rows)
        @test all(r -> r.causal_covered isa Bool && isfinite(r.causal_bias) && r.width > 0, res.rows)

        # Verify output files written
        for fname in ("raw.tsv", "summary.tsv", "paired_methods.tsv", "metadata.toml", "report.md")
            @test isfile(joinpath(outdir, fname))
        end
        meta = TOML.parsefile(joinpath(outdir, "metadata.toml"))
        @test meta["status"] == "complete"
        @test meta["rows"] == 24
        @test meta["datasets"] == 4

        raw_tbl = CSV.File(joinpath(outdir, "raw.tsv"); delim = '\t', missingstring = "NA")
        @test length(raw_tbl) == 24

        # 2 cells × 2 priors × 3 methods = 12 summary rows
        @test length(res.summaries) == 12
        # 2 cells × 2 priors = 4 paired method rows
        @test length(res.paired) == 4
        @test all(p -> p.method_a == "variational" && p.method_b == "linear_response" && p.attempts == 2, res.paired)
    end

    # Gate-failure isolation: failed Tier 1 PSIS does NOT suppress VI or LRVB causal recovery
    fail_ref = (X, y, mu0, Sigma0, X_alt, X_obs; rng = nothing) -> (;
        available = false,
        posterior_mean = fill(NaN, size(X, 2)),
        posterior_covariance = fill(NaN, size(X, 2), size(X, 2)),
        policy_mean = NaN,
        policy_lower = NaN,
        policy_upper = NaN,
        policy_mcse = NaN,
        pareto_k = 0.85,
        ess = 120.0,
        failure_reason = "pareto_k_high",
    )
    cell0 = test_cells[1]
    sim0 = simulate_causal_policy_dataset(cell0, 1; seed = 20261121)
    p0 = first(prior_specs(cell0.p))
    rows_fail = evaluate_dataset_prior(cell0, sim0, p0, 1; seed = 20261121, n_draws = 200,
        reference_fn = fail_ref, interval_content_fn = mock_wic)
    @test length(rows_fail) == 3
    @test !rows_fail[1].available && ismissing(rows_fail[1].causal_covered) && rows_fail[1].pareto_k == 0.85
    @test rows_fail[2].available && !rows_fail[2].reference_available && !rows_fail[2].fidelity_available
    @test rows_fail[3].available && !rows_fail[3].reference_available && !rows_fail[3].fidelity_available
    @test rows_fail[2].causal_covered isa Bool && isnan(rows_fail[2].endpoint_error)

    # Duplicate rows rejected by summary & paired aggregators
    @test_throws ArgumentError summarise_validation_rows(vcat(rows_fail, rows_fail[1:1]))
    @test_throws ArgumentError paired_method_summaries(vcat(rows_fail, rows_fail[2:2]))
end

const _PSIS_REF_PATH = joinpath(@__DIR__, "stress", "logistic_psis_reference.jl")
if isfile(_PSIS_REF_PATH)
    @testset "End-to-end 2-replication PSIS smoke run" begin
        mktempdir() do tmpdir
            outdir = joinpath(tmpdir, "psis_smoke")
            res = run_validation((;
                profile = "pilot",
                replications = 2,
                seed = 20261121,
                draws = 500,
                out = outdir,
                cells = ["mc_baseline_n80", "mc_interact_spec_n80"],
            ); quiet = true)
            @test length(res.rows) == 24
            @test any(r -> r.method == "tier1_psis" && r.available && r.pareto_k < 0.7 && r.psis_ess >= 400, res.rows)
            for r in res.rows
                if r.fidelity_available
                    @test isfinite(r.mean_error_sd) && r.mean_error_sd >= 0
                    @test isfinite(r.covariance_error) && r.covariance_error >= 0
                    @test isfinite(r.policy_error_vs_ref)
                    @test 0.0 <= r.reference_content <= 1.0
                    @test isfinite(r.endpoint_error) && r.endpoint_error >= 0
                end
            end
            @test isfile(joinpath(outdir, "raw.tsv"))
            @test isfile(joinpath(outdir, "summary.tsv"))
            @test isfile(joinpath(outdir, "paired_methods.tsv"))
            @test isfile(joinpath(outdir, "metadata.toml"))
            @test isfile(joinpath(outdir, "report.md"))
        end
    end
end
