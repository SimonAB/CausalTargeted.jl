"""Phase 4: posterior MAR imputation draws and LMTP pooling."""

using CausalTargeted
import CausalDynamics
using CausalDynamics: MissingnessSpec, certify_missingness
using DataFrames
using Random
using StableRNGs
using Statistics
using Test
using Distributions: TDist, quantile

@testset "impute_posterior MAR Gaussian" begin
    df, truth = CausalTargeted.simulate_missing_outcome_mtp(80; rng = StableRNG(50))
    cert = certify_missingness(
        MissingnessSpec(:Y; regime = :mar, conditioning_set = [:W]),
    )
    draws = impute_posterior(
        df, :Y, [:W];
        treatment = :A,
        certificate = cert,
        n_draws = 5,
        rng = StableRNG(50),
    )
    @test draws isa ImputationDraws
    @test draws.n_draws == 5
    @test length(draws.draws) == 5
    @test draws.method === :gaussian_mar
    @test draws.mar_set == [:W]
    for d in draws.draws
        @test !any(ismissing, d.Y)
        @test nrow(d) == nrow(df)
        # Observed outcomes preserved
        for i in 1:nrow(df)
            if !ismissing(df.Y[i])
                @test d.Y[i] == df.Y[i]
            end
        end
    end
end

@testset "posterior Gaussian imputation propagates fit uncertainty" begin
    rng = StableRNG(155)
    x_observed = collect(range(-1.0, 1.0; length = 8))
    y_observed = 0.5 .+ 0.25 .* x_observed .+ 0.08 .* randn(rng, 8)
    df = DataFrame(
        Y = Union{Missing, Float64}[y_observed..., fill(missing, 2)...],
        W = [x_observed..., 10.0, 10.0],
    )
    draws = impute_posterior(df, :Y, [:W]; n_draws = 40, rng = StableRNG(156))
    extrapolated = [d.Y[9] for d in draws.draws]
    @test std(extrapolated) > 0.3
end

@testset "impute_posterior uses one fitted categorical schema" begin
    @testset "unseen levels remain an explicit error" begin
        df = DataFrame(
            Y = Union{Missing, Float64}[
                0, 0.1, 5, 5.1, 10, 10.1, missing, missing, missing,
            ],
            group = ["A", "A", "B", "B", "C", "C", "B", "C", "D"],
        )
        # Both subsets independently produce two dummy columns, but their
        # reference levels differ. D is absent from outcome-observed rows and
        # must remain unseen rather than being silently encoded as an observed
        # category.
        @test_throws ArgumentError impute_posterior(
            df, :Y, [:group]; n_draws = 1, rng = StableRNG(53),
        )
    end

    @testset "missing subset levels represented among observed rows retain columns" begin
        df = DataFrame(
            Y = Union{Missing, Float64}[
                0, 0.1, 5, 5.1, 10, 10.1, missing, missing, missing, missing,
            ],
            group = ["A", "A", "B", "B", "C", "C", "B", "C", "B", "C"],
        )
        draws = impute_posterior(
            df, :Y, [:group]; n_draws = 2, rng = StableRNG(54),
        )
        @test all(isfinite, draws.draws[1].Y)
        @test all(isfinite, draws.draws[2].Y)
        @test all(d -> d.Y[7] < d.Y[8], draws.draws)
        @test all(d -> d.Y[9] < d.Y[10], draws.draws)
    end
end

@testset "Rubin pooling uses finite-imputation t limits and no false bands" begin
    grids = DataFrame[]
    for estimate in (1.0, 2.0, 5.0)
        push!(grids, DataFrame(
            delta = [0.0], estimand = ["TE"], est = [estimate], se = [0.1],
            clamp = [0.0], severity = [0.0], effective_shift = [0.0],
            shift_retention = [1.0], lower_q = [0.0], upper_q = [1.0],
            sd_exposure = [1.0], support_status = ["ok"], stratum = ["all"],
        ))
    end
    pooled = only(pool_lmtp_grids(grids))
    qbar = mean((1.0, 2.0, 5.0))
    between = var([1.0, 2.0, 5.0]; corrected = true)
    ubar = 0.01
    total = ubar + (1 + 1 / 3) * between
    df_rubin = (3 - 1) * (1 + ubar / ((1 + 1 / 3) * between))^2
    critical = quantile(TDist(df_rubin), 0.975)
    @test pooled.lwr ≈ qbar - critical * sqrt(total)
    @test pooled.upr ≈ qbar + critical * sqrt(total)
    @test isnan(pooled.lwr_sim) && isnan(pooled.upr_sim) && isnan(pooled.crit_sim)

    @test_throws ArgumentError pool_lmtp_grids([first(grids)])

    keyed = vcat(grids[1], grids[1])
    keyed.delta .= [0.0, 1.0]
    keyed.stratum .= ["baseline", "followup"]
    permuted = copy(keyed)
    permuted.delta .= reverse(keyed.delta)
    permuted.stratum .= reverse(keyed.stratum)
    @test_throws ArgumentError pool_lmtp_grids([keyed, permuted])
end

@testset "impute_posterior rejects unidentified MNAR" begin
    df, _ = CausalTargeted.simulate_missing_outcome_mtp(40; rng = StableRNG(51))
    cert = certify_missingness(MissingnessSpec(:Y; regime = :mnar))
    @test_throws ArgumentError impute_posterior(
        df, :Y, [:W]; certificate = cert, n_draws = 2, rng = StableRNG(51),
    )
end

@testset "run_lmtp_grid imputation pooling" begin
    df, truth = CausalTargeted.simulate_missing_outcome_mtp(100; rng = StableRNG(52))
    sdA = std(skipmissing(df.A))
    δ = 1.0 * sdA
    draws = impute_posterior(
        df, :Y, [:W];
        treatment = :A,
        certificate = certify_missingness(
            MissingnessSpec(:Y; regime = :mar, conditioning_set = [:W]),
        ),
        n_draws = 4,
        rng = StableRNG(52),
    )
    grid = run_lmtp_grid(
        df, :A, :Y; baseline = [:W], deltas = [δ],
        folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
        parallel = false, simultaneous = false, cache_nuisances = false,
        rng = StableRNG(52), shift_scale = "raw",
        imputation = draws,
    )
    @test isfinite(only(grid.est))
    meta = missingness_metadata(grid)
    @test meta.strategy === :posterior_gaussian_mar
    @test meta.n_draws == 4
    te = truth.effects(δ).te
    @test abs(only(grid.est) - te) < 0.45
end
