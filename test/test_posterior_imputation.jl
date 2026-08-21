"""Phase 4: posterior MAR imputation draws and LMTP pooling."""

using CausalTargeted
using CausalDynamics
using DataFrames
using Random
using StableRNGs
using Statistics
using Test

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
