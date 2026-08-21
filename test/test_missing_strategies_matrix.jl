"""Phase 2: estimand × handle_missing strategy matrix (Observable L2 cells)."""

using CausalTargeted
using CausalMediation
using DataFrames
using Random
using StableRNGs
using Statistics
using Test

const STRATS = (:drop, :ipcw, :impute, :ipcw_impute)

function _finite_te(grid::DataFrame)
    return Float64(only(grid.est))
end

@testset "missing strategy matrix" begin
    @testset "point LMTP outcome-only × strategies" begin
        df, _ = CausalTargeted.simulate_missing_outcome_mtp(120; rng = StableRNG(40))
        sdA = std(skipmissing(df.A))
        δ = 1.0 * sdA
        for strat in STRATS
            grid = run_lmtp_grid(
                df, :A, :Y; baseline = [:W], deltas = [δ],
                folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
                parallel = false, simultaneous = false, cache_nuisances = false,
                rng = StableRNG(40), shift_scale = "raw", handle_missing = strat,
            )
            @test isfinite(_finite_te(grid))
            meta = missingness_metadata(grid)
            @test meta.strategy === strat
            @test meta.rung === :L2
            @test meta.time_indexed === false
            @test meta.n_out > 0
        end
    end

    @testset "point LMTP covariate-only × strategies" begin
        df, _ = CausalTargeted.simulate_missing_covariate_mtp(120; rng = StableRNG(41))
        sdA = std(df.A)
        δ = 1.0 * sdA
        for strat in STRATS
            grid = run_lmtp_grid(
                df, :A, :Y; baseline = [:W], deltas = [δ],
                folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
                parallel = false, simultaneous = false, cache_nuisances = false,
                rng = StableRNG(41), shift_scale = "raw", handle_missing = strat,
            )
            @test isfinite(_finite_te(grid))
            @test missingness_metadata(grid).strategy === strat
        end
    end

    @testset "g-comp missing Y" begin
        df, _ = CausalTargeted.simulate_missing_outcome_mtp(100; rng = StableRNG(42))
        for strat in (:drop, :ipcw)
            r = run_gcomp(
                df, :A, :Y; covariates = [:W], delta = 1.0,
                folds = 2, learners = (:glm, :mean), n_boot = 0,
                rng = StableRNG(42), handle_missing = strat,
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
            @test r.missingness.time_indexed === false
        end
    end

    @testset "sequential missing Y drop/ipcw" begin
        rng = StableRNG(43)
        n = 50
        W = randn(rng, n)
        A1 = 0.5 .* W .+ randn(rng, n)
        L1 = 0.3 .* A1 .+ randn(rng, n)
        A2 = 0.4 .* L1 .+ randn(rng, n)
        Y = 0.5 .* A2 .+ 0.3 .* A1 .+ 0.2 .* W .+ randn(rng, n)
        df = DataFrame(W = W, A1 = A1, L1 = L1, A2 = A2, Y = Y)
        df.Y = Vector{Union{Float64, Missing}}(df.Y)
        df.Y[1:10] .= missing
        shift = additive_shift_policy(; scale = "raw", lower_q = 0.0, upper_q = 1.0)
        for strat in (:drop, :ipcw)
            r = run_sequential_lmtp(
                df, [:A1, :A2], :Y;
                baseline = [:W], time_vary = [Symbol[], [:L1]],
                delta = 0.25, shift = shift, folds = 2,
                learners = (:glm, :mean), handle_missing = strat, rng = StableRNG(43),
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
            @test r.missingness.time_indexed === true
            @test r.missingness.rung === :L2
        end
    end

    @testset "survival MAR S_T vs censoring docs" begin
        df, truth = CausalTargeted.simulate_discrete_survival_mtp(80; T = 2, rng = StableRNG(47))
        Sh = truth.surv[end]
        rng = StableRNG(48)
        df[!, Sh] = Vector{Union{Float64, Missing}}(df[!, Sh])
        p_miss = 1.0 ./ (1.0 .+ exp.(-(-1.4 .+ 0.8 .* df.W)))
        for i in 1:nrow(df)
            rand(rng) < p_miss[i] && (df[i, Sh] = missing)
        end
        for strat in (:drop, :ipcw)
            r = run_survival_lmtp(
                df, truth.treatments, truth.surv;
                baseline = [:W], delta = 0.25, folds = 2,
                learners = (:glm, :mean), handle_missing = strat, rng = StableRNG(49),
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
            @test r.missingness.time_indexed === true
            @test r.estimand === :survival
        end
    end

    @testset "mediation MAR Y all strategies" begin
        df, truth = CausalMediation.simulate_mediation(160; rng = StableRNG(44))
        rng = StableRNG(45)
        df.Y = Vector{Union{Float64, Missing}}(df.Y)
        p_miss = 1.0 ./ (1.0 .+ exp.(-(-1.2 .+ 0.9 .* df.W)))
        for i in 1:nrow(df)
            rand(rng) < p_miss[i] && (df.Y[i] = missing)
        end
        for strat in STRATS
            tab = CausalMediation.run_mediation_scalar(
                df, :A, :Y;
                covar = [:W], mediators = [:M],
                folds = 2, n_mc = 6, estimator = :onestep,
                learners = (:glm, :mean), handle_missing = strat, rng = StableRNG(46),
            )
            te = only(tab[tab.effect .== "TE", :estimate])
            @test isfinite(te)
            meta = missingness_metadata(tab)
            @test meta.strategy === strat
            @test meta.rung === :L2
        end
    end
end
