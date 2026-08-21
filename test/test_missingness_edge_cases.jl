"""Thorough edge cases across Observable missingness kinds."""

using CausalTargeted
import CausalDynamics
using CausalDynamics:
    MissingnessSpec, certify_missingness, DiGraph, add_edge!, identify,
    TotalEffectQuery, DiscreteTimeCDM, simulate_incomplete_panel, miss_rates,
    CDMPanel
import CausalMediation  # import, not using: avoid clashing with CT façade exports
using DataFrames
using Random
using StableRNGs
using Statistics
using Test

const STRATS = (:drop, :ipcw, :impute, :ipcw_impute)

function _lmtp_lean(df, strat; rng = StableRNG(1), δ = nothing)
    sdA = std(skipmissing(df.A))
    dlt = δ === nothing ? 1.0 * sdA : δ
    return run_lmtp_grid(
        df, :A, :Y; baseline = [:W], deltas = [dlt],
        folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
        parallel = false, simultaneous = false, cache_nuisances = false,
        rng = rng, shift_scale = "raw", handle_missing = strat,
    )
end

@testset "missingness edge cases (Targeted / Mediation)" begin
    @testset "unknown strategy" begin
        df = DataFrame(y = [1.0, missing], w = [0.0, 1.0], a = [0.0, 1.0])
        @test_throws ErrorException handle_missing_data(df, :y, [:w], :bogus)
    end

    @testset "complete data identity" begin
        df = DataFrame(y = [1.0, 2.0, 3.0], w = [0.0, 1.0, 2.0], a = [0.0, 1.0, 0.0])
        for strat in STRATS
            r = handle_missing_data(df, :y, [:w, :a], strat; rng = StableRNG(1))
            @test r.meta.n_in == 3
            @test r.meta.n_out == 3
            @test all(≈(1.0), r.weights)
        end
    end

    @testset "all outcome missing → empty under drop/ipcw" begin
        df = DataFrame(
            y = Vector{Union{Float64, Missing}}([missing, missing, missing]),
            w = [0.0, 1.0, 2.0],
            a = [0.0, 1.0, 0.0],
        )
        for strat in (:drop, :ipcw, :impute, :ipcw_impute)
            r = handle_missing_data(df, :y, [:w], strat; rng = StableRNG(2))
            @test r.meta.n_out == 0
            @test isempty(r.data)
        end
    end

    @testset "covariate-only missing: drop vs impute n_out" begin
        df = DataFrame(
            y = [1.0, 2.0, 3.0, 4.0],
            w = Vector{Union{Float64, Missing}}([0.0, missing, 2.0, missing]),
            a = [0.0, 1.0, 0.0, 1.0],
        )
        drop = handle_missing_data(df, :y, [:w], :drop)
        imp = handle_missing_data(df, :y, [:w], :impute)
        @test drop.meta.n_out == 2
        @test imp.meta.n_out == 4
        @test :w_miss in imp.extra_cols
        @test all(x -> x in (0.0, 1.0), imp.data.w_miss)
    end

    @testset "mixed Y+W: rates and strategy effects" begin
        df = DataFrame(
            y = Vector{Union{Float64, Missing}}([1.0, missing, 3.0, 4.0]),
            w = Vector{Union{Float64, Missing}}([0.0, 1.0, missing, 2.0]),
            a = [0.0, 1.0, 0.0, 1.0],
        )
        r = handle_missing_data(df, :y, [:w, :a], :ipcw_impute; rng = StableRNG(3))
        @test r.meta.miss_rates[:y] ≈ 0.25
        @test r.meta.miss_rates[:w] ≈ 0.25
        @test r.meta.n_out == 3  # Y missing dropped after W imputed
        @test length(r.weights) == r.meta.n_out
        @test CausalTargeted._uses_ipcw_weights(r.weights)
    end

    @testset "rung and time_indexed recorded" begin
        df = DataFrame(y = [1.0, missing, 3.0], w = [0.0, 1.0, 2.0])
        r = handle_missing_data(
            df, :y, [:w], :drop; rung = :L1, time_indexed = true,
        )
        @test r.meta.rung === :L1
        @test r.meta.time_indexed === true
    end

    @testset "MAR Y: drop vs ipcw estimates differ (IPCW used)" begin
        df, _ = CausalTargeted.simulate_missing_outcome_mtp(160; rng = StableRNG(70))
        g_drop = _lmtp_lean(df, :drop; rng = StableRNG(70))
        g_ipcw = _lmtp_lean(df, :ipcw; rng = StableRNG(70))
        @test isfinite(only(g_drop.est))
        @test isfinite(only(g_ipcw.est))
        # With MAR Y and IPCW wired into the IF, estimates should not be bit-identical
        # at this n (allow tiny numerical equality only if weights unused — then fail loudly).
        ψd, ψi = only(g_drop.est), only(g_ipcw.est)
        @test abs(ψd - ψi) > 1e-8 || begin
            @warn "drop≈ipcw on MAR Y; check IPCW wiring" ψd ψi
            false
        end
    end

    @testset "g-comp all strategies" begin
        df, _ = CausalTargeted.simulate_missing_outcome_mtp(100; rng = StableRNG(71))
        for strat in STRATS
            r = run_gcomp(
                df, :A, :Y; covariates = [:W], delta = 1.0,
                folds = 2, learners = (:glm, :mean), n_boot = 0,
                rng = StableRNG(71), handle_missing = strat,
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
        end
    end

    @testset "sequential all strategies" begin
        rng = StableRNG(72)
        n = 60
        W = randn(rng, n)
        A1 = 0.5 .* W .+ randn(rng, n)
        L1 = 0.3 .* A1 .+ randn(rng, n)
        A2 = 0.4 .* L1 .+ randn(rng, n)
        Y = 0.5 .* A2 .+ 0.3 .* A1 .+ 0.2 .* W .+ randn(rng, n)
        df = DataFrame(W = W, A1 = A1, L1 = L1, A2 = A2, Y = Y)
        df.Y = Vector{Union{Float64, Missing}}(df.Y)
        p_miss = 1.0 ./ (1.0 .+ exp.(-(-1.3 .+ 0.9 .* W)))
        for i in 1:n
            rand(rng) < p_miss[i] && (df.Y[i] = missing)
        end
        # Also MCAR a few L1 cells for :impute / :ipcw_impute
        df.L1 = Vector{Union{Float64, Missing}}(df.L1)
        df.L1[1:5] .= missing
        shift = additive_shift_policy(; scale = "raw", lower_q = 0.0, upper_q = 1.0)
        for strat in STRATS
            r = run_sequential_lmtp(
                df, [:A1, :A2], :Y;
                baseline = [:W], time_vary = [Symbol[], [:L1]],
                delta = 0.25, shift = shift, folds = 2,
                learners = (:glm, :mean), handle_missing = strat, rng = StableRNG(72),
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
            @test r.missingness.time_indexed === true
        end
    end

    @testset "survival MAR S_T all strategies" begin
        df, truth = CausalTargeted.simulate_discrete_survival_mtp(90; T = 2, rng = StableRNG(73))
        Sh = truth.surv[end]
        rng = StableRNG(74)
        df[!, Sh] = Vector{Union{Float64, Missing}}(df[!, Sh])
        p_miss = 1.0 ./ (1.0 .+ exp.(-(-1.3 .+ 0.7 .* df.W)))
        for i in 1:nrow(df)
            rand(rng) < p_miss[i] && (df[i, Sh] = missing)
        end
        for strat in STRATS
            r = run_survival_lmtp(
                df, truth.treatments, truth.surv;
                baseline = [:W], delta = 0.25, folds = 2,
                learners = (:glm, :mean), handle_missing = strat, rng = StableRNG(75),
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
            @test r.missingness.time_indexed === true
        end
    end

    @testset "discrete LMTP missing Y drop/ipcw" begin
        df, truth = simulate_categorical_treatment_mtp(200; rng = StableRNG(76))
        policy = discrete_recode_policy(truth.recode)
        df.Y = Vector{Union{Float64, Missing}}(df.Y)
        df.Y[1:30] .= missing
        for strat in (:drop, :ipcw)
            r = run_discrete_lmtp(
                df, :A, :Y;
                policy = policy, baseline = [:W], folds = 2,
                learners_outcome = (:glm, :mean),
                handle_missing = strat, rng = StableRNG(76),
            )
            @test isfinite(r.estimate)
            @test r.missingness.strategy === strat
        end
    end

    @testset "mediation missing M (covariate path)" begin
        df, _ = CausalMediation.simulate_mediation(140; rng = StableRNG(77))
        rng = StableRNG(78)
        df.M = Vector{Union{Float64, Missing}}(df.M)
        for i in 1:20
            df.M[i] = missing
        end
        for strat in (:drop, :impute, :ipcw_impute)
            tab = CausalMediation.run_mediation_scalar(
                df, :A, :Y;
                covar = [:W], mediators = [:M],
                folds = 2, n_mc = 6, estimator = :onestep,
                learners = (:glm, :mean), handle_missing = strat, rng = StableRNG(79),
            )
            te = only(tab[tab.effect .== "TE", :estimate])
            @test isfinite(te)
            @test missingness_metadata(tab).strategy === strat
        end
    end

    @testset "posterior edges" begin
        df, _ = CausalTargeted.simulate_missing_outcome_mtp(60; rng = StableRNG(80))
        @test_throws ArgumentError impute_posterior(
            df, :Y, [:W]; n_draws = 0, rng = StableRNG(80),
        )
        df_w = copy(df)
        df_w.W = Vector{Union{Float64, Missing}}(df_w.W)
        df_w.W[1] = missing
        cert = certify_missingness(MissingnessSpec(:Y; regime = :mar, conditioning_set = [:W]))
        @test_throws ArgumentError impute_posterior(
            df_w, :Y, [:W]; certificate = cert, n_draws = 2, rng = StableRNG(80),
        )
        # IdentificationResult as certificate carrier
        g = DiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 1, 3)
        add_edge!(g, 2, 3)
        id = identify(
            g, TotalEffectQuery(:A, :Y);
            node_names = Dict(1 => :W, 2 => :A, 3 => :Y),
            missingness = MissingnessSpec(:Y; regime = :mar, conditioning_set = [:W]),
        )
        draws = impute_posterior(df, :Y, [:W]; treatment = :A, certificate = id, n_draws = 3, rng = StableRNG(81))
        @test draws.mar_set == [:W]
        # No missing Y: draws still length n_draws, observed preserved
        df_full = dropmissing(df, [:Y, :W, :A])
        draws0 = impute_posterior(
            df_full, :Y, [:W]; treatment = :A, certificate = cert, n_draws = 2, rng = StableRNG(82),
        )
        @test draws0.meta.n_missing == 0
        @test draws0.draws[1].Y == df_full.Y
    end

    @testset "Dynamics→Targeted incomplete panel → LMTP drop" begin
        cdm = DiscreteTimeCDM(
            [:a, :y];
            initialise = (rng) -> (a = 0.0, y = 0.0),
            sample_noise = (rng, state, t) -> (u_a = randn(rng), u_y = randn(rng)),
            step = (state, t, noise, intervention) -> begin
                a = 0.4 * state.a + noise.u_a
                y = 1.5 * a + noise.u_y
                (a = a, y = y)
            end,
        )
        incomplete, mask, panel = simulate_incomplete_panel(
            cdm, 80, 2;
            missingness = MissingnessSpec(:y; regime = :mcar),
            intercept = -0.8,
            baseline = Symbol[],
            timed = [:a],
            terminal = [:y],
            rng = StableRNG(83),
            rng_missing = StableRNG(84),
        )
        @test haskey(incomplete, :a2) && haskey(incomplete, :y)
        df = DataFrame(
            A = incomplete[:a2],
            Y = incomplete[:y],
            W = Float64.(incomplete[:a1]),
        )
        @test miss_rates(mask)[:y] > 0.05
        grid = run_lmtp_grid(
            df, :A, :Y; baseline = [:W], deltas = [0.5],
            folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = StableRNG(86), shift_scale = "raw", handle_missing = :drop,
        )
        @test isfinite(only(grid.est))
        @test missingness_metadata(grid).strategy === :drop
        @test panel isa CDMPanel
    end

    @testset "pool_lmtp_grids empty / rubin" begin
        @test_throws ArgumentError pool_lmtp_grids(DataFrame[])
    end
end
