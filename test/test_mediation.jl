    @testset "mediation fold cache" begin
        rng = StableRNG(5)
        df, _ = CausalTargeted.simulate_continuous_mtp_mediation(80; rng = rng)
        cache = CausalMediation.build_mediation_fold_cache(
            df, :Y, :A, [:W], [:M], 2, rng; learners = SMALL_N_SL_LEARNERS,
        )
        @test length(cache.fold_sets) == 2
        g1 = CausalMediation.run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [0.5], folds = 2, n_mc = 8,
            learners = SMALL_N_SL_LEARNERS,
            cache_nuisances = true, parallel = false, rng = rng,
        )
        g2 = @test_deprecated run_crumble_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [0.5], folds = 2, n_mc = 8,
            learners = SMALL_N_SL_LEARNERS,
            cache_nuisances = false, parallel = false, rng = StableRNG(5),
        )
        @test nrow(g1) == nrow(g2)
    end
