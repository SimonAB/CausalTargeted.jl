    @testset "MLJFlux optional learners" begin
        # Extensions load when their weakdeps are present in the environment.
        using MLJ
        using MLJLinearModels
        using MLJFlux
        ext = Base.get_extension(CausalTargeted, :CausalTargetedMLJFluxExt)
        @test ext !== nothing
        @test Base.get_extension(CausalTargeted, :CausalTargetedMLJExt) !== nothing
        X = randn(StableRNG(7), 60, 3)
        y = Float64.(sin.(X[:, 1]) .+ 0.1 .* randn(StableRNG(8), 60))
        sl = fit_super_learner(
            X, y;
            learners = (:mlj_mlp, :mean),
            family = :gaussian,
            folds = 2,
            metalearner = :invmse,
            rng = StableRNG(9),
        )
        @test sl isa SuperLearnerFit
        pred = predict_super_learner(sl, X)
        @test length(pred) == 60
        @test all(isfinite, pred)

        s = Float64.(randn(StableRNG(10), 60) .> 0.0)
        slb = fit_super_learner(
            X, s;
            learners = (:mlj_nn_binary, :mean),
            family = :binomial,
            folds = 2,
            metalearner = :invmse,
            rng = StableRNG(11),
        )
        predb = predict_super_learner(slb, X)
        @test length(predb) == 60
        @test all(isfinite, predb)
    end
