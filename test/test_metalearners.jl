@testset "Super Learner metalearner taxonomy" begin
    @testset "deprecated :discrete aliases :nnls" begin
        CausalTargeted._DISCRETE_METALEARNER_WARNED[] = false
        rng_data = StableRNG(101)
        X = hcat(ones(48), randn(rng_data, 48, 2))
        y = 0.5 .+ X[:, 2] .- 0.25 .* X[:, 3] .+ 0.1 .* randn(rng_data, 48)
        learners = (:glm, :mean)
        sl_nnls = fit_super_learner(
            X, y; learners = learners, metalearner = :nnls,
            folds = 3, rng = StableRNG(202),
        )
        sl_alias = fit_super_learner(
            X, y; learners = learners, metalearner = :discrete,
            folds = 3, rng = StableRNG(202),
        )
        @test sl_nnls.metalearner == :nnls
        @test sl_alias.metalearner == :nnls
        @test sl_alias.weights ≈ sl_nnls.weights atol = 1e-12
        @test sl_nnls.family == :gaussian
    end

    @testset "cv_selector is one-hot on the true GLM" begin
        df, _ = simulate_linear_mtp(400; σ_y = 0.15, rng = StableRNG(7))
        X = hcat(ones(nrow(df)), df.A, df.W)
        sl = fit_super_learner(
            X, df.Y;
            learners = (:glm, :mean),
            metalearner = :cv_selector,
            folds = 3,
            rng = StableRNG(8),
        )
        @test sl.metalearner == :cv_selector
        @test count(==(1.0), sl.weights) == 1
        @test sum(sl.weights) ≈ 1.0
        @test sl.weights[findfirst(==(:glm), sl.learners)] == 1.0
        sl_alias = fit_super_learner(
            X, df.Y;
            learners = (:glm, :mean),
            metalearner = :winner,
            folds = 3,
            rng = StableRNG(8),
        )
        @test sl_alias.metalearner == :cv_selector
        @test sl_alias.weights == sl.weights
    end

    @testset "binomial :glm predicts in (0,1)" begin
        df, _ = simulate_binomial_mtp(180; rng = StableRNG(11))
        X = hcat(ones(nrow(df)), df.A, df.W)
        sl = fit_super_learner(
            X, df.Y;
            learners = (:glm, :mean),
            family = :binomial,
            metalearner = :nnls,
            folds = 3,
            rng = StableRNG(12),
        )
        pred = predict_super_learner(sl, X)
        @test sl.fits[:glm][1] in (:logistic, :mean)
        @test all(isfinite, pred)
        @test all(p -> 0 < p < 1, pred)
    end

    @testset "binomial default metalearner is :nnloglik" begin
        rng = StableRNG(21)
        x = randn(rng, 60)
        X = hcat(ones(60), x)
        p = 1.0 ./ (1.0 .+ exp.(-x))
        y = Float64.(rand(rng, 60) .< p)
        sl = fit_super_learner(X, y; learners = (:logistic, :mean), family = :binomial, folds = 3, rng = StableRNG(22))
        @test sl.metalearner == :nnloglik
        @test sl.family == :binomial
    end

    @testset "linear MTP recovery under :nnls" begin
        r = CausalTargeted.run_julia_synthetic_once(
            :linear_mtp; n = 300, delta = 1.0, folds = 3,
            rng = StableRNG(1), learners = (:glm, :mean),
        )
        @test only(r.abs_error) < 0.12
    end
end
