@testset "multinomial Super Learner" begin
    df, truth = simulate_multinomial_outcome(400; K = 3, rng = StableRNG(31))
    X = hcat(ones(nrow(df)), df.W)
    sl = fit_super_learner(
        X, df.Y;
        family = :multinomial,
        learners = (:glm, :mean),
        metalearner = :nnls,
        folds = 3,
        rng = StableRNG(32),
    )
    P = predict_super_learner(sl, X)
    @test sl.family == :multinomial
    @test size(P) == (nrow(df), 3)
    @test all(isapprox.(sum(P; dims = 2), 1; atol = 1e-8))
    @test all(0 .<= P .<= 1)
    brier_sl = mean(sum((P .- truth.P) .^ 2; dims = 2))
    p_mean = vec(mean(truth.P; dims = 1))
    P_mean = repeat(reshape(p_mean, 1, :), nrow(df), 1)
    brier_mean = mean(sum((P_mean .- truth.P) .^ 2; dims = 2))
    @test brier_sl < brier_mean

    sl_cv = fit_super_learner(
        X, df.Y;
        family = :multinomial,
        learners = (:glm, :mean),
        metalearner = :cv_selector,
        folds = 3,
        rng = StableRNG(33),
    )
    @test count(==(1.0), sl_cv.weights) == 1
    @test sum(sl_cv.weights) ≈ 1
    sl_nll = fit_super_learner(
        X, df.Y;
        family = :multinomial,
        learners = (:glm, :mean),
        metalearner = :nnloglik,
        folds = 3,
        rng = StableRNG(34),
    )
    @test all(sl_nll.weights .>= 0)
    @test sum(sl_nll.weights) ≈ 1

    @testset "K=2 matches binomial" begin
        rng = StableRNG(41)
        x = randn(rng, 160)
        X2 = hcat(ones(160), x)
        p = 1.0 ./ (1.0 .+ exp.(-(-0.2 .+ 1.1 .* x)))
        y01 = Float64.(rand(rng, 160) .< p)
        y12 = Int.(y01) .+ 1
        sl_bin = fit_super_learner(
            X2, y01;
            family = :binomial, learners = (:glm, :mean),
            metalearner = :nnls, folds = 3, rng = StableRNG(42),
        )
        sl_mul = fit_super_learner(
            X2, y12;
            family = :multinomial, learners = (:glm, :mean),
            metalearner = :nnls, folds = 3, rng = StableRNG(42),
            levels = [1, 2],
        )
        p_bin = predict_super_learner(sl_bin, X2)
        P_mul = predict_super_learner(sl_mul, X2)
        @test maximum(abs.(P_mul[:, 2] .- p_bin)) < 0.05
    end
end
