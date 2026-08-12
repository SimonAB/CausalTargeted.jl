@testset "NNloglik Super Learner metalearner" begin
    @testset "existing NNLS path is preserved" begin
        rng_data = StableRNG(101)
        X = hcat(ones(48), randn(rng_data, 48, 2))
        y = 0.5 .+ X[:, 2] .- 0.25 .* X[:, 3] .+ 0.1 .* randn(rng_data, 48)
        learners = (:glm, :mean)

        Z = zeros(48, length(learners))
        for test_idx in crossfit_indices(48, 3, StableRNG(202))
            train_idx = setdiff(1:48, test_idx)
            for (j, learner) in enumerate(learners)
                fit = CausalTargeted._fit_learner(
                    learner,
                    X[train_idx, :],
                    y[train_idx];
                    family = :gaussian,
                )
                Z[test_idx, j] = CausalTargeted._predict_learner(fit, X[test_idx, :])
            end
        end
        expected = CausalTargeted._nonneg_ls_weights(Z, y)
        sl = fit_super_learner(
            X,
            y;
            learners = learners,
            metalearner = :discrete,
            folds = 3,
            rng = StableRNG(202),
        )
        pred = predict_super_learner(sl, X)

        @test sl isa SuperLearnerFit
        @test sl.weights ≈ expected atol = 1e-12 rtol = 1e-12
        @test all(sl.weights .>= 0)
        @test sum(sl.weights) ≈ 1.0
        @test length(pred) == length(y)
    end

    @testset "basic binary fit and logit-scale prediction" begin
        rng = StableRNG(303)
        x = randn(rng, 90)
        X = hcat(ones(90), x)
        true_probability = 1.0 ./ (1.0 .+ exp.(-(-0.3 .+ 1.2 .* x)))
        y = Float64.(rand(rng, 90) .< true_probability)
        sl = fit_super_learner(
            X,
            y;
            learners = (:logistic, :mean),
            family = :binomial,
            metalearner = :nnloglik,
            folds = 3,
            rng = StableRNG(304),
        )
        pred = predict_super_learner(sl, X)

        @test sl isa SuperLearnerFit
        @test sl.metalearner == :nnloglik
        @test all(isfinite, sl.weights)
        @test all(sl.weights .>= 0)
        @test sum(sl.weights) ≈ 1.0
        @test length(pred) == length(y)
        @test all(isfinite, pred)
        @test all(p -> 0 <= p <= 1, pred)

        manual_fit = SuperLearnerFit(
            Dict{Symbol, Any}(:low => (:mean, 0.1), :high => (:mean, 0.8)),
            [0.25, 0.75],
            [:low, :high],
            :nnloglik,
        )
        manual_pred = only(unique(predict_super_learner(manual_fit, ones(3, 1))))
        expected_logit_pred = CausalTargeted._logistic(
            0.25 * log(0.1 / 0.9) + 0.75 * log(0.8 / 0.2),
        )
        @test manual_pred ≈ expected_logit_pred
        @test !isapprox(manual_pred, 0.25 * 0.1 + 0.75 * 0.8; atol = 1e-3)
    end

    @testset "direct and extreme probability fixtures" begin
        y = repeat([0.0, 1.0], 6)
        good = [0.10, 0.90, 0.20, 0.80, 0.80, 0.70,
                0.30, 0.20, 0.15, 0.85, 0.25, 0.75]
        weak = [0.40, 0.60, 0.45, 0.55, 0.60, 0.55,
                0.48, 0.45, 0.42, 0.58, 0.47, 0.53]
        weights = CausalTargeted._nnloglik_weights(hcat(good, weak), y)
        @test weights[1] > weights[2]
        @test all(weights .>= 0)
        @test sum(weights) ≈ 1.0

        Z_extreme = hcat(
            [1e-8, 1 - 1e-8, 0.0, 1.0, 0.2, 0.8],
            [0.3, 0.7, 0.4, 0.6, 0.45, 0.55],
        )
        y_extreme = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        logits = CausalTargeted._trim_logit_predictions(Z_extreme)
        fit = CausalTargeted._nnloglik_fit(Z_extreme, y_extreme)
        pred = CausalTargeted._predict_nnloglik(Z_extreme, fit.weights)
        bound = abs(log(CausalTargeted.NNLOGLIK_TRIM /
                        (1 - CausalTargeted.NNLOGLIK_TRIM)))
        @test all(isfinite, logits)
        @test maximum(abs, logits) <= bound + 1e-10
        @test all(isfinite, fit.weights)
        @test all(fit.weights .>= 0)
        @test sum(fit.weights) ≈ 1.0
        @test all(isfinite, pred)
        @test all(p -> 0 <= p <= 1, pred)

        beta = [0.4, 0.7]
        obs_weights = [1.0, 2.0, 1.0, 0.5, 1.5, 1.0]
        analytic = CausalTargeted._nnloglik_gradient(
            beta,
            CausalTargeted._trim_logit_predictions(Z_extreme),
            y_extreme,
            obs_weights,
        )
        epsilon = 1e-6
        finite_difference = similar(beta)
        Xlogit = CausalTargeted._trim_logit_predictions(Z_extreme)
        for j in eachindex(beta)
            direction = zeros(length(beta))
            direction[j] = epsilon
            finite_difference[j] = (
                CausalTargeted._nnloglik_objective(
                    beta .+ direction, Xlogit, y_extreme, obs_weights,
                ) -
                CausalTargeted._nnloglik_objective(
                    beta .- direction, Xlogit, y_extreme, obs_weights,
                )
            ) / (2epsilon)
        end
        @test analytic ≈ finite_difference atol = 1e-7 rtol = 1e-6
    end

    @testset "overconfident error is penalised" begin
        y = repeat([0.0, 1.0], 6)
        catastrophic = [0.01, 0.99, 0.02, 0.98, 1 - 1e-8, 0.97,
                        0.03, 0.96, 0.02, 0.99, 0.04, 0.95]
        conservative = [0.25, 0.75, 0.30, 0.70, 0.60, 0.68,
                        0.32, 0.70, 0.28, 0.72, 0.35, 0.65]
        weights = CausalTargeted._nnloglik_weights(
            hcat(catastrophic, conservative),
            y,
        )
        @test weights[2] > 0.99
        @test weights[1] < 0.01
    end

    @testset "R-compatible all-zero coefficient convention" begin
        Z = hcat(fill(0.8, 6), fill(0.7, 6))
        y = zeros(6)
        fit = CausalTargeted._nnloglik_fit(Z, y)
        pred = CausalTargeted._predict_nnloglik(Z, fit.weights)

        @test all(iszero, fit.weights)
        @test pred == zeros(6)
        @test all(isfinite, pred)
    end

    @testset "validation" begin
        X = hcat(ones(12), collect(range(-1, 1; length = 12)))
        y = repeat([0.0, 1.0], 6)
        @test_throws ArgumentError fit_super_learner(
            X,
            y;
            learners = (:logistic, :mean),
            family = :gaussian,
            metalearner = :nnloglik,
            folds = 2,
        )
        @test_throws ArgumentError fit_super_learner(
            X,
            collect(range(0, 1; length = 12));
            learners = (:logistic, :mean),
            family = :binomial,
            metalearner = :nnloglik,
            folds = 2,
        )
        @test_throws ArgumentError fit_super_learner(
            X,
            y;
            metalearner = :not_a_metalearner,
        )
        @test_throws ArgumentError CausalTargeted._nnloglik_weights(
            [0.2 NaN; 0.8 0.7],
            [0.0, 1.0],
        )
        @test_throws ArgumentError CausalTargeted._nnloglik_weights(
            [0.2 1.2; 0.8 0.7],
            [0.0, 1.0],
        )
    end

    @testset "R SuperLearner 2.0.29 reference fixture" begin
        # Generated by dev/qc_nnloglik.R with method.NNloglik, trimLogit=1e-5.
        y = repeat([0.0, 1.0], 10)
        learner_a = [0.08, 0.82, 0.18, 0.76, 0.88, 0.68, 0.25, 0.84, 0.12, 0.73,
                     0.30, 0.20, 0.22, 0.78, 0.40, 0.65, 0.75, 0.80, 0.35, 0.70]
        learner_b = [0.25, 0.65, 0.35, 0.62, 0.58, 0.55, 0.45, 0.60, 0.28, 0.64,
                     0.48, 0.42, 0.38, 0.67, 0.47, 0.58, 0.55, 0.70, 0.43, 0.61]
        learner_c = [0.15, 0.72, 0.55, 0.66, 0.65, 0.40, 0.20, 0.76, 0.31, 0.52,
                     0.62, 0.35, 0.41, 0.59, 0.33, 0.75, 0.45, 0.57, 0.29, 0.54]
        Z = hcat(learner_a, learner_b, learner_c)
        obs_weights = [1.0, 2.0, 1.0, 1.5, 3.0, 1.0, 2.0, 1.0, 0.5, 2.0,
                       1.0, 2.5, 1.0, 1.0, 2.0, 1.5, 1.0, 2.0, 1.0, 3.0]
        r_raw = [0.0, 3.4981831075599645, 0.05986300190465167]
        r_weights = [0.0, 0.9831753158719859, 0.01682468412801411]
        r_predictions = [
            0.2479990636062083, 0.6512445666357413, 0.3531439885969911,
            0.6206884689742601, 0.5812137568710531, 0.5474747184893746,
            0.4450680160797876, 0.6030133671227271, 0.2804898498449455,
            0.6380777007667590, 0.4823924038351897, 0.4187862431289467,
            0.3804979233583847, 0.6687182659154191, 0.4675362556436273,
            0.5831764082315107, 0.5483282124926122, 0.6979983795543453,
            0.4274718052052174, 0.6088507696788767,
        ]
        r_logloss = 0.5677248646079829

        julia_fit = CausalTargeted._nnloglik_fit(
            Z,
            y;
            obs_weights = obs_weights,
            trim = 1e-5,
        )
        julia_predictions = CausalTargeted._predict_nnloglik(
            Z,
            julia_fit.weights;
            trim = 1e-5,
        )
        Xlogit = CausalTargeted._trim_logit_predictions(Z; trim = 1e-5)
        julia_logloss = CausalTargeted._nnloglik_objective(
            julia_fit.weights,
            Xlogit,
            y,
            obs_weights,
        )

        @test julia_fit.raw_weights ≈ r_raw atol = 2e-5 rtol = 1e-5
        @test julia_fit.weights ≈ r_weights atol = 1e-5 rtol = 1e-5
        @test julia_predictions ≈ r_predictions atol = 1e-6 rtol = 1e-6
        @test julia_logloss ≈ r_logloss atol = 1e-7 rtol = 1e-7
        @test all(julia_fit.weights .>= 0)
        @test sum(julia_fit.weights) ≈ 1.0 atol = 1e-12
    end
end
