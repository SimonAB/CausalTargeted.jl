"""Bayesian Gaussian posterior-mean candidate for Super Learner."""

using CausalTargeted
using DataFrames
using LinearAlgebra
using StableRNGs
using Test

const CT = CausalTargeted

@testset "Bayesian Gaussian Super Learner candidate" begin
    X_train = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
    y_train = [0.2, 1.1, 1.8, 3.2]
    X_test = [1.0 -1.0; 1.0 4.0]
    prior = CT.ConjugateGaussianPrior(
        mean = [0.0, 0.0],
        covariance = 10.0 .* Matrix{Float64}(I, 2, 2),
        shape = 2.0,
        scale = 1.0,
    )
    learner = CT.BayesianGaussianLearner(; prior)

    @testset "candidate fit predicts posterior mean only" begin
        fit = CT._fit_any_candidate(learner, X_train, y_train; family = :gaussian)
        prediction = CT._predict_learner(fit, X_test)
        @test fit isa CT.BayesianGaussianCandidateFit
        @test !(fit isa CT.BayesianGCompFit)
        @test prediction ≈ X_test * fit.posterior_mean
        @test all(isfinite, prediction)
        @test all(diag(fit.posterior_covariance) .> 0)
        expected_posterior = CT._bayesian_gaussian_posterior(X_train, y_train, prior)
        @test fit.posterior_mean ≈ expected_posterior.posterior_mean
        @test fit.posterior_covariance ≈ expected_posterior.posterior_covariance

        # Prediction uses only the target design and the fitted posterior mean.
        fit_again = CT._fit_any_candidate(learner, X_train, y_train; family = :gaussian)
        @test CT._predict_learner(fit_again, X_test) == prediction
        perturbed_training_fit = CT._fit_any_candidate(
            learner, X_train, y_train .+ [0.5, 0.0, 0.0, 0.0]; family = :gaussian,
        )
        @test perturbed_training_fit.posterior_mean != fit.posterior_mean
    end

    @testset "ensemble route uses the same posterior-mean predictions" begin
        sl = CT.fit_super_learner(
            X_train, y_train;
            learners = [learner, :mean],
            metalearner = :invmse,
            family = :gaussian,
        )
        candidate_key = CT._candidate_key(learner)
        candidate_fit = sl.fits[candidate_key]
        direct = X_test * candidate_fit.posterior_mean
        @test candidate_key === :bayesian_gaussian
        @test candidate_fit isa CT.BayesianGaussianCandidateFit
        @test CT._predict_learner(candidate_fit, X_test) ≈ direct
        @test CT.predict_super_learner(sl, X_test) ≈
              sl.weights[1] .* direct .+ sl.weights[2] .* CT._predict_learner(sl.fits[:mean], X_test)
    end

    @testset "Gaussian-only and dimension-explicit prior contract" begin
        @test_throws ArgumentError CT._fit_any_candidate(
            learner, X_train, y_train; family = :binomial,
        )
        wrong_dimension_prior = CT.ConjugateGaussianPrior(
            mean = zeros(3),
            covariance = Matrix{Float64}(I, 3, 3),
            shape = 2.0,
            scale = 1.0,
        )
        wrong_dimension_learner = CT.BayesianGaussianLearner(; prior = wrong_dimension_prior)
        @test_throws DimensionMismatch CT._fit_any_candidate(
            wrong_dimension_learner, X_train, y_train; family = :gaussian,
        )
    end
end

@testset "Bayesian Gaussian candidate through public LMTP and fold fitting" begin
    rng = StableRNG(2501)
    n = 40
    W = randn(rng, n)
    A = 1 ./ (1 .+ exp.(-randn(rng, n)))
    Y = 0.8 .+ 0.7 .* A .- 0.3 .* W .+ 0.2 .* randn(rng, n)
    data = DataFrame(A = A, W = W, Y = Y)
    prior_outcome = CT.ConjugateGaussianPrior(
        mean = zeros(3), covariance = 20.0 .* Matrix{Float64}(I, 3, 3),
        shape = 2.0, scale = 1.0,
    )
    prior_treatment = CT.ConjugateGaussianPrior(
        mean = zeros(2), covariance = 20.0 .* Matrix{Float64}(I, 2, 2),
        shape = 2.0, scale = 1.0,
    )
    outcome_learner = CT.BayesianGaussianLearner(; prior = prior_outcome)
    treatment_learner = CT.BayesianGaussianLearner(; prior = prior_treatment)

    @testset "fold fits exclude their held-out outcomes" begin
        fold_sets = [collect(1:20), collect(21:40)]
        original_fit = CT.fit_outcome_regression(
            data, :Y, :A, [:W], 2, StableRNG(2502);
            learners = (outcome_learner,), fold_sets,
        )
        changed = copy(data)
        changed.Y[1:20] .+= 4.0
        changed_fit = CT.fit_outcome_regression(
            changed, :Y, :A, [:W], 2, StableRNG(2502);
            learners = (outcome_learner,), fold_sets,
        )
        fit1 = original_fit.fold_models[1].fits[:bayesian_gaussian]
        changed1 = changed_fit.fold_models[1].fits[:bayesian_gaussian]
        fit2 = original_fit.fold_models[2].fits[:bayesian_gaussian]
        changed2 = changed_fit.fold_models[2].fits[:bayesian_gaussian]
        @test fit1.posterior_mean == changed1.posterior_mean
        @test fit2.posterior_mean != changed2.posterior_mean
    end

    policy = CT.smooth_bounded_logit_shift_policy(0.0, 1.0)
    for cache_nuisances in (true, false)
        grid = CT.run_lmtp_grid(
            data, :A, :Y;
            baseline = [:W], deltas = [0.2], policy,
            learners_outcome = (outcome_learner,),
            learners_trt = (treatment_learner,),
            density_ratio = :gaussian, weight_truncation = false,
            folds = 2, simultaneous = false, parallel = false,
            cache_nuisances, rng = StableRNG(2503),
        )
        @test only(grid.inference_status) === :available
        @test isfinite(only(grid.est))
        @test isfinite(only(grid.se))
    end

    @test_throws ArgumentError CT.validate_contrast_learners((:mean,))
    @test CT.validate_contrast_learners(
        (CT.nested_sl_candidate((:glm,)),),
    ) === nothing
end
