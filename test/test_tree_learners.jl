@testset "MLJ tree Super Learner candidates" begin
    rf_ext = Base.get_extension(CausalTargeted, :CausalTargetedMLJDecisionTreeExt)
    xgb_ext = Base.get_extension(CausalTargeted, :CausalTargetedMLJXGBoostExt)
    @test rf_ext !== nothing
    @test xgb_ext !== nothing

    rng = StableRNG(701)
    n = 48
    x1 = randn(rng, n)
    x2 = 10.0 .+ 3.0 .* randn(rng, n)
    x3 = rand(rng, n)
    X = hcat(ones(n), x1, x2, x3)
    y = 0.4 .* x1 .- 0.1 .* x2 .+ sin.(2 .* x3) .+ 0.05 .* randn(rng, n)
    probability = 1.0 ./ (1.0 .+ exp.(-(-0.2 .+ x1 .- 0.08 .* x2 .+ x3)))
    y_binary = Float64.(rand(rng, n) .< probability)

    @testset "unscaled features and configurations" begin
        Xt = CausalTargeted._prepare_mlj_tree_features(X)
        @test Xt == X[:, 2:end]
        @test vec(mean(Xt; dims = 1)) != zeros(3)

        rf = rf_ext.randomforest_model(:gaussian, size(Xt, 2))
        @test rf isa MLJDecisionTreeInterface.RandomForestRegressor
        @test rf.n_trees == 500
        @test rf.n_subfeatures == max(1, floor(Int, sqrt(size(Xt, 2))))
        @test rf.min_samples_leaf == 5
        @test rf.sampling_fraction == 1.0
        @test rf.rng isa AbstractRNG

        rfc = rf_ext.randomforest_model(:binomial, size(Xt, 2))
        @test rfc isa MLJDecisionTreeInterface.RandomForestClassifier
        @test rfc.n_trees == rf.n_trees
        @test rfc.n_subfeatures == rf.n_subfeatures
        @test rfc.min_samples_leaf == rf.min_samples_leaf
        @test rfc.sampling_fraction == rf.sampling_fraction

        for family in (:gaussian, :binomial)
            xgb = xgb_ext.xgboost_model(family)
            @test (family == :gaussian ?
                xgb isa MLJXGBoostInterface.XGBoostRegressor :
                xgb isa MLJXGBoostInterface.XGBoostClassifier)
            @test xgb.num_round == 100
            @test xgb.max_depth == 2
            @test xgb.eta ≈ 0.05
            @test xgb.min_child_weight ≈ 5.0
            @test xgb.subsample ≈ 0.8
            @test xgb.colsample_bytree ≈ 0.8
            @test xgb.lambda ≈ 1.0
            @test xgb.alpha ≈ 0.0
            @test xgb.gamma ≈ 0.0
            @test xgb.nthread == 1
            @test xgb.seed == 42
        end
    end

    for learner in (:randomforest, :xgboost)
        @testset "$learner gaussian and binomial" begin
            fit_a = fit_super_learner(
                X,
                y;
                learners = (learner,),
                family = :gaussian,
                metalearner = :invmse,
            )
            fit_b = fit_super_learner(
                X,
                y;
                learners = (learner,),
                family = :gaussian,
                metalearner = :invmse,
            )
            pred_a = predict_super_learner(fit_a, X)
            pred_b = predict_super_learner(fit_b, X)
            @test fit_a.learners == [learner]
            @test length(pred_a) == n
            @test all(isfinite, pred_a)
            @test pred_a ≈ pred_b atol = 0 rtol = 0

            fit_binary = fit_super_learner(
                X,
                y_binary;
                learners = (learner,),
                family = :binomial,
                metalearner = :invmse,
            )
            pred_binary = predict_super_learner(fit_binary, X)
            @test length(pred_binary) == n
            @test all(isfinite, pred_binary)
            @test all(p -> 0 <= p <= 1, pred_binary)
            @test any(p -> 0 < p < 1, pred_binary)
        end
    end

    @testset "mixed libraries and metalearners" begin
        gaussian_learners = (:glm, :randomforest, :xgboost, :mean)
        gaussian_fit = fit_super_learner(
            X,
            y;
            learners = gaussian_learners,
            family = :gaussian,
            metalearner = :discrete,
            folds = 2,
            rng = StableRNG(702),
        )
        gaussian_pred = predict_super_learner(gaussian_fit, X)
        @test gaussian_fit.learners == collect(gaussian_learners)
        @test length(gaussian_fit.weights) == length(gaussian_learners)
        @test all(isfinite, gaussian_fit.weights)
        @test length(gaussian_pred) == n
        @test all(isfinite, gaussian_pred)

        binary_learners = (:logistic, :randomforest, :xgboost, :mean)
        binary_fit = fit_super_learner(
            X,
            y_binary;
            learners = binary_learners,
            family = :binomial,
            metalearner = :nnloglik,
            folds = 2,
            rng = StableRNG(703),
        )
        binary_pred = predict_super_learner(binary_fit, X)
        @test binary_fit.learners == collect(binary_learners)
        @test length(binary_fit.weights) == length(binary_learners)
        @test all(isfinite, binary_fit.weights)
        @test length(binary_pred) == n
        @test all(isfinite, binary_pred)
        @test all(p -> 0 <= p <= 1, binary_pred)
    end

    @testset "direct MLJ adapter QC" begin
        using DataFrames: DataFrame
        using MLJ: fit!, machine, predict

        Xt = CausalTargeted._prepare_mlj_tree_features(X)
        Xtable = DataFrame(Xt, :auto)
        for learner in (:randomforest, :xgboost)
            adapter = CausalTargeted._fit_learner(learner, X, y; family = :gaussian)
            adapter_pred = CausalTargeted._predict_learner(adapter, X)
            model = learner == :randomforest ?
                rf_ext.randomforest_model(:gaussian, size(Xt, 2)) :
                xgb_ext.xgboost_model(:gaussian)
            direct = machine(model, Xtable, y)
            fit!(direct, verbosity = 0)
            direct_pred = vec(Float64.(predict(direct, Xtable)))
            @test adapter_pred ≈ direct_pred atol = 0 rtol = 0
        end
    end

    @test :randomforest in RICH_SL_LEARNERS
    @test !(:xgboost in RICH_SL_LEARNERS)
    @test !(:randomforest in DEFAULT_SL_LEARNERS)
    @test !(:xgboost in DEFAULT_SL_LEARNERS)
    @test !(:randomforest in SMALL_N_SL_LEARNERS)
    @test !(:xgboost in SMALL_N_SL_LEARNERS)
    @test !(:randomforest in adaptive_learners(200; rich = true))
    @test !(:xgboost in adaptive_learners(200; rich = true))
end
