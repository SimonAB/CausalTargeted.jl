import CausalTargeted: CovariateSchema, fit_covariate_schema, transform_covariates

@testset "CovariateSchema and categorical covariates" begin
    @testset "schema machinery remains internal" begin
        public_names = names(CausalTargeted)
        @test :CovariateSchema ∉ public_names
        @test :fit_covariate_schema ∉ public_names
        @test :transform_covariates ∉ public_names
    end

    @testset "supported columns and stable StatsModels coding" begin
        df = DataFrame(
            x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            count = [1, 2, 3, 4, 5, 6],
            flag = [true, false, true, false, true, false],
            sex = ["M", "F", "F", "M", "F", "M"],
            breed = ["A", "B", "C", "A", "B", "C"],
            pooled = categorical(["low", "high", "low", "high", "low", "high"]),
        )
        schema = fit_covariate_schema(df, [:x, :count, :flag, :sex, :breed, :pooled])
        encoded = transform_covariates(schema, df)
        @test encoded isa Matrix{Float64}
        @test size(encoded) == (6, 7) # 3 numeric/Bool + 1 sex + 2 breed + 1 pooled
        @test encoded[:, 1] == df.x
        @test encoded[:, 2] == Float64.(df.count)
        @test encoded[:, 3] == Float64.(df.flag)
        @test schema.feature_names == [
            "x", "count", "flag", "sex: M", "breed: B", "breed: C", "pooled: low",
        ]
        @test transform_covariates(schema, df) == encoded

        # Ordinary integer columns remain continuous, even when values repeat.
        integer_schema = fit_covariate_schema(DataFrame(site = [1, 2, 3, 1]), [:site])
        @test transform_covariates(integer_schema, DataFrame(site = [1, 2, 3, 1])) ==
            reshape([1.0, 2.0, 3.0, 1.0], :, 1)

        categorical_with_unused = categorical(
            ["A", "B", "A", "B"];
            levels = ["A", "B", "C"],
        )
        categorical_schema = fit_covariate_schema(
            DataFrame(group = categorical_with_unused),
            [:group],
        )
        @test categorical_schema.feature_names == ["group: B", "group: C"]
        @test size(transform_covariates(
            categorical_schema,
            DataFrame(group = categorical(["C", "A"]; levels = ["A", "B", "C"])),
        )) == (2, 2)

        explicitly_ordered = categorical(
            ["CHB", "CB", "TEX", "CHB"];
            levels = ["CB", "CHB", "TEX"],
        )
        ordered_schema = fit_covariate_schema(
            DataFrame(breed = explicitly_ordered),
            [:breed],
        )
        @test ordered_schema.feature_names == ["breed: CHB", "breed: TEX"]
        @test transform_covariates(
            ordered_schema,
            DataFrame(breed = categorical(
                ["CB", "CHB", "TEX"];
                levels = ["CB", "CHB", "TEX"],
            )),
        ) == [0.0 0.0; 1.0 0.0; 0.0 1.0]
    end

    @testset "fold-level level stability" begin
        full = DataFrame(breed = ["A", "B", "C", "A", "B", "C"], age = 1:6)
        schema = fit_covariate_schema(full, [:breed, :age])
        ab = transform_covariates(schema, full[[1, 2, 4, 5], :])
        bc = transform_covariates(schema, full[[2, 3, 5, 6], :])
        @test size(ab, 2) == size(bc, 2) == 3
        @test all(iszero, ab[:, 2]) # C dummy is structural even though C is absent
        @test bc[2, 2] == 1.0
        @test schema.feature_names == fit_covariate_schema(full, [:breed, :age]).feature_names
    end

    @testset "single-level categorical covariate" begin
        for group in (
            fill("A", 4),
            categorical(fill("A", 4); levels = ["A"]),
        )
            schema = fit_covariate_schema(DataFrame(group = group), [:group])
            @test isempty(schema.feature_names)
            @test size(transform_covariates(schema, DataFrame(group = group))) == (4, 0)
        end
    end

    @testset "errors are explicit" begin
        err = try
            fit_covariate_schema(DataFrame(age = [1, 2]), [:age, :breed])
            nothing
        catch error
            error
        end
        @test err isa ArgumentError
        @test occursin("breed", sprint(showerror, err))
        @test design_matrix(DataFrame(age = [1, 2]), [:breed]) == ones(2, 1)
        @test CausalTargeted.columns_present(DataFrame(age = [1, 2]), [:age, :breed]) == [:age]
        @test_throws ArgumentError fit_covariate_schema(
            DataFrame(age = Union{Missing, Int}[1, missing]),
            [:age],
        )

        schema = fit_covariate_schema(DataFrame(group = ["A", "B"]), [:group])
        unseen = try
            transform_covariates(schema, DataFrame(group = ["C"]))
            nothing
        catch error
            error
        end
        @test unseen isa ArgumentError
        @test occursin("unseen categorical level", sprint(showerror, unseen))

        treatment_schema = fit_covariate_schema(DataFrame(x = [1.0, 2.0]), [:x])
        @test_throws ArgumentError design_matrix(
            treatment_schema,
            DataFrame(x = [1.0, 2.0], A = ["control", "treated"]);
            treatment = :A,
        )

        categorical_treatment = try
            run_lmtp_grid(
                DataFrame(
                    A = repeat(["control", "treated"], 4),
                    x = 1.0:8.0,
                    Y = 1.0:8.0,
                ),
                :A,
                :Y;
                baseline = [:x],
                deltas = [0.0],
                folds = 2,
                parallel = false,
                simultaneous = false,
            )
            nothing
        catch error
            error
        end
        @test categorical_treatment isa ArgumentError
        @test occursin(
            "categorical treatment is not supported",
            sprint(showerror, categorical_treatment),
        )

        all_missing_categorical = try
            handle_missing_data(
                DataFrame(
                    group = Union{Missing, String}[missing, missing],
                    Y = [1.0, 2.0],
                ),
                :Y,
                [:group],
                :impute,
            )
            nothing
        catch error
            error
        end
        @test all_missing_categorical isa ArgumentError
        @test occursin("group", sprint(showerror, all_missing_categorical))
        @test occursin("no observed values", sprint(showerror, all_missing_categorical))

        unsupported = try
            fit_covariate_schema(
                DataFrame(visit_date = [Date(2025, 1, 1), Date(2025, 1, 2)]),
                [:visit_date],
            )
            nothing
        catch error
            error
        end
        @test unsupported isa ArgumentError
        @test occursin("Unsupported covariate type for :visit_date: Date", sprint(showerror, unsupported))

        invalid_df = DataFrame(
            A = collect(0.1:0.1:1.2),
            visit_date = Date(2025, 1, 1) .+ Day.(0:11),
            Y = collect(1.0:12.0),
        )
        schema_errors = map((true, false)) do cache_nuisances
            try
                run_lmtp_grid(
                    invalid_df, :A, :Y;
                    baseline = [:visit_date],
                    deltas = [0.2],
                    folds = 2,
                    learners_outcome = (:glm, :mean),
                    learners_trt = (:glm, :mean),
                    cache_nuisances = cache_nuisances,
                    parallel = false,
                    simultaneous = false,
                    shift_scale = "raw",
                    rng = StableRNG(416),
                )
                nothing
            catch error
                error
            end
        end
        @test all(error -> error isa ArgumentError, schema_errors)
        @test sprint(showerror, schema_errors[1]) == sprint(showerror, schema_errors[2])
        @test occursin("Unsupported covariate type for :visit_date: Date", sprint(showerror, schema_errors[1]))
    end

    @testset "design-matrix and tabular-fit compatibility" begin
        df = DataFrame(
            A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            age = [1, 2, 3, 4, 5, 6],
            breed = ["A", "B", "C", "A", "B", "C"],
        )
        schema = fit_covariate_schema(df, [:age, :breed])
        W = covariate_design_matrix(schema, df)
        X = design_matrix(schema, df; treatment = :A)
        Xcf = design_matrix(schema, df; treatment = :A, treatment_values = ones(6))
        @test X isa Matrix{Float64}
        @test X[:, 1] == ones(6)
        @test X[:, 2] == df.A
        @test X[:, 3:end] == transform_covariates(schema, df)
        @test outcome_design_matrix(W, df.A) == X
        @test outcome_design_matrix(W, ones(6)) == Xcf

        y = 1 .+ df.A .+ Float64.(df.age)
        fit = CausalTargeted._fit_sl_outcome(
            df[[1, 2, 4, 5], :],
            [:age, :breed],
            y[[1, 2, 4, 5]];
            treatment = :A,
            learners = (:glm, :mean),
            rng = StableRNG(401),
            schema = schema,
        )
        pred = CausalTargeted._predict_sl(
            fit,
            df[[3, 6], :],
            [:age, :breed];
            treatment = :A,
        )
        @test length(pred) == 2
        @test all(isfinite, pred)

        # Schema storage must not change the layouts of existing public cache
        # and nuisance types.
        @test fieldnames(OutcomeRegression) == (
            :treatment, :covariates, :learners, :fold_models, :fold_test_idx, :W,
        )
        @test fieldnames(ExposureDensity) == (
            :covariates, :learners, :fold_models, :fold_test_idx, :W,
        )
        @test fieldnames(CausalTargeted.LMTPFoldCache) == (
            :df, :trt, :outcome, :covariates, :outcome_model, :exposure_model,
            :y, :a, :W, :folds, :learners_outcome, :learners_trt, :rng_seed,
        )
    end

    @testset "automatic coding agrees with explicit dummy coding" begin
        raw = DataFrame(
            A = repeat([0.0, 0.5, 1.0], 4),
            age = collect(1.0:12.0),
            site = repeat(["A", "B", "C"], 4),
        )
        y = 0.5 .+ 0.7 .* raw.A .+ 0.1 .* raw.age .+
            0.4 .* (raw.site .== "B") .- 0.2 .* (raw.site .== "C")
        automatic_schema = fit_covariate_schema(raw, [:age, :site])
        X_automatic = design_matrix(automatic_schema, raw; treatment = :A)

        manual = DataFrame(
            A = raw.A,
            age = raw.age,
            site_B = Float64.(raw.site .== "B"),
            site_C = Float64.(raw.site .== "C"),
        )
        X_manual = design_matrix(manual, [:age, :site_B, :site_C]; treatment = :A)
        @test automatic_schema.feature_names == ["age", "site: B", "site: C"]
        @test X_automatic == X_manual

        fit_automatic = fit_super_learner(
            X_automatic, y;
            learners = (:glm,), folds = 2, rng = StableRNG(415),
        )
        fit_manual = fit_super_learner(
            X_manual, y;
            learners = (:glm,), folds = 2, rng = StableRNG(415),
        )
        @test predict_super_learner(fit_automatic, X_automatic) ≈
            predict_super_learner(fit_manual, X_manual)
    end

    @testset "Super Learner candidates consume encoded Float64 matrices" begin
        rng = StableRNG(402)
        n = 36
        site = repeat(["A", "B", "C"], 12)
        x = randn(rng, n)
        A = 0.3 .* x .+ 0.2 .* (site .== "C") .+ randn(rng, n)
        y = 0.8 .* A .+ 0.4 .* x .+ 0.2 .* (site .== "B") .+ 0.1 .* randn(rng, n)
        df = DataFrame(A = A, x = x, site = site)
        schema = fit_covariate_schema(df, [:x, :site])
        X = design_matrix(schema, df; treatment = :A)
        for learner in (:glm, :glmnet, :randomforest, :xgboost, :evotree, :mean)
            fit = fit_super_learner(
                X, y;
                learners = (learner,),
                folds = 2,
                rng = StableRNG(403),
            )
            pred = predict_super_learner(fit, X)
            @test length(pred) == n
            @test all(isfinite, pred)
        end
        binary = Float64.(y .> median(y))
        fit_binary = fit_super_learner(
            X, binary;
            learners = (:logistic, :randomforest, :xgboost, :mean),
            family = :binomial,
            metalearner = :invmse,
            folds = 2,
            rng = StableRNG(404),
        )
        probabilities = predict_super_learner(fit_binary, X)
        @test all(isfinite, probabilities)
        @test all((0.0 .<= probabilities) .& (probabilities .<= 1.0))
    end

    @testset "simulate_mixed_baseline_mtp" begin
        df, truth = simulate_mixed_baseline_mtp(40; rng = StableRNG(77))
        @test names(df) == ["W", "A", "Y", "site", "vaccinated", "breed"]
        @test eltype(df.site) <: AbstractString
        @test eltype(df.vaccinated) == Bool
        @test truth.baseline == [:W, :site, :vaccinated, :breed]
        @test truth.shift_effect(0.5) ≈ 0.5 * truth.β_a
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = truth.baseline,
            deltas = [0.0, 0.25],
            folds = 2,
            learners_outcome = (:glm, :mean),
            parallel = false,
            rng = StableRNG(78),
        )
        @test all(isfinite, grid.est)
    end

    @testset "categorical estimator paths" begin
        rng = StableRNG(405)
        n = 72
        site = repeat(["A", "B", "C"], 24)
        sex = repeat(["F", "M"], 36)
        breed = categorical(
            repeat(["CB", "CHB", "TEX"], 24);
            levels = ["CB", "CHB", "TEX"],
        )
        vaccinated = repeat([true, false, true], 24)
        age = randn(rng, n)
        A = 0.4 .* age .+ 0.3 .* (site .== "C") .+ 0.1 .* vaccinated .+ randn(rng, n)
        Y = 0.7 .* A .+ 0.5 .* age .+ 0.2 .* (sex .== "M") .+
            0.1 .* (breed .== "CHB") .+ randn(rng, n)
        df = DataFrame(
            age = age,
            sex = sex,
            breed = breed,
            vaccinated = vaccinated,
            site = site,
            A = A,
            Y = Y,
        )
        mixed_covariates = [:age, :sex, :breed, :vaccinated, :site]

        gc = run_gcomp(
            df, :A, :Y;
            covariates = mixed_covariates,
            delta = 0.25,
            folds = 2,
            learners = (:glm, :mean),
            rng = StableRNG(406),
            n_boot = 5,
        )
        @test all(isfinite, (gc.estimate, gc.se, gc.ci_lower, gc.ci_upper))

        for ratio in (:gaussian, :classification, :hybrid)
            grid = run_lmtp_grid(
                df, :A, :Y;
                baseline = mixed_covariates,
                deltas = [0.2],
                folds = 2,
                learners_outcome = (:glm, :mean),
                learners_trt = (:glm, :mean),
                density_ratio = ratio,
                parallel = false,
                simultaneous = false,
                cache_nuisances = false,
                shift_scale = "raw",
                rng = StableRNG(407),
            )
            @test isfinite(only(grid.est))
            @test isfinite(only(grid.se))
        end

        cache = build_lmtp_fold_cache(
            df, :A, :Y, mixed_covariates, 2, StableRNG(408);
            learners_outcome = (:glm, :mean),
            learners_trt = (:glm, :mean),
        )
        cache_schema = fit_covariate_schema(df, mixed_covariates)
        @test cache.W == transform_covariates(cache_schema, df)
        # Preserve the historical RNG sequence: outcome and exposure folds are
        # drawn independently from the supplied RNG.
        expected_rng = StableRNG(408)
        expected_outcome_folds = crossfit_indices(n, 2, expected_rng)
        for test_idx in expected_outcome_folds
            train_idx = setdiff(1:n, test_idx)
            Xtr = outcome_design_matrix(
                cache.outcome_model.W[train_idx, :],
                df.A[train_idx],
            )
            fit_super_learner(
                Xtr, df.Y[train_idx];
                learners = (:glm, :mean), rng = expected_rng,
            )
        end
        expected_exposure_folds = crossfit_indices(n, 2, expected_rng)
        @test cache.outcome_model.fold_test_idx == expected_outcome_folds
        @test cache.exposure_model.fold_test_idx == expected_exposure_folds
        L, U = exposure_bounds(df.A, 0.01, 0.99)
        for delta in (0.1, 0.2)
            shifted = apply_shift_policy(df.A, delta, L, U)
            components = lmtp_components_from_cache(
                cache, shifted, df.A;
                density_ratio = :hybrid,
                L = L,
                U = U,
                shift_policy = delta,
            )
            @test all(isfinite, components.Q1)
            @test size(cache.W, 2) == length(cache_schema.feature_names)
        end
    end

    @testset "sequential, survival, and missing-data paths" begin
        rng = StableRNG(409)
        n = 60
        site = repeat(["A", "B", "C"], 20)
        phase = repeat(["early", "late"], 30)
        age = randn(rng, n)
        A1 = 0.4 .* age .+ 0.2 .* (site .== "C") .+ randn(rng, n)
        L1 = 0.3 .* A1 .+ 0.2 .* (phase .== "late") .+ randn(rng, n)
        A2 = 0.4 .* L1 .+ randn(rng, n)
        Y = 0.4 .* A1 .+ 0.6 .* A2 .+ 0.2 .* age .+ randn(rng, n)
        sequential_df = DataFrame(
            age = age, site = site, A1 = A1, phase = phase, L1 = L1, A2 = A2, Y = Y,
        )
        sequential = run_sequential_lmtp(
            sequential_df, [:A1, :A2], :Y;
            baseline = [:age, :site],
            time_vary = [Symbol[], [:phase, :L1]],
            delta = 0.2,
            folds = 2,
            learners = (:glm, :mean),
            rng = StableRNG(410),
        )
        @test all(isfinite, (sequential.estimate, sequential.se))

        survival_df, truth = simulate_discrete_survival_mtp(80; T = 2, rng = StableRNG(411))
        survival_df.site = repeat(["A", "B", "C", "D"], 20)
        survival = run_survival_lmtp(
            survival_df, truth.treatments, truth.surv;
            baseline = [:W, :site],
            delta = 0.2,
            folds = 2,
            learners = (:glm, :mean),
            rng = StableRNG(412),
        )
        @test all(isfinite, (survival.estimate, survival.se))

        missing_df = DataFrame(
            age = Union{Missing, Float64}[1.0, 2.0, missing, 4.0, 5.0, 6.0, 7.0, 8.0],
            site = categorical(
                Union{Missing, String}["A", "B", missing, "A", "B", "C", "A", "C"];
                levels = ["A", "B", "C"],
            ),
            A = collect(0.1:0.1:0.8),
            Y = Union{Missing, Float64}[1.0, 1.2, 1.4, missing, 1.8, 2.0, 2.2, 2.4],
        )
        for strategy in (:drop, :impute, :ipcw, :ipcw_impute)
            clean, weights, extra = handle_missing_data(
                missing_df, :Y, [:age, :site], strategy;
                folds = 2,
                rng = StableRNG(413),
            )
            schema = fit_covariate_schema(clean, vcat([:age, :site], extra))
            @test all(isfinite, design_matrix(schema, clean))
            @test all(isfinite, weights)
            @test levels(clean.site) == ["A", "B", "C"]
        end
        gc_missing = run_gcomp(
            missing_df, :A, :Y;
            covariates = [:age, :site],
            delta = 0.1,
            folds = 2,
            learners = (:glm, :mean),
            handle_missing = :ipcw_impute,
            n_boot = 0,
            rng = StableRNG(414),
        )
        @test all(isfinite, (gc_missing.estimate, gc_missing.se))
    end
end
