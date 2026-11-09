using Test
using CausalTargeted
using DataFrames
using StableRNGs
using Statistics
using Distributions

function targeting_fixture(; family = :gaussian, status = :available)
    y = family === :binomial ? [0.0, 1.0, 0.0, 1.0, 1.0, 0.0] :
        family in CausalTargeted.COUNT_OUTCOME_FAMILIES ? [1.0, 2.0, 0.5, 3.0, 1.5, 2.5] :
        [0.2, 1.0, 1.4, 2.0, 0.7, 1.8]
    q_obs = family === :binomial ? [0.2, 0.35, 0.4, 0.6, 0.7, 0.25] :
        family in CausalTargeted.COUNT_OUTCOME_FAMILIES ? [1.2, 1.6, 0.9, 2.2, 1.4, 1.8] :
        [0.4, 0.8, 1.0, 1.3, 0.9, 1.1]
    q1 = family === :binomial ? [0.3, 0.4, 0.5, 0.7, 0.75, 0.35] :
        family in CausalTargeted.COUNT_OUTCOME_FAMILIES ? [1.4, 1.8, 1.2, 2.6, 1.7, 2.0] :
        [0.7, 1.2, 1.3, 1.8, 1.0, 1.4]
    q0 = family === :binomial ? [0.15, 0.3, 0.35, 0.55, 0.65, 0.2] :
        family in CausalTargeted.COUNT_OUTCOME_FAMILIES ? [1.0, 1.3, 0.8, 1.9, 1.2, 1.5] :
        [0.3, 0.7, 0.8, 1.1, 0.8, 0.9]
    return (
        y = y, Q_obs = q_obs, Q1 = q1, Q0 = q0,
        H1 = [0.4, 0.7, 1.0, 1.3, 1.6, 1.0],
        H0 = [1.6, 1.3, 1.0, 0.7, 0.4, 1.0],
        n = length(y), trunc = 10.0, density_ratio = :gaussian,
        family_outcome = family, inference_status = status,
    )
end

@testset "LMTP nuisance cache shares the same cross-fit partition" begin
    df = DataFrame(
        W = collect(1.0:24.0),
        A = [sin(i / 3) + 0.1 * i for i in 1:24],
        Y = [cos(i / 5) + i / 8 for i in 1:24],
    )
    requested_folds = [collect(1:8), collect(9:16), collect(17:24)]

    qfit = fit_outcome_regression(
        df, :Y, :A, [:W], 3, StableRNG(31);
        learners = (:glm,), fold_sets = requested_folds,
    )
    gfit = fit_exposure_density(
        df, :A, [:W], 3, StableRNG(32);
        learners = (:glm,), fold_sets = requested_folds,
    )
    @test qfit.fold_test_idx == requested_folds
    @test gfit.fold_test_idx == requested_folds

    cache = build_lmtp_fold_cache(
        df, :A, :Y, [:W], 3, StableRNG(33);
        learners_outcome = (:glm,), learners_trt = (:glm,),
    )
    @test cache.outcome_model.fold_test_idx == cache.exposure_model.fold_test_idx

    components = lmtp_components_from_cache(cache, df.A .+ 0.2, df.A; trunc = 1e6)
    altered = copy(df)
    validation_rows = cache.outcome_model.fold_test_idx[1]
    altered.Y[validation_rows] .+= 1e4
    cache_altered = build_lmtp_fold_cache(
        altered, :A, :Y, [:W], 3, StableRNG(33);
        learners_outcome = (:glm,), learners_trt = (:glm,),
    )
    altered_components = lmtp_components_from_cache(
        cache_altered, altered.A .+ 0.2, altered.A; trunc = 1e6,
    )
    @test components.Q_obs[validation_rows] == altered_components.Q_obs[validation_rows]
end

@testset "cross-fit helpers coerce numeric values in Any-typed columns" begin
    df = DataFrame(
        A = Any[repeat([0, 1], 6)...],
        W = collect(1.0:12.0),
    )
    fold_sets = [collect(1:4), collect(5:8), collect(9:12)]

    propensity = crossfit_propensity(
        df, :A, [:W], 3, StableRNG(3001);
        learners = (:mean,), fold_sets = fold_sets,
    )
    treatment_mean = crossfit_treatment_mean(
        df, :A, [:W], 3, StableRNG(3002); learners = (:mean,),
    )

    @test length(propensity) == nrow(df)
    @test all(isfinite, propensity)
    @test all(p -> 1e-3 <= p <= 1 - 1e-3, propensity)
    @test length(treatment_mean) == nrow(df)
    @test all(isfinite, treatment_mean)

    numeric_outcomes = DataFrame(
        A = repeat([0.0, 1.0], 6),
        W = collect(1.0:12.0),
        Y = Any[2 * i + 1 for i in 1:12],
    )
    outcome_predictions = crossfit_outcome_predictions(
        numeric_outcomes, :Y, :A, [:W], 3, StableRNG(3003); learners = (:mean,),
    )
    counterfactual_predictions = crossfit_predict_outcome(
        numeric_outcomes, :Y, :A, [:W], fill(0.5, 12), 3, StableRNG(3004);
        learners = (:mean,),
    )
    @test all(isfinite, outcome_predictions)
    @test all(isfinite, counterfactual_predictions)
end

@testset "LMTP policy-specific TMLE targeting contract" begin
    @testset "Gaussian weighted-intercept updates solve each arm score" begin
        c = targeting_fixture()
        fit = lmtp_tmle_from_components(c; estimator = :tmle, epochs = 3)
        ε1 = sum(c.H1 .* (c.y .- c.Q_obs)) / sum(c.H1)
        ε0 = sum(c.H0 .* (c.y .- c.Q_obs)) / sum(c.H0)
        qobs1 = c.Q_obs .+ ε1
        qobs0 = c.Q_obs .+ ε0
        q1 = c.Q1 .+ ε1
        q0 = c.Q0 .+ ε0
        raw_score = (q1 .+ c.H1 .* (c.y .- qobs1)) .-
            (q0 .+ c.H0 .* (c.y .- qobs0))
        @test abs(sum(c.H1 .* (c.y .- qobs1))) < 1e-12
        @test abs(sum(c.H0 .* (c.y .- qobs0))) < 1e-12
        @test fit.psi1 ≈ mean(q1)
        @test fit.psi0 ≈ mean(q0)
        @test fit.estimate ≈ mean(q1) - mean(q0)
        @test fit.point_score ≈ raw_score
        @test mean(fit.point_score) ≈ fit.estimate
        @test fit.ic ≈ raw_score .- mean(raw_score)
        @test fit.se ≈ std(fit.ic) / sqrt(c.n)
        @test abs(fit.targeting_diagnostics.score_policy1) < 1e-12
        @test abs(fit.targeting_diagnostics.score_policy0) < 1e-12
        @test lmtp_tmle_from_components(c; estimator = :tmle, epochs = 1).estimate ≈ fit.estimate
        @test lmtp_tmle_from_components(c; estimator = :itmle, epochs = 5).estimate ≈ fit.estimate
    end

    @testset "count weighted-intercept scaling solves each arm score" begin
        c = targeting_fixture(; family = :poisson)
        fit = lmtp_tmle_from_components(c; estimator = :tmle, family_outcome = :poisson)
        scale1 = sum(c.H1 .* c.y) / sum(c.H1 .* c.Q_obs)
        scale0 = sum(c.H0 .* c.y) / sum(c.H0 .* c.Q_obs)
        @test abs(sum(c.H1 .* (c.y .- scale1 .* c.Q_obs))) < 1e-12
        @test abs(sum(c.H0 .* (c.y .- scale0 .* c.Q_obs))) < 1e-12
        @test fit.psi1 ≈ mean(scale1 .* c.Q1)
        @test fit.psi0 ≈ mean(scale0 .* c.Q0)
        @test mean(fit.point_score) ≈ fit.estimate
        @test all(isfinite, fit.point_score)
    end

    @testset "count and binary boundary outcomes remain finite and gate inference" begin
        c = targeting_fixture(; family = :poisson)
        zeros_count = merge(c, (y = zeros(c.n),))
        fit_count = lmtp_tmle_from_components(
            zeros_count; estimator = :tmle, family_outcome = :poisson,
        )
        @test fit_count.psi1 == 0.0
        @test fit_count.psi0 == 0.0
        @test fit_count.inference_status === :boundary_outcome
        @test all(isnan, fit_count.ic)
        @test all(isfinite, fit_count.point_score)

        for boundary_y in (zeros(c.n), ones(c.n))
            binary = targeting_fixture(; family = :binomial)
            binary = merge(binary, (y = boundary_y,))
            fit_binary = lmtp_tmle_from_components(
                binary; estimator = :tmle, family_outcome = :binomial,
            )
            target = first(boundary_y)
            @test fit_binary.psi1 == target
            @test fit_binary.psi0 == target
            @test fit_binary.inference_status === :boundary_outcome
            @test all(isnan, fit_binary.ic)
            @test all(isfinite, fit_binary.point_score)
        end
    end

    @testset "binomial intercept brackets expand and inputs are validated" begin
        c = targeting_fixture(; family = :binomial)
        q_extreme = merge(c, (
            Q_obs = fill(eps(Float64), c.n),
            Q1 = fill(1 - eps(Float64), c.n),
            Q0 = fill(eps(Float64), c.n),
        ))
        fit = lmtp_tmle_from_components(q_extreme; estimator = :tmle, family_outcome = :binomial)
        @test abs(fit.targeting_diagnostics.score_policy1) < 1e-7
        @test abs(fit.targeting_diagnostics.score_policy0) < 1e-7
        @test_throws DomainError lmtp_tmle_from_components(
            merge(c, (Q_obs = fill(1.1, c.n),)); estimator = :tmle, family_outcome = :binomial,
        )
        @test_throws DomainError lmtp_tmle_from_components(
            merge(targeting_fixture(; family = :poisson), (y = fill(-1.0, c.n),));
            estimator = :tmle, family_outcome = :poisson,
        )
        @test_throws ArgumentError lmtp_tmle_from_components(c; estimator = :unknown)
        @test_throws DomainError lmtp_tmle_from_components(
            merge(targeting_fixture(), (H1 = fill(floatmax(Float64), c.n),));
            estimator = :tmle,
        )
    end

    @testset "uncapped Gaussian density ratios remain stable and exact" begin
        ratio = CausalTargeted._mtp_clever_covariate_gaussian_uncapped
        @test ratio([1e160], [1e160], [0.0], 1.0) == [1.0]
        observed, policy, μ, σ = 0.7, 0.2, -0.1, 0.8
        expected = pdf(Distributions.Normal(μ, σ), 2observed - policy) /
            pdf(Distributions.Normal(μ, σ), observed)
        @test ratio([observed], [policy], [μ], σ)[1] ≈ expected
        @test_throws DomainError ratio([1e308], [0.0], [0.0], 1.0)
    end

    @testset "binomial weighted-intercept tilts solve each arm score" begin
        c = targeting_fixture(; family = :binomial)
        fit = lmtp_tmle_from_components(c; estimator = :tmle, family_outcome = :binomial)
        @test abs(fit.targeting_diagnostics.score_policy1) < 1e-8
        @test abs(fit.targeting_diagnostics.score_policy0) < 1e-8
        @test 0.0 <= fit.psi1 <= 1.0
        @test 0.0 <= fit.psi0 <= 1.0
        @test mean(fit.point_score) ≈ fit.estimate
        @test all(isfinite, fit.point_score)
    end

    @testset "constant outcome translation preserves the contrast" begin
        c = targeting_fixture()
        shifted = merge(c, (y = c.y .+ 4.5, Q_obs = c.Q_obs .+ 4.5,
                            Q1 = c.Q1 .+ 4.5, Q0 = c.Q0 .+ 4.5))
        fit = lmtp_tmle_from_components(c; estimator = :tmle)
        fit_shifted = lmtp_tmle_from_components(shifted; estimator = :tmle)
        @test fit_shifted.estimate ≈ fit.estimate
        @test fit_shifted.se ≈ fit.se
    end

    @testset "gated and attenuated inference keeps a finite point score" begin
        c = targeting_fixture(; status = :truncated_weights)
        gated = lmtp_tmle_from_components(c; estimator = :tmle)
        @test gated.inference_status === :truncated_weights
        @test all(isnan, gated.ic)
        @test all(isnan, (gated.se, gated.lower, gated.upper))
        @test all(isfinite, gated.point_score)
        @test mean(gated.point_score) ≈ gated.estimate

        attenuated = lmtp_tmle_from_components(targeting_fixture();
            estimator = :tmle, targeting_weight = 0.5)
        @test attenuated.inference_status === :attenuated_targeting
        @test all(isnan, attenuated.ic)
        @test all(isfinite, attenuated.point_score)
        @test mean(attenuated.point_score) ≈ attenuated.estimate

        precedence = lmtp_tmle_from_components(targeting_fixture(; status = :positivity_violation);
            estimator = :tmle, targeting_weight = 0.5)
        @test precedence.inference_status === :positivity_violation
        @test all(isnan, precedence.ic)
    end

    @testset "zero policy clever covariates gate inference for every estimator" begin
        c = targeting_fixture()
        for arm in (:H1, :H0), estimator in (:tmle, :eif)
            zero_arm = arm === :H1 ? (H1 = zeros(c.n),) : (H0 = zeros(c.n),)
            gated_components = merge(c, zero_arm, (inference_status = :available,))
            fit = lmtp_tmle_from_components(gated_components; estimator = estimator)
            @test fit.inference_status === :positivity_violation
            @test all(isnan, fit.ic)
            @test all(isnan, (fit.se, fit.lower, fit.upper))
            @test all(isfinite, fit.point_score)
            @test mean(fit.point_score) ≈ fit.estimate
        end
    end

    @testset "identical policy arms are exactly trivial" begin
        c = targeting_fixture()
        same = merge(c, (Q0 = copy(c.Q1), H0 = copy(c.H1), inference_status = :trivial))
        fit = lmtp_tmle_from_components(same; estimator = :tmle)
        @test fit.estimate === 0.0
        @test fit.se === 0.0
        @test fit.inference_status === :trivial
        @test all(iszero, fit.ic)
        @test all(iszero, fit.point_score)

        unsupported = merge(c, (Q0 = copy(c.Q1), H0 = copy(c.H1),
                                inference_status = :positivity_violation))
        unsupported_fit = lmtp_tmle_from_components(unsupported; estimator = :tmle)
        @test unsupported_fit.inference_status === :positivity_violation
        @test all(isnan, unsupported_fit.ic)
    end

    @testset "uncapped component routes reject unsupported backends and policies" begin
        df = DataFrame(A = collect(range(-1.0, 1.0; length = 12)),
                       W = repeat([0.0, 1.0], 6), Y = collect(range(0.0, 1.0; length = 12)))
        a = Float64.(df.A)
        @test_throws ArgumentError CausalTargeted._shared_fold_lmtp_components(
            df, :A, :Y, [:W], a .+ 0.2, a, 2, StableRNG(991);
            density_ratio = :classification, weight_truncation = false,
        )
        @test_throws ArgumentError CausalTargeted._shared_fold_lmtp_components(
            df, :A, :Y, [:W], a .+ 0.2, a, 2, StableRNG(992);
            density_ratio = :gaussian, cv_trunc = true, weight_truncation = false,
        )
        @test_throws ArgumentError CausalTargeted._shared_fold_lmtp_components(
            df, :A, :Y, [:W], a .+ 0.2, a, 2, StableRNG(993);
            density_ratio = :gaussian, L = -1.0, U = 1.0,
            shift_amount = 0.2, weight_truncation = false,
        )
        @test_throws ArgumentError CausalTargeted._shared_fold_lmtp_components(
            df, :A, :Y, [:W], a .+ 0.2 .* df.W, a, 2, StableRNG(994);
            density_ratio = :gaussian, weight_truncation = false,
        )
    end
end

@testset "single-policy LMTP mean uses the same targeting and inference contract" begin
    w = collect(range(-1.0, 1.0; length = 60))
    a = [0.6w[i] + 0.4sin(i / 4) for i in eachindex(w)]
    y = [1.2 + 0.7a[i] + 0.3w[i] + 0.2cos(i / 5) for i in eachindex(w)]
    df = DataFrame(W = w, A = a, Y = y)
    a = Float64.(df.A)
    policy = a .+ 0.25
    learners = (:glm,)
    c = CausalTargeted._shared_fold_lmtp_components(
        df, :A, :Y, [:W], policy, a, 3, StableRNG(82);
        learners_outcome = learners, learners_trt = learners,
        density_ratio = :gaussian, weight_truncation = false,
    )
    qobs, qpolicy, _, _ = CausalTargeted._weighted_intercept_tilt(
        c.y, c.Q_obs, c.Q1, c.H1, :gaussian, 1.0,
    )
    raw = qpolicy .+ c.H1 .* (c.y .- qobs)
    ψ_expected = mean(qpolicy)
    ic_expected = raw .- mean(raw)

    ψ, ic = lmtp_tmle_policy_mean(
        df, :A, :Y, [:W], policy, 3, StableRNG(82);
        learners_outcome = learners, learners_trt = learners,
        density_ratio = :gaussian, estimator = :tmle, weight_truncation = false,
    )
    @test ψ ≈ ψ_expected
    @test ic ≈ ic_expected

    c_capped = CausalTargeted._shared_fold_lmtp_components(
        df, :A, :Y, [:W], policy, a, 3, StableRNG(82);
        learners_outcome = learners, learners_trt = learners,
        density_ratio = :gaussian,
    )
    qobs_capped, qpolicy_capped, _, _ = CausalTargeted._weighted_intercept_tilt(
        c_capped.y, c_capped.Q_obs, c_capped.Q1, c_capped.H1, :gaussian, 1.0,
    )
    ψ_capped_expected = mean(qpolicy_capped)
    ψ_capped, ic_capped = lmtp_tmle_policy_mean(
        df, :A, :Y, [:W], policy, 3, StableRNG(82);
        learners_outcome = learners, learners_trt = learners,
        density_ratio = :gaussian, estimator = :tmle,
    )
    @test ψ_capped ≈ ψ_capped_expected
    @test all(isnan, ic_capped)

    ψ_uncapped, ic_uncapped = lmtp_tmle_policy_mean(
        df, :A, :Y, [:W], policy, 3, StableRNG(82);
        learners_outcome = learners, learners_trt = learners,
        density_ratio = :gaussian, estimator = :tmle, weight_truncation = false,
    )
    @test isfinite(ψ_uncapped)
    @test all(isfinite, ic_uncapped)

    ψ_identity, ic_identity = lmtp_tmle_policy_mean(
        df, :A, :Y, [:W], a, 3, StableRNG(82);
        learners_outcome = learners, learners_trt = learners,
        density_ratio = :gaussian, estimator = :tmle,
    )
    @test ψ_identity ≈ mean(df.Y)
    @test ic_identity ≈ df.Y .- mean(df.Y)
    @test_throws ArgumentError lmtp_tmle_policy_mean(
        df, :A, :Y, [:W], policy, 3, StableRNG(82); estimator = :unknown,
    )

    # The single-policy public route must reject finite inputs whose uncapped
    # influence score or weighted intercept overflows Float64.
    n_extreme = 24
    extreme_df = DataFrame(
        W = zeros(n_extreme),
        A = repeat([-0.5, 0.5], n_extreme ÷ 2),
        Y = repeat([1e308, -1e308], n_extreme ÷ 2),
    )
    extreme_policy = extreme_df.A .+ 5.0
    for extreme_estimator in (:eif, :tmle)
        @test_throws DomainError lmtp_tmle_policy_mean(
            extreme_df, :A, :Y, [:W], extreme_policy, 2, StableRNG(83);
            learners_outcome = (:mean,), learners_trt = (:mean,),
            density_ratio = :gaussian, estimator = extreme_estimator,
            weight_truncation = false,
        )
    end
end
