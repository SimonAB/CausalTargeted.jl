using CategoricalArrays
using DataFrames
using Distributions
using LinearAlgebra
using MixedModels
using StableRNGs
using Statistics
using StatsModels
using Test

include("fixtures/pre_refactor_parametric_gcomp.jl")

# These checks deliberately calculate Xβ and G V G' without asking either
# g-computation entry point for its own predictions or gradients.
function _test_site_design(model, data)
    columns = Dict(
        "(Intercept)" => ones(nrow(data)),
        "A" => Float64.(data.A),
        "Group" => Float64.(data.Group),
        "A & Group" => Float64.(data.A .* data.Group),
        "W" => Float64.(data.W),
    )
    names = coefnames(model)
    @assert all(haskey(columns, name) for name in names)
    return hcat((columns[name] for name in names)...)
end

function _test_site_data()
    rng = StableRNG(20260929)
    rows = NamedTuple[]
    for site in 1:30
        group = Float64(isodd(site))
        b = 0.45randn(rng)
        b_slope = 0.25randn(rng)
        for individual in 1:4
            treatment = Float64(iseven(site + individual))
            w = 0.3site / 30 + 0.2individual + 0.3randn(rng)
            response = 1.3 + 0.65treatment + 0.4group +
                0.3treatment * group + 0.2w + b + b_slope * treatment +
                0.12randn(rng)
            push!(rows, (; Site = site, A = treatment, Group = group, W = w, Y = response))
        end
    end
    return DataFrame(rows)
end

function _test_frozen_data()
    rng = StableRNG(20260930)
    rows = NamedTuple[]
    for subject in 1:48
        a = Float64(isodd(subject))
        w = Float64(mod(subject, 4) - 1.5)
        b = 0.35randn(rng)
        for time in (0.0, 1.0, 2.0)
            time == 2 && subject % 4 == 0 && continue
            y = 1.0 + 0.8a + 0.2time + 0.25a * time + 0.3w + b + 0.15randn(rng)
            eta = 0.3 + 0.4a + 0.15time + 0.10a * time + 0.08w + 0.2b
            count = rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(eta))))
            push!(rows, (; subject, A = a, W = w, time, Y = y, Count = count))
        end
    end
    return DataFrame(rows)
end

function _test_nb_design(model, data)
    columns = Dict(
        "(Intercept)" => ones(nrow(data)),
        "A" => Float64.(data.A),
        "time" => Float64.(data.time),
        "A & time" => Float64.(data.A .* data.time),
        "W" => Float64.(data.W),
    )
    names = coefnames(model)
    @assert all(haskey(columns, name) for name in names)
    return hcat((columns[name] for name in names)...)
end

function _test_nb_variance(model)
    if model isa NB2RandomInterceptModel
        return model.random_intercept_variance
    end
    component = only(values(VarCorr(model).σρ))
    return Float64(only(values(component.σ)))^2
end

@testset "Shared parametric g-computation: mixed backend" begin
    data = _test_site_data()
    @test all(length(unique(data.A[data.Site .== site])) == 2 for site in unique(data.Site))
    original = deepcopy(data)
    formula_term = @formula(Y ~ 1 + A * Group + W + (1 | Site))
    fit = fit_parametric_gcomp(
        formula_term, data; backend = :mixed, family = :gaussian, id = :Site,
    )
    model = fit.model
    target = select(data[1:3:nrow(data), :], Not(:Y))
    @test nrow(target) != nrow(data)

    @testset "Gaussian mean, contrast, interaction, and adapters" begin
        target_original = deepcopy(target)
        reference = copy(target)
        comparison = copy(target)
        reference.A .= 0.0
        comparison.A .= 1.0
        x0 = _test_site_design(model, reference)
        x1 = _test_site_design(model, comparison)
        beta = coef(model)
        mu0, mu1 = mean(x0 * beta), mean(x1 * beta)
        g = vec(mean(x1 .- x0; dims = 1))
        variance = dot(g, vcov(model) * g)

        mean_result = gcomp_mean(fit, target; set = (; A = 1.0), random_effects = :zero)
        contrast = gcomp_contrast(
            fit, target; treatment = :A, reference = 0.0, comparison = 1.0,
            random_effects = :zero,
        )
        adapter = gcomp_contrast(
            model, data, target; id = :Site, treatment = :A,
            reference = 0.0, comparison = 1.0, random_effects = :zero,
        )
        @test mean_result.estimate ≈ mu1 rtol = 2e-12
        @test contrast.reference_mean ≈ mu0 rtol = 2e-12
        @test contrast.comparison_mean ≈ mu1 rtol = 2e-12
        @test contrast.estimate ≈ mu1 - mu0 rtol = 2e-12
        @test contrast.se ≈ sqrt(variance) rtol = 2e-10
        @test adapter.estimate ≈ contrast.estimate rtol = 2e-12
        @test adapter.se ≈ contrast.se rtol = 2e-10
        no_covariance = gcomp_contrast(
            model, data, target; id = :Site, covariance = :none,
            treatment = :A, reference = 0.0, comparison = 1.0,
            random_effects = :zero,
        )
        @test no_covariance.se === nothing
        @test no_covariance.covariance_type == :none
        @test contrast.random_effects == :zero
        @test contrast.uncertainty == :delta_fixed
        @test gcomp_mean(
            model, data, target; id = :Site, set = (; A = 1.0), random_effects = :marginal,
        ).estimate ≈ mu1 rtol = 2e-12
        @test gcomp_contrast(
            fit, target; treatment = :A, reference = 0.0, comparison = 1.0,
            random_effects = :marginal,
        ).estimate ≈ contrast.estimate rtol = 2e-12

        ratio = gcomp_contrast(
            fit, target; treatment = :A, reference = 0.0, comparison = 1.0,
            scale = :ratio, random_effects = :zero,
        )
        logratio = gcomp_contrast(
            fit, target; treatment = :A, reference = 0.0, comparison = 1.0,
            scale = :logratio, random_effects = :zero,
        )
        @test ratio.estimate ≈ mu1 / mu0 rtol = 2e-12
        @test logratio.estimate ≈ log(mu1 / mu0) rtol = 2e-12
        @test ratio.log_se ≈ logratio.se rtol = 2e-10
        @test ratio.ci_lower ≈ exp(logratio.ci_lower) rtol = 2e-12
        @test ratio.ci_upper ≈ exp(logratio.ci_upper) rtol = 2e-12

        selected = target[target.A .== 0.0, :]
        selected_set = copy(selected)
        selected_set.A .= 1.0
        selected_expected = mean(_test_site_design(model, selected_set) * beta)
        selected_mean = gcomp_mean(
            fit, target; set = (; A = 1.0), by = (; A = 0.0), random_effects = :zero,
        )
        @test selected_mean.n == nrow(selected)
        @test selected_mean.estimate ≈ selected_expected rtol = 2e-12

        interaction = gcomp_interaction(
            fit, target; treatment = :A, reference = 0.0, comparison = 1.0,
            modifier = :Group, modifier_reference = 0.0,
            modifier_comparison = 1.0, random_effects = :zero,
        )
        @test interaction.estimate ≈ beta[only(findall(==("A & Group"), coefnames(model)))] rtol = 2e-12
        @test interaction.component_means.modifier_reference.n ==
            count(==(0.0), target.Group)
        @test interaction.component_means.modifier_comparison.n ==
            count(==(1.0), target.Group)
        @test gcomp_interaction(
            model, data, target; id = :Site, treatment = :A,
            reference = 0.0, comparison = 1.0, modifier = :Group,
            modifier_reference = 0.0, modifier_comparison = 1.0,
            random_effects = :zero,
        ).estimate ≈ interaction.estimate rtol = 2e-12

        run_result = run_parametric_gcomp(
            formula_term, data; backend = :mixed, family = :gaussian,
            fit_kwargs = (; id = :Site), treatment = :A,
            reference = 0.0, comparison = 1.0, random_effects = :zero,
        )
        full_target_result = gcomp_contrast(
            fit, data; treatment = :A, reference = 0.0,
            comparison = 1.0, random_effects = :zero,
        )
        @test run_result.estimate ≈ full_target_result.estimate rtol = 2e-10

        permuted = target[end:-1:1, :]
        @test gcomp_contrast(
            fit, permuted; treatment = :A, reference = 0.0, comparison = 1.0,
            random_effects = :zero,
        ).estimate ≈ contrast.estimate rtol = 2e-12
        @test isequal(data, original)
        @test isequal(target, target_original)
    end

    @testset "Gaussian random coefficients retain population predictions" begin
        slope_fit = fit_parametric_gcomp(
            @formula(Y ~ 1 + A * Group + W + (1 + A | Site)), data;
            backend = :mixed, family = :gaussian, id = :Site,
        )
        treatment = copy(target)
        treatment.A .= 1.0
        expected = mean(_test_site_design(slope_fit.model, treatment) * coef(slope_fit.model))
        @test gcomp_mean(
            slope_fit, target; set = (; A = 1.0), random_effects = :zero,
        ).estimate ≈ expected rtol = 2e-10
        @test gcomp_mean(
            slope_fit, target; set = (; A = 1.0), random_effects = :marginal,
        ).estimate ≈ expected rtol = 2e-10
    end

    @testset "categorical fitted schema and failures" begin
        categorised = copy(data)
        categorised.Category = categorical(ifelse.(categorised.Group .== 0.0, "low", "high"))
        catfit = fit_parametric_gcomp(
            @formula(Y ~ 1 + A * Category + W + (1 | Site)), categorised;
            backend = :mixed, family = :gaussian, id = :Site,
        )
        cat_target = select(categorised, Not(:Y))
        @test isfinite(gcomp_contrast(
            catfit, cat_target; treatment = :A, reference = 0.0,
            comparison = 1.0, by = (; Category = "high"), random_effects = :zero,
        ).estimate)
        unseen = copy(cat_target)
        unseen.Category = fill("unseen", nrow(unseen))
        @test_throws ArgumentError gcomp_mean(catfit, unseen; random_effects = :zero)

        categorical_treatment = copy(categorised)
        categorical_treatment.A = categorical(ifelse.(categorical_treatment.A .== 0.0,
            "control", "treated"))
        treatment_fit = fit_parametric_gcomp(
            @formula(Y ~ 1 + A * Category + W + (1 | Site)),
            categorical_treatment; backend = :mixed, family = :gaussian, id = :Site,
        )
        categorical_target = select(categorical_treatment, Not(:Y))
        @test isfinite(gcomp_mean(
            treatment_fit, categorical_target; set = (; A = "treated"),
            random_effects = :zero,
        ).estimate)
        @test isfinite(gcomp_contrast(
            treatment_fit, categorical_target; treatment = :A,
            reference = "control", comparison = "treated", random_effects = :zero,
        ).estimate)

        missing_predictor = select(target, Not(:W))
        @test_throws ArgumentError gcomp_mean(fit, missing_predictor; random_effects = :zero)
        @test_throws ArgumentError gcomp_mean(
            model, data[1:(end - 1), :], target;
            id = :Site, random_effects = :zero,
        )
        missing_value = copy(target)
        allowmissing!(missing_value, :W)
        missing_value.W[1] = missing
        @test_throws ArgumentError gcomp_mean(fit, missing_value; random_effects = :zero)
        unrelated_missing = copy(target)
        unrelated_missing.Unused = fill(missing, nrow(unrelated_missing))
        @test isfinite(gcomp_mean(
            fit, unrelated_missing; random_effects = :zero,
        ).estimate)
        @test_throws ArgumentError gcomp_mean(
            fit, target; set = (; Site = 1), random_effects = :zero,
        )
        @test_throws ArgumentError gcomp_mean(
            fit, target; set = (; A = 9.0), random_effects = :conditional,
        )
        @test_throws ArgumentError fit_parametric_gcomp(
            @formula(Y ~ 1 + A + W + (1 | Site)), data;
            backend = :mixed, family = :binomial, id = :Site,
        )
        nonfinite_training = copy(data)
        nonfinite_training.W[1] = Inf
        @test_throws ArgumentError fit_parametric_gcomp(
            formula_term, nonfinite_training;
            backend = :mixed, family = :gaussian, id = :Site,
        )
        rank_deficient = copy(data)
        rank_deficient.Wcopy = copy(rank_deficient.W)
        @test_throws ArgumentError fit_parametric_gcomp(
            @formula(Y ~ 1 + A * Group + W + Wcopy + (1 | Site)),
            rank_deficient; backend = :mixed, family = :gaussian, id = :Site,
        )
        count_data = copy(data)
        count_rng = StableRNG(20260928)
        count_means = exp.(0.2 .+ 0.3 .* count_data.A .+ 0.1 .* count_data.W)
        count_data.Count = [rand(
            count_rng, NegativeBinomial(2.0, 2.0 / (2.0 + mean)),
        ) for mean in count_means]
        @test_throws ArgumentError fit_parametric_gcomp(
            @formula(Count ~ 1 + A + W + (1 + A | Site)), count_data;
            backend = :mixed, family = :negbin, id = :Site, theta = 2.0,
        )
    end
end

@testset "Frozen pre-refactor mixed trajectories" begin
    data = _test_frozen_data()
    glmfit = fit_parametric_gcomp(
        @formula(Y ~ 1 + A * time + W), data; covariance = :model,
    )
    glm_result = gcomp_contrast(
        glmfit, data; treatment = :A, reference = 0.0, comparison = 1.0,
    )
    @test glm_result.estimate ≈ PRE_REFACTOR_GCOMP.glm.estimate rtol = 2e-10
    @test glm_result.reference_mean ≈ PRE_REFACTOR_GCOMP.glm.reference_mean rtol = 2e-10
    @test glm_result.comparison_mean ≈ PRE_REFACTOR_GCOMP.glm.comparison_mean rtol = 2e-10
    @test glm_result.se ≈ PRE_REFACTOR_GCOMP.glm.se rtol = 2e-10

    lmm = MixedModels.fit(
        MixedModel, @formula(Y ~ 1 + A * time + W + (1 | subject)), data;
        progress = false,
    )
    nb_formula = @formula(Count ~ 1 + A * time + W + (1 | subject))
    fixed_fit = fit_parametric_gcomp(
        nb_formula, data; backend = :mixed, family = :negbin,
        id = :subject, theta = 2.0,
    )
    estimated_fit = fit_parametric_gcomp(
        nb_formula, data; backend = :mixed, family = :negbin,
        id = :subject, treatment = :A,
        quadrature_points = 9, multiple_starts = 1, profile = false,
    )
    fixed_nb = fixed_fit.model
    estimated_nb = estimated_fit.model
    @test fixed_nb isa GeneralizedLinearMixedModel
    @test estimated_nb isa NB2RandomInterceptModel
    @test estimated_nb.theta ≈ PRE_REFACTOR_GCOMP.nb_estimated.theta rtol = 1e-8

    # Evaluate the original fitted parameters to avoid optimiser variation
    # across platforms, retaining the frozen comparisons' tight tolerances.
    @test MixedModels.objective!(lmm, [2.3003030639897304]) ≈ -5.620997920736926 rtol = 1e-10
    @test MixedModels.objective!(
        fixed_nb, [
            0.2514691833442748, 0.675368934822376, 0.2164724382531897,
            -0.042904424228179965, -0.047412201460321254, -3.161297034850349e-6,
        ]; fast = false, nAGQ = 1,
    ) ≈ 133.61186696168468 rtol = 1e-10

    for (model, outcome, mode, frozen) in (
        (lmm, :Y, :zero, PRE_REFACTOR_GCOMP.lmm),
        (fixed_nb, :Count, :marginal, PRE_REFACTOR_GCOMP.nb_fixed),
        (estimated_nb, :Count, :marginal, PRE_REFACTOR_GCOMP.nb_estimated),
    )
        tolerance = model isa NB2RandomInterceptModel ? 1e-8 : 1e-10
        result = mixed_g_computation(
            model, data; treatment = :A, outcome, time = :time,
            id = :subject, random_effects = mode,
        )
        @test result.times == [0.0, 1.0, 2.0]
        @test result.values == (0, 1)
        @test result.random_effects == mode
        @test result.uncertainty == :delta_fixed
        @test result.adjustment == Symbol[]
        @test result.mean_reference ≈ frozen.reference rtol = tolerance
        @test result.mean_comparison ≈ frozen.comparison rtol = tolerance
        @test result.effect ≈ frozen.comparison .- frozen.reference rtol = tolerance
        @test result.vcov ≈ frozen.covariance rtol = tolerance atol = 1e-12
        @test result.se ≈ sqrt.(diag(frozen.covariance)) rtol = tolerance
        # At the final visit, one quarter of subjects lack a row. The fixture
        # tests observed-row weighting rather than a reconstructed balanced panel.
        @test count(==(2.0), data.time) == 36
    end

    # Independent joint covariance: treatment effects are β_A + t β_A:t.
    names = coefnames(lmm)
    g = zeros(3, length(names))
    g[:, only(findall(==("A"), names))] .= 1.0
    g[:, only(findall(==("A & time"), names))] .= [0.0, 1.0, 2.0]
    independent_vcov = g * vcov(lmm) * transpose(g)
    @test independent_vcov ≈ PRE_REFACTOR_GCOMP.lmm.covariance rtol = 1e-10

    @testset "NB2 marginal means and fixed-parameter gradients" begin
        target = select(data, Not(:Count))
        for (fit, model) in ((fixed_fit, fixed_nb), (estimated_fit, estimated_nb))
            no_treatment = copy(target)
            treatment = copy(target)
            no_treatment.A .= 0.0
            treatment.A .= 1.0
            x0 = _test_nb_design(model, no_treatment)
            x1 = _test_nb_design(model, treatment)
            beta = coef(model)
            variance = _test_nb_variance(model)
            correction = exp(variance / 2)
            expected_zero = mean(exp.(x1 * beta))
            expected_marginal = mean(exp.(x1 * beta .+ variance / 2))
            expected_reference = mean(exp.(x0 * beta .+ variance / 2))

            zero = gcomp_mean(fit, target; set = (; A = 1.0), random_effects = :zero)
            marginal = gcomp_mean(
                fit, target; set = (; A = 1.0), random_effects = :marginal,
            )
            adapter = gcomp_mean(
                model, data, target; id = :subject, set = (; A = 1.0),
                random_effects = :marginal,
            )
            contrast = gcomp_contrast(
                fit, target; treatment = :A, reference = 0.0,
                comparison = 1.0, random_effects = :marginal,
            )
            @test zero.estimate ≈ expected_zero rtol = 2e-10
            @test marginal.estimate ≈ expected_marginal rtol = 2e-10
            @test marginal.estimate / zero.estimate ≈ correction rtol = 2e-10
            @test adapter.estimate ≈ expected_marginal rtol = 2e-10
            @test contrast.reference_mean ≈ expected_reference rtol = 2e-10
            @test contrast.comparison_mean ≈ expected_marginal rtol = 2e-10
            @test contrast.random_effects == :marginal
            @test contrast.uncertainty == :delta_fixed

            component = CausalTargeted._gcomp_mean_components(
                fit, target; set = (; A = 1.0), random_effects = :marginal,
            )
            numerical_gradient = similar(beta)
            for j in eachindex(beta)
                h = 1e-6 * (1 + abs(beta[j]))
                upper, lower = copy(beta), copy(beta)
                upper[j] += h
                lower[j] -= h
                numerical_gradient[j] = (
                    mean(exp.(x1 * upper .+ variance / 2)) -
                    mean(exp.(x1 * lower .+ variance / 2))
                ) / (2h)
            end
            @test component.gradient ≈ numerical_gradient rtol = 3e-7 atol = 2e-8
            @test marginal.se^2 ≈ dot(
                numerical_gradient, vcov(model) * numerical_gradient,
            ) rtol = 3e-7
        end

        interaction = gcomp_interaction(
            fixed_fit, target; treatment = :A, reference = 0.0,
            comparison = 1.0, modifier = :time,
            modifier_reference = 0.0, modifier_comparison = 1.0,
            scale = :ratio, random_effects = :marginal,
        )
        manual_ratio = map((0.0, 1.0)) do time
            observed = target[target.time .== time, :]
            reference = copy(observed)
            comparison = copy(observed)
            reference.A .= 0.0
            comparison.A .= 1.0
            mean(exp.(_test_nb_design(fixed_nb, comparison) * coef(fixed_nb))) /
                mean(exp.(_test_nb_design(fixed_nb, reference) * coef(fixed_nb)))
        end
        @test interaction.estimate ≈ manual_ratio[2] / manual_ratio[1] rtol = 2e-10
        @test interaction.component_means.modifier_reference.n ==
            count(==(0.0), data.time)
        @test interaction.component_means.modifier_comparison.n ==
            count(==(1.0), data.time)
        @test_throws ArgumentError gcomp_mean(fixed_fit, target)
        @test_throws ArgumentError gcomp_interaction(
            fixed_fit, data; treatment = :A, reference = 0.0,
            comparison = 1.0, modifier = :time,
            modifier_reference = 0.0, modifier_comparison = 1.0,
            random_effects = :marginal, n_boot = 2,
        )
    end
end
