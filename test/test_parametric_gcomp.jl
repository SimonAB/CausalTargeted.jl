using Distributions
using GLM

@testset "Formula-based parametric g-computation" begin
    function factorial_data()
        rows = NamedTuple[]
        infection_levels = ("0_0", "0_1", "1_0", "1_1")
        for replicate in 1:8,
            line in ("ROL", "ROH"),
            protein in ("LP", "HP"),
            infection in infection_levels
            line_high = line == "ROH"
            protein_high = protein == "HP"
            infection_11 = infection == "1_1"
            outcome = 2.0 + 0.6line_high + 0.3protein_high +
                0.4line_high * protein_high + 0.10infection_11 +
                0.05line_high * infection_11 +
                0.08protein_high * infection_11 + 0.01replicate
            push!(rows, (;
                Y = outcome,
                Line = line,
                Protein = protein,
                InfectionHistory = infection,
                replicate,
            ))
        end
        return DataFrame(rows)
    end

    data = factorial_data()
    formula_term = @formula(
        Y ~ Line + Protein + InfectionHistory +
            Line & Protein + Line & InfectionHistory + Protein & InfectionHistory
    )
    fit = fit_parametric_gcomp(formula_term, data; family = :gaussian)

    @testset "factorial marginal, subgroup, and formal interaction" begin
        marginal = gcomp_contrast(
            fit,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        subgroup = gcomp_contrast(
            fit,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            by = (; Line = "ROH"),
        )
        interaction = gcomp_interaction(
            fit,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            modifier = :Line,
            modifier_reference = "ROL",
            modifier_comparison = "ROH",
        )

        @test marginal.estimate ≈ 0.52 atol = 1.0e-10
        @test subgroup.estimate ≈ 0.72 atol = 1.0e-10
        @test interaction.estimate ≈ 0.40 atol = 1.0e-10
        @test interaction.estimate ≈
            interaction.modifier_comparison_effect - interaction.modifier_reference_effect
        @test marginal.n == nrow(data)
        @test subgroup.n == count(==("ROH"), data.Line)
        @test fit.covariance_type == :hc3

        manual_target = data[data.Line .== "ROH", :]
        manual_hp = copy(manual_target)
        manual_hp.Protein .= "HP"
        manual_lp = copy(manual_target)
        manual_lp.Protein .= "LP"
        @test subgroup.comparison_mean ≈ mean(GLM.predict(fit.model, manual_hp))
        @test subgroup.reference_mean ≈ mean(GLM.predict(fit.model, manual_lp))
    end

    @testset "three-level categorical intervention and fitted levels" begin
        three_level = DataFrame(
            Y = [1.0, 1.1, 2.0, 2.1, 4.0, 4.1, 1.2, 2.2, 4.2],
            A = repeat(["low", "middle", "high"], 3),
            W = repeat([0.0, 1.0, 2.0], inner = 3),
        )
        three_fit = fit_parametric_gcomp(
            @formula(Y ~ A + W), three_level; family = :gaussian, covariance = :model,
        )
        result = gcomp_contrast(
            three_fit,
            three_level;
            treatment = :A,
            reference = "middle",
            comparison = "high",
        )
        high = copy(three_level)
        high.A .= "high"
        middle = copy(three_level)
        middle.A .= "middle"
        @test result.estimate ≈
            mean(GLM.predict(three_fit.model, high)) - mean(GLM.predict(three_fit.model, middle))
        error = try
            gcomp_mean(three_fit, three_level; set = (; A = "unseen"))
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("unseen level", sprint(showerror, error))
    end

    @testset "response-scale logistic risks and gradient" begin
        logistic_data = factorial_data()
        eta = -1.2 .+ 0.9 .* (logistic_data.Protein .== "HP") .+
            0.7 .* (logistic_data.Line .== "ROH") .+
            0.35 .* (logistic_data.InfectionHistory .== "1_1")
        probability = 1.0 ./ (1.0 .+ exp.(-eta))
        rng = StableRNG(991)
        logistic_data.Y = Float64.(rand.(Ref(rng), Bernoulli.(probability)))
        logistic_fit = fit_parametric_gcomp(
            formula_term, logistic_data; family = :binomial,
        )
        risk_difference = gcomp_contrast(
            logistic_fit,
            logistic_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        risk_ratio = gcomp_contrast(
            logistic_fit,
            logistic_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            scale = :ratio,
        )
        lp = copy(logistic_data)
        hp = copy(logistic_data)
        lp.Protein .= "LP"
        hp.Protein .= "HP"
        manual_lp = mean(GLM.predict(logistic_fit.model, lp))
        manual_hp = mean(GLM.predict(logistic_fit.model, hp))
        @test risk_difference.estimate ≈ manual_hp - manual_lp atol = 1.0e-12
        @test risk_ratio.estimate ≈ manual_hp / manual_lp atol = 1.0e-12
        @test risk_ratio.ci_lower > 0

        component = CausalTargeted._gcomp_mean_components(
            logistic_fit, logistic_data; set = (; Protein = "HP"),
        )
        counterfactual = copy(logistic_data)
        counterfactual.Protein .= "HP"
        design = CausalTargeted._gcomp_design(logistic_fit, counterfactual)
        beta = coef(logistic_fit.model)
        finite_difference = similar(beta)
        for index in eachindex(beta)
            step = 1.0e-6 * (1 + abs(beta[index]))
            upper, lower = copy(beta), copy(beta)
            upper[index] += step
            lower[index] -= step
            upper_mean = mean(1.0 ./ (1.0 .+ exp.(-(design * upper))))
            lower_mean = mean(1.0 ./ (1.0 .+ exp.(-(design * lower))))
            finite_difference[index] = (upper_mean - lower_mean) / (2step)
        end
        @test component.gradient ≈ finite_difference rtol = 2.0e-6 atol = 2.0e-8
    end

    @testset "Gamma and NB response-scale mean ratios" begin
        log_data = factorial_data()
        eta = 0.3 .+ 0.55 .* (log_data.Protein .== "HP") .+
            0.2 .* (log_data.Line .== "ROH") .+
            0.1 .* log_data.replicate
        rng = StableRNG(883)
        means = exp.(eta)
        log_data.Y = [rand(rng, Gamma(4.0, mu / 4.0)) for mu in means]
        gamma_fit = fit_parametric_gcomp(formula_term, log_data; family = :gamma)
        gamma_ratio = gcomp_contrast(
            gamma_fit,
            log_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            scale = :ratio,
        )
        lp = copy(log_data)
        hp = copy(log_data)
        lp.Protein .= "LP"
        hp.Protein .= "HP"
        @test gamma_ratio.estimate ≈
            mean(GLM.predict(gamma_fit.model, hp)) / mean(GLM.predict(gamma_fit.model, lp))

        mean_eta = mean(CausalTargeted._gcomp_design(gamma_fit, hp) * coef(gamma_fit.model))
        @test gamma_ratio.comparison_mean ≈ mean(GLM.predict(gamma_fit.model, hp))
        @test abs(gamma_ratio.comparison_mean - exp(mean_eta)) > 1.0e-4

        count_data = factorial_data()
        count_means = exp.(0.7 .+ 0.4 .* (count_data.Protein .== "HP") .+
            0.15 .* (count_data.Line .== "ROH"))
        rng_count = StableRNG(884)
        count_data.Y = Float64[
            rand(rng_count, NegativeBinomial(2.3, 2.3 / (2.3 + mu))) for mu in count_means
        ]
        fixed_nb = fit_parametric_gcomp(
            formula_term, count_data; family = :negbin, theta = 2.3,
        )
        fixed_ratio = gcomp_contrast(
            fixed_nb,
            count_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            scale = :ratio,
        )
        lp_count, hp_count = copy(count_data), copy(count_data)
        lp_count.Protein .= "LP"
        hp_count.Protein .= "HP"
        @test fixed_ratio.estimate ≈
            mean(GLM.predict(fixed_nb.model, hp_count)) / mean(GLM.predict(fixed_nb.model, lp_count))
        @test fixed_nb.theta ≈ 2.3
        @test fixed_nb.family == :negbin
        @test !fixed_nb.estimated_theta

        for alias in (:negativebinomial, :negative_binomial, :nb, NegativeBinomial(2.3))
            alias_fit = fit_parametric_gcomp(
                formula_term, count_data; family = alias, theta = 2.3,
            )
            @test alias_fit.family == :negbin
            @test coef(alias_fit.model) ≈ coef(fixed_nb.model)
        end

        estimated_nb = fit_parametric_gcomp(formula_term, count_data; family = :negbin)
        direct_nb = GLM.negbin(formula_term, count_data, LogLink())
        @test estimated_nb.family == :negbin
        @test estimated_nb.estimated_theta
        @test isfinite(estimated_nb.theta) && estimated_nb.theta > 0
        @test estimated_nb.theta ≈ direct_nb.model.rr.d.r
        @test coef(estimated_nb.model) ≈ coef(direct_nb)
        estimated_mean = gcomp_mean(estimated_nb, count_data; set = (; Protein = "HP"))
        @test estimated_mean.estimate ≈ mean(GLM.predict(direct_nb, hp_count))
        @test isfinite(estimated_mean.se) && estimated_mean.se > 0

        # HC3 conditions on fitted theta: fixing theta at the estimated value
        # gives the same coefficient covariance and delta-method uncertainty.
        conditioned_nb = fit_parametric_gcomp(
            formula_term, count_data; family = :negbin, theta = estimated_nb.theta,
        )
        @test !conditioned_nb.estimated_theta
        @test estimated_nb.covariance ≈ conditioned_nb.covariance rtol = 1.0e-5
        @test estimated_mean.se ≈
            gcomp_mean(conditioned_nb, count_data; set = (; Protein = "HP")).se rtol = 1.0e-5

        # The refit path, unlike HC3, re-estimates theta when originally estimated.
        indices = CausalTargeted._gcomp_bootstrap_indices(
            StableRNG(885), count_data, [:Line, :Protein, :InfectionHistory],
        )
        resample = count_data[indices, :]
        estimated_refit = CausalTargeted._gcomp_refit(estimated_nb, resample)
        fixed_refit = CausalTargeted._gcomp_refit(fixed_nb, resample)
        direct_refit = GLM.negbin(formula_term, resample, LogLink())
        @test estimated_refit.estimated_theta
        @test estimated_refit.theta ≈ direct_refit.model.rr.d.r
        # Do not assert that the resampled θ differs from the original: a stratified
        # redraw can land on the same shape by chance. Fixed-θ refits must keep θ.
        @test !fixed_refit.estimated_theta
        @test fixed_refit.theta == fixed_nb.theta
    end

    @testset "ratio interaction reconciliation and bootstrap strata" begin
        positive_data = factorial_data()
        positive_data.Y = exp.(positive_data.Y)
        log_fit = fit_parametric_gcomp(formula_term, positive_data; family = :gamma)
        interaction = gcomp_interaction(
            log_fit,
            positive_data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
            modifier = :Line,
            modifier_reference = "ROL",
            modifier_comparison = "ROH",
            scale = :ratio,
        )
        @test interaction.estimate ≈
            interaction.modifier_comparison_effect / interaction.modifier_reference_effect

        rng = StableRNG(123)
        indices = CausalTargeted._gcomp_bootstrap_indices(
            rng, positive_data, [:Line, :Protein, :InfectionHistory],
        )
        resample = positive_data[indices, :]
        original_counts = combine(
            groupby(positive_data, [:Line, :Protein, :InfectionHistory]), nrow => :n,
        )
        resample_counts = combine(
            groupby(resample, [:Line, :Protein, :InfectionHistory]), nrow => :n,
        )
        sort!(original_counts, [:Line, :Protein, :InfectionHistory])
        sort!(resample_counts, [:Line, :Protein, :InfectionHistory])
        @test original_counts.n == resample_counts.n

        @testset "refit bootstrap intervals and replicate counts" begin
            bootstrap = bootstrap_gcomp_interaction(
                log_fit,
                positive_data;
                treatment = :Protein,
                reference = "LP",
                comparison = "HP",
                modifier = :Line,
                modifier_reference = "ROL",
                modifier_comparison = "ROH",
                scale = :ratio,
                n_boot = 20,
                strata = [:Line, :Protein, :InfectionHistory],
                rng = StableRNG(124),
            )
            @test bootstrap.requested_replicates == 20
            @test bootstrap.successful_replicates == 20
            @test length(bootstrap.estimates) == bootstrap.successful_replicates
            @test bootstrap.success_fraction == 1.0
            @test isempty(bootstrap.failure_reasons)
            @test all(isfinite, bootstrap.estimates)
            @test isfinite(bootstrap.se) && bootstrap.se > 0
            @test 0 < bootstrap.ci_lower < bootstrap.ci_upper < Inf
            @test bootstrap.ci_lower ≈ quantile(bootstrap.estimates, 0.025)
            @test bootstrap.ci_upper ≈ quantile(bootstrap.estimates, 0.975)
        end
    end

    @testset "direct formula GLM and convenience runner" begin
        direct = glm(formula_term, data, Normal(), IdentityLink())
        direct_result = gcomp_contrast(
            direct,
            data,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        run_result = run_parametric_gcomp(
            formula_term,
            data;
            treatment = :Protein,
            reference = "LP",
            comparison = "HP",
        )
        @test direct_result.estimate ≈ 0.52 atol = 1.0e-10
        @test run_result.estimate ≈ direct_result.estimate

        # A smaller target with a different covariate distribution and no Y.
        target = select(data[data.Line .== "ROH", :], Not(:Y))
        target_mean = gcomp_mean(direct, data, target; set = (; Protein = "HP"))
        hp_target = copy(target)
        hp_target.Protein .= "HP"
        @test target_mean.n == nrow(target) < nrow(data)
        @test target_mean.estimate ≈ mean(GLM.predict(direct, hp_target))
        @test target_mean.se ≈ gcomp_mean(fit, target; set = (; Protein = "HP")).se
        @test !isapprox(target_mean.estimate, gcomp_mean(direct, data, data; set = (; Protein = "HP")).estimate)
        target_contrast = gcomp_contrast(
            direct, data, target; treatment = :Protein, reference = "LP", comparison = "HP",
        )
        @test target_contrast.estimate ≈ 0.72 atol = 1.0e-10
        @test target_contrast.n == nrow(target)

        target_both = select(data[data.replicate .<= 3, :], Not(:Y))
        target_interaction = gcomp_interaction(
            direct, data, target_both;
            treatment = :Protein, reference = "LP", comparison = "HP",
            modifier = :Line, modifier_reference = "ROL", modifier_comparison = "ROH",
        )
        @test target_interaction.estimate ≈ 0.40 atol = 1.0e-10
        @test_throws ArgumentError gcomp_mean(direct, data[1:10, :], target)
        @test_throws MethodError gcomp_mean(direct, target)
    end

    @testset "explicit complete-case fitting and target policy" begin
        for column in (:Y, :Protein), covariance in (:hc3, :model, :none)
            incomplete = allowmissing(copy(data), column)
            incomplete[1, column] = missing
            error = try
                fit_parametric_gcomp(formula_term, incomplete; covariance)
                nothing
            catch caught
                caught
            end
            @test error isa ArgumentError
            @test occursin("training data must be complete-case", sprint(showerror, error))
            @test occursin(":$column contains missing", sprint(showerror, error))
        end

        unrelated = copy(data)
        unrelated.unused = fill(missing, nrow(data))
        @test coef(fit_parametric_gcomp(formula_term, unrelated).model) ≈ coef(fit.model)
        nullable = allowmissing(copy(data))
        @test coef(fit_parametric_gcomp(formula_term, nullable).model) ≈ coef(fit.model)

        incomplete = allowmissing(copy(data), :Y)
        incomplete.Y[1] = missing
        dropped_model = glm(formula_term, incomplete, Normal(), IdentityLink())
        @test_throws ArgumentError gcomp_mean(dropped_model, incomplete, data)
        complete = dropmissing(incomplete, :Y)
        @test isfinite(gcomp_mean(dropped_model, complete, select(data, Not(:Y))).estimate)

        target = allowmissing(select(data, Not(:Y)), :Protein)
        target.Protein[1] = missing
        error = try
            gcomp_mean(fit, target; set = (; Protein = "HP"))
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("target data must be complete-case", sprint(showerror, error))
        # Subset selection precedes complete-case checks and intervention.
        @test isfinite(gcomp_mean(fit, target; by = (; Line = "ROH")).estimate)
        @test_throws ArgumentError gcomp_mean(fit, nullable; set = (; Protein = missing))
        @test_throws ArgumentError fit_parametric_gcomp(formula_term, select(data, Not(:Protein)))
        @test_throws ArgumentError gcomp_mean(fit, select(data, Not(:Protein)))
        @test_throws ArgumentError bootstrap_gcomp_interaction(
            fit, incomplete;
            treatment = :Protein, reference = "LP", comparison = "HP",
            modifier = :Line, modifier_reference = "ROL", modifier_comparison = "ROH",
            n_boot = 1,
        )
    end
end
