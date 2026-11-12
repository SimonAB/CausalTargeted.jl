using Test, DataFrames, LinearAlgebra, Random, Distributions, Statistics
using CausalTargeted
using CausalDynamics: identify, TotalEffectQuery, IdentificationResult
using Graphs

@testset "Exact Bayesian Gaussian g-computation" begin
    @testset "Hand-calculated normal–inverse-gamma posterior" begin
        data = DataFrame(A = [-1.0, 1.0], Y = [0.0, 2.0])
        prior = ConjugateGaussianPrior(mean = zeros(2), covariance = Matrix{Float64}(I, 2, 2), shape = 2, scale = 1)
        fit = fit_bayesian_gcomp(data, :Y, :A, Symbol[]; prior)
        @test fit.posterior_mean ≈ fill(2 / 3, 2)
        @test fit.Vn ≈ Matrix{Float64}(I, 2, 2) / 3
        @test fit.shape == 3
        @test fit.scale ≈ 5 / 3
        @test fit.posterior_df == 6
        @test fit.posterior_scale ≈ Matrix{Float64}(I, 2, 2) * (5 / 27)
        @test fit.posterior_covariance ≈ Matrix{Float64}(I, 2, 2) * (5 / 18)
        @test fit.metadata.identification_status == :user_assumed
        @test fit.metadata.graph_identification_status == :user_assumed
        @test fit.metadata.outcome_model_status == :user_assumed
        @test fit.variance_diagnostics.prior_variance ≈ ones(2)

        policy = SmoothBoundedLogitShiftPolicy(-2, 2)
        result = bayesian_policy_contrast(fit, select(data, :A), policy; deltas = [0.0, log(3), log(3)])
        @test result.mean ≈ [0, 8 / 15, 8 / 15]
        @test result.covariance[2, 2] ≈ 8 / 45
        @test result.covariance[2, 3] ≈ result.covariance[2, 2]
        @test iszero(result.covariance[1, 1])
        @test result.lower[1] == result.upper[1] == 0
        @test result.upper[2] ≈ 8 / 15 + quantile(TDist(6), 0.975) * sqrt(16 / 135)
        @test result.metadata.factual_reference == :observed
        @test result.metadata.policy_constants.lower == -2
        @test result.metadata.policy_constants.deltas == [0, log(3), log(3)]
        @test_throws ArgumentError bayesian_policy_contrast(fit, data, ShiftPolicy("raw", 0.01, 0.99); deltas = [1])
        @test_throws ArgumentError bayesian_policy_contrast(fit, data, policy; deltas = [Inf])
        @test_throws ArgumentError bayesian_policy_contrast(fit, data, policy; deltas = [1], level = 1)

        known = fit_bayesian_gcomp(data, :Y, :A, Symbol[]; prior, observation_variance = 2)
        @test known.posterior_mean ≈ fill(2 / 3, 2)
        @test known.posterior_covariance ≈ Matrix{Float64}(I, 2, 2) * (2 / 3)
        @test isinf(known.posterior_df)
        normal_result = bayesian_policy_contrast(known, data, policy; deltas = [log(3)])
        @test normal_result.upper[1] ≈ 8 / 15 + quantile(Normal(), 0.975) * sqrt(32 / 75)

        # Two future rows with identical predictors share coefficient uncertainty,
        # while their independent outcome noise adds only to diagonal variance.
        future = DataFrame(A = [0.0, 0], Y = [missing, missing])
        rng = MersenneTwister(93)
        predictions = hcat([predictive_missing_outcomes(fit, future; rng).values for _ in 1:5000]...)'
        predictive_covariance = cov(predictions)
        @test predictive_covariance[1, 1] ≈ 10 / 9 atol = 0.10
        @test predictive_covariance[1, 2] ≈ 5 / 18 atol = 0.08
    end

    @testset "Frozen interaction basis and proper prior under collinearity" begin
        data = DataFrame(A = [-0.5, 0.5, -0.5, 0.5], W = [0.0, 0, 1, 1], Y = [0.0, 1, 0, 2])
        basis = (df, a) -> hcat(ones(nrow(df)), a, df.W, a .* df.W)
        prior = ConjugateGaussianPrior(mean = zeros(4), covariance = Matrix{Float64}(I, 4, 4), shape = 2, scale = 1)
        fit = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, basis)
        @test fit.metadata.identification_status == :user_assumed
        result = bayesian_policy_contrast(fit, data, discrete_static_policy(0.5); deltas = [0])
        expected_C = vec(mean(basis(data, fill(0.5, 4)) - basis(data, data.A); dims = 1))
        @test vec(result.contrast_matrix) ≈ expected_C
        @test only(result.mean) ≈ dot(expected_C, fit.posterior_mean)
        @test only(result.covariance) ≈ dot(expected_C, fit.posterior_covariance * expected_C)
        rankdeficient = DataFrames.transform(data, :A => (_ -> ones(4)) => :A)
        @test all(isfinite, fit_bayesian_gcomp(rankdeficient, :Y, :A, [:W]; prior, basis).posterior_covariance)
        @test_throws DimensionMismatch fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, basis = (df, a) -> ones(nrow(df), 3))
    end

    @testset "Observed likelihood under explicitly assumed MAR" begin
        data = DataFrame(A = [-1.0, 1, 0], W = [0.0, 0, 0], Y = Union{Missing, Float64}[0, 2, missing])
        prior = ConjugateGaussianPrior(mean = zeros(3), covariance = Matrix{Float64}(I, 3, 3), shape = 2, scale = 1)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
        fit = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, handle_missing = :mar)
        complete = fit_bayesian_gcomp(data[1:2, :], :Y, :A, [:W]; prior)
        @test fit.posterior_mean == complete.posterior_mean
        @test fit.posterior_covariance == complete.posterior_covariance
        absent = copy(data); absent[!, :Y] = fill(missing, 3)
        unupdated = fit_bayesian_gcomp(absent, :Y, :A, [:W]; prior, handle_missing = :mar)
        @test unupdated.posterior_mean ≈ prior.mean
        @test unupdated.posterior_covariance ≈ prior.scale / (prior.shape - 1) .* prior.covariance
        @test unupdated.metadata.n_observed == 0
        @test fit.shape == complete.shape
        @test fit.metadata.missingness_status == :user_assumed_mar
        @test fit.metadata.n_observed == 2
        generated = predictive_missing_outcomes(fit, data; rng = MersenneTwister(81))
        repeated = predictive_missing_outcomes(fit, data; rng = MersenneTwister(81))
        @test generated.rows == [3]
        @test generated.values == repeated.values
        @test length(generated.beta) == 3
        @test generated.sigma2 > 0
        @test fit.posterior_covariance == complete.posterior_covariance
        bad = copy(data); bad.W[3] = NaN
        @test_throws ArgumentError fit_bayesian_gcomp(bad, :Y, :A, [:W]; prior, handle_missing = :mar)
        bad = copy(data); bad[!, :W] = [0.0, missing, 0]
        @test_throws ArgumentError fit_bayesian_gcomp(bad, :Y, :A, [:W]; prior, handle_missing = :mar)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:absent]; prior, handle_missing = :mar)
    end

    @testset "Categorical schema and invalid priors" begin
        data = DataFrame(A = ["low", "high", "low", "high"], W = ["a", "a", "b", "b"], Y = [0.0, 1, 1, 2])
        prior = ConjugateGaussianPrior(mean = zeros(3), covariance = Matrix{Float64}(I, 3, 3), shape = 2, scale = 1)
        fit = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
        target = data[[3, 1], [:A, :W]]
        result = bayesian_policy_contrast(fit, target, discrete_static_policy("high"); deltas = [0])
        @test length(result.mean) == 1
        @test size(result.contrast_matrix) == (1, 3)
        @test_throws ArgumentError bayesian_policy_contrast(fit, target, discrete_static_policy("unseen"); deltas = [0])
        @test_throws ArgumentError ConjugateGaussianPrior(mean = [0.0], covariance = ones(1, 1), shape = 1, scale = 1)
        @test_throws ArgumentError ConjugateGaussianPrior(mean = [NaN], covariance = ones(1, 1), shape = 2, scale = 1)
        @test_throws ArgumentError ConjugateGaussianPrior(mean = zeros(2), covariance = [1.0 2; 0 1], shape = 2, scale = 1)
        @test_throws ArgumentError ConjugateGaussianPrior(mean = zeros(2), covariance = ones(2, 2), shape = 2, scale = 1)
        @test_throws DimensionMismatch ConjugateGaussianPrior(mean = zeros(2), covariance = ones(1, 1), shape = 2, scale = 1)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, observation_variance = 0)
        constant = DataFrame(A = ["low", "low"], Y = [0.0, 1])
        constant_prior = ConjugateGaussianPrior(mean = [0.0], covariance = ones(1, 1), shape = 2, scale = 1)
        constant_fit = fit_bayesian_gcomp(constant, :Y, :A, Symbol[]; prior = constant_prior)
        @test_throws ArgumentError bayesian_policy_contrast(constant_fit, constant, discrete_static_policy("high"); deltas = [0])

        # Existing discrete policies also match integer and string level labels.
        integer_data = DataFrame(A = [0, 1], Y = [0.0, 1])
        integer_prior = ConjugateGaussianPrior(mean = zeros(2), covariance = Matrix{Float64}(I, 2, 2), shape = 2, scale = 1)
        integer_fit = fit_bayesian_gcomp(integer_data, :Y, :A, Symbol[]; prior = integer_prior)
        integer_result = bayesian_policy_contrast(integer_fit, integer_data, discrete_static_policy("1"); deltas = [0])
        @test vec(integer_result.contrast_matrix) == [0, 0.5]
        subset_result = bayesian_policy_contrast(integer_fit, integer_data[1:1, :], discrete_static_policy("1"); deltas = [0])
        @test vec(subset_result.contrast_matrix) == [0, 1]
        string_data = DataFrame(A = ["0", "1"], Y = [0.0, 1])
        string_fit = fit_bayesian_gcomp(string_data, :Y, :A, Symbol[]; prior = integer_prior)
        string_subset = bayesian_policy_contrast(string_fit, string_data[1:1, :], discrete_static_policy(1); deltas = [0])
        @test vec(string_subset.contrast_matrix) == [0, 1]
        @test_throws ArgumentError bayesian_policy_contrast(string_fit, string_data[1:1, :], discrete_static_policy("2"); deltas = [0])
    end

    @testset "Certificate treatment, outcome and actual adjustment" begin
        graph = SimpleDiGraph(3)
        add_edge!(graph, 1, 2); add_edge!(graph, 1, 3); add_edge!(graph, 2, 3)
        result = identify(graph, TotalEffectQuery(:A, :Y); node_names = [:W, :A, :Y])
        @test result.adjustment == [:W]
        cert = CausalTargeted.identification_certificate(result, :A, :Y)
        data = DataFrame(A = [-1.0, 1, -1, 1], W = [0.0, 0, 1, 1], Y = [0.0, 2, 1, 3], Z = ones(4))
        prior = ConjugateGaussianPrior(mean = zeros(3), covariance = Matrix{Float64}(I, 3, 3), shape = 2, scale = 1)
        fit = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, identification = cert)
        @test fit.metadata.identification_status == :identified
        @test fit.metadata.graph_identification_status == :identified
        @test fit.metadata.outcome_model_status == :user_assumed
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, identification = result)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior,
            identification = CausalTargeted.identification_certificate(result, :Z, :Y))
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior,
            identification = CausalTargeted.identification_certificate(result, :A, :Z))
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W, :Z]; prior, identification = cert)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior,
            identification = CausalTargeted.identification_certificate(result, :A, :Y; adjustment = [:Z]))
        mismatch = IdentificationResult(query = TotalEffectQuery(:Z, :Y), graph_hash = result.graph_hash,
            adjustment = [:W], strategy = :backdoor, identifiable = true)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior,
            identification = CausalTargeted.identification_certificate(mismatch, :A, :Y))
        false_set = IdentificationResult(query = result.query, graph_hash = result.graph_hash,
            adjustment = [:Z], strategy = :backdoor, identifiable = true)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior,
            identification = CausalTargeted.identification_certificate(false_set, :A, :Y; adjustment = [:W]))
        unsupported = IdentificationResult(query = result.query, graph_hash = result.graph_hash,
            adjustment = [:W], strategy = :backdoor, identifiable = false)
        @test_throws ArgumentError fit_bayesian_gcomp(data, :Y, :A, [:W]; prior,
            identification = CausalTargeted.identification_certificate(unsupported, :A, :Y))
        # Numeric node labels use the explicit certificate's graph-to-column map.
        numeric_result = identify(graph, TotalEffectQuery(2, 3))
        numeric_cert = CausalTargeted.identification_certificate(numeric_result, :A, :Y; adjustment = [:W])
        @test fit_bayesian_gcomp(data, :Y, :A, [:W]; prior, identification = numeric_cert).metadata.identification_status == :identified
        interaction = (df, a) -> hcat(ones(nrow(df)), a, df.W, a .* df.W)
        interaction_prior = ConjugateGaussianPrior(mean = zeros(4), covariance = Matrix{Float64}(I, 4, 4), shape = 2, scale = 1)
        certified_basis = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior = interaction_prior, basis = interaction, identification = cert)
        @test certified_basis.metadata.identification_status == :user_assumed
        @test certified_basis.metadata.graph_identification_status == :identified
        @test certified_basis.metadata.outcome_model_status == :user_assumed
        @test certified_basis.design_metadata.basis_source == :user_function
        propagated = bayesian_policy_contrast(certified_basis, data,
            discrete_static_policy(1.0); deltas = [0.0])
        @test propagated.metadata.graph_identification_status == :identified
        @test propagated.metadata.outcome_model_status == :user_assumed
        @test certified_basis.metadata.identification === cert
    end
end
