# Independent scalar tail checks, including an exactly collinear multivariate
# model whose likelihood depends on one Gaussian-prior projection. Run with the
# package's test extras (QuadGK is needed), for example:
# julia --project=test/stress --threads=auto test/stress/logistic_tail_validation.jl
using CausalTargeted, GraphPPL, RxInfer, Distributions, StableRNGs
using LinearAlgebra, Statistics, Random, DataFrames

include(joinpath(@__DIR__, "..", "logistic_tail_reference.jl"))
include(joinpath(@__DIR__, "..", "logistic_reference.jl"))
using .LogisticTailReference

function tail_engine(X, y, μ, V)
    extension = Base.get_extension(CausalTargeted, :CausalTargetedRxInferExt)
    return extension._fit_logistic_engine(Matrix{Float64}(X), Float64.(y), Float64.(μ),
        Matrix{Float64}(V); maxiter = 1000, tol = 1e-10, covariance = :linear_response)
end

function scalar_fixtures()
    x = repeat([-0.9, -0.5, 0.1, 0.5, 0.9], inner = 4)
    y = Float64.([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    cases = [
        (; name = :moderate_scalar, x, y, μ = 0.2, variance = 2.25),
        (; name = :all_one, x = ones(20), y = ones(20), μ = 0.0, variance = 2.25),
        (; name = :all_zero, x = ones(20), y = zeros(20), μ = 0.0, variance = 2.25),
        (; name = :separated_weak_prior, x, y = Float64.(x .> 0), μ = 0.0, variance = 9.0),
        (; name = :zero_mean_symmetry, x = ones(20), y = Float64.(repeat([0, 1], 10)), μ = 0.0, variance = 9.0),
        (; name = :missing_outcomes, x = x[1:2:end], y = y[1:2:end], μ = 0.2, variance = 2.25),
        (; name = :no_observed_outcomes, x = Float64[], y = Float64[], μ = 0.2, variance = 2.25),
    ]
    for sd in (0.5, 1.0, 1.5, 3.0, 6.0)
        push!(cases, (; name = Symbol("separation_prior_sd_", sd), x,
            y = Float64.(x .> 0), μ = 0.0, variance = sd^2))
    end
    return cases
end

function check_tail_fixtures()
    reference_gate = true
    for case in scalar_fixtures()
        truth = adaptive_reference(case.x, case.y, case.μ, case.variance)
        tighter = adaptive_reference(case.x, case.y, case.μ, case.variance; rtol = 1e-12, atol = 1e-14)
        tails, refined = posterior_tails(truth), posterior_tails(tighter)
        quantile_change = max(abs(tails.lower - refined.lower), abs(tails.upper - refined.upper))
        validated = quantile_change <= 1e-7 && tails.maximum_tail_residual <= 1e-9 &&
                    tails.maximum_integration_error <= 1e-9
        reference_gate &= validated
        fit = tail_engine(reshape(case.x, :, 1), case.y, [case.μ], reshape([case.variance], 1, 1))
        println((; fixture = case.name, target = :scalar_coefficient,
            n_observed = length(case.y), prior_sd = sqrt(case.variance),
            reference_mean = truth.mean, reference_variance = truth.variance,
            reference_interval = (tails.lower, tails.upper), quantile_change,
            maximum_tail_residual = tails.maximum_tail_residual,
            maximum_integration_error = tails.maximum_integration_error,
            reference_tail_gate_passed = validated, status = fit.status))
        for (mode, V) in ((:variational, fit.variational_covariance), (:linear_response, fit.corrected_covariance))
            available = fit.converged && V !== nothing && all(isfinite, V) && isposdef(Symmetric(V))
            if !available
                println((; fixture = case.name, mode, available = false))
                continue
            end
            gaussian = Normal(fit.mean[1], sqrt(V[1, 1]))
            lower, upper = quantile(gaussian, 0.025), quantile(gaussian, 0.975)
            content = interval_content(truth, lower, upper)
            println((; fixture = case.name, mode, available,
                mean_error_sd = abs(fit.mean[1] - truth.mean) / sqrt(truth.variance),
                relative_variance_error = abs(V[1, 1] / truth.variance - 1),
                lower_endpoint_error_sd = (lower - tails.lower) / sqrt(truth.variance),
                upper_endpoint_error_sd = (upper - tails.upper) / sqrt(truth.variance),
                posterior_content = content.content, lower_tail = content.lower_tail,
                upper_tail = content.upper_tail))
        end
        # Evaluate discrete Hermite limits with the *adaptive* CDF. Moment
        # agreement is recorded separately and is never a quantile gate.
        for points in (80, 160, 320, 640)
            grid = LogisticPosteriorReference.logistic_reference(reshape(case.x, :, 1),
                case.y, [case.μ], reshape([case.variance], 1, 1); points)
            lower = LogisticPosteriorReference.weighted_quantile(vec(grid.coefficients), grid.weights, 0.025)
            upper = LogisticPosteriorReference.weighted_quantile(vec(grid.coefficients), grid.weights, 0.975)
            content = interval_content(truth, lower, upper)
            println((; fixture = case.name, method = :prior_hermite_grid, points,
                mean_error_sd = abs(grid.mean[1] - truth.mean) / sqrt(truth.variance),
                relative_variance_error = abs(grid.covariance[1, 1] / truth.variance - 1),
                maximum_tail_probability_error = max(abs(content.lower_tail - 0.025), abs(content.upper_tail - 0.025)),
                lower_endpoint_error_sd = (lower - tails.lower) / sqrt(truth.variance),
                upper_endpoint_error_sd = (upper - tails.upper) / sqrt(truth.variance)))
        end
        flush(stdout)
    end
    return reference_gate
end

function check_collinearity()
    case = first(scalar_fixtures())
    c = [1.0, 1.0]
    μ, V = [0.3, -0.1], [2.25 0.4; 0.4 1.0]
    projection_variance = dot(c, V * c)
    truth = adaptive_reference(case.x, case.y, dot(c, μ), projection_variance)
    gain = V * c / projection_variance
    exact_mean = μ + gain * (truth.mean - dot(c, μ))
    exact_covariance = V - (V * c) * (V * c)' / projection_variance + gain * gain' * truth.variance
    # This direction is independent of c'β under the Gaussian prior, hence
    # retains its prior marginal exactly even after observing the outcomes.
    untouched = [-(V * c)[2], (V * c)[1]]
    for perturbation in (0.0, 1e-8, 1e-3)
        X = hcat(case.x, case.x + perturbation * repeat([-1.0, 1.0], 10))
        fit = tail_engine(X, case.y, μ, V)
        for (mode, covariance) in ((:variational, fit.variational_covariance), (:linear_response, fit.corrected_covariance))
            available = fit.converged && covariance !== nothing && isposdef(Symmetric(covariance))
            if !available
                println((; fixture = :collinearity, perturbation, mode, available = false, status = fit.status))
                continue
            end
            gaussian = Normal(dot(c, fit.mean), sqrt(dot(c, covariance * c)))
            content = interval_content(truth, quantile(gaussian, 0.025), quantile(gaussian, 0.975))
            println((; fixture = :collinearity, perturbation, mode, available, status = fit.status,
                reference = perturbation == 0 ? :exact_scalar_reduction : :collinear_limit_only,
                mean_error_sd = maximum(abs.(fit.mean - exact_mean) ./ sqrt.(diag(exact_covariance))),
                covariance_relative_error = norm(covariance - exact_covariance) / norm(exact_covariance),
                projected_interval_content = content.content,
                untouched_mean_change = dot(untouched, fit.mean - μ),
                untouched_variance_ratio = dot(untouched, covariance * untouched) / dot(untouched, V * untouched)))
        end
    end
end

"""Separate scalar SBC and fixed-parameter coverage against adaptive quantiles."""
function calibrate_tails(; replications = 500, seed = 1607)
    rng = StableRNG(seed)
    x = collect(range(-0.9, 0.9; length = 20))
    μ, variance = 0.2, 2.25
    for experiment in (:prior_predictive, :fixed_parameter)
        modes = (:adaptive_reference, :variational, :linear_response)
        covered = Dict(mode => 0 for mode in modes)
        failures = Dict(mode => 0 for mode in modes)
        widths = Dict(mode => Float64[] for mode in modes)
        ranks = Dict(mode => Float64[] for mode in modes)
        reference_failures = 0
        seconds = @elapsed for replication in 1:replications
            β = experiment == :prior_predictive ? rand(rng, Normal(μ, sqrt(variance))) : 0.9
            y = Float64.(rand(rng, length(x)) .< logistic_probability.(x .* β))
            truth = adaptive_reference(x, y, μ, variance)
            tails = posterior_tails(truth)
            reference_valid = tails.maximum_tail_residual <= 1e-9 && tails.maximum_integration_error <= 1e-9
            reference_failures += !reference_valid
            if reference_valid
                covered[:adaptive_reference] += tails.lower <= β <= tails.upper
                push!(widths[:adaptive_reference], tails.upper - tails.lower)
                push!(ranks[:adaptive_reference], posterior_cdf(truth, β))
            else
                failures[:adaptive_reference] += 1
            end
            fit = try
                tail_engine(reshape(x, :, 1), y, [μ], reshape([variance], 1, 1))
            catch exception
                println((; experiment, replication, fit_exception = sprint(showerror, exception)))
                nothing
            end
            for mode in (:variational, :linear_response)
                V = fit === nothing ? nothing : mode == :variational ? fit.variational_covariance : fit.corrected_covariance
                available = fit !== nothing && fit.converged && V !== nothing && isposdef(Symmetric(V))
                if !available
                    failures[mode] += 1
                    continue
                end
                gaussian = Normal(fit.mean[1], sqrt(V[1, 1]))
                lower, upper = quantile(gaussian, 0.025), quantile(gaussian, 0.975)
                covered[mode] += lower <= β <= upper
                push!(widths[mode], upper - lower)
                push!(ranks[mode], cdf(gaussian, β))
            end
        end
        for mode in modes
            available = replications - failures[mode]
            coverage = available == 0 ? NaN : covered[mode] / available
            println((; experiment, mode, attempted = replications, available,
                failures = failures[mode], coverage_among_available = coverage,
                coverage_all_attempts = covered[mode] / replications,
                coverage_mcse = available == 0 ? NaN : sqrt(coverage * (1 - coverage) / available),
                mean_width = isempty(widths[mode]) ? NaN : mean(widths[mode]),
                posterior_cdf_mean = isempty(ranks[mode]) ? NaN : mean(ranks[mode]),
                posterior_cdf_deciles = [count(r -> (j - 1) / 10 <= r < j / 10, ranks[mode]) for j in 1:10],
                cdf_interpretation = experiment == :prior_predictive ? :sbc : :fixed_parameter_not_sbc))
        end
        println((; experiment, replications, seconds, reference_failures,
            target = :scalar_coefficient, policy_scope = "a monotone logistic(β)-1/2 contrast has the same coverage"))
        flush(stdout)
    end
end


"""Continuous bivariate policy quantiles and the public Gaussian sampling layer."""
function check_bivariate_policy_tails()
    a = repeat([-0.9, -0.5, 0.1, 0.5, 0.9], inner = 4)
    X = hcat(ones(20), a)
    y = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    μ, V = zeros(2), 2.25I(2)
    prior = GaussianCoefficientPrior(mean = μ, covariance = Matrix(V))
    gate = true
    for fixture in (:prior_only, :moderate_posterior)
        design = fixture == :prior_only ? zeros(0, 2) : X
        outcomes = fixture == :prior_only ? Float64[] : Float64.(y)
        coarse = bivariate_policy_reference(design, outcomes, μ, V; rtol = 1e-7, atol = 1e-10)
        tight = bivariate_policy_reference(design, outcomes, μ, V; rtol = 1e-9, atol = 1e-12)
        coarse_tails, tails = policy_tails(coarse), policy_tails(tight)
        quantile_change = max(abs(tails.lower - coarse_tails.lower), abs(tails.upper - coarse_tails.upper))
        validated = quantile_change <= 2e-6 && tails.maximum_tail_residual <= 2e-7 && tails.maximum_integration_error <= 1e-7
        gate &= validated
        println((; fixture, target = :bivariate_policy_A0_to_A1,
            reference_interval = (tails.lower, tails.upper), quantile_change,
            maximum_tail_residual = tails.maximum_tail_residual,
            maximum_integration_error = tails.maximum_integration_error,
            reference_tail_gate_passed = validated))
        # Declare both policy values in training support. The two extra rows
        # have no observed outcome, hence leave the reference likelihood intact.
        frame = DataFrame(A = [a; 0.0; 1.0], Y = fixture == :prior_only ? fill(missing, 22) :
            Union{Missing,Int}[y; missing; missing])
        for covariance in (:variational, :linear_response)
            fit = fit_bayesian_logistic(frame, :Y, :A, Symbol[]; prior,
                handle_missing = :mar, covariance,
                maxiter = 1000, tol = 1e-10)
            gaussian = bivariate_policy_reference(zeros(0, 2), Float64[], fit.posterior_mean,
                fit.posterior_covariance; rtol = 1e-9, atol = 1e-12)
            gaussian_tails = policy_tails(gaussian)
            draws = 80_000
            public_result = bayesian_policy_contrast(fit, DataFrame(A = [0.0]), discrete_static_policy(1.0);
                deltas = [1.0], n_draws = draws, rng = Xoshiro(4809))
            public_lower, public_upper = public_result.lower[1], public_result.upper[1]
            lower_cdf = policy_cdf(gaussian, public_lower)
            upper_cdf = policy_cdf(gaussian, public_upper)
            public_probability_error = max(abs(lower_cdf - 0.025), abs(upper_cdf - 0.975))
            public_sampling_gate = public_probability_error <= 0.004
            gate &= public_sampling_gate
            true_lower_cdf = policy_cdf(tight, gaussian_tails.lower)
            true_upper_cdf = policy_cdf(tight, gaussian_tails.upper)
            println((; fixture, target = :bivariate_policy_A0_to_A1, covariance,
                gaussian_interval = (gaussian_tails.lower, gaussian_tails.upper),
                public_interval = (public_lower, public_upper), draws,
                public_maximum_tail_probability_error = public_probability_error,
                public_sampling_gate_passed = public_sampling_gate,
                gaussian_posterior_content = true_upper_cdf - true_lower_cdf,
                gaussian_lower_true_tail = true_lower_cdf,
                gaussian_upper_true_tail = 1 - true_upper_cdf,
                gaussian_lower_endpoint_error = gaussian_tails.lower - tails.lower,
                gaussian_upper_endpoint_error = gaussian_tails.upper - tails.upper))
        end
        flush(stdout)
    end
    return gate
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    println((; julia = VERSION, threads = Threads.nthreads(), blas_threads = BLAS.get_num_threads(), seed = 1607,
        reference = "adaptive infinite-domain scalar integration; direct tails and quantile-specific tolerance refinement"))
    gate = check_tail_fixtures()
    check_collinearity()
    gate &= check_bivariate_policy_tails()
    calibrate_tails(; replications = parse(Int, get(ENV, "LOGISTIC_TAIL_REPLICATIONS", "500")))
    println((; adaptive_reference_tail_gate_passed = gate,
        scope = "scalar calibration, exact collinear projection and two deterministic bivariate policy checks; not general multivariate calibration"))
    gate || error("adaptive tail reference failed its prespecified numerical accuracy checks")
end
