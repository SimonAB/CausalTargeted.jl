# Independent posterior quadrature, prior-predictive SBC, and fixed-parameter
# coverage are deliberately separate checks. Run from the package directory:
# julia --project=test/stress --threads=auto test/stress/logistic_calibration.jl
using CausalTargeted, GraphPPL, RxInfer, Distributions, StableRNGs
using LinearAlgebra, Statistics, Random, Printf

include(joinpath(@__DIR__, "..", "logistic_reference.jl"))
using .LogisticPosteriorReference

const LOGISTIC_COVARIANCE_GATE = 0.25
const LOGISTIC_MEAN_GATE = 0.20

function logistic_engine(X, y, μ, V)
    extension = Base.get_extension(CausalTargeted, :CausalTargetedRxInferExt)
    isnothing(extension) && error("the optional GraphPPL/RxInfer extension did not load")
    return extension._fit_logistic_engine(Matrix{Float64}(X), Float64.(y), Float64.(μ),
        Matrix{Float64}(V); maxiter = 1000, tol = 1e-10, covariance = :linear_response)
end

function valid_moments(fit, covariance)
    fit.converged && covariance isa AbstractMatrix && all(isfinite, covariance) &&
        isposdef(Symmetric(covariance))
end

function policy_truth(β, X, X_policy)
    mean(logistic_probability.(X_policy * β)) - mean(logistic_probability.(X * β))
end

function fixture_cases()
    a = repeat([-0.9, -0.5, 0.1, 0.5, 0.9], inner = 4)
    X = hcat(ones(20), a)
    y = Float64.([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1,
                 0, 0, 1, 1, 0, 1, 1, 1])
    correlated = hcat(a, a .+ 0.08 .* repeat([-1, 1, 1, -1], 5))
    # Near-collinear predictors without an intercept keep p=2 for the reference.
    near_collinear = hcat(a, a .+ 1e-3 .* repeat([-1.0, 1.0, 1.0, -1.0], 5))
    # Larger continuous columns under a unit prior. Scale 10 remains scoreable by
    # prior-centred Hermite (more nodes). Gram-scale columns document where that
    # reference itself fails; use adaptive QuadGK then (logistic_tail_validation.jl).
    W10 = 10.0 .* a
    moderately_scaled = hcat(ones(20), W10)
    y10 = Float64.(W10 .> 0)
    W500 = 500.0 .* a
    poorly_scaled = hcat(ones(20), W500)
    y500 = Float64.(W500 .> 0)
    return [
        (; name = :moderate_prespecified, X, y, μ = zeros(2), V = 2.25I(2), gate = true),
        (; name = :correlated_design, X = correlated, y, μ = zeros(2), V = 2.25I(2), gate = false),
        (; name = :near_collinear, X = near_collinear, y, μ = zeros(2), V = 2.25I(2), gate = false),
        (; name = :all_one, X, y = ones(20), μ = zeros(2), V = 2.25I(2), gate = false),
        (; name = :separated_weak_prior, X, y = Float64.(a .> 0), μ = zeros(2), V = 9.0I(2), gate = false),
        (; name = :strong_prior, X, y, μ = zeros(2), V = 0.25I(2), gate = false),
        (; name = :moderately_scaled, X = moderately_scaled, y = y10, μ = zeros(2),
            V = Matrix{Float64}(I, 2, 2), gate = false),
        (; name = :poorly_scaled, X = poorly_scaled, y = y500, μ = zeros(2),
            V = Matrix{Float64}(I, 2, 2), gate = false),
        (; name = :zero_mean_symmetry, X, y = Float64.(repeat([0, 1], 10)), μ = zeros(2), V = 9.0I(2), gate = false),
        (; name = :zero_design_prior_only, X = zeros(20, 2), y, μ = [0.3, -0.2], V = [1.0 0.3; 0.3 2.0], gate = false),
        (; name = :one_coefficient, X = reshape(a, :, 1), y, μ = [0.2], V = reshape([2.25], 1, 1), gate = false),
        (; name = :prior_sensitivity_tight, X, y, μ = zeros(2), V = 1.0I(2), gate = false),
        (; name = :prior_sensitivity_wide, X, y, μ = zeros(2), V = 9.0I(2), gate = false),
    ]
end

"""Deterministic comparisons; only the preselected moderate case has an accuracy gate."""
function check_logistic_fixtures()
    println("Accuracy gates fixed before execution: relative covariance error <= ",
        LOGISTIC_COVARIANCE_GATE, "; maximum mean error / reference SD <= ", LOGISTIC_MEAN_GATE)
    println("Interval content is reference posterior mass inside equal-tailed Gaussian ",
        "coefficient or policy intervals; it is not repeated-sampling coverage.")
    println("Grid-tail outputs are exploratory: moment convergence does not certify ",
        "interval masses or endpoints; these outputs are not interval-accuracy or calibration evidence.")
    gate_passed = true
    for case in fixture_cases()
        # Broad-prior adversarial cases can require many more prior-centred
        # nodes than ordinary cases; refine the reference, never the accuracy gate.
        reference = converged_reference(case.X, case.y, case.μ, case.V;
            orders = (80, 160, 240, 400, 640, 960))
        fit = logistic_engine(case.X, case.y, case.μ, case.V)
        X_policy = copy(case.X)
        # Keep the intervened column on a comparable scale to the factual design.
        X_policy[:, end] .= median(case.X[:, end])
        exact_policy = policy_reference(reference, case.X, X_policy)
        println((; fixture = case.name, reference_converged = reference.converged,
            reference_points = reference.points, reference_history = reference.history,
            status = fit.status, iterations = fit.n_iterations,
            reference_mean = reference.mean, reference_covariance = reference.covariance,
            reference_policy = exact_policy))
        for (mode, covariance) in ((:variational, fit.variational_covariance),
                                   (:linear_response, fit.corrected_covariance))
            if !valid_moments(fit, covariance)
                println((; fixture = case.name, mode, available = false,
                    status = fit.status, iterations = fit.n_iterations))
                case.gate && (gate_passed = false)
                continue
            end
            scale = sqrt.(diag(reference.covariance))
            mean_error = maximum(abs.(fit.mean - reference.mean) ./ scale)
            covariance_error = norm(covariance - reference.covariance) / norm(reference.covariance)
            coefficient_widths = 2quantile(Normal(), 0.975) .* sqrt.(diag(covariance))
            reference_widths = [weighted_quantile(view(reference.coefficients, j, :), reference.weights, 0.975) -
                weighted_quantile(view(reference.coefficients, j, :), reference.weights, 0.025)
                for j in eachindex(fit.mean)]
            coeff_content = coefficient_interval_content(reference, fit.mean, covariance)
            policy = gaussian_policy_summary(fit.mean, covariance, case.X, X_policy)
            policy_content = policy_interval_content(reference, case.X, X_policy,
                policy.lower, policy.upper)
            accepted = reference.converged && mean_error <= LOGISTIC_MEAN_GATE &&
                       covariance_error <= LOGISTIC_COVARIANCE_GATE
            case.gate && mode == :linear_response && (gate_passed &= accepted)
            println((; fixture = case.name, mode, available = true, mean_error, covariance_error,
                coefficient_widths, reference_widths,
                interval_evidence = :exploratory_unchecked_grid_tails,
                coefficient_interval_content = coeff_content.contents,
                coefficient_tails = coeff_content.intervals,
                policy,
                policy_mean_error = policy.mean - exact_policy.mean,
                policy_sd_error = policy.sd - exact_policy.sd,
                policy_lower_error = policy.lower - exact_policy.lower,
                policy_upper_error = policy.upper - exact_policy.upper,
                policy_interval_content = policy_content.content,
                policy_interval_tails = (policy_content.lower_tail, policy_content.upper_tail),
                gated = case.gate && mode == :linear_response, accepted))
        end
        if case.name == :zero_mean_symmetry && fit.corrected_covariance !== nothing
            println((; symmetry_correction_norm = norm(fit.corrected_covariance - fit.variational_covariance),
                interpretation = "zero mean gives no LRVB correction; exact posterior covariance may differ"))
        end
        flush(stdout)
    end
    println((; moderate_accuracy_gate_passed = gate_passed))
    return gate_passed
end

function calibration_accumulator()
    return (; attempted = Ref(0), failed = Ref(0), covered = zeros(Int, 2),
        widths = [Float64[], Float64[]], squared_errors = [Float64[], Float64[]],
        cdf = [Float64[], Float64[]], policy_covered = Ref(0), policy_width = Float64[],
        policy_squared_errors = Float64[])
end

function record_calibration!(acc, μ, V, β, policy, true_policy)
    acc.attempted[] += 1
    if V === nothing || !all(isfinite, V) || !isposdef(Symmetric(V))
        acc.failed[] += 1
        return
    end
    critical = quantile(Normal(), 0.975)
    for j in eachindex(β)
        sd = sqrt(V[j, j])
        acc.covered[j] += abs(β[j] - μ[j]) <= critical * sd
        push!(acc.widths[j], 2critical * sd)
        push!(acc.squared_errors[j], (β[j] - μ[j])^2)
        push!(acc.cdf[j], cdf(Normal(μ[j], sd), β[j]))
    end
    acc.policy_covered[] += policy.lower <= true_policy <= policy.upper
    push!(acc.policy_width, policy.upper - policy.lower)
    push!(acc.policy_squared_errors, (policy.mean - true_policy)^2)
end

function report_calibration(experiment, mode, acc)
    attempted = acc.attempted[]
    available = attempted - acc.failed[]
    for j in 1:2
        coverage = available == 0 ? NaN : acc.covered[j] / available
        ranks = acc.cdf[j]
        println((; experiment, mode, coefficient = j, attempted, available,
            validation_status = mode == :quadrature ? :exploratory_grid_quantiles : :normal_moment_intervals,
            failures = acc.failed[], coverage_among_available = coverage,
            coverage_all_attempts = acc.covered[j] / attempted,
            coverage_mcse = available == 0 ? NaN : sqrt(coverage * (1 - coverage) / available),
            mean_width = isempty(acc.widths[j]) ? NaN : mean(acc.widths[j]),
            rmse = isempty(acc.squared_errors[j]) ? NaN : sqrt(mean(acc.squared_errors[j])),
            posterior_cdf_mean = isempty(ranks) ? NaN : mean(ranks),
            posterior_cdf_deciles = [count(r -> (k-1)/10 <= r < k/10, ranks) for k in 1:10],
            cdf_interpretation = experiment == :prior_predictive ? :sbc : :fixed_parameter_not_sbc))
    end
    policy_coverage = available == 0 ? NaN : acc.policy_covered[] / available
    println((; experiment, mode, target = :fixed_cohort_policy,
        validation_status = :exploratory_grid_quantiles,
        coverage_among_available = policy_coverage,
        coverage_all_attempts = acc.policy_covered[] / attempted,
        coverage_mcse = available == 0 ? NaN : sqrt(policy_coverage * (1 - policy_coverage) / available),
        mean_width = isempty(acc.policy_width) ? NaN : mean(acc.policy_width),
        rmse = isempty(acc.policy_squared_errors) ? NaN : sqrt(mean(acc.policy_squared_errors))))
end

"""500 prior-predictive datasets and 500 separate fixed-parameter datasets by default."""
function calibrate_bayesian_logistic(; replications = 500, seed = 913)
    rng = StableRNG(seed)
    μ, V = zeros(2), 2.25I(2)
    # A bounded fixed cohort makes all methods target the same empirical contrast.
    a = collect(range(-0.9, 0.9; length = 20))
    X = hcat(ones(20), a)
    X_policy = hcat(ones(20), fill(0.75, 20))
    for experiment in (:prior_predictive, :fixed_parameter)
        accumulators = Dict(mode => calibration_accumulator() for mode in (:variational, :linear_response, :quadrature))
        reference_failures = 0
        iteration_counts = Int[]
        elapsed = @elapsed for replication in 1:replications
            β = experiment == :prior_predictive ? μ + cholesky(Symmetric(Matrix(V))).L * randn(rng, 2) : [-0.3, 0.9]
            y = Float64.(rand(rng, 20) .< logistic_probability.(X * β))
            true_policy = policy_truth(β, X, X_policy)
            reference = converged_reference(X, y, μ, V)
            reference_failures += !reference.converged
            exact_policy = policy_reference(reference, X, X_policy)
            # These grid-tail diagnostics are exploratory: moment convergence
            # does not validate their quantiles. Separate adaptive integration
            # fixtures test tails in logistic_tail_validation.jl.
            exact = accumulators[:quadrature]
            exact.attempted[] += 1
            if !reference.converged
                exact.failed[] += 1
            else
                for j in 1:2
                    values = view(reference.coefficients, j, :)
                    lower = weighted_quantile(values, reference.weights, 0.025)
                    upper = weighted_quantile(values, reference.weights, 0.975)
                    exact.covered[j] += lower <= β[j] <= upper
                    push!(exact.widths[j], upper - lower)
                    push!(exact.squared_errors[j], (reference.mean[j] - β[j])^2)
                    push!(exact.cdf[j], sum(reference.weights[values .<= β[j]]))
                end
                exact.policy_covered[] += exact_policy.lower <= true_policy <= exact_policy.upper
                push!(exact.policy_width, exact_policy.upper - exact_policy.lower)
                push!(exact.policy_squared_errors, (exact_policy.mean - true_policy)^2)
            end
            fit = try
                logistic_engine(X, y, μ, V)
            catch exception
                println((; experiment, replication, fit_exception = sprint(showerror, exception)))
                nothing
            end
            fit !== nothing && push!(iteration_counts, fit.n_iterations)
            for mode in (:variational, :linear_response)
                covariance = fit === nothing || !fit.converged ? nothing :
                    mode == :variational ? fit.variational_covariance : fit.corrected_covariance
                usable = covariance !== nothing && all(isfinite, covariance) && isposdef(Symmetric(covariance))
                policy = usable ? gaussian_policy_summary(fit.mean, covariance, X, X_policy; points = 40) : nothing
                record_calibration!(accumulators[mode], fit === nothing ? zeros(2) : fit.mean,
                    usable ? covariance : nothing, β, policy, true_policy)
            end
            if replication % 50 == 0
                println((; experiment, completed = replication, reference_failures))
                flush(stdout)
            end
        end
        for mode in (:variational, :linear_response, :quadrature)
            report_calibration(experiment, mode, accumulators[mode])
        end
        println((; experiment, seconds = elapsed, reference_failures,
            mean_iterations = isempty(iteration_counts) ? NaN : mean(iteration_counts),
            reference = "prior Gauss-Hermite product rules 80/160, refined to 240 on disagreement",
            quantile_caveat = "moment convergence does not certify weighted-grid quantile convergence"))
    end
end

function benchmark_logistic()
    case = first(fixture_cases())
    seconds = @elapsed for _ in 1:10
        logistic_engine(case.X, case.y, case.μ, case.V)
    end
    bytes = @allocated logistic_engine(case.X, case.y, case.μ, case.V)
    println((; warmed_fit_seconds = seconds / 10, warmed_fit_bytes = bytes,
        covariance = :linear_response, n = 20, p = 2))
end

if abspath(PROGRAM_FILE) == @__FILE__
    # The reference repeatedly multiplies very thin (p <= 2) matrices. Avoid
    # nested BLAS worker overhead while retaining Julia's requested thread count.
    BLAS.set_num_threads(1)
    println("Independent logistic calibration; Julia ", VERSION,
        "; threads=", Threads.nthreads(), "; BLAS threads=", BLAS.get_num_threads(), "; seed=913")
    flush(stdout)
    compilation_seconds = @elapsed begin
        case = first(fixture_cases())
        logistic_engine(case.X, case.y, case.μ, case.V)
        reference = logistic_reference(case.X, case.y, case.μ, case.V; points = 8)
        policy_reference(reference, case.X, case.X)
    end
    println((; first_call_including_compilation_seconds = compilation_seconds))
    flush(stdout)
    gate = check_logistic_fixtures()
    replications = parse(Int, get(ENV, "LOGISTIC_REPLICATIONS", "500"))
    calibrate_bayesian_logistic(; replications)
    benchmark_logistic()
    println((; final_moderate_accuracy_gate_passed = gate,
        approximation_claim = "LRVB is not guaranteed to improve every covariance or interval"))
    gate || error("the prespecified moderate logistic approximation accuracy gate failed")
end
