"""Parametric bootstrap and g-computation dispatch for `NB2RandomInterceptModel`."""

function _empirical_covariance(samples::Matrix{Float64})
    n = size(samples, 1)
    n >= 2 || throw(ArgumentError("at least two successful bootstrap samples are required"))
    centered = samples .- sum(samples; dims = 1) ./ n
    covariance = transpose(centered) * centered / (n - 1)
    return Matrix{Float64}((covariance .+ transpose(covariance)) ./ 2)
end

function _replace_uncertainty(
    result::MixedGComputationResult,
    covariance::Matrix{Float64},
    diagnostics,
)
    diagonal = diag(covariance)
    all(diagonal .>= -1024eps(Float64) * max(1.0, maximum(abs, covariance))) ||
        throw(ArgumentError("bootstrap covariance has a negative diagonal"))
    standard_errors = sqrt.(max.(diagonal, 0.0))
    return MixedGComputationResult(;
        treatment = result.treatment,
        outcome = result.outcome,
        time = result.time,
        id = result.id,
        times = result.times,
        values = result.values,
        mean_reference = result.mean_reference,
        mean_comparison = result.mean_comparison,
        adjustment = result.adjustment,
        vcov = covariance,
        se = standard_errors,
        random_effects = result.random_effects,
        uncertainty = :parametric_bootstrap,
        uncertainty_diagnostics = diagnostics,
    )
end

function _replace_uncertainty(
    result::StratifiedMixedGComputationResult,
    covariances::Vector{Matrix{Float64}},
    diagnostics,
)
    trajectories = [
        _replace_uncertainty(result.results[index], covariances[index], diagnostics)
        for index in eachindex(result.results)
    ]
    return StratifiedMixedGComputationResult(result.strata, result.levels, trajectories)
end

function _simulate_nb2_response(
    rng::AbstractRNG,
    model::NB2RandomInterceptModel,
    data::DataFrame,
)
    design = _fixed_effect_design(model, data)
    eta = design * model.coefficients
    random_intercepts = Dict{Any, Float64}()
    sigma = sqrt(model.random_intercept_variance)
    response = Vector{Int}(undef, nrow(data))
    for row_index in 1:nrow(data)
        identifier = data[row_index, model.id]
        b = get!(random_intercepts, identifier) do
            sigma * randn(rng)
        end
        mean_value = exp(eta[row_index] + b)
        isfinite(mean_value) || throw(ArgumentError(
            "bootstrap response mean overflowed for row $row_index",
        ))
        probability = model.theta / (model.theta + mean_value)
        response[row_index] = rand(
            rng,
            MixedModels.Distributions.NegativeBinomial(model.theta, probability),
        )
    end
    return response
end

function _bootstrap_g_computation(
    model::NB2RandomInterceptModel,
    data::DataFrame,
    point_result;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values,
    adjustment,
    strata,
    random_effects,
    n_boot::Int,
    seed::Integer,
    min_success_rate::Float64,
)
    n_boot >= 2 || throw(ArgumentError("n_boot must be at least 2"))
    0 < min_success_rate <= 1 || throw(ArgumentError(
        "min_success_rate must lie in (0, 1]",
    ))
    rng = MersenneTwister(seed)
    effect_samples = point_result isa MixedGComputationResult ?
        [Vector{Float64}[]] : [Vector{Float64}[] for _ in point_result.results]
    failure_messages = String[]
    start = vcat(
        model.coefficients,
        0.5log(model.random_intercept_variance),
        log(model.theta),
    )
    for replicate in 1:n_boot
        try
            bootstrap_data = copy(data)
            bootstrap_data[!, outcome] = _simulate_nb2_response(rng, model, data)
            bootstrap_model = CausalTargeted.fit_profiled_nb2(
                model.formula,
                bootstrap_data;
                id,
                treatment,
                theta_initial = model.theta,
                theta_bounds = model.optimizer_diagnostics.theta_bounds,
                quadrature_points = model.quadrature_points,
                quadrature_method = model.quadrature_method,
                progress = false,
                maxeval = 2_000,
                multiple_starts = 1,
                profile = false,
                initial_parameters = start,
            )
            bootstrap_model.converged || throw(ErrorException(
                "refit returned converged=false ($(bootstrap_model.optimizer_diagnostics))",
            ))
            bootstrap_result = _mixed_g_computation(
                bootstrap_model,
                bootstrap_data;
                treatment,
                outcome,
                time,
                id,
                values,
                adjustment,
                strata,
                random_effects,
                compute_uncertainty = false,
            )
            if bootstrap_result isa MixedGComputationResult
                push!(only(effect_samples), bootstrap_result.effect)
            else
                bootstrap_result.levels == point_result.levels || throw(ErrorException(
                    "bootstrap stratum ordering changed",
                ))
                for index in eachindex(effect_samples)
                    push!(effect_samples[index], bootstrap_result.results[index].effect)
                end
            end
        catch error
            push!(failure_messages, "replicate $replicate: $(sprint(showerror, error))")
        end
    end
    successful = length(first(effect_samples))
    failed = n_boot - successful
    successful / n_boot >= min_success_rate || throw(ErrorException(
        "only $successful of $n_boot parametric-bootstrap refits succeeded; required " *
        "at least $(ceil(Int, min_success_rate * n_boot)). First failures: " *
        join(first(failure_messages, min(3, length(failure_messages))), " | "),
    ))
    failed > 0 && @warn(
        "$failed of $n_boot parametric-bootstrap refits failed",
        failure_messages,
    )
    effect_draws = map(samples -> reduce(vcat, transpose.(samples)), effect_samples)
    covariances = map(_empirical_covariance, effect_draws)
    diagnostics = (;
        n_boot,
        n_successful = successful,
        n_failed = failed,
        seed = Int(seed),
        failure_messages,
        effect_draws,
        refitted_parameters = (:beta, :random_intercept_variance, :theta),
    )
    return point_result isa MixedGComputationResult ?
        _replace_uncertainty(point_result, only(covariances), diagnostics) :
        _replace_uncertainty(point_result, covariances, diagnostics)
end

function _nb2_g_computation(
    model::NB2RandomInterceptModel,
    data::DataFrame;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    adjustment = Symbol[],
    strata = nothing,
    random_effects = :zero,
    uncertainty = model.uncertainty_method,
    n_boot::Int = 200,
    seed::Integer = 20260825,
    min_success_rate::Float64 = 0.8,
)
    uncertainty in (:delta_fixed, :parametric_bootstrap) || throw(ArgumentError(
        "uncertainty must be :delta_fixed or :parametric_bootstrap; got $(repr(uncertainty))",
    ))
    point_result = _mixed_g_computation(
        model,
        data;
        treatment,
        outcome,
        time,
        id,
        values,
        adjustment,
        strata,
        random_effects,
    )
    uncertainty == :delta_fixed && return point_result
    return _bootstrap_g_computation(
        model,
        data,
        point_result;
        treatment,
        outcome,
        time,
        id,
        values,
        adjustment,
        strata,
        random_effects,
        n_boot,
        seed,
        min_success_rate,
    )
end

function _run_mixed_g_computation(
    model::NB2RandomInterceptModel,
    data::DataFrame,
    adjustment;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    strata = nothing,
    random_effects = :zero,
    uncertainty = model.uncertainty_method,
    n_boot::Int = 200,
    seed::Integer = 20260825,
    min_success_rate::Float64 = 0.8,
)
    return _nb2_g_computation(
        model,
        data;
        treatment,
        outcome,
        time,
        id,
        values,
        adjustment,
        strata,
        random_effects,
        uncertainty,
        n_boot,
        seed,
        min_success_rate,
    )
end
