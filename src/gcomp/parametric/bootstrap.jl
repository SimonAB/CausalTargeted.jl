# ---------------------------------------------------------------------------
# Non-parametric stratified refit bootstrap for parametric g-computation
# ---------------------------------------------------------------------------

function _gcomp_bootstrap_indices(rng, data, strata)
    strata === nothing && return rand(rng, 1:nrow(data), nrow(data))
    columns = strata isa Symbol ? [strata] : Symbol.(collect(strata))
    isempty(columns) && return rand(rng, 1:nrow(data), nrow(data))
    for column in columns
        hasproperty(data, column) || throw(ArgumentError("bootstrap stratum column :$column is absent"))
    end
    groups = Dict{Tuple,Vector{Int}}()
    for (index, row) in enumerate(eachrow(data))
        key = Tuple(row[column] for column in columns)
        push!(get!(groups, key, Int[]), index)
    end
    indices = Int[]
    for group in values(groups)
        append!(indices, rand(rng, group, length(group)))
    end
    return indices
end

function _gcomp_refit(fit::ParametricGComputationFit, data)
    fit.formula_spec === nothing && throw(ArgumentError(
        "bootstrap refitting requires a fit created by fit_parametric_gcomp",
    ))
    theta = fit.family == :negbin && !fit.estimated_theta ? fit.theta : nothing
    return fit_parametric_gcomp(
        fit.formula_spec,
        data;
        family = fit.family,
        theta,
        covariance = :none,
    )
end

"""
    bootstrap_gcomp_interaction(fit, data; n_boot=500, strata=nothing, ...)

Refit a non-parametric bootstrap for a formal interaction. If `strata` is
provided, sampling occurs independently within each observed stratum and keeps
the original stratum sizes. Failed fits are counted and summarised.
The resampled table must be complete-case in the original formula's outcome and
predictors. NB2 `theta` is re-estimated in each replicate if originally estimated;
a supplied fixed `theta` stays fixed. This differs from HC3/delta intervals, which
do not propagate theta-estimation uncertainty.
"""
function bootstrap_gcomp_interaction(
    fit::ParametricGComputationFit,
    data::AbstractDataFrame;
    treatment::Symbol,
    reference,
    comparison,
    modifier::Symbol,
    modifier_reference,
    modifier_comparison,
    scale::Symbol = :ratio,
    n_boot::Int = 500,
    strata = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    _gcomp_check_complete_cases(data, StatsModels.formula(fit.model); context = "bootstrap data")
    estimates = Float64[]
    failures = Dict{String,Int}()
    for _ in 1:n_boot
        indices = _gcomp_bootstrap_indices(rng, data, strata)
        bootstrap_data = DataFrame(data[indices, :])
        try
            bootstrap_fit = _gcomp_refit(fit, bootstrap_data)
            component = _gcomp_interaction_components(
                bootstrap_fit,
                bootstrap_data;
                treatment, reference, comparison,
                modifier, modifier_reference, modifier_comparison,
                scale,
            )
            isfinite(component.estimate) || error("non-finite interaction estimate")
            push!(estimates, component.estimate)
        catch error
            reason = sprint(showerror, error)
            failures[reason] = get(failures, reason, 0) + 1
        end
    end
    successful = length(estimates)
    lower = successful == 0 ? nothing : quantile(estimates, 0.025)
    upper = successful == 0 ? nothing : quantile(estimates, 0.975)
    null_value = scale == :ratio ? 1.0 : 0.0
    p_value = successful == 0 ? nothing : min(
        1.0,
        2min(
            (count(<=(null_value), estimates) + 1) / (successful + 1),
            (count(>=(null_value), estimates) + 1) / (successful + 1),
        ),
    )
    return (;
        requested_replicates = n_boot,
        successful_replicates = successful,
        success_fraction = successful / n_boot,
        estimates,
        se = successful < 2 ? nothing : std(estimates),
        ci_lower = lower,
        ci_upper = upper,
        p_value,
        failure_reasons = failures,
        strata,
    )
end
