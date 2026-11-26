# ---------------------------------------------------------------------------
# Standardised means, contrasts, interactions & high-level entry point
# ---------------------------------------------------------------------------

"""
    gcomp_mean(fit, data; set=(;), by=nothing, subset=nothing,
               random_effects=nothing)

Standardise response-scale predictions over the empirical rows in `data`, or
over the target rows selected by `by`/`subset`. Target rows are restricted
before the intervention is applied.
The target can differ from the training table in both rows and size and need not
contain the outcome. Formula predictors in the selected target rows must be
complete-case before intervention. Missing values are rejected, not dropped.
For NB2 fits, HC3/delta inference treats the supplied or estimated `theta` as fixed.
For mixed fits, specify `random_effects=:zero` or `:marginal` as supported by the
backend. The mixed backend provides its own covariance and uncertainty metadata.
"""
function gcomp_mean(
    fit::AbstractParametricGComputationFit,
    data::AbstractDataFrame;
    set = NamedTuple(),
    by = nothing,
    subset = nothing,
    random_effects = nothing,
)
    result = _gcomp_mean_components(fit, data; set, by, subset, random_effects)
    inference = _gcomp_wald(result.estimate, result.gradient, _gcomp_parameter_covariance(fit))
    return (;
        estimate = result.estimate,
        n = result.n,
        set = result.set,
        target = (; by = result.by, subset = result.subset),
        inference...,
        _gcomp_result_metadata(fit, random_effects)...,
    )
end

"""
    gcomp_mean(model::StatsModels.TableRegressionModel, training_data, target_data; kwargs...)

Standardise an existing formula-fitted GLM over `target_data`. Supply the exact
complete-case table used to fit `model` as `training_data`; it is used to validate
the fit for HC3 coefficient inference, not as the standardisation population.
`target_data` may have a different number of rows and need not contain the outcome.
Target predictors must be complete-case as described in `gcomp_mean(fit, data)`.
NB2 HC3/delta inference treats `theta` as fixed, including when GLM estimated it.
"""
function gcomp_mean(
    model::StatsModels.TableRegressionModel,
    training_data::AbstractDataFrame,
    target_data::AbstractDataFrame;
    kwargs...,
)
    return gcomp_mean(_gcomp_wrap_model(model, training_data), target_data; kwargs...)
end

function _gcomp_contrast_from_components(reference_component, comparison_component, scale::Symbol)
    mu0, mu1 = reference_component.estimate, comparison_component.estimate
    G0, G1 = reference_component.gradient, comparison_component.gradient
    scale in (:ratio, :logratio) && (mu0 <= 0 || mu1 <= 0) && throw(ArgumentError(
        "ratio contrasts require positive standardised means; got reference=$mu0 and comparison=$mu1",
    ))
    if scale == :difference
        estimate, gradient, transform = mu1 - mu0, G1 - G0, :identity
    elseif scale == :ratio
        estimate, gradient, transform = mu1 / mu0, G1 / mu1 - G0 / mu0, :exp
    elseif scale == :logratio
        estimate, gradient, transform = log(mu1) - log(mu0), G1 / mu1 - G0 / mu0, :identity
    else
        throw(ArgumentError("scale must be :difference, :ratio, or :logratio"))
    end
    return (; estimate, gradient, transform, reference_component, comparison_component)
end

function _gcomp_contrast_components(
    fit::AbstractParametricGComputationFit, data::AbstractDataFrame;
    treatment::Symbol, reference, comparison,
    by = nothing, subset = nothing, scale::Symbol = :difference,
    random_effects = nothing,
)
    components = _gcomp_standardise(fit, data, [
        (; set = _gcomp_setting(treatment, reference), by, subset),
        (; set = _gcomp_setting(treatment, comparison), by, subset),
    ]; random_effects)
    return _gcomp_contrast_from_components(components[1], components[2], scale)
end

"""
    gcomp_contrast(fit, data; treatment, reference, comparison,
                   by=nothing, subset=nothing, scale=:difference,
                   random_effects=nothing)

Compute a marginal or subgroup empirical standardised difference, response-mean
ratio, or log response-mean ratio. Ratio inference is performed on the log scale.
Target rows follow the complete-case predictor policy of `gcomp_mean`; no outcome
column is required. NB2 HC3/delta inference treats supplied or estimated `theta` as fixed.
For mixed fits, set `random_effects` explicitly when using a nonlinear link.
"""
function gcomp_contrast(
    fit::AbstractParametricGComputationFit,
    data::AbstractDataFrame;
    treatment::Symbol,
    reference,
    comparison,
    by = nothing,
    subset = nothing,
    scale::Symbol = :difference,
    random_effects = nothing,
)
    result = _gcomp_contrast_components(
        fit, data; treatment, reference, comparison, by, subset, scale, random_effects,
    )
    inference = _gcomp_wald(
        result.estimate, result.gradient, _gcomp_parameter_covariance(fit);
        transform = result.transform,
    )
    return (;
        estimate = result.estimate,
        reference_mean = result.reference_component.estimate,
        comparison_mean = result.comparison_component.estimate,
        n = result.reference_component.n,
        treatment,
        reference,
        comparison,
        scale,
        target = (; by, subset = subset === nothing ? nothing : :function),
        inference...,
        _gcomp_result_metadata(fit, random_effects)...,
    )
end

"""
    gcomp_contrast(model::StatsModels.TableRegressionModel, training_data, target_data; kwargs...)

Compute a contrast from an existing formula-fitted GLM. The explicit training/target
split and complete-case requirements are the same as for `gcomp_mean(model,
training_data, target_data)`. NB2 HC3/delta inference treats `theta` as fixed.
"""
function gcomp_contrast(
    model::StatsModels.TableRegressionModel,
    training_data::AbstractDataFrame,
    target_data::AbstractDataFrame;
    kwargs...,
)
    return gcomp_contrast(_gcomp_wrap_model(model, training_data), target_data; kwargs...)
end

function _gcomp_interaction_components(
    fit::AbstractParametricGComputationFit, data::AbstractDataFrame;
    treatment, reference, comparison,
    modifier, modifier_reference, modifier_comparison,
    scale,
    random_effects = nothing,
)
    subgroup_scale = scale == :difference ? :difference : :ratio
    specs = [
        (; set = _gcomp_setting(treatment, treatment_value),
            by = _gcomp_setting(modifier, modifier_value))
        for modifier_value in (modifier_reference, modifier_comparison)
        for treatment_value in (reference, comparison)
    ]
    components = _gcomp_standardise(fit, data, specs; random_effects)
    subgroup0 = _gcomp_contrast_from_components(components[1], components[2], subgroup_scale)
    subgroup1 = _gcomp_contrast_from_components(components[3], components[4], subgroup_scale)
    if scale == :difference
        estimate = subgroup1.estimate - subgroup0.estimate
        gradient = subgroup1.gradient - subgroup0.gradient
        transform = :identity
    elseif scale in (:ratio, :logratio)
        log_estimate = log(subgroup1.estimate) - log(subgroup0.estimate)
        gradient = subgroup1.gradient - subgroup0.gradient
        estimate = scale == :ratio ? exp(log_estimate) : log_estimate
        transform = scale == :ratio ? :exp : :identity
    else
        throw(ArgumentError("interaction scale must be :difference, :ratio, or :logratio"))
    end
    return (; estimate, gradient, transform, subgroup0, subgroup1)
end

function _gcomp_interaction_point(
    result, fit, scale, modifier, modifier_reference, modifier_comparison,
    random_effects = nothing,
)
    inference = _gcomp_wald(
        result.estimate, result.gradient, _gcomp_parameter_covariance(fit);
        transform = result.transform,
    )
    subgroup0 = result.subgroup0
    subgroup1 = result.subgroup1
    return (;
        estimate = result.estimate,
        scale,
        modifier,
        modifier_reference,
        modifier_comparison,
        modifier_reference_effect = subgroup0.estimate,
        modifier_comparison_effect = subgroup1.estimate,
        component_means = (;
            modifier_reference = (;
                reference = subgroup0.reference_component.estimate,
                comparison = subgroup0.comparison_component.estimate,
                n = subgroup0.reference_component.n,
            ),
            modifier_comparison = (;
                reference = subgroup1.reference_component.estimate,
                comparison = subgroup1.comparison_component.estimate,
                n = subgroup1.reference_component.n,
            ),
        ),
        inference...,
        _gcomp_result_metadata(fit, random_effects)...,
    )
end

"""
    gcomp_interaction(fit, data; treatment, reference, comparison, modifier,
                      modifier_reference, modifier_comparison, scale=:difference,
                      random_effects=nothing)

Compute a difference-of-differences or a response-mean ratio-of-ratios. The
return value includes both component subgroup effects and all four standardised
means. Delta inference uses the single fitted coefficient covariance jointly.
For NB2 this treats supplied or estimated `theta` as fixed. Selected target rows
must have complete-case formula predictors; the outcome is needed only for refitting.
Set `n_boot > 0` for an additional stratified refit-bootstrap summary.
"""
function gcomp_interaction(
    fit::AbstractParametricGComputationFit,
    data::AbstractDataFrame;
    treatment::Symbol,
    reference,
    comparison,
    modifier::Symbol,
    modifier_reference,
    modifier_comparison,
    scale::Symbol = :difference,
    n_boot::Int = 0,
    bootstrap_strata = nothing,
    rng::AbstractRNG = Random.default_rng(),
    random_effects = nothing,
)
    n_boot >= 0 || throw(ArgumentError("n_boot must be non-negative"))
    n_boot > 0 && !(fit isa ParametricGComputationFit) && throw(ArgumentError(
        "generic mixed-model gcomp_interaction does not support the row-wise GLM bootstrap; " *
        "use a supported mixed-model bootstrap entry point",
    ))
    result = _gcomp_interaction_components(
        fit, data;
        treatment, reference, comparison,
        modifier, modifier_reference, modifier_comparison,
        scale, random_effects,
    )
    point = _gcomp_interaction_point(
        result, fit, scale, modifier, modifier_reference, modifier_comparison,
        random_effects,
    )
    n_boot == 0 && return merge(point, (; bootstrap = nothing))
    bootstrap = bootstrap_gcomp_interaction(
        fit, data;
        treatment, reference, comparison,
        modifier, modifier_reference, modifier_comparison,
        scale, n_boot, strata = bootstrap_strata, rng,
    )
    return merge(point, (; bootstrap))
end

"""
    gcomp_interaction(model::StatsModels.TableRegressionModel, training_data, target_data; kwargs...)

Compute an interaction from an existing formula-fitted GLM with the same explicit
training/target split and complete-case requirements as `gcomp_mean(model,
training_data, target_data)`. NB2 HC3/delta inference treats `theta` as fixed.
For refit bootstraps, create a fit with `fit_parametric_gcomp` instead.
"""
function gcomp_interaction(
    model::StatsModels.TableRegressionModel,
    training_data::AbstractDataFrame,
    target_data::AbstractDataFrame;
    kwargs...,
)
    return gcomp_interaction(_gcomp_wrap_model(model, training_data), target_data; kwargs...)
end

"""
    run_parametric_gcomp(formula, data; treatment, reference, comparison,
                         backend=:glm, random_effects=nothing, ...)

Fit a supported parametric outcome model and return an empirical standardised
contrast. The GLM backend retains HC3 coefficient covariance by default; the mixed
backend uses its own covariance method. Supply `id` for mixed fitting, and specify
`random_effects` for nonlinear mixed predictions. See `fit_parametric_gcomp` for
the complete-case policy and fitted-parameter uncertainty limits.
"""
function run_parametric_gcomp(
    formula_term::StatsModels.FormulaTerm,
    data::AbstractDataFrame;
    treatment::Symbol,
    reference,
    comparison,
    backend::Symbol = :glm,
    family = :gaussian,
    id::Union{Nothing,Symbol} = nothing,
    link = nothing,
    theta::Union{Nothing,Real} = nothing,
    covariance::Union{Nothing,Symbol} = nothing,
    by = nothing,
    subset = nothing,
    scale::Symbol = :difference,
    random_effects = nothing,
    fit_kwargs = NamedTuple(),
)
    fit_kwargs isa NamedTuple || throw(ArgumentError("fit_kwargs must be a NamedTuple"))
    fit_options = (; family, fit_kwargs...)
    id !== nothing && (fit_options = merge(fit_options, (; id)))
    link !== nothing && (fit_options = merge(fit_options, (; link)))
    theta !== nothing && (fit_options = merge(fit_options, (; theta)))
    covariance !== nothing && (fit_options = merge(fit_options, (; covariance)))
    fit = fit_parametric_gcomp(
        formula_term, data; backend, fit_options...,
    )
    return gcomp_contrast(
        fit, data; treatment, reference, comparison, by, subset, scale, random_effects,
    )
end
