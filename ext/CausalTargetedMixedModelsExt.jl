"""Optional repeated-outcome g-computation with MixedModels.jl."""
module CausalTargetedMixedModelsExt

using CausalTargeted
using CausalTargeted: MixedModelGComputationFit
using CausalDynamics:
    CausalGraph,
    TotalEffectQuery,
    get_node_names,
    identify
using DataFrames
using FastGaussQuadrature
using Graphs: AbstractGraph
using LinearAlgebra
using MixedModels
using NLopt
using Random
using SpecialFunctions

const SupportedMixedModel = Union{
    LinearMixedModel,
    GeneralizedLinearMixedModel,
    NB2RandomInterceptModel,
}

function _validate_columns(data::DataFrame, columns)
    available = Set(propertynames(data))
    for column in columns
        column in available || throw(ArgumentError(
            "column :$column is missing from data; available columns are " *
            "$(collect(propertynames(data)))",
        ))
    end
    return nothing
end

function _validate_values(values)
    # Require an indexable 2-element container so invalid inputs raise ArgumentError
    # rather than MethodError (e.g. Set is length-applicable but not getindex-able).
    values isa Union{AbstractVector, Tuple} || throw(ArgumentError(
        "values must be a 2-element Tuple or AbstractVector of static intervention values; " *
        "got $(typeof(values))",
    ))
    length(values) == 2 || throw(ArgumentError(
        "values must contain exactly two static intervention values; got $(length(values))",
    ))
    isequal(values[1], values[2]) && throw(ArgumentError(
        "the two intervention values must be distinct; got $(repr(values[1])) twice",
    ))
    return (values[1], values[2])
end

function _validate_random_effects(random_effects)
    random_effects isa Symbol || throw(ArgumentError(
        "random_effects must be :zero or :marginal; got $(repr(random_effects))",
    ))
    random_effects in (:zero, :marginal) || throw(ArgumentError(
        "random_effects must be :zero or :marginal; got $(repr(random_effects))",
    ))
    return random_effects
end

function _family_link(model::GeneralizedLinearMixedModel)
    family = MixedModels.Distributions.Distribution(model)
    link = MixedModels.GLM.Link(model)
    return family, link
end

function _validate_glmm_family_link(model::GeneralizedLinearMixedModel)
    family, link = _family_link(model)
    family_name = string(family)
    link_name = string(typeof(link))
    family <: MixedModels.Distributions.NegativeBinomial || throw(ArgumentError(
        "unsupported GeneralizedLinearMixedModel family $family_name with link " *
        "$link_name; only fixed-shape NB2 NegativeBinomial with LogLink is supported",
    ))
    link isa LogLink || throw(ArgumentError(
        "unsupported GeneralizedLinearMixedModel family $family_name with link " *
        "$link_name; negative-binomial models require LogLink",
    ))
    _random_intercept_variance(model)
    return nothing
end

function _mixed_adapter(
    model::SupportedMixedModel,
    training_data::AbstractDataFrame;
    id::Symbol,
    covariance::Symbol = :model,
    formula_spec = nothing,
)
    covariance in (:model, :none) || throw(ArgumentError(
        "mixed-model covariance must be :model or :none; received $(repr(covariance))",
    ))
    data = DataFrame(training_data)
    nrow(data) > 0 || throw(ArgumentError("training data must contain at least one row"))
    nobs(model) == nrow(data) || throw(ArgumentError(
        "training data have $(nrow(data)) rows but the fitted mixed model used " *
        "$(nobs(model)); supply the exact complete-case fitting data",
    ))
    grouping = unique(fnames(model))
    collect(grouping) == [id] || throw(ArgumentError(
        "model grouping factor must be :$id; fitted grouping factors are $(collect(grouping))",
    ))
    variables = _formula_variables(model)
    _validate_columns(data, variables)
    _validate_complete_finite(data, variables)
    coefficients = Float64.(coef(model))
    all(isfinite, coefficients) || throw(ArgumentError(
        "fitted fixed-effect coefficients contain non-finite values",
    ))
    fixed_design = _fixed_effect_design(model, data)
    rank(fixed_design) == size(fixed_design, 2) || throw(ArgumentError(
        "fitted fixed-effect design is rank deficient; g-computation covariance " *
        "and coefficient alignment are unavailable",
    ))
    family = if model isa LinearMixedModel
        :gaussian
    elseif model isa NB2RandomInterceptModel
        model.converged || throw(ArgumentError("estimated-shape NB2 model did not converge"))
        :negbin
    else
        _validate_glmm_family_link(model)
        :negbin
    end
    response = Symbol(responsename(model))
    return MixedModelGComputationFit(
        model, formula_spec, response, family, id, data, covariance,
    )
end

function _validate_fixed_nb2_structure(formula_term, data::DataFrame, id::Symbol)
    template = try
        LinearMixedModel(formula_term, data)
    catch error
        throw(ArgumentError(
            "could not construct the NB2 mixed-model formula/design: " *
            sprint(showerror, error),
        ))
    end
    grouping = collect(fnames(template))
    grouping == [id] || throw(ArgumentError(
        "NB2 requires exactly one grouping factor :$id; detected $grouping",
    ))
    rhs = formula(template).rhs
    terms = rhs isa Tuple ? rhs : (rhs,)
    random_terms = [term for term in terms if term isa MixedModels.RandomEffectsTerm]
    length(random_terms) == 1 || throw(ArgumentError(
        "NB2 requires exactly one random-intercept term; detected " *
        "$(length(random_terms)) random-effects terms",
    ))
    raw_names = MixedModels.StatsModels.coefnames(only(random_terms).lhs)
    random_names = raw_names isa AbstractString ? [String(raw_names)] : String.(raw_names)
    random_names == ["(Intercept)"] || throw(ArgumentError(
        "NB2 random slopes are unsupported; detected random-effect columns $random_names",
    ))
    return nothing
end

function CausalTargeted._fit_parametric_gcomp(
    ::Val{:mixed},
    formula_term::MixedModels.StatsModels.FormulaTerm,
    data::AbstractDataFrame;
    family::Symbol = :gaussian,
    id::Symbol,
    theta::Union{Nothing, Real} = nothing,
    covariance::Symbol = :model,
    link = nothing,
    progress::Bool = false,
    kwargs...,
)
    work = DataFrame(data)
    _validate_columns(work, Symbol.(MixedModels.StatsModels.termvars(formula_term)))
    _validate_complete_finite(work, Symbol.(MixedModels.StatsModels.termvars(formula_term)))
    normalised = family in (:gaussian, :normal) ? :gaussian :
        family in (:negbin, :nb2, :nb, :negativebinomial, :negative_binomial) ? :negbin :
        throw(ArgumentError("mixed backend supports :gaussian and :negbin only"))
    model = if normalised == :gaussian
        theta === nothing || throw(ArgumentError("theta is only valid for NB2 models"))
        link === nothing || link == :identity || link isa IdentityLink || throw(ArgumentError(
            "Gaussian mixed models require the identity link",
        ))
        MixedModels.fit(MixedModel, formula_term, work; progress, kwargs...)
    elseif theta === nothing
        link === nothing || link == :log || link isa LogLink || throw(ArgumentError(
            "NB2 mixed models require a log link",
        ))
        CausalTargeted.fit_profiled_nb2(
            formula_term, work; id, family = :nb2, link = LogLink(),
            progress, kwargs...,
        )
    else
        isfinite(theta) && theta > 0 || throw(ArgumentError(
            "NB2 theta must be positive and finite",
        ))
        link === nothing || link == :log || link isa LogLink || throw(ArgumentError(
            "NB2 mixed models require a log link",
        ))
        _validate_fixed_nb2_structure(formula_term, work, id)
        MixedModels.fit(
            MixedModel, formula_term, work,
            MixedModels.Distributions.NegativeBinomial(Float64(theta)), LogLink();
            progress, kwargs...,
        )
    end
    return _mixed_adapter(model, work; id, covariance, formula_spec = formula_term)
end

CausalTargeted._gcomp_required_columns(fit::MixedModelGComputationFit) =
    setdiff(_formula_variables(fit.model), Set([fit.outcome]))

CausalTargeted._gcomp_protected_columns(fit::MixedModelGComputationFit) =
    Set([fit.id])

function CausalTargeted._gcomp_parameter_covariance(fit::MixedModelGComputationFit)
    fit.covariance_type == :none && return nothing
    covariance = Matrix{Float64}(vcov(fit.model))
    p = length(coef(fit.model))
    size(covariance) == (p, p) || throw(ArgumentError(
        "mixed fixed-effect covariance has size $(size(covariance)); expected ($p, $p)",
    ))
    all(isfinite, covariance) || throw(ArgumentError(
        "mixed fixed-effect covariance contains non-finite values",
    ))
    return covariance
end

function _mixed_random_effects_mode(fit::MixedModelGComputationFit, mode)
    mode === nothing && fit.family == :gaussian && return :zero
    mode === nothing && throw(ArgumentError(
        "nonlinear mixed-model g-computation requires random_effects=:zero or :marginal",
    ))
    return _validate_random_effects(mode)
end

function CausalTargeted._gcomp_result_metadata(
    fit::MixedModelGComputationFit, random_effects,
)
    mode = _mixed_random_effects_mode(fit, random_effects)
    return (;
        family = fit.family,
        covariance_type = fit.covariance_type,
        random_effects = mode,
        uncertainty = :delta_fixed,
    )
end

function CausalTargeted._gcomp_predictions_gradients(
    fit::MixedModelGComputationFit,
    regimes::Vector{DataFrame};
    random_effects = nothing,
)
    mode = _mixed_random_effects_mode(fit, random_effects)
    isempty(regimes) && return NamedTuple[]
    model = fit.model
    lengths = nrow.(regimes)
    combined = reduce(vcat, regimes)
    # The fitted subject-specific BLUPs must never enter population predictions.
    # A fresh level also preserves the established batched-regime workaround for
    # rank loss when one intervention fixes a factor to a single level.
    _validate_columns(combined, (fit.id,))
    level = _fresh_group_level(fit.training_data[!, fit.id])
    # Include the observed training predictor rows as an anchor. MixedModels
    # reconstructs a prediction design from the supplied table, and a single
    # intervention can otherwise collapse treatment or interaction columns.
    # Only the requested target rows enter any standardised mean.
    predictors = collect(CausalTargeted._gcomp_required_columns(fit))
    _validate_complete_finite(combined, predictors)
    prediction_data = vcat(
        select(combined, predictors),
        select(fit.training_data, predictors),
    )
    prediction_data[!, fit.id] = fill(level, nrow(prediction_data))
    # MixedModels' prediction constructor requires an initialised response
    # column even though its value is not used in population prediction.
    # Keep that implementation detail away from callers' target tables.
    prediction_data[!, fit.outcome] = fill(
        first(fit.training_data[!, fit.outcome]), nrow(prediction_data),
    )
    design = _fixed_effect_design(model, prediction_data)
    all_predictions, nonlinear = _prediction_components(
        model, prediction_data, design, mode,
    )
    predictions = all_predictions[1:nrow(combined)]
    design = design[1:nrow(combined), :]
    gradients = nonlinear ? design .* reshape(predictions, :, 1) : design
    all(isfinite, predictions) && all(isfinite, gradients) || throw(ArgumentError(
        "mixed population predictions or gradients are non-finite",
    ))
    endpoints = cumsum(lengths)
    return [begin
        rows = (endpoints[i] - lengths[i] + 1):endpoints[i]
        (; predictions = Vector{Float64}(predictions[rows]),
           gradients = Matrix{Float64}(gradients[rows, :]))
    end for i in eachindex(regimes)]
end

for operation in (:gcomp_mean, :gcomp_contrast, :gcomp_interaction)
    @eval function CausalTargeted.$operation(
        model::SupportedMixedModel,
        training_data::AbstractDataFrame,
        target_data::AbstractDataFrame;
        id::Symbol,
        covariance::Symbol = :model,
        kwargs...,
    )
        fit = _mixed_adapter(model, training_data; id, covariance)
        return CausalTargeted.$operation(fit, target_data; kwargs...)
    end
end

function _validate_complete_finite(data::DataFrame, columns)
    for column in columns
        values = data[!, column]
        any(ismissing, values) && throw(ArgumentError(
            "column :$column contains missing values required for prediction",
        ))
        for value in values
            if value isa Real && !isfinite(value)
                throw(ArgumentError(
                    "column :$column contains a non-finite value required for prediction",
                ))
            end
        end
    end
    return nothing
end


function _validate_static_treatment(data::DataFrame, treatment::Symbol, id::Symbol)
    observed = Dict{Any, Any}()
    for row in eachrow(data)
        unit = row[id]
        value = row[treatment]
        if haskey(observed, unit) && !isequal(observed[unit], value)
            throw(ArgumentError(
                "treatment :$treatment must be static within :$id; unit " *
                "$(repr(unit)) has both $(repr(observed[unit])) and $(repr(value))",
            ))
        end
        observed[unit] = value
    end
    return nothing
end

function _normalize_strata(strata)
    columns = if strata === nothing
        Symbol[]
    elseif strata isa Symbol
        [strata]
    elseif strata isa Tuple && all(x -> x isa Symbol, strata)
        collect(Symbol, strata)
    else
        throw(ArgumentError(
            "strata must be nothing, a column Symbol, or a tuple of column Symbols",
        ))
    end
    isempty(columns) && strata !== nothing && throw(ArgumentError(
        "strata must contain at least one column Symbol",
    ))
    length(unique(columns)) == length(columns) || throw(ArgumentError(
        "strata contains duplicate columns: $columns",
    ))
    return columns
end

function _fresh_group_level(values)
    present = Set(values)
    nonmissing = collect(skipmissing(values))
    isempty(nonmissing) && throw(ArgumentError("ID column must contain at least one value"))
    sample = first(nonmissing)

    if sample isa Integer && !(sample isa Bool)
        candidate = BigInt(maximum(nonmissing)) + 1
        converted = try
            convert(typeof(sample), candidate)
        catch
            nothing
        end
        converted !== nothing && !(converted in present) && return converted
    elseif sample isa AbstractFloat
        candidate = nextfloat(maximum(nonmissing))
        isfinite(candidate) && !(candidate in present) && return candidate
    elseif sample isa AbstractString
        prefix = "__causaltargeted_population__"
        candidate = prefix
        suffix = 0
        while candidate in present
            suffix += 1
            candidate = string(prefix, suffix)
        end
        return candidate
    elseif sample isa Symbol
        prefix = "__causaltargeted_population__"
        candidate = Symbol(prefix)
        suffix = 0
        while candidate in present
            suffix += 1
            candidate = Symbol(prefix, suffix)
        end
        return candidate
    end

    throw(ArgumentError(
        "unsupported ID element type $(typeof(sample)) for population-level prediction; " *
        "supported types are Integer, AbstractFloat, AbstractString, and Symbol",
    ))
end

function _formula_variables(model)
    return Set(Symbol.(MixedModels.StatsModels.termvars(formula(model))))
end

function _validate_model(
    model::SupportedMixedModel,
    data::DataFrame,
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    adjustment,
    strata,
)
    response = Symbol(responsename(model))
    response == outcome || throw(ArgumentError(
        "outcome :$outcome does not match the fitted model response :$response",
    ))

    grouping = unique(fnames(model))
    grouping == [id] || grouping == (id,) || throw(ArgumentError(
        "population standardization currently requires :$id to be the model's only " *
        "grouping factor; fitted grouping factors are $(collect(grouping))",
    ))

    variables = _formula_variables(model)
    required = unique([
        outcome, treatment, time, id, Symbol.(adjustment)..., Symbol.(strata)...,
    ])
    omitted = setdiff(required, variables)
    isempty(omitted) || throw(ArgumentError(
        "the fitted MixedModels formula omits required variables $(collect(omitted)); " *
        "graph-derived adjustment variables must be included explicitly",
    ))
    _validate_columns(data, required)
    _validate_columns(data, variables)
    _validate_complete_finite(data, variables)
    return nothing
end

function _fixed_effect_design(model::SupportedMixedModel, prediction_data::DataFrame)
    model_names = String.(coefnames(model))
    rhs = formula(model).rhs
    terms = rhs isa Tuple ? rhs : (rhs,)
    matches = Tuple{Any, Vector{String}}[]
    encountered = Vector{String}[]

    for term in terms
        names = try
            raw_names = MixedModels.StatsModels.coefnames(term)
            raw_names isa AbstractString ? [String(raw_names)] : String.(raw_names)
        catch
            continue
        end
        push!(encountered, names)
        names == model_names && push!(matches, (term, names))
    end

    length(matches) == 1 || throw(ArgumentError(
        "could not uniquely align fitted coefficient names $model_names with the " *
        "fixed-effect formula terms $(encountered); refusing covariance calculation",
    ))
    fixed_term = only(matches)[1]
    design = try
        Matrix{Float64}(MixedModels.StatsModels.modelcols(fixed_term, prediction_data))
    catch error
        throw(ArgumentError("could not construct the fixed-effect prediction matrix: $error"))
    end

    size(design) == (nrow(prediction_data), length(model_names)) || throw(ArgumentError(
        "fixed-effect design has size $(size(design)), but prediction data and fitted " *
        "coefficients require ($(nrow(prediction_data)), $(length(model_names)))",
    ))
    all(isfinite, design) || throw(ArgumentError(
        "fixed-effect prediction matrix contains non-finite values",
    ))
    return design
end

function _validate_design_predictions(model, design, predictions; scale_name="prediction")
    all(x -> x isa Real && isfinite(x), predictions) || throw(ArgumentError(
        "MixedModels returned missing or non-finite $scale_name values",
    ))
    coefficients = Float64.(coef(model))
    length(coefficients) == size(design, 2) || throw(ArgumentError(
        "fixed-effect coefficient and design-matrix dimensions do not align",
    ))
    all(isfinite, coefficients) || throw(ArgumentError(
        "fitted fixed-effect coefficients contain non-finite values",
    ))
    design_predictions = design * coefficients
    scale = max(1.0, maximum(abs, predictions), maximum(abs, design_predictions))
    tolerance = 512eps(Float64) * scale
    all(abs.(design_predictions .- predictions) .<= tolerance) || throw(ArgumentError(
        "fixed-effect design columns do not reproduce MixedModels $scale_name values; " *
        "refusing covariance calculation because coefficient alignment is uncertain",
    ))
    return nothing
end


function _prediction_components(
    model::LinearMixedModel,
    prediction_data::DataFrame,
    design::Matrix{Float64},
    random_effects::Symbol,
)
    predictions = predict(model, prediction_data; new_re_levels = :population)
    _validate_design_predictions(
        model, design, predictions; scale_name = "population predictions",
    )
    return Float64.(predictions), false
end

function _random_intercept_variance(model::GeneralizedLinearMixedModel)
    variance_components = VarCorr(model).σρ
    length(variance_components) == 1 || throw(ArgumentError(
        "random_effects=:marginal currently requires exactly one random-intercept " *
        "grouping factor",
    ))
    component = only(values(variance_components))
    standard_deviations = component.σ
    names = propertynames(standard_deviations)
    names == (Symbol("(Intercept)"),) || throw(ArgumentError(
        "random_effects=:marginal currently supports a single random intercept only; " *
        "fitted random-effect columns are $(collect(names))",
    ))
    standard_deviation = Float64(only(values(standard_deviations)))
    isfinite(standard_deviation) && standard_deviation >= 0 || throw(ArgumentError(
        "fitted random-intercept standard deviation is invalid: $standard_deviation",
    ))
    return standard_deviation^2
end

function _prediction_components(
    model::GeneralizedLinearMixedModel,
    prediction_data::DataFrame,
    design::Matrix{Float64},
    random_effects::Symbol,
)
    eta = predict(
        model,
        prediction_data;
        new_re_levels = :population,
        type = :linpred,
    )
    _validate_design_predictions(model, design, eta; scale_name = "link-scale predictions")

    response = predict(
        model,
        prediction_data;
        new_re_levels = :population,
        type = :response,
    )
    all(x -> x isa Real && isfinite(x), response) || throw(ArgumentError(
        "MixedModels returned missing or non-finite response-scale predictions",
    ))
    expected_response = exp.(Float64.(eta))
    scale = max(1.0, maximum(abs, response), maximum(abs, expected_response))
    tolerance = 512eps(Float64) * scale
    all(abs.(response .- expected_response) .<= tolerance) || throw(ArgumentError(
        "MixedModels response-scale predictions do not equal exp(link-scale prediction) " *
        "for the validated LogLink model",
    ))

    predictions = Float64.(response)
    if random_effects == :marginal
        variance = _random_intercept_variance(model)
        predictions .*= exp(0.5variance)
    end
    return predictions, true
end

function _sorted_unique(values, column::Symbol)
    result = unique(values)
    try
        sort!(result)
    catch error
        throw(ArgumentError(
            "observed values in :$column must have a stable sort order: $error",
        ))
    end
    return result
end

function _stratum_identifier(row, strata)
    length(strata) == 1 && return row[only(strata)]
    names = Tuple(strata)
    return NamedTuple{names}(Tuple(row[column] for column in strata))
end

function _stratum_groups(data::DataFrame, strata)
    isempty(strata) && return (Any[nothing], [collect(1:nrow(data))])
    identifiers = [_stratum_identifier(row, strata) for row in eachrow(data)]
    levels = unique(identifiers)
    try
        sort!(levels; by = x -> x isa NamedTuple ? Tuple(x) : x)
    catch error
        throw(ArgumentError("stratum levels must have a stable sort order: $error"))
    end
    groups = [findall(x -> isequal(x, level), identifiers) for level in levels]
    return levels, groups
end

function _mixed_g_computation(
    model::SupportedMixedModel,
    data::DataFrame;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    adjustment = Symbol[],
    strata = nothing,
    random_effects = :zero,
    compute_uncertainty::Bool = true,
)
    nrow(data) > 0 || throw(ArgumentError("data must contain at least one row"))
    intervention_values = _validate_values(values)
    random_effects_mode = _validate_random_effects(random_effects)
    model isa GeneralizedLinearMixedModel && _validate_glmm_family_link(model)
    strata_columns = _normalize_strata(strata)
    reserved = Set((treatment, outcome, time, id))
    invalid_strata = [column for column in strata_columns if column in reserved]
    isempty(invalid_strata) || throw(ArgumentError(
        "stratification columns must differ from treatment, outcome, time, and id; " *
        "invalid columns are $invalid_strata",
    ))
    _validate_columns(data, (treatment, outcome, time, id))
    _validate_columns(data, strata_columns)
    _validate_complete_finite(data, (treatment, outcome, time, id))
    _validate_complete_finite(data, strata_columns)
    _validate_static_treatment(data, treatment, id)
    _validate_model(
        model, data, treatment, outcome, time, id, adjustment, strata_columns,
    )
    if model isa GeneralizedLinearMixedModel && random_effects_mode == :marginal
        _random_intercept_variance(model)
    end

    times = _sorted_unique(data[!, time], time)
    levels, groups = _stratum_groups(data, strata_columns)
    fit = _mixed_adapter(model, data; id)
    specs = NamedTuple[]
    for (group_index, indices) in pairs(groups), t in times
        selected = [i for i in indices if isequal(data[i, time], t)]
        isempty(selected) && throw(ArgumentError(
            "every requested stratum must contain at least one target-population row " *
            "at time $(repr(t))",
        ))
        level = levels[group_index]
        selector = row -> isequal(row[time], t) &&
            (isempty(strata_columns) ||
             isequal(_stratum_identifier(row, strata_columns), level))
        for value in intervention_values
            push!(specs, (;
                set = CausalTargeted._gcomp_setting(treatment, value),
                by = nothing,
                subset = selector,
            ))
        end
    end
    components = CausalTargeted._gcomp_standardise(
        fit, data, specs; random_effects = random_effects_mode,
    )
    results = map(eachindex(groups)) do group_index
        reference = Float64[]
        comparison = Float64[]
        effect_gradients = Vector{Float64}[]
        for time_index in eachindex(times)
            index = 2 * ((group_index - 1) * length(times) + time_index) - 1
            reference_component = components[index]
            comparison_component = components[index + 1]
            contrast = CausalTargeted._gcomp_contrast_from_components(
                reference_component, comparison_component, :difference,
            )
            push!(reference, reference_component.estimate)
            push!(comparison, comparison_component.estimate)
            push!(effect_gradients, contrast.gradient)
        end
        G = reduce(vcat, transpose.(effect_gradients))
        effect_vcov = compute_uncertainty ?
            CausalTargeted._gcomp_joint_covariance(fit, G) : nothing
        effect_se = if effect_vcov === nothing
            nothing
        else
            diagonal = diag(effect_vcov)
            tolerance = 8192eps(Float64) * max(1.0, maximum(abs, effect_vcov))
            minimum(diagonal) >= -tolerance || throw(ArgumentError(
                "effect covariance has a negative diagonal beyond numerical tolerance",
            ))
            sqrt.(max.(diagonal, 0.0))
        end
        MixedGComputationResult(;
            treatment,
            outcome,
            time,
            id,
            times,
            values = intervention_values,
            mean_reference = reference,
            mean_comparison = comparison,
            adjustment,
            vcov = effect_vcov,
            se = effect_se,
            random_effects = random_effects_mode,
        )
    end

    isempty(strata_columns) && return only(results)
    return StratifiedMixedGComputationResult(strata_columns, levels, results)
end

function CausalTargeted.mixed_g_computation(
    model::SupportedMixedModel,
    data::DataFrame;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    strata = nothing,
    random_effects = :zero,
)
    return _mixed_g_computation(
        model,
        data;
        treatment,
        outcome,
        time,
        id,
        values,
        strata,
        random_effects,
    )
end

function _identified_adjustment(g, treatment, outcome, node_names)
    result = identify(g, TotalEffectQuery(treatment, outcome); node_names)
    result.identifiable || throw(ArgumentError(
        "the effect of :$treatment on :$outcome is not backdoor-identifiable in the graph",
    ))
    all(x -> x isa Symbol, result.adjustment) || throw(ArgumentError(
        "graph adjustment variables must resolve to data-column symbols; provide node_names",
    ))
    return Symbol.(result.adjustment)
end

function CausalTargeted.mixed_g_computation(
    g::AbstractGraph,
    model::SupportedMixedModel,
    data::DataFrame;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    strata = nothing,
    random_effects = :zero,
    node_names = nothing,
)
    node_names === nothing && throw(ArgumentError(
        "node_names is required when graph nodes are addressed by data-column symbols",
    ))
    adjustment = _identified_adjustment(g, treatment, outcome, node_names)
    return _mixed_g_computation(
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
end

function CausalTargeted.mixed_g_computation(
    g::CausalGraph,
    model::SupportedMixedModel,
    data::DataFrame;
    treatment::Symbol,
    outcome::Symbol,
    time::Symbol,
    id::Symbol,
    values = (0, 1),
    strata = nothing,
    random_effects = :zero,
    node_names = nothing,
)
    names = node_names === nothing ? get_node_names(g) : node_names
    isempty(names) && throw(ArgumentError(
        "CausalGraph has no node names; set node :name properties or pass node_names",
    ))
    adjustment = _identified_adjustment(g, treatment, outcome, names)
    return _mixed_g_computation(
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
end

include("mixedmodels_nb2.jl")
include("mmrm.jl")

end
