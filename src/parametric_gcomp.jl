"""Formula-based parametric regression standardisation.

This file deliberately does not share the numeric-treatment matrix path used by
`run_gcomp`.  A fitted StatsModels formula is the schema for prediction, so a
categorical intervention rebuilds every main-effect and interaction column that
depends on the intervened variable.
"""

using DataFrames
using Distributions
using GLM
using LinearAlgebra
using Random
using Statistics
using StatsModels

const _GCOMP_NORMAL_QUANTILE = 1.959963984540054

"""
    ParametricGComputationFit

A fitted parametric reference / complete-case GLM for empirical-distribution
standardisation, separate from cross-fitted `run_gcomp` and LMTP estimators.
`covariance` is coefficient covariance (HC3 by default). For NB2 fits, `theta`
records the shape and `family` stores the canonical symbol `:negbin`.
NB2 HC3 covariance and delta-method intervals treat `theta` as fixed at its
supplied or estimated value; they do not propagate uncertainty in its estimation.
The wrapper also retains the unapplied formula needed to refit a non-parametric
bootstrap, which re-estimates `theta` when `estimated_theta` is true.
"""
struct ParametricGComputationFit{M,F}
    model::M
    formula_spec::F
    outcome::Symbol
    family::Symbol
    link::Symbol
    theta::Union{Nothing,Float64}
    estimated_theta::Bool
    covariance::Union{Nothing,Matrix{Float64}}
    covariance_type::Symbol
end

function _gcomp_family_symbol(family)
    family isa Symbol && return if family in (:gaussian, :normal)
        :gaussian
    elseif family in (:binomial, :logistic)
        :binomial
    elseif family == :gamma
        :gamma
    elseif family in (:negativebinomial, :negative_binomial, :negbin, :nb)
        :negbin
    else
        throw(ArgumentError(
            "unsupported parametric g-computation family $(repr(family)); " *
            "use :gaussian, :binomial, :gamma, or :negbin",
        ))
    end
    family isa Normal && return :gaussian
    family isa Binomial && return :binomial
    family isa Gamma && return :gamma
    family isa NegativeBinomial && return :negbin
    throw(ArgumentError(
        "unsupported parametric g-computation family $(typeof(family)); " *
        "use a supported family symbol or Distributions family instance",
    ))
end

function _gcomp_expected_link(family::Symbol)
    family == :gaussian && return :identity
    family == :binomial && return :logit
    family in (:gamma, :negbin) && return :log
    error("internal unsupported g-computation family $family")
end

function _gcomp_validate_link(family::Symbol, link)
    expected = _gcomp_expected_link(family)
    link === nothing && return expected
    actual = if link isa Symbol
        link
    elseif link isa IdentityLink
        :identity
    elseif link isa LogitLink
        :logit
    elseif link isa LogLink
        :log
    else
        throw(ArgumentError("unsupported link $(typeof(link)) for parametric g-computation"))
    end
    actual == expected || throw(ArgumentError(
        "family :$family requires the $expected link for this g-computation backend; " *
        "received $(repr(link))",
    ))
    return actual
end

function _gcomp_response_name(formula_term)
    variables = Symbol.(StatsModels.termvars(formula_term.lhs))
    length(variables) == 1 || throw(ArgumentError(
        "the outcome side of the formula must name exactly one data column",
    ))
    return only(variables)
end

function _gcomp_underlying(model)
    hasproperty(model, :model) || throw(ArgumentError(
        "parametric g-computation requires a formula-fitted GLM/StatsModels model",
    ))
    return getproperty(model, :model)
end

function _gcomp_model_family_link(model)
    inner = _gcomp_underlying(model)
    if inner isa GLM.LinearModel
        return (:gaussian, :identity, nothing)
    end
    inner isa GLM.GeneralizedLinearModel || throw(ArgumentError(
        "unsupported fitted model $(typeof(inner)); expected a GLM linear or generalized linear model",
    ))
    # `Distributions.Distribution(inner)` returns the distribution *type* in
    # GLM 1.9; the fitted response stores the actual instance (including NB θ).
    distribution = inner.rr.d
    link = GLM.Link(inner)
    family = _gcomp_family_symbol(distribution)
    link_symbol = _gcomp_validate_link(family, link)
    theta = distribution isa NegativeBinomial ? Float64(distribution.r) : nothing
    return family, link_symbol, theta
end

function _gcomp_check_complete_cases(data::AbstractDataFrame, terms; context)
    for column in unique(Symbol.(StatsModels.termvars(terms)))
        hasproperty(data, column) || throw(ArgumentError(
            "$context is missing required formula column :$column",
        ))
        any(ismissing, data[!, column]) && throw(ArgumentError(
            "$context must be complete-case in formula columns; :$column contains missing values. " *
            "Handle missingness explicitly before calling parametric g-computation " *
            "(for example, use dropmissing on the required columns); rows are not dropped automatically.",
        ))
    end
    return nothing
end

function _gcomp_check_training_model(model, data::AbstractDataFrame)
    _gcomp_check_complete_cases(data, StatsModels.formula(model); context = "training data")
    nrow(data) > 0 || throw(ArgumentError("parametric g-computation data must contain rows"))
    size(modelmatrix(model), 1) == nrow(data) || throw(ArgumentError(
        "training data have $(nrow(data)) rows but the fitted model used " *
        "$(size(modelmatrix(model), 1)); supply the exact complete-case fitting data",
    ))
    coefficients = Float64.(coef(model))
    all(isfinite, coefficients) || throw(ArgumentError(
        "fitted coefficients are non-finite, usually because the formula matrix is rank deficient",
    ))
    inner = _gcomp_underlying(model)
    if inner isa GLM.GeneralizedLinearModel && !isempty(inner.rr.offset)
        throw(ArgumentError("offset models are not yet supported by parametric g-computation"))
    end
    return nothing
end

function _gcomp_irls_components(family, y, mu, theta, prior_weights)
    if family == :gaussian
        return prior_weights, prior_weights .* (y .- mu)
    elseif family == :binomial
        w = prior_weights .* mu .* (1 .- mu)
        return w, prior_weights .* (y .- mu)
    elseif family == :gamma
        return prior_weights, prior_weights .* (y .- mu) ./ mu
    elseif family == :negbin
        theta === nothing && error("internal NB2 fit lacks a shape parameter")
        denominator = theta .+ mu
        return prior_weights .* theta .* mu ./ denominator,
            prior_weights .* theta .* (y .- mu) ./ denominator
    end
    error("internal unsupported family $family")
end

"""Coefficient HC3 covariance matching the estimating-equation GLM sandwich."""
function _gcomp_hc3_covariance(model, data, family, theta)
    _gcomp_check_training_model(model, data)
    X = Matrix{Float64}(modelmatrix(model))
    y = Float64.(response(model))
    mu = Float64.(predict(model))
    size(X, 1) == length(y) == length(mu) || error("internal fitted-model dimension mismatch")
    all(isfinite, mu) || throw(ArgumentError("fitted response means are non-finite"))
    family in (:gamma, :negbin) && any(x -> x <= 0, mu) &&
        throw(ArgumentError("log-link fitted response means must be positive"))

    inner = _gcomp_underlying(model)
    stored_weights = inner isa GLM.GeneralizedLinearModel ? inner.rr.wts : Float64[]
    prior_weights = isempty(stored_weights) ? ones(length(y)) : Float64.(stored_weights)
    length(prior_weights) == length(y) || throw(ArgumentError(
        "fitted prior weights do not align with the model response",
    ))
    weights, scores = _gcomp_irls_components(family, y, mu, theta, prior_weights)
    all(x -> isfinite(x) && x > 0, weights) || throw(ArgumentError(
        "HC3 is unavailable because an IRLS weight is non-positive or non-finite",
    ))
    weighted_X = X .* sqrt.(weights)
    gram = transpose(weighted_X) * weighted_X
    factor = try
        cholesky(Symmetric(gram); check = true)
    catch error
        throw(ArgumentError(
            "HC3 is unavailable because the fitted formula matrix is rank deficient: " *
            sprint(showerror, error),
        ))
    end
    bread = Matrix(factor \ Matrix{Float64}(I, size(gram)...))
    leverage = weights .* vec(sum((X * bread) .* X; dims = 2))
    any(h -> !isfinite(h) || h >= 1 - sqrt(eps(Float64)), leverage) &&
        throw(ArgumentError(
            "HC3 is unstable because at least one leverage value is effectively one; " *
            "use covariance=:model or add replication",
        ))
    adjusted_scores = scores ./ (1 .- leverage)
    meat = transpose(X) * (X .* reshape(abs2.(adjusted_scores), :, 1))
    covariance = bread * meat * bread
    covariance = Matrix(Symmetric((covariance .+ transpose(covariance)) ./ 2))
    all(isfinite, covariance) || throw(ArgumentError("HC3 covariance contains non-finite values"))
    return covariance
end

function _gcomp_covariance(model, data, family, theta, covariance::Symbol)
    covariance == :none && return nothing, :none
    covariance == :model && return Matrix{Float64}(vcov(model)), :model
    covariance == :hc3 || throw(ArgumentError(
        "covariance must be :hc3, :model, or :none; received $(repr(covariance))",
    ))
    return _gcomp_hc3_covariance(model, data, family, theta), :hc3
end

function _gcomp_wrap_model(
    model,
    data::AbstractDataFrame;
    formula_spec = nothing,
    estimated_theta::Bool = false,
    covariance::Symbol = :hc3,
)
    _gcomp_check_training_model(model, data)
    family, link, theta = _gcomp_model_family_link(model)
    covariance_matrix, covariance_type = _gcomp_covariance(
        model, data, family, theta, covariance,
    )
    fitted_formula = StatsModels.formula(model)
    return ParametricGComputationFit(
        model,
        formula_spec,
        _gcomp_response_name(fitted_formula),
        family,
        link,
        theta,
        estimated_theta,
        covariance_matrix,
        covariance_type,
    )
end

"""
    fit_parametric_gcomp(formula, data; family=:gaussian, link=nothing,
                         theta=nothing, covariance=:hc3, kwargs...)

Fit a parametric reference / complete-case GLM for empirical-distribution
g-computation, separate from cross-fitted `run_gcomp` and LMTP estimators. Supported
family/link combinations are Gaussian/identity, binomial/logit, Gamma/log, and
negative-binomial/log (`:negbin`; aliases `:negativebinomial`, `:negative_binomial`,
and `:nb` are normalised to `:negbin`). For negative binomial, omit `theta` to
estimate the NB2 shape continuously with `GLM.negbin`, or provide it for a
fixed-shape fit.

NB2 HC3 covariance and delta-method SEs/intervals treat `theta` as fixed at its
supplied or estimated value; uncertainty in estimated `theta` is not propagated.
Refit bootstraps re-estimate `theta` only when it was estimated in the original fit.

All outcome and predictor columns referenced by the formula must be present and
contain no `missing` values. Handle missingness explicitly before fitting; this
path never silently drops rows. Missing values in unrelated columns are allowed.
"""
function fit_parametric_gcomp(
    formula_term::StatsModels.FormulaTerm,
    data::AbstractDataFrame;
    family = :gaussian,
    link = nothing,
    theta::Union{Nothing,Real} = nothing,
    covariance::Symbol = :hc3,
    kwargs...,
)
    _gcomp_check_complete_cases(data, formula_term; context = "training data")
    family_symbol = _gcomp_family_symbol(family)
    _gcomp_validate_link(family_symbol, link)
    model = if family_symbol == :gaussian
        glm(formula_term, data, Normal(), IdentityLink(); kwargs...)
    elseif family_symbol == :binomial
        glm(formula_term, data, Binomial(), LogitLink(); kwargs...)
    elseif family_symbol == :gamma
        glm(formula_term, data, Gamma(), LogLink(); kwargs...)
    elseif theta === nothing
        negbin(formula_term, data, LogLink(); kwargs...)
    else
        theta > 0 || throw(ArgumentError("negative-binomial theta must be positive"))
        glm(formula_term, data, NegativeBinomial(Float64(theta)), LogLink(); kwargs...)
    end
    return _gcomp_wrap_model(
        model,
        data;
        formula_spec = formula_term,
        estimated_theta = family_symbol == :negbin && theta === nothing,
        covariance,
    )
end

function _gcomp_design(fit::ParametricGComputationFit, data::AbstractDataFrame)
    rhs = StatsModels.formula(fit.model).rhs
    _gcomp_check_complete_cases(data, rhs; context = "counterfactual data")
    raw = try
        StatsModels.modelcols(rhs, data)
    catch error
        throw(ArgumentError(
            "could not construct the counterfactual model matrix with the fitted " *
            "categorical schema; an intervention may use an unseen level. StatsModels reported: " *
            sprint(showerror, error),
        ))
    end
    design = try
        Matrix{Float64}(reshape(raw, nrow(data), :))
    catch error
        throw(ArgumentError("could not convert the counterfactual formula matrix: " * sprint(showerror, error)))
    end
    length(coef(fit.model)) == size(design, 2) || throw(ArgumentError(
        "counterfactual design has $(size(design, 2)) columns but the fitted model has " *
        "$(length(coef(fit.model))) coefficients",
    ))
    return design
end

function _gcomp_target_rows(data::AbstractDataFrame; by = nothing, subset = nothing)
    nrow(data) > 0 || throw(ArgumentError("the requested standardisation population is empty"))
    by !== nothing && subset !== nothing && throw(ArgumentError(
        "specify at most one of declarative `by` and callable `subset`",
    ))
    mask = trues(nrow(data))
    if by !== nothing
        by isa NamedTuple || throw(ArgumentError("by must be a NamedTuple such as (; Line=\"ROH\")"))
        for (column, value) in pairs(by)
            hasproperty(data, column) || throw(ArgumentError("subgroup column :$column is absent from data"))
            mask .&= isequal.(data[!, column], Ref(value))
        end
    elseif subset !== nothing
        applicable(subset, first(eachrow(data))) || throw(ArgumentError(
            "subset must be callable on a DataFrameRow",
        ))
        mask .= Bool[subset(row) for row in eachrow(data)]
    end
    target = DataFrame(data[mask, :])
    nrow(target) > 0 || throw(ArgumentError("the requested standardisation population is empty"))
    return target
end

function _gcomp_apply_set!(data::DataFrame, setting)
    setting isa NamedTuple || throw(ArgumentError(
        "set must be a NamedTuple such as (; Protein=\"HP\")",
    ))
    for (column, value) in pairs(setting)
        hasproperty(data, column) || throw(ArgumentError("intervention column :$column is absent from data"))
        try
            data[!, column] .= value
        catch error
            throw(ArgumentError(
                "intervention value $(repr(value)) is incompatible with column :$column: " *
                sprint(showerror, error),
            ))
        end
    end
    return data
end

_gcomp_setting(column::Symbol, value) = NamedTuple{(column,)}((value,))

function _gcomp_inverse_link(link::Symbol, eta)
    link == :identity && return eta
    link == :log && return exp.(eta)
    link == :logit && return 1.0 ./ (1.0 .+ exp.(-eta))
    error("internal unsupported link $link")
end

function _gcomp_mean_components(
    fit::ParametricGComputationFit,
    data::AbstractDataFrame;
    set = NamedTuple(),
    by = nothing,
    subset = nothing,
)
    target = _gcomp_target_rows(data; by, subset)
    _gcomp_check_complete_cases(
        target, StatsModels.formula(fit.model).rhs; context = "target data",
    )
    counterfactual = _gcomp_apply_set!(copy(target), set)
    design = _gcomp_design(fit, counterfactual)
    eta = design * Float64.(coef(fit.model))
    predictions = _gcomp_inverse_link(fit.link, eta)
    all(isfinite, predictions) || throw(ArgumentError(
        "counterfactual response-scale predictions contain non-finite values",
    ))
    estimate = mean(predictions)
    derivative = fit.link == :identity ? ones(length(predictions)) :
        fit.link == :log ? predictions : predictions .* (1 .- predictions)
    gradient = vec(mean(design .* reshape(derivative, :, 1); dims = 1))
    return (; estimate, gradient, n = nrow(target), set, by,
        subset = subset === nothing ? nothing : :function)
end

function _gcomp_wald(estimate, gradient, covariance; transform::Symbol = :identity)
    covariance === nothing && return (;
        se = nothing, ci_lower = nothing, ci_upper = nothing, p_value = nothing,
        log_se = nothing,
    )
    variance = dot(gradient, covariance * gradient)
    tolerance = 1024eps(Float64) * max(1.0, opnorm(covariance)) * max(1.0, sum(abs2, gradient))
    variance >= -tolerance || throw(ArgumentError("delta-method variance is negative: $variance"))
    se = sqrt(max(variance, 0.0))
    if transform == :exp
        log_estimate = log(estimate)
        lower = exp(log_estimate - _GCOMP_NORMAL_QUANTILE * se)
        upper = exp(log_estimate + _GCOMP_NORMAL_QUANTILE * se)
        p = se == 0 ? (log_estimate == 0 ? 1.0 : 0.0) :
            2ccdf(Normal(), abs(log_estimate / se))
        return (; se = estimate * se, ci_lower = lower, ci_upper = upper,
            p_value = p, log_se = se)
    end
    lower = estimate - _GCOMP_NORMAL_QUANTILE * se
    upper = estimate + _GCOMP_NORMAL_QUANTILE * se
    p = se == 0 ? (estimate == 0 ? 1.0 : 0.0) : 2ccdf(Normal(), abs(estimate / se))
    return (; se, ci_lower = lower, ci_upper = upper, p_value = p, log_se = nothing)
end

"""
    gcomp_mean(fit, data; set=(;), by=nothing, subset=nothing)

Standardise response-scale predictions over the empirical rows in `data`, or
over the target rows selected by `by`/`subset`. Target rows are restricted
before the intervention is applied.
The target can differ from the training table in both rows and size and need not
contain the outcome. Formula predictors in the selected target rows must be
complete-case before intervention. Missing values are rejected, not dropped.
For NB2 fits, HC3/delta inference treats the supplied or estimated `theta` as fixed.
"""
function gcomp_mean(
    fit::ParametricGComputationFit,
    data::AbstractDataFrame;
    set = NamedTuple(),
    by = nothing,
    subset = nothing,
)
    result = _gcomp_mean_components(fit, data; set, by, subset)
    inference = _gcomp_wald(result.estimate, result.gradient, fit.covariance)
    return (;
        estimate = result.estimate,
        n = result.n,
        set = result.set,
        target = (; by = result.by, subset = result.subset),
        inference...,
        covariance_type = fit.covariance_type,
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

function _gcomp_contrast_components(
    fit, data;
    treatment::Symbol,
    reference,
    comparison,
    by = nothing,
    subset = nothing,
    scale::Symbol = :difference,
)
    reference_component = _gcomp_mean_components(
        fit, data; set = _gcomp_setting(treatment, reference), by, subset,
    )
    comparison_component = _gcomp_mean_components(
        fit, data; set = _gcomp_setting(treatment, comparison), by, subset,
    )
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

"""
    gcomp_contrast(fit, data; treatment, reference, comparison,
                   by=nothing, subset=nothing, scale=:difference)

Compute a marginal or subgroup empirical standardised difference, response-mean
ratio, or log response-mean ratio. Ratio inference is performed on the log scale.
Target rows follow the complete-case predictor policy of `gcomp_mean`; no outcome
column is required. NB2 HC3/delta inference treats supplied or estimated `theta` as fixed.
"""
function gcomp_contrast(
    fit::ParametricGComputationFit,
    data::AbstractDataFrame;
    treatment::Symbol,
    reference,
    comparison,
    by = nothing,
    subset = nothing,
    scale::Symbol = :difference,
)
    result = _gcomp_contrast_components(
        fit, data; treatment, reference, comparison, by, subset, scale,
    )
    inference = _gcomp_wald(
        result.estimate, result.gradient, fit.covariance; transform = result.transform,
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
        covariance_type = fit.covariance_type,
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
    fit, data;
    treatment, reference, comparison,
    modifier, modifier_reference, modifier_comparison,
    scale,
)
    subgroup0 = _gcomp_contrast_components(
        fit, data;
        treatment, reference, comparison,
        by = _gcomp_setting(modifier, modifier_reference),
        scale = scale == :difference ? :difference : :ratio,
    )
    subgroup1 = _gcomp_contrast_components(
        fit, data;
        treatment, reference, comparison,
        by = _gcomp_setting(modifier, modifier_comparison),
        scale = scale == :difference ? :difference : :ratio,
    )
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

function _gcomp_interaction_point(result, fit, scale, modifier, modifier_reference, modifier_comparison)
    inference = _gcomp_wald(
        result.estimate, result.gradient, fit.covariance; transform = result.transform,
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
        covariance_type = fit.covariance_type,
    )
end

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

"""
    gcomp_interaction(fit, data; treatment, reference, comparison, modifier,
                      modifier_reference, modifier_comparison, scale=:difference)

Compute a difference-of-differences or a response-mean ratio-of-ratios. The
return value includes both component subgroup effects and all four standardised
means. Delta inference uses the single fitted coefficient covariance jointly.
For NB2 this treats supplied or estimated `theta` as fixed. Selected target rows
must have complete-case formula predictors; the outcome is needed only for refitting.
Set `n_boot > 0` for an additional stratified refit-bootstrap summary.
"""
function gcomp_interaction(
    fit::ParametricGComputationFit,
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
)
    n_boot >= 0 || throw(ArgumentError("n_boot must be non-negative"))
    result = _gcomp_interaction_components(
        fit, data;
        treatment, reference, comparison,
        modifier, modifier_reference, modifier_comparison,
        scale,
    )
    point = _gcomp_interaction_point(
        result, fit, scale, modifier, modifier_reference, modifier_comparison,
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
Fit a parametric reference / complete-case GLM and return a standardised contrast.
See `fit_parametric_gcomp` for the complete-case requirement and the NB2 HC3/delta
inference limitation: supplied or estimated `theta` is treated as fixed.
"""
function run_parametric_gcomp(
    formula_term::StatsModels.FormulaTerm,
    data::AbstractDataFrame;
    treatment::Symbol,
    reference,
    comparison,
    family = :gaussian,
    link = nothing,
    theta::Union{Nothing,Real} = nothing,
    covariance::Symbol = :hc3,
    by = nothing,
    subset = nothing,
    scale::Symbol = :difference,
    fit_kwargs = NamedTuple(),
)
    fit_kwargs isa NamedTuple || throw(ArgumentError("fit_kwargs must be a NamedTuple"))
    fit = fit_parametric_gcomp(
        formula_term, data; family, link, theta, covariance, fit_kwargs...,
    )
    return gcomp_contrast(
        fit, data; treatment, reference, comparison, by, subset, scale,
    )
end
