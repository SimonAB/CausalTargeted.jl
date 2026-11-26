# ---------------------------------------------------------------------------
# Parametric g-computation fit types, validation, HC3 covariance & model fitting
# ---------------------------------------------------------------------------

const _GCOMP_NORMAL_QUANTILE = 1.959963984540054

"""Abstract supertype for parametric g-computation fits."""
abstract type AbstractParametricGComputationFit end

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
struct ParametricGComputationFit{M,F} <: AbstractParametricGComputationFit
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
    mu = Float64.(GLM.predict(model))
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
    fit_parametric_gcomp(formula, data; backend=:glm, family=:gaussian,
                         link=nothing, theta=nothing, covariance=nothing, kwargs...)

Fit a parametric outcome model for empirical-distribution g-computation,
separate from cross-fitted `run_gcomp` and LMTP estimators. The default GLM
backend supports Gaussian/identity, binomial/logit, Gamma/log, and
negative-binomial/log (`:negbin`; aliases `:negativebinomial`,
`:negative_binomial`, and `:nb` are normalised to `:negbin`). For GLM negative
binomial, omit `theta` to estimate the NB2 shape continuously with `GLM.negbin`,
or provide it for a fixed-shape fit. GLM coefficient covariance defaults to
`:hc3`.

NB2 HC3 covariance and delta-method SEs/intervals treat `theta` as fixed at its
supplied or estimated value; uncertainty in estimated `theta` is not propagated.
Refit bootstraps re-estimate `theta` only when it was estimated in the original fit.

All outcome and predictor columns referenced by the formula must be present and
contain no `missing` values. Handle missingness explicitly before fitting; this
path never silently drops rows. Missing values in unrelated columns are allowed.
Set `backend=:mixed` and supply `id` to use a supported Gaussian LMM or NB2
random-intercept model when the optional extension is loaded. A supplied NB2
`theta` selects the fixed-shape fit; omitting it selects the dedicated
estimated-shape fitter. Mixed-model coefficient covariance defaults to `:model`.
For nonlinear mixed predictions, pass `random_effects=:zero` or `:marginal` to
the subsequent g-computation call.
"""
function fit_parametric_gcomp(
    formula_term::StatsModels.FormulaTerm,
    data::AbstractDataFrame;
    backend::Symbol = :glm,
    kwargs...,
)
    return _fit_parametric_gcomp(Val(backend), formula_term, data; kwargs...)
end

function _fit_parametric_gcomp(
    ::Val{:glm},
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

function _fit_parametric_gcomp(
    ::Val{B}, formula_term::StatsModels.FormulaTerm, data::AbstractDataFrame; kwargs...,
) where B
    B == :mixed && throw(ArgumentError(
        "backend=:mixed requires the CausalTargeted MixedModels extension; " *
        "load its documented optional dependencies before fitting",
    ))
    throw(ArgumentError("unsupported parametric g-computation backend :$B; use :glm or :mixed"))
end
