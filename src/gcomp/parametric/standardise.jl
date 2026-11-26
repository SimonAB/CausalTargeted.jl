# ---------------------------------------------------------------------------
# Counterfactual design construction, standardisation & delta-method inference
# ---------------------------------------------------------------------------

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

"""Predict each regime in the fitted parameter basis; backends may batch regimes."""
function _gcomp_predictions_gradients(
    fit::ParametricGComputationFit,
    counterfactuals::Vector{DataFrame};
    random_effects = nothing,
)
    random_effects === nothing || throw(ArgumentError(
        "random_effects applies only to the mixed-model backend",
    ))
    coefficients = Float64.(coef(fit.model))
    return map(counterfactuals) do counterfactual
        design = _gcomp_design(fit, counterfactual)
        eta = design * coefficients
        predictions = _gcomp_inverse_link(fit.link, eta)
        derivative = fit.link == :identity ? ones(length(predictions)) :
            fit.link == :log ? predictions : predictions .* (1 .- predictions)
        gradients = design .* reshape(derivative, :, 1)
        (; predictions, gradients)
    end
end

_gcomp_required_columns(fit::ParametricGComputationFit) =
    unique(Symbol.(StatsModels.termvars(StatsModels.formula(fit.model).rhs)))
_gcomp_protected_columns(::ParametricGComputationFit) = Symbol[]
_gcomp_parameter_covariance(fit::ParametricGComputationFit) = fit.covariance
_gcomp_result_metadata(fit::ParametricGComputationFit, random_effects) =
    (; covariance_type = fit.covariance_type)

function _gcomp_check_required_columns(data::AbstractDataFrame, columns; context)
    for column in columns
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

"""Select, intervene, predict, and average several regimes with one fitted model."""
function _gcomp_standardise(
    fit::AbstractParametricGComputationFit,
    data::AbstractDataFrame,
    specs::AbstractVector;
    random_effects = nothing,
)
    isempty(specs) && throw(ArgumentError("at least one counterfactual regime is required"))
    counterfactuals = DataFrame[]
    selected = NamedTuple[]
    protected = _gcomp_protected_columns(fit)
    for spec in specs
        spec isa NamedTuple || throw(ArgumentError("each counterfactual regime must be a NamedTuple"))
        setting = get(spec, :set, NamedTuple())
        by = get(spec, :by, nothing)
        subset = get(spec, :subset, nothing)
        setting isa NamedTuple || throw(ArgumentError("set must be a NamedTuple"))
        for column in keys(setting)
            column in protected && throw(ArgumentError(
                "intervention cannot change structural identifier :$column",
            ))
        end
        target = _gcomp_target_rows(data; by, subset)
        _gcomp_check_required_columns(target, _gcomp_required_columns(fit); context = "target data")
        push!(counterfactuals, _gcomp_apply_set!(copy(target), setting))
        push!(selected, (; n = nrow(target), set = setting, by,
            subset = subset === nothing ? nothing : :function))
    end
    outputs = _gcomp_predictions_gradients(fit, counterfactuals; random_effects)
    length(outputs) == length(specs) || throw(ArgumentError(
        "prediction backend returned the wrong number of counterfactual regimes",
    ))
    return map(eachindex(outputs)) do i
        predictions = outputs[i].predictions
        gradients = outputs[i].gradients
        n = selected[i].n
        length(predictions) == n && size(gradients, 1) == n || throw(ArgumentError(
            "prediction backend returned predictions or gradients with incorrect row count",
        ))
        all(isfinite, predictions) || throw(ArgumentError(
            "counterfactual response-scale predictions contain non-finite values",
        ))
        all(isfinite, gradients) || throw(ArgumentError(
            "counterfactual response-scale gradients contain non-finite values",
        ))
        (; estimate = mean(predictions),
            gradient = vec(mean(gradients; dims = 1)), selected[i]...)
    end
end

function _gcomp_mean_components(
    fit::AbstractParametricGComputationFit,
    data::AbstractDataFrame;
    set = NamedTuple(), by = nothing, subset = nothing, random_effects = nothing,
)
    return only(_gcomp_standardise(fit, data, [(; set, by, subset)]; random_effects))
end

function _gcomp_joint_covariance(fit::AbstractParametricGComputationFit, G::AbstractMatrix)
    covariance = _gcomp_parameter_covariance(fit)
    covariance === nothing && return nothing
    size(G, 2) == size(covariance, 1) == size(covariance, 2) || throw(ArgumentError(
        "contrast gradients do not align with the fitted parameter covariance",
    ))
    all(isfinite, G) || throw(ArgumentError("contrast gradients contain non-finite values"))
    all(isfinite, covariance) || throw(ArgumentError(
        "fitted parameter covariance contains non-finite values",
    ))
    raw = Matrix{Float64}(G * covariance * transpose(G))
    joint = (raw .+ transpose(raw)) ./ 2
    all(isfinite, joint) || throw(ArgumentError("joint contrast covariance contains non-finite values"))
    tolerance = 1024eps(Float64) * max(1.0, opnorm(covariance)) *
        max(1.0, maximum(vec(sum(abs2, G; dims = 2))))
    for i in axes(joint, 1)
        joint[i, i] >= -tolerance || throw(ArgumentError(
            "joint contrast variance is negative at index $i: $(joint[i, i])",
        ))
        joint[i, i] = max(0.0, joint[i, i])
    end
    return joint
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
