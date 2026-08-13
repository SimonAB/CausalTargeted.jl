"""Fitted, learner-agnostic encoding for tabular adjustment covariates."""

using DataFrames
using StatsModels

"""
    CovariateSchema

Fitted representation of an ordered covariate list. `terms` are StatsModels
terms with categorical levels and dummy contrasts learned from the schema-fit
data. The schema defines columns only; it does not impute, standardise, select,
or otherwise transform features.
"""
struct CovariateSchema
    covariates::Vector{Symbol}
    terms::Tuple
    feature_names::Vector{String}
end

function _missing_columns(df::AbstractDataFrame, columns::AbstractVector{Symbol})
    return [column for column in columns if !hasproperty(df, column)]
end

function _validate_requested_columns(df::AbstractDataFrame, columns::AbstractVector{Symbol})
    missing_columns = _missing_columns(df, columns)
    isempty(missing_columns) || throw(ArgumentError(
        "missing requested column(s): $(join(string.(missing_columns), ", "))",
    ))
    return nothing
end

function _validate_covariate_values(df::AbstractDataFrame, covariates::Vector{Symbol})
    for covariate in covariates
        column = df[!, covariate]
        any(ismissing, column) && throw(ArgumentError(
            "covariate :$covariate contains missing values; apply the package's " *
            "existing missing-data handling (:drop, :impute, :ipcw, or :ipcw_impute) " *
            "before fitting or transforming a CovariateSchema",
        ))
    end
    return nothing
end

function _flatten_coefnames(terms::Tuple)
    names = String[]
    for term in terms
        term === nothing && continue
        term_names = coefnames(term)
        if term_names isa AbstractString
            push!(names, String(term_names))
        else
            append!(names, String.(term_names))
        end
    end
    return names
end

function _fit_covariate_term(df::AbstractDataFrame, covariate::Symbol)
    column = df[!, covariate]
    value_type = Base.nonmissingtype(eltype(column))
    if value_type <: Real
        return concrete_term(term(covariate), column)
    end

    is_string = value_type <: AbstractString
    is_explicit_categorical = StatsModels.DataAPI.refpool(column) !== nothing
    (is_string || is_explicit_categorical) || throw(ArgumentError(
        "Unsupported covariate type for :$covariate: $value_type. " *
        "Convert it to an appropriate numeric or categorical representation before fitting.",
    ))

    # DataAPI-backed `levels` (used by StatsModels) returns the full ordered
    # pool for CategoricalArray columns and deterministic observed levels for
    # ordinary strings. Constructing the documented ContrastsMatrix directly
    # avoids StatsModels dropping unused explicit levels during schema fitting.
    categorical_levels = collect(StatsModels.levels(column))
    isempty(categorical_levels) && throw(ArgumentError(
        "categorical covariate :$covariate has no observed or declared levels",
    ))
    length(categorical_levels) == 1 && return nothing
    contrasts = StatsModels.ContrastsMatrix(DummyCoding(), categorical_levels)
    return CategoricalTerm(covariate, contrasts)
end

"""
    fit_covariate_schema(df, covariates) -> CovariateSchema

Fit StatsModels terms for the requested covariates in their supplied order.
Numeric and Boolean columns remain one column each. String and categorical
columns use StatsModels' default `DummyCoding`, retaining the fitted levels and
column order for later subsets or prediction data.
"""
function fit_covariate_schema(
    df::AbstractDataFrame,
    covariates::AbstractVector{Symbol},
)
    requested = collect(Symbol, covariates)
    length(unique(requested)) == length(requested) || throw(ArgumentError(
        "covariates must be unique; received $(repr(requested))",
    ))
    _validate_requested_columns(df, requested)
    _validate_covariate_values(df, requested)
    isempty(requested) && return CovariateSchema(requested, (), String[])

    fitted_terms = try
        Tuple(_fit_covariate_term(df, covariate) for covariate in requested)
    catch error
        error isa ArgumentError && rethrow()
        throw(ArgumentError(
            "could not fit a covariate schema for $(repr(requested)); " *
            "supported columns are real numeric, Bool, String, and categorical values. " *
            "StatsModels reported: $(sprint(showerror, error))",
        ))
    end
    return CovariateSchema(requested, fitted_terms, _flatten_coefnames(fitted_terms))
end

function _numeric_model_column(column, covariate::Symbol, n::Int)
    values = if column isa AbstractVector
        reshape(column, n, 1)
    elseif column isa AbstractMatrix
        column
    else
        throw(ArgumentError(
            "unsupported model-column representation $(typeof(column)) for covariate :$covariate",
        ))
    end
    try
        return Matrix{Float64}(values)
    catch error
        throw(ArgumentError(
            "covariate :$covariate could not be converted to numeric model columns: " *
            sprint(showerror, error),
        ))
    end
end

"""
    transform_covariates(schema, df) -> Matrix{Float64}

Encode `df` with the exact fitted terms, levels, contrasts, feature order, and
matrix width stored in `schema`. Unseen levels produce an informative error.
"""
function transform_covariates(schema::CovariateSchema, df::AbstractDataFrame)
    _validate_requested_columns(df, schema.covariates)
    _validate_covariate_values(df, schema.covariates)
    n = nrow(df)
    isempty(schema.covariates) && return zeros(n, 0)

    fitted_columns = try
        map(schema.terms, schema.covariates) do fitted_term, covariate
            fitted_term === nothing ? zeros(n, 0) : modelcols(fitted_term, df)
        end
    catch error
        throw(ArgumentError(
            "could not transform covariates with the fitted schema; data may contain " *
            "an unseen categorical level or an incompatible column type. StatsModels reported: " *
            sprint(showerror, error),
        ))
    end
    matrices = map(
        (column, covariate) -> _numeric_model_column(column, covariate, n),
        fitted_columns,
        schema.covariates,
    )
    encoded = reduce(hcat, matrices)
    size(encoded, 2) == length(schema.feature_names) || error(
        "internal covariate schema width mismatch: expected $(length(schema.feature_names)), " *
        "got $(size(encoded, 2))",
    )
    return Matrix{Float64}(encoded)
end
