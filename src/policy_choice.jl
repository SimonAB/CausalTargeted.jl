"""Decision helper: choose among typed estimands by estimated value."""

using DataFrames
using CausalDynamics

"""
    PolicyChoice

Result of [`choose_policy`](@ref): selected label, value table, and optional certificate.
"""
struct PolicyChoice
    selected::Symbol
    values::DataFrame
    id_result::Union{Nothing, IdentificationResult}
end

"""
    choose_policy(candidates, data; id_result, sense=:max, kwargs...) -> PolicyChoice

Evaluate each labelled estimand via [`execute_estimand`](@ref) and pick the best
scalar TE (first row `est` column) by `sense` (`:max` or `:min`).

`candidates` is a vector of `label::Symbol => estimand` pairs (or a `Dict`).
"""
function choose_policy(
    candidates,
    data::DataFrame;
    id_result::Union{Nothing, IdentificationResult} = nothing,
    sense::Symbol = :max,
    kwargs...,
)
    sense in (:max, :min) || throw(ArgumentError("sense must be :max or :min; got :$sense"))
    pairs_iter = candidates isa AbstractDict ? collect(pairs(candidates)) : collect(candidates)
    isempty(pairs_iter) && throw(ArgumentError("candidates is empty"))

    rows = NamedTuple[]
    for (label, estimand) in pairs_iter
        lab = Symbol(label)
        estimand isa Estimand || throw(ArgumentError(
            "candidate :$lab must be an Estimand; got $(typeof(estimand))",
        ))
        grid = execute_estimand(
            estimand, data;
            id_result = id_result,
            metadata = false,
            kwargs...,
        )
        est = Float64(first(grid.est))
        se = hasproperty(grid, :se) ? Float64(first(grid.se)) : NaN
        push!(rows, (policy = lab, est = est, se = se))
    end
    values = DataFrame(rows)
    selected = if sense === :max
        values.policy[argmax(values.est)]
    else
        values.policy[argmin(values.est)]
    end
    return PolicyChoice(selected, values, id_result)
end

export PolicyChoice, choose_policy
