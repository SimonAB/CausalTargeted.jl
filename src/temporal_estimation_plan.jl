"""
    EstimationPlan

CausalTargeted plan linking temporal identification to an estimation engine
and wide-panel column names.
"""
struct EstimationPlan
    engine::Symbol
    treatment::Symbol
    outcome::Symbol
    presence_col::Union{Nothing, Symbol}
    intensity_col::Union{Nothing, Symbol}
    family_outcome::Union{Nothing, Symbol}
    baseline::Vector{Symbol}
    query::TemporalEffectQuery
    identifiable::Bool
    strategy::Symbol
    adjustment_columns::Vector{Symbol}
    missing_columns::Vector{Symbol}
    min_complete_n::Union{Nothing, Int}
    estimability::Union{Nothing, Symbol}
    missingness_note::String
    capture_conditioned::Bool
end

function EstimationPlan(
    engine::Symbol,
    treatment::Symbol,
    outcome::Symbol,
    baseline::Vector{Symbol},
    query::TemporalEffectQuery,
    identifiable::Bool,
    strategy::Symbol,
    adjustment_columns::Vector{Symbol},
    missing_columns::Vector{Symbol};
    presence_col::Union{Nothing, Symbol} = nothing,
    intensity_col::Union{Nothing, Symbol} = nothing,
    family_outcome::Union{Nothing, Symbol} = nothing,
    min_complete_n::Union{Nothing, Int} = nothing,
    estimability::Union{Nothing, Symbol} = nothing,
    missingness_note::String = "",
    capture_conditioned::Bool = false,
)
    return EstimationPlan(
        engine, treatment, outcome, presence_col, intensity_col, family_outcome,
        baseline, query, identifiable, strategy,
        adjustment_columns, missing_columns,
        min_complete_n, estimability, missingness_note, capture_conditioned,
    )
end

function Base.show(io::IO, plan::EstimationPlan)
    hurdle = plan.presence_col !== nothing ? ", hurdle" : ""
    fam = plan.family_outcome === nothing ? "" : ", family=$(plan.family_outcome)"
    power = plan.estimability === :underpowered ? " [underpowered]" :
        plan.estimability === :empty ? " [empty]" :
        plan.estimability === :structural_skip ? " [structural_skip]" : ""
    print(io, "EstimationPlan(", plan.engine, ", ",
        plan.treatment, " → ", plan.outcome, hurdle, fam,
        ", baseline=", length(plan.baseline), " cols",
        plan.identifiable ? "" : " [not identifiable]", power, ")")
end

"""Map planner outcome kind to CausalTargeted `family_outcome` hint."""
function _family_outcome_for_spec(
    spec::NodeOutcomeSpec;
    count_family::Symbol = :negbin,
)
    if spec.kind == CausalDynamics.binary
        return :binomial
    elseif spec.kind == CausalDynamics.count_outcome
        count_family in (:poisson, :negbin) || throw(ArgumentError(
            "count_family must be :poisson or :negbin; got $count_family",
        ))
        return count_family
    end
    return nothing
end

"""
    _targeted_engine_for_query(query, discrete_treatment, outcome_spec) -> Symbol

Choose a default CausalTargeted runner symbol for a temporal query.
"""
function _targeted_engine_for_query(
    query::TemporalEffectQuery,
    discrete_treatment::Bool,
    outcome_spec::NodeOutcomeSpec,
)
    if outcome_spec.kind == CausalDynamics.hurdle
        return :two_part_discrete_lmtp
    end
    if query.t_treat == query.t_outcome
        return discrete_treatment ? :discrete_lmtp : :lmtp_grid
    end
    if query.treatment == query.outcome && query.t_treat < query.t_outcome
        return :sequential_lmtp
    end
    return :lmtp_grid
end

function EstimationPlan(plan::CausalDynamics.EstimationPlan)
    return EstimationPlan(
        plan.engine, plan.treatment, plan.outcome, plan.baseline, plan.query,
        plan.identifiable, plan.strategy, plan.adjustment_columns, plan.missing_columns;
        presence_col = plan.presence_col,
        intensity_col = plan.intensity_col,
        family_outcome = plan.family_outcome,
        min_complete_n = plan.min_complete_n,
        estimability = plan.estimability,
        missingness_note = plan.missingness_note,
        capture_conditioned = plan.capture_conditioned,
    )
end

"""
    CausalTargeted.plan_targeted_estimation(unrolling, query, column_names; ...)

Use CausalDynamics temporal identification and wide-panel mapping to prepare an
[`CausalTargeted.EstimationPlan`](@ref) for a CausalTargeted engine.

For same-time contrasts, `discrete_treatment=true` selects `:discrete_lmtp`
and `false` selects `:lmtp_grid`. A query on the same node at later times
selects `:sequential_lmtp`. Other queries select `:lmtp_grid`.

`outcome_specs` maps DAG nodes to CausalDynamics `NodeOutcomeSpec` values.
Hurdle outcomes select `:two_part_discrete_lmtp` and populate `presence_col`
and `intensity_col`. Binary outcomes use `:binomial`, and count outcomes use
`:negbin` by default or `count_family = :poisson`.

Pass `data` as a wide table to attach complete-case support using `min_n` as
the threshold. `missingness`, `skip`, and `name_fn` control identification
and panel column mapping.
"""
function plan_targeted_estimation(
    unrolling::TemporalUnrolling,
    query::TemporalEffectQuery,
    column_names;
    missingness = nothing,
    skip::AbstractVector{Symbol} = [:sex],
    name_fn = panel_column_name,
    discrete_treatment::Bool = true,
    outcome_specs::Dict{Symbol, NodeOutcomeSpec} = Dict{Symbol, NodeOutcomeSpec}(),
    count_family::Symbol = :negbin,
    data = nothing,
    min_n::Int = 10,
)
    if !isdefined(CausalDynamics, :resolve_panel_query)
        # CausalDynamics 0.5 predates the separate identification and support helpers.
        return EstimationPlan(CausalDynamics.plan_targeted_estimation(
            unrolling, query, column_names;
            missingness, skip, name_fn, discrete_treatment, outcome_specs, count_family, data, min_n,
        ))
    end

    inputs = CausalDynamics.resolve_panel_query(
        unrolling, query, column_names; missingness, skip, name_fn, outcome_specs,
    )
    engine = _targeted_engine_for_query(inputs.query, discrete_treatment, inputs.outcome_spec)
    family_outcome = _family_outcome_for_spec(inputs.outcome_spec; count_family)
    support = CausalDynamics.panel_query_support(inputs; data, min_n)
    return EstimationPlan(
        engine, inputs.treatment, inputs.outcome, inputs.adjustment_columns, inputs.query,
        inputs.result.identifiable, inputs.result.strategy,
        inputs.adjustment_columns, inputs.missing_columns;
        presence_col = inputs.presence_col,
        intensity_col = inputs.intensity_col,
        family_outcome,
        support...,
    )
end
