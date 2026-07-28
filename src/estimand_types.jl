"""First-class estimand types and shift policies for MTP pipelines."""

using DataFrames

"""
    ShiftPolicy(scale, lower_q, upper_q)

How realised shifts map grid `delta` to exposure changes.
`scale` is one of `"z"`, `"raw"`, `"global_sd"`, `"stratum_sd"`.
"""
struct ShiftPolicy
    scale::String
    lower_q::Float64
    upper_q::Float64
end

ShiftPolicy(; scale = "z", lower_q = 0.01, upper_q = 0.99) =
    ShiftPolicy(scale, lower_q, upper_q)

function shift_policy_from_settings()
    s = mtp_settings()
    return ShiftPolicy(scale = s.shift_scale, lower_q = s.lower_q, upper_q = s.upper_q)
end

abstract type Estimand end

"""
    InterventionalMean(trt, outcome, adjustment, shift)

Modified treatment policy contrast vs natural policy (LMTP TE grid).
"""
struct InterventionalMean <: Estimand
    trt::Symbol
    outcome::Symbol
    adjustment::Vector{Symbol}
    shift::ShiftPolicy
end

"""
    MediationContrast(trt, outcome, adjustment, mediators, shift)

Interventional NDE / NIE / TE grid under shifted exposure.
"""
struct MediationContrast <: Estimand
    trt::Symbol
    outcome::Symbol
    adjustment::Vector{Symbol}
    mediators::Vector{Symbol}
    shift::ShiftPolicy
end

"""
    LongitudinalPolicy(trt, outcome, adjustment, shift, treat_lag, outcome_lag)

Pathway exposure at `treat_lag` on AUC outcome at `outcome_lag` (temporal ID).
"""
struct LongitudinalPolicy <: Estimand
    trt::Symbol
    outcome::Symbol
    adjustment::Vector{Symbol}
    shift::ShiftPolicy
    treat_lag::Int
    outcome_lag::Int
end

"""
    ScalarMediation(trt, outcome, adjustment, mediators)

Binary contrast mediation (NDE / NIE / TE) without a δ-grid.
"""
struct ScalarMediation <: Estimand
    trt::Symbol
    outcome::Symbol
    adjustment::Vector{Symbol}
    mediators::Vector{Symbol}
end

"""
    estimand_engine(::Estimand) -> Symbol

Dispatch helper: `:lmtp`, `:crumble`, or `:scalar`.
"""
estimand_engine(::InterventionalMean) = :lmtp
estimand_engine(::MediationContrast) = :crumble
estimand_engine(::LongitudinalPolicy) = :lmtp
estimand_engine(::ScalarMediation) = :scalar

"""
    estimand_from_pathway_task(task, nuisances; shift) -> Estimand

Build a typed estimand from a registry task and resolved nuisances.
"""
function estimand_from_pathway_task(
    task,
    nuisances::NamedTuple;
    shift::ShiftPolicy = shift_policy_from_settings(),
)
    if task.engine == :crumble
        return MediationContrast(
            task.trt, task.outcome,
            nuisances.adjustment, nuisances.mediators, shift,
        )
    end
    return InterventionalMean(task.trt, task.outcome, nuisances.adjustment, shift)
end

"""
    longitudinal_estimand(trt, outcome, adjustment; treat_lag, outcome_lag, shift) -> LongitudinalPolicy
"""
function longitudinal_estimand(
    trt::Symbol,
    outcome::Symbol,
    adjustment::Vector{Symbol};
    treat_lag::Int,
    outcome_lag::Int,
    shift::ShiftPolicy = shift_policy_from_settings(),
)
    return LongitudinalPolicy(trt, outcome, adjustment, shift, treat_lag, outcome_lag)
end

"""
    estimand_from_query(query::CausalQuery, adjustment; mediators, shift) -> Estimand
"""
function estimand_from_query(
    query::CausalQuery,
    adjustment::Vector{Symbol};
    mediators::Vector{Symbol} = Symbol[],
    shift::ShiftPolicy = shift_policy_from_settings(),
)
    if query isa TotalEffectQuery
        return InterventionalMean(query.treatment, query.outcome, adjustment, shift)
    elseif query isa MediationQuery
        return MediationContrast(
            query.treatment, query.outcome, adjustment,
            isempty(mediators) ? Symbol.(query.mediators) : mediators, shift,
        )
    elseif query isa InterventionalPolicyQuery
        return InterventionalMean(query.treatment, query.outcome, adjustment, shift)
    elseif query isa TemporalEffectQuery
        return LongitudinalPolicy(
            query.treatment, query.outcome, adjustment, shift,
            query.t_treat, query.t_outcome,
        )
    end
    error("Cannot build Estimand from $(typeof(query))")
end

export ShiftPolicy, Estimand
export InterventionalMean, MediationContrast, LongitudinalPolicy, ScalarMediation
export shift_policy_from_settings, estimand_engine, estimand_from_pathway_task
export estimand_from_query, longitudinal_estimand
