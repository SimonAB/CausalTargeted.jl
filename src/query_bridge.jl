"""Bridge CausalDynamics queries to typed estimands (after policy types are loaded)."""

"""Collect a `policies` keyword into a vector of [`DiscreteTreatmentPolicy`](@ref)."""
function _as_discrete_policies(policies)
    policies === nothing && return DiscreteTreatmentPolicy[]
    policies isa DiscreteTreatmentPolicy && return DiscreteTreatmentPolicy[policies]
    return collect(DiscreteTreatmentPolicy, policies)
end

"""Return a [`ShiftPolicy`](@ref) when `shift` is numeric; otherwise a default."""
function _numeric_shift(shift)
    shift isa ShiftPolicy && return shift
    return shift_policy_from_settings()
end

"""
    estimand_from_query(query::CausalQuery, adjustment; mediators, shift, policies, treatments, time_vary) -> Estimand

Map a CausalDynamics query to a typed estimand.

- `TotalEffectQuery` → [`InterventionalMean`](@ref)
- `MediationQuery` → [`MediationContrast`](@ref)
- `InterventionalPolicyQuery` → [`DiscreteInterventionalMean`](@ref) when
  `query.shift` or `policies` is a [`DiscreteTreatmentPolicy`](@ref); otherwise
  [`InterventionalMean`](@ref)
- `TemporalEffectQuery` → [`LongitudinalPolicy`](@ref) by default (single lagged
  column). Pass nonempty `policies` and wide `treatments` (`:A1`, `:A2`, …) for
  [`SequentialPolicy`](@ref).
"""
function estimand_from_query(
    query::CausalQuery,
    adjustment::Vector{Symbol};
    mediators::Vector{Symbol} = Symbol[],
    shift = shift_policy_from_settings(),
    policies = nothing,
    treatments::Union{Nothing, Vector{Symbol}} = nothing,
    time_vary::Union{Nothing, Vector{Vector{Symbol}}} = nothing,
)
    pols = _as_discrete_policies(policies)
    numeric_shift = _numeric_shift(shift)

    if query isa TotalEffectQuery
        shift isa DiscreteTreatmentPolicy && throw(ArgumentError(
            "TotalEffectQuery does not take DiscreteTreatmentPolicy; use InterventionalPolicyQuery",
        ))
        return InterventionalMean(query.treatment, query.outcome, adjustment, numeric_shift)
    elseif query isa MediationQuery
        shift isa DiscreteTreatmentPolicy && throw(ArgumentError(
            "MediationQuery with DiscreteTreatmentPolicy belongs on CausalMediation " *
            "MediationSpec (run_mediation), not MediationContrast",
        ))
        return MediationContrast(
            query.treatment, query.outcome, adjustment,
            isempty(mediators) ? Symbol.(query.mediators) : mediators, numeric_shift,
        )
    elseif query isa InterventionalPolicyQuery
        disc = if !isempty(pols)
            length(pols) == 1 || throw(ArgumentError(
                "InterventionalPolicyQuery expects one DiscreteTreatmentPolicy; got $(length(pols))",
            ))
            only(pols)
        elseif query.shift isa DiscreteTreatmentPolicy
            query.shift
        elseif shift isa DiscreteTreatmentPolicy
            shift
        else
            nothing
        end
        if disc !== nothing
            return DiscreteInterventionalMean(
                Symbol(query.treatment), Symbol(query.outcome), adjustment, disc,
            )
        end
        sh = query.shift isa ShiftPolicy ? query.shift : numeric_shift
        return InterventionalMean(query.treatment, query.outcome, adjustment, sh)
    elseif query isa TemporalEffectQuery
        if !isempty(pols)
            treatments === nothing && throw(ArgumentError(
                "TemporalEffectQuery with policies requires treatments as Vector{Symbol} " *
                "(wide A_1,…,A_T columns); the default estimand_from_query layout remains LongitudinalPolicy",
            ))
            tv = time_vary === nothing ? [Symbol[] for _ in treatments] : time_vary
            return SequentialPolicy(
                treatments, Symbol(query.outcome), adjustment;
                time_vary = tv, shift = numeric_shift, policies = pols,
            )
        end
        return LongitudinalPolicy(
            query.treatment, query.outcome, adjustment, numeric_shift,
            query.t_treat, query.t_outcome,
        )
    end
    error("Cannot build Estimand from $(typeof(query))")
end
