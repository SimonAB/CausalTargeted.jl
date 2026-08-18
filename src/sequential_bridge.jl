"""Bridge CausalDynamics temporal identification → [`SequentialPolicy`](@ref)."""

using CausalDynamics

"""Default wide-column naming; delegates to CausalDynamics when registered."""
function _default_panel_column_name(var::Symbol, t::Integer)
    if isdefined(CausalDynamics, :panel_column_name)
        return CausalDynamics.panel_column_name(var, t)
    end
    return Symbol(string(var), Int(t))
end

"""
    plan_sequential(spec, id_result; shift) -> SequentialPolicy

Certificate-first planning: merge baseline adjustment from a CausalDynamics
`IdentificationResult` into a [`SequentialPolicy`](@ref) (mirrors
CausalMediation `plan_mediation`).

Empty `baseline` on `spec` is filled from `id_result.adjustment`; nonempty
fields are kept. Optional `shift` replaces the numeric policy shift.
Factor `spec.policies` are copied through unchanged.
Requires `id_result.query isa TemporalEffectQuery`.
"""
function plan_sequential(
    spec::SequentialPolicy,
    id_result::IdentificationResult;
    shift::Union{Nothing, ShiftPolicy} = nothing,
)
    q = id_result.query
    q isa TemporalEffectQuery || throw(ArgumentError(
        "plan_sequential expects TemporalEffectQuery; got $(typeof(q))",
    ))
    baseline = isempty(spec.baseline) ? Symbol.(id_result.adjustment) : spec.baseline
    pol = shift === nothing ? spec.shift : shift
    return SequentialPolicy(
        spec.treatments,
        spec.outcome,
        baseline;
        time_vary = spec.time_vary,
        shift = pol,
        policies = spec.policies,
    )
end

"""
    sequential_spec_from_identification(id_result; treatments, outcome, baseline, time_vary, shift, policies, name_fn)

Build a [`SequentialPolicy`](@ref) from an `IdentificationResult` whose query is
a [`TemporalEffectQuery`](@ref).

Default treatments are `name_fn(treatment, t)` for ``t = 1:t_outcome``;
default outcome is the bare `query.outcome` symbol (terminal column, matching
[`simulate_panel`](@ref) `terminal=`). Pass `treatments` / `outcome` /
`baseline` / `time_vary` explicitly when the wide table uses a different layout.
"""
function sequential_spec_from_identification(
    id_result::IdentificationResult;
    treatments::Union{Nothing, Vector{Symbol}} = nothing,
    outcome::Union{Nothing, Symbol} = nothing,
    baseline::Union{Nothing, Vector{Symbol}} = nothing,
    time_vary::Union{Nothing, Vector{Vector{Symbol}}} = nothing,
    shift::ShiftPolicy = ShiftPolicy(scale = "z", lower_q = 0.01, upper_q = 0.99),
    policies = DiscreteTreatmentPolicy[],
    name_fn = _default_panel_column_name,
)
    q = id_result.query
    q isa TemporalEffectQuery || throw(ArgumentError(
        "sequential_spec_from_identification expects TemporalEffectQuery; got $(typeof(q))",
    ))
    T = Int(q.t_outcome)
    T >= 1 || throw(ArgumentError("t_outcome must be ≥ 1, got $T"))
    trt_base = Symbol(q.treatment)
    out_base = Symbol(q.outcome)
    trts = treatments === nothing ? [name_fn(trt_base, t) for t in 1:T] : treatments
    out = outcome === nothing ? out_base : outcome
    base = baseline === nothing ? Symbol.(id_result.adjustment) : baseline
    tv = time_vary === nothing ? [Symbol[] for _ in 1:T] : time_vary
    length(trts) == T || throw(ArgumentError(
        "treatments length ($(length(trts))) must equal t_outcome ($T)",
    ))
    return SequentialPolicy(trts, out, base; time_vary = tv, shift = shift, policies = policies)
end

export plan_sequential, sequential_spec_from_identification
