"""Engine symbols and legacy aliases for CausalTargeted grids.

Public engines:
- `:lmtp` — continuous / longitudinal modified treatment policy (total effect)
- `:discrete_lmtp` — categorical-treatment LMTP (Díaz–Williams classification ratios)
- `:mediation` — interventional mediation under MTP (NDE / NIE / TE)
- `:scalar` — binary-treatment mediation without a δ-grid
- `:sequential_lmtp` — multi-time sequential LMTP (numeric shift or factor `policies`)
- `:survival_lmtp` — discrete-time event-time / survival LMTP (competing risks deferred)

Legacy alias: `:crumble` → `:mediation` (name taken from the R `crumble` package;
Julia APIs prefer descriptive `mediation_*` names).
"""

"""
    normalize_engine(engine) -> Symbol

Map legacy engine names to the canonical symbol.
"""
function normalize_engine(engine::Symbol)
    engine === :crumble && return :mediation
    return engine
end

normalize_engine(engine::AbstractString) = normalize_engine(Symbol(engine))

"""
    is_mediation_engine(engine) -> Bool

True for `:mediation` and the legacy `:crumble` alias.
"""
is_mediation_engine(engine::Symbol) = normalize_engine(engine) === :mediation

export normalize_engine, is_mediation_engine
