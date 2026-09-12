"""Engine symbols and legacy aliases for CausalTargeted grids.

Public engines:
- `:lmtp` — continuous / longitudinal modified treatment policy (total effect)
- `:discrete_lmtp` — categorical-treatment LMTP (Díaz–Williams classification ratios)
- `:two_part_discrete_lmtp` — hurdle outcome (binomial presence + conditional Gaussian intensity)
- `:mediation` — interventional mediation under MTP (NDE / NIE / TE)
- `:scalar` — binary-treatment mediation without a δ-grid
- `:sequential_lmtp` — multi-time sequential LMTP (numeric shift or factor `policies`)
- `:survival_lmtp` — discrete-time event-time / survival LMTP (competing risks deferred)
- `:repeated_msm` — binary point treatment, repeated outcomes, joint IF covariance
- `:parametric_msm` — GLS projection of repeated-outcome IF estimates onto a treatment×time design

Optional weakdep paths (not `execute_estimand` engines): **MMRM** (`fit_mmrm` /
`run_mmrm` after `using MixedModels`) for parametric repeated Gaussian trials;
**NB2 g-comp** (`mixed_g_computation` on profiled NB2 fits) for static treatment
count panels. See `docs/src/methods.md` (outcome paths table).

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

export normalize_engine
