# SPDX-License-Identifier: MIT

"""
    MMRMResult

Gaussian mixed model for repeated measures (MMRM) reference fit for a **static**
treatment and repeated outcome visits.

Wraps a fitted `MixedModels.LinearMixedModel` and a [`MixedGComputationResult`](@ref)
from [`run_mmrm`](@ref). This is a **parametric trial-style reference path** beside
LMTP/MSM; it is not a substitute for discrete longitudinal LMTP.

# Fields
- `model`: fitted `LinearMixedModel`
- `contrast`: [`MixedGComputationResult`](@ref) (LS-mean-style trajectory contrast)
- `formula`: applied `MixedModels` formula
- `covariance`: `:random_intercept` or `:unstructured` (random-effects approximation)
- `time_categorical`: whether `:unstructured` used an internal categorical visit factor
"""
struct MMRMResult{M, G, F}
    model::M
    contrast::G
    formula::F
    covariance::Symbol
    time_categorical::Bool
end

function Base.show(io::IO, result::MMRMResult)
    print(
        io,
        "MMRMResult(covariance=",
        repr(result.covariance),
        ", ",
        length(result.contrast.times),
        " visits, do(",
        result.contrast.treatment,
        ") contrast)",
    )
end

function Base.show(io::IO, ::MIME"text/plain", result::MMRMResult)
    show(io, result)
    print(io, "\nformula: ", result.formula)
    show(io, "\n", result.contrast)
end

"""
    fit_mmrm(
        data;
        outcome, treatment, time, id,
        baseline=Symbol[], covariance=:random_intercept
    ) -> LinearMixedModel

Fit a Gaussian MMRM with fixed effects `outcome ~ treatment * time + baseline`
and a subject random-effects structure.

# Covariance (`covariance=`)
- `:random_intercept` — `(1 | id)` (default)
- `:unstructured` — `(1 + _mmrm_time | id)` with an internal categorical visit
  column `_mmrm_time` (MixedModels **random-effects** approximation to unstructured
  within-subject correlation; not SAS/R `mmrm` residual ``\\Sigma``)

Random **slopes** are intentionally unsupported in this release; see issue #25.

Requires `using MixedModels` (loads `CausalTargetedMixedModelsExt`).
"""
function fit_mmrm(args...; kwargs...)
    throw(ArgumentError(
        "fit_mmrm requires the CausalTargeted MixedModels extension. " *
        "Load it with `using MixedModels`.",
    ))
end

"""
    run_mmrm(
        data;
        outcome, treatment, time, id,
        values=(0, 1), baseline=Symbol[], covariance=:random_intercept,
        random_effects=:zero, strata=nothing
    ) -> MMRMResult

Fit an MMRM and return visit-specific contrast estimates via [`mixed_g_computation`](@ref).

Keywords for g-computation (`random_effects`, `strata`, …) are explicit; remaining
keywords forward to `MixedModels.fit` (`progress`, `REML`, etc.).
"""
function run_mmrm(args...; kwargs...)
    throw(ArgumentError(
        "run_mmrm requires the CausalTargeted MixedModels extension. " *
        "Load it with `using MixedModels`.",
    ))
end

export MMRMResult, fit_mmrm, run_mmrm
