"""Scalar binary-treatment mediation (crumble tidy output)."""

using DataFrames
using Statistics
using StableRNGs

"""
    run_crumble_scalar(data, trt, outcome; mediators, covar, folds, epochs, rng) -> DataFrame

Binary contrast `d0=0` vs `d1=1` with NDE / NIE / TE rows.
"""
function run_crumble_scalar(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    mediators::Vector{Symbol},
    covar::Vector{Symbol},
    folds::Int = mtp_settings().folds,
    epochs::Int = mtp_settings().epochs,
    learners = DEFAULT_SL_LEARNERS,
    n_mc::Int = 32,
    rng = StableRNG(42),
)
    cols = unique(vcat([trt, outcome], covar, mediators))
    df = dropmissing(data[:, cols])
    n = nrow(df)
    a0 = zeros(n)
    a1 = ones(n)
    est, se = _crumble_mediation_effects(
        df, outcome, trt, covar, mediators, a0, a1, folds, epochs, rng;
        learners = learners,
        n_mc = n_mc,
    )
    rows = Dict{String, Any}[]
    for (lab, e, s) in (("NDE", est.nde, se.nde), ("NIE", est.nie, se.nie), ("TE", est.te, se.te))
        lwr, upr = wald_ci(e, s)
        push!(rows, Dict(
            "effect" => lab, "estimate" => e, "se" => s,
            "lower" => lwr, "upper" => upr,
        ))
    end
    return DataFrame(rows)
end

export run_crumble_scalar
