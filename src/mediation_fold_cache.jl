"""Fold-level nuisance cache for mediation δ-grids (outcome, mediators, exposure)."""

using DataFrames
using Random
using StableRNGs

"""
    MediationFoldCache

Per-fold SuperLearner fits reused across δ values. Policy-specific predictions
still depend on `a_nat` / `a_shift`.

Legacy alias: [`CrumbleFoldCache`](@ref) (from the R `crumble` package name).
"""
struct MediationFoldCache
    fold_sets::Vector{Vector{Int}}
    outcome_models::Vector{Any}          # SL fit per fold (on train)
    mediator_models::Vector{Vector{Any}} # med_models per fold
    sigma_m::Vector{Vector{Float64}}     # residual SDs per fold
    exposure_models::Vector{Any}         # continuous A density SL per fold
    adjust::Vector{Symbol}
    covar::Vector{Symbol}
    mediators::Vector{Symbol}
    trt::Symbol
    learners::Tuple
    rng_seed::UInt
end

"""Legacy name for [`MediationFoldCache`](@ref)."""
const CrumbleFoldCache = MediationFoldCache

"""
    build_mediation_fold_cache(df, outcome, trt, covar, mediators, folds, rng; learners) -> MediationFoldCache
"""
function build_mediation_fold_cache(
    df::DataFrame,
    outcome::Symbol,
    trt::Symbol,
    covar::Vector{Symbol},
    mediators::Vector{Symbol},
    folds::Int,
    rng::AbstractRNG;
    learners = DEFAULT_SL_LEARNERS,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, trt])
    adjust = unique(vcat(covar, mediators))
    fold_sets = crossfit_indices(n, folds, rng)
    seed = UInt(mod(hash(rng), typemax(UInt)))

    outcome_models = Any[]
    mediator_models = Vector{Any}[]
    sigma_m = Vector{Float64}[]
    exposure_models = Any[]

    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        y_tr = y[train_idx]
        ols_y = _fit_sl_outcome(train, adjust, y_tr; treatment = trt, learners = learners, rng = rng)
        med_models = [
            _fit_sl_outcome(train, covar, Float64.(train[!, m]); treatment = trt, learners = learners, rng = rng)
            for m in mediators
        ]
        σ = [
            _mediator_residual_sd(train, med_models[j], mediators[j], covar, trt)
            for j in eachindex(mediators)
        ]
        sl_a = fit_super_learner(
            design_matrix(train, covar), a[train_idx];
            learners = learners, rng = rng,
        )
        push!(outcome_models, ols_y)
        push!(mediator_models, med_models)
        push!(sigma_m, σ)
        push!(exposure_models, sl_a)
    end

    return MediationFoldCache(
        fold_sets, outcome_models, mediator_models, sigma_m, exposure_models,
        adjust, covar, mediators, trt, Tuple(learners), seed,
    )
end

"""Legacy alias for [`build_mediation_fold_cache`](@ref)."""
const build_crumble_fold_cache = build_mediation_fold_cache

export MediationFoldCache, CrumbleFoldCache
export build_mediation_fold_cache, build_crumble_fold_cache
