"""In-memory fold nuisance cache for δ-grids (outcome + exposure models)."""

using DataFrames
using Random
using StableRNGs

"""
    LMTPFoldCache

Pre-fitted cross-fitted nuisances reused across δ values in one grid run.
"""
mutable struct LMTPFoldCache
    df::DataFrame
    trt::Symbol
    outcome::Symbol
    covariates::Vector{Symbol}
    outcome_model::OutcomeRegression
    exposure_model::ExposureDensity
    y::Vector{Float64}
    a::Vector{Float64}
    W::Matrix{Float64}
    folds::Int
    learners_outcome::Tuple
    learners_trt::Tuple
    rng_seed::UInt
end

"""
    build_lmtp_fold_cache(df, trt, outcome, covariates, folds, rng; learners_outcome, learners_trt) -> LMTPFoldCache
"""
function build_lmtp_fold_cache(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng::AbstractRNG;
    learners_outcome = DEFAULT_SL_LEARNERS,
    learners_trt = DEFAULT_SL_LEARNERS,
)
    outcome_model = fit_outcome_regression(
        df, outcome, trt, covariates, folds, rng; learners = learners_outcome,
    )
    exposure_model = fit_exposure_density(
        df, trt, covariates, folds, rng; learners = learners_trt,
    )
    seed = UInt(mod(hash(rng), typemax(UInt)))
    return LMTPFoldCache(
        df, trt, outcome, covariates,
        outcome_model, exposure_model,
        Float64.(df[!, outcome]),
        Float64.(df[!, trt]),
        _covariate_matrix(df, covariates),
        folds, learners_outcome, learners_trt, seed,
    )
end

"""
    lmtp_components_from_cache(cache, a_policy, a_reference; density_ratio, trunc, L, U, shift_policy, shift_reference) -> NamedTuple

Recompute Q and H from cached nuisances (δ-specific policies only).
"""
function lmtp_components_from_cache(
    cache::LMTPFoldCache,
    a_policy::AbstractVector{<:Real},
    a_reference::AbstractVector{<:Real};
    density_ratio::Symbol = :gaussian,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0),
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_policy::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
)
    n = length(cache.y)
    a1 = Float64.(a_policy)
    a0 = Float64.(a_reference)
    clamp_aware = L !== nothing && U !== nothing && shift_policy !== nothing

    Q_obs = predict_outcome(cache.outcome_model, cache.df)
    Q1 = predict_outcome(cache.outcome_model, cache.df; treatment_values = a1)
    Q0 = predict_outcome(cache.outcome_model, cache.df; treatment_values = a0)

    H1 = ones(n)
    H0 = ones(n)
    a = cache.a
    W = cache.W

    for (fold_i, test_idx) in enumerate(cache.outcome_model.fold_test_idx)
        train_idx = setdiff(1:n, test_idx)
        if density_ratio == :gaussian || density_ratio == :hybrid
            sl_a = cache.exposure_model.fold_models[fold_i]
            train = cache.df[train_idx, :]
            test = cache.df[test_idx, :]
            μ_tr = predict_super_learner(sl_a, design_matrix(train, cache.covariates))
            μ_te = predict_super_learner(sl_a, design_matrix(test, cache.covariates))
            σ_fold = robust_residual_sd(a[train_idx] .- μ_tr)
            if clamp_aware
                Hg1 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], μ_te, σ_fold, shift_policy, L, U,
                )
                δ0 = shift_reference === nothing ? 0.0 : Float64(shift_reference)
                Hg0 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], μ_te, σ_fold, δ0, L, U,
                )
            else
                Hg1 = _mtp_clever_covariate_gaussian(a[test_idx], a1[test_idx], μ_te, σ_fold)
                Hg0 = _mtp_clever_covariate_gaussian(a[test_idx], a0[test_idx], μ_te, σ_fold)
            end
            if density_ratio == :hybrid
                W_tr = W[train_idx, :]
                clf1 = _fit_density_ratio_classifier(a[train_idx], a1[train_idx], W_tr; rng = StableRNG(cache.rng_seed))
                clf0 = _fit_density_ratio_classifier(a[train_idx], a0[train_idx], W_tr; rng = StableRNG(cache.rng_seed + 1))
                Hc1 = _ratio_from_classifier(clf1, a[test_idx], W[test_idx, :]; trunc = trunc)
                Hc0 = _ratio_from_classifier(clf0, a[test_idx], W[test_idx, :]; trunc = trunc)
                H1[test_idx] = [g > 1e-12 ? sqrt(g * c) : 0.0 for (g, c) in zip(Hg1, Hc1)]
                H0[test_idx] = [g > 1e-12 ? sqrt(g * c) : 0.0 for (g, c) in zip(Hg0, Hc0)]
            else
                H1[test_idx] = Hg1
                H0[test_idx] = Hg0
            end
        else
            W_tr = W[train_idx, :]
            rng = StableRNG(cache.rng_seed)
            clf1 = _fit_density_ratio_classifier(a[train_idx], a1[train_idx], W_tr; rng = rng)
            clf0 = _fit_density_ratio_classifier(a[train_idx], a0[train_idx], W_tr; rng = rng)
            H1[test_idx] = _ratio_from_classifier(clf1, a[test_idx], W[test_idx, :]; trunc = trunc)
            H0[test_idx] = _ratio_from_classifier(clf0, a[test_idx], W[test_idx, :]; trunc = trunc)
        end
    end

    trunc_used = Float64(trunc)
    stabilize = !clamp_aware
    if cv_trunc
        trunc_used, H1 = cv_select_truncation(H1; candidates = trunc_candidates, stabilize = stabilize)
        _, H0 = cv_select_truncation(H0; candidates = trunc_candidates, stabilize = stabilize)
    else
        H1 = prepare_clever_covariate(H1; trunc = trunc, stabilize = stabilize)
        H0 = prepare_clever_covariate(H0; trunc = trunc, stabilize = stabilize)
    end

    return (
        y = cache.y, Q_obs = Q_obs, Q1 = Q1, Q0 = Q0, H1 = H1, H0 = H0, n = n,
        trunc = trunc_used, density_ratio = density_ratio, clamp_aware = clamp_aware,
    )
end

export LMTPFoldCache, build_lmtp_fold_cache, lmtp_components_from_cache
