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
    family_outcome::Symbol
    weight_truncation::Bool
end

"""
    build_lmtp_fold_cache(df, trt, outcome, covariates, folds, rng; learners_outcome, learners_trt, family_outcome, weight_truncation) -> LMTPFoldCache

The cache records `family_outcome` and `weight_truncation` for component
construction. `weight_truncation=true` preserves the capped and stabilised
weight path; `false` requests raw static Gaussian ratios and is validated when
the policies are supplied to `lmtp_components_from_cache`.
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
    family_outcome::Symbol = :gaussian,
    weight_truncation::Bool = true,
)
    covariate_schema = fit_covariate_schema(df, covariates)
    shared_fold_sets = crossfit_indices(nrow(df), folds, rng)
    outcome_model = fit_outcome_regression(
        df, outcome, trt, covariates, folds, rng;
        learners = learners_outcome, family = family_outcome,
        fold_sets = shared_fold_sets,
    )
    exposure_model = fit_exposure_density(
        df, trt, covariates, folds, rng;
        learners = learners_trt, fold_sets = shared_fold_sets,
    )
    seed = CausalDynamics.stable_rng_seed(rng)
    return LMTPFoldCache(
        df, trt, outcome, covariates,
        outcome_model, exposure_model,
        Float64.(df[!, outcome]),
        Float64.(df[!, trt]),
        _covariate_matrix(covariate_schema, df),
        folds, learners_outcome, learners_trt, seed, family_outcome, weight_truncation,
    )
end

"""
    lmtp_components_from_cache(cache, a_policy, a_reference; density_ratio, trunc, L, U, shift_amount, shift_reference) -> NamedTuple

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
    shift_amount::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
    weight_truncation::Bool = cache.weight_truncation,
)
    n = length(cache.y)
    a1 = Float64.(a_policy)
    a0 = Float64.(a_reference)
    clamp_aware = L !== nothing && U !== nothing && shift_amount !== nothing
    clamp_config = L !== nothing || U !== nothing || shift_amount !== nothing
    if !weight_truncation
        density_ratio === :gaussian || throw(ArgumentError(
            "weight_truncation=false supports only Gaussian LMTP density ratios",
        ))
        cv_trunc && throw(ArgumentError(
            "weight_truncation=false cannot be combined with cv_trunc=true",
        ))
        clamp_config && throw(ArgumentError(
            "weight_truncation=false supports only unclamped LMTP policies",
        ))
        (_constant_additive_shift(cache.a, a1) && _constant_additive_shift(cache.a, a0)) ||
            throw(ArgumentError(
                "weight_truncation=false supports only constant additive shifts in both LMTP policies",
            ))
    end
    inference_status = _lmtp_weight_inference_status(
        cache.a, a1, a0; weight_truncation = weight_truncation, clamp_config = clamp_config,
    )

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
            μ_tr = predict_super_learner(
                sl_a,
                cache.exposure_model.W[train_idx, :],
            )
            μ_te = predict_super_learner(
                sl_a,
                cache.exposure_model.W[test_idx, :],
            )
            σ_fold = robust_residual_sd(a[train_idx] .- μ_tr)
            if clamp_aware
                Hg1 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], μ_te, σ_fold, shift_amount, L, U,
                )
                δ0 = shift_reference === nothing ? 0.0 : Float64(shift_reference)
                Hg0 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], μ_te, σ_fold, δ0, L, U,
                )
            else
                ratio = weight_truncation ? _mtp_clever_covariate_gaussian :
                    _mtp_clever_covariate_gaussian_uncapped
                Hg1 = ratio(a[test_idx], a1[test_idx], μ_te, σ_fold)
                Hg0 = ratio(a[test_idx], a0[test_idx], μ_te, σ_fold)
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

    trunc_used = weight_truncation ? Float64(trunc) : Inf
    stabilize = !clamp_aware
    if !weight_truncation
        # Keep raw static Gaussian ratios: no cap, floor, or stabilisation.
    elseif cv_trunc
        trunc_used, H1 = cv_select_truncation(H1; candidates = trunc_candidates, stabilize = stabilize)
        _, H0 = cv_select_truncation(H0; candidates = trunc_candidates, stabilize = stabilize)
    else
        H1 = prepare_clever_covariate(H1; trunc = trunc, stabilize = stabilize)
        H0 = prepare_clever_covariate(H0; trunc = trunc, stabilize = stabilize)
    end

    return (
        y = cache.y, Q_obs = Q_obs, Q1 = Q1, Q0 = Q0, H1 = H1, H0 = H0, n = n,
        trunc = trunc_used, density_ratio = density_ratio, clamp_aware = clamp_aware,
        family_outcome = cache.family_outcome,
        weight_truncation = weight_truncation, inference_status = inference_status,
    )
end

export LMTPFoldCache, build_lmtp_fold_cache, lmtp_components_from_cache
