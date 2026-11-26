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

struct _LMTPFoldPrecomputed
    Q_obs::Vector{Float64}
    train_idx::Vector{Vector{Int}}
    μ_tr::Vector{Vector{Float64}}
    μ_te::Vector{Vector{Float64}}
    σ_fold::Vector{Float64}
    density_a::Vector{Float64}
end

const _FOLD_CACHE_PRECOMPUTED = WeakKeyDict{LMTPFoldCache, _LMTPFoldPrecomputed}()
const _FOLD_CACHE_PRECOMPUTED_LOCK = ReentrantLock()

function _precompute_fold_cache(
    df::DataFrame,
    outcome_model::OutcomeRegression,
    exposure_model::ExposureDensity,
    density_a::Vector{Float64},
)
    n = nrow(df)
    Q_obs = predict_outcome(outcome_model, df)
    n_folds = length(outcome_model.fold_test_idx)
    train_idxs = Vector{Vector{Int}}(undef, n_folds)
    has_exposure = !isempty(exposure_model.fold_models)
    μ_trs = has_exposure ? Vector{Vector{Float64}}(undef, n_folds) : Vector{Vector{Float64}}()
    μ_tes = has_exposure ? Vector{Vector{Float64}}(undef, n_folds) : Vector{Vector{Float64}}()
    σ_folds = has_exposure ? Vector{Float64}(undef, n_folds) : Float64[]
    for (fold_i, test_idx) in enumerate(outcome_model.fold_test_idx)
        tr_idx = _complement_indices(n, test_idx)
        train_idxs[fold_i] = tr_idx
        if has_exposure
            sl_a = exposure_model.fold_models[fold_i]
            μ_tr = Vector{Float64}(predict_super_learner(sl_a, exposure_model.W[tr_idx, :]))
            μ_te = Vector{Float64}(predict_super_learner(sl_a, exposure_model.W[test_idx, :]))
            μ_trs[fold_i] = μ_tr
            μ_tes[fold_i] = μ_te
            σ_folds[fold_i] = robust_residual_sd(density_a[tr_idx] .- μ_tr)
        end
    end
    return _LMTPFoldPrecomputed(Q_obs, train_idxs, μ_trs, μ_tes, σ_folds, density_a)
end

"""
    build_lmtp_fold_cache(df, trt, outcome, covariates, folds, rng; learners_outcome, learners_trt, family_outcome, weight_truncation, exposure_values, fit_exposure) -> LMTPFoldCache

The cache records `family_outcome` and `weight_truncation` for component
construction. `weight_truncation=true` preserves the capped and stabilised
weight path; `false` requests raw static Gaussian ratios and is validated when
the policies are supplied to `lmtp_components_from_cache`. Pass
`fit_exposure=false` when only `density_ratio=:classification` will be evaluated
to skip fitting unused Gaussian exposure models.
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
    exposure_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
    fit_exposure::Bool = true,
)
    covariate_schema = fit_covariate_schema(df, covariates)
    shared_fold_sets = crossfit_indices(nrow(df), folds, rng)
    outcome_model = fit_outcome_regression(
        df, outcome, trt, covariates, folds, rng;
        learners = learners_outcome, family = family_outcome,
        fold_sets = shared_fold_sets,
    )
    exposure_model = if fit_exposure
        fit_exposure_density(
            df, trt, covariates, folds, rng;
            learners = learners_trt, fold_sets = shared_fold_sets,
            treatment_values = exposure_values,
        )
    else
        ExposureDensity(
            covariates, Tuple(learners_trt), SuperLearnerFit[],
            deepcopy(shared_fold_sets), outcome_model.W,
        )
    end
    seed = CausalDynamics.stable_rng_seed(rng)
    y = collect(Float64, df[!, outcome])::Vector{Float64}
    a = collect(Float64, df[!, trt])::Vector{Float64}
    density_a = exposure_values === nothing ? a : collect(Float64, exposure_values)::Vector{Float64}
    cache = LMTPFoldCache(
        df, trt, outcome, covariates,
        outcome_model, exposure_model,
        y, a,
        _covariate_matrix(covariate_schema, df),
        folds, learners_outcome, learners_trt, seed, family_outcome, weight_truncation,
    )
    pre = _precompute_fold_cache(df, outcome_model, exposure_model, density_a)
    lock(_FOLD_CACHE_PRECOMPUTED_LOCK) do
        _FOLD_CACHE_PRECOMPUTED[cache] = pre
    end
    return cache
end

"""
    lmtp_components_from_cache(cache, a_policy, a_reference; density_ratio, trunc, L, U, shift_amount, shift_reference, policy, policy_delta, ...) -> NamedTuple

Recompute Q and H from cached nuisances (δ-specific policies only).
"""
function lmtp_components_from_cache(
    cache::LMTPFoldCache,
    a_policy::AbstractVector{<:Real},
    a_reference::AbstractVector{<:Real};
    density_ratio::Symbol = :gaussian,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_amount::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
    policy::Union{Nothing, AbstractContinuousPolicy} = nothing,
    policy_delta::Union{Nothing, Real} = nothing,
    smooth_policy::Union{Nothing, SmoothBoundedLogitShiftPolicy} = nothing,
    smooth_shift::Union{Nothing, Real} = nothing,
    feasible_policy::Union{Nothing, FeasibleAdditiveShiftPolicy} = nothing,
    feasible_delta::Union{Nothing, Real} = nothing,
    headroom_policy::Union{Nothing, ProportionalHeadroomPolicy} = nothing,
    headroom_delta::Union{Nothing, Real} = nothing,
    policy_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    weight_truncation::Bool = cache.weight_truncation,
)
    n = length(cache.y)
    a = cache.a
    a1 = collect(Float64, a_policy)::Vector{Float64}
    a0 = collect(Float64, a_reference)::Vector{Float64}
    length(a1) == n == length(a0) || throw(ArgumentError("policy vectors must match the data length"))

    resolved_policy, resolved_delta = _resolve_policy_kwargs(;
        policy, policy_delta,
        smooth_policy, smooth_shift,
        feasible_policy, feasible_delta,
        headroom_policy, headroom_delta,
    )
    clamp_aware = L !== nothing && U !== nothing && shift_amount !== nothing
    clamp_config = L !== nothing || U !== nothing || shift_amount !== nothing

    policy_param = _policy_validate_components(
        resolved_policy, resolved_delta, a, a1, a0;
        density_ratio, policy_mask, weight_truncation, clamp_config,
    )
    _validate_uncapped_options(
        resolved_policy, a, a1, a0;
        density_ratio, cv_trunc, clamp_config, weight_truncation,
    )
    density_vals = _policy_density_values(resolved_policy, a)
    density_a = density_vals === nothing ? a : density_vals

    inference_status = _policy_inference_status(
        resolved_policy, policy_param, a, a1, a0;
        weight_truncation, clamp_config, policy_mask,
    )

    if (density_ratio == :gaussian || density_ratio == :hybrid) &&
       isempty(cache.exposure_model.fold_models)
        throw(ArgumentError(
            "fold cache was built with fit_exposure=false and cannot evaluate density_ratio=:$density_ratio",
        ))
    end

    pre = lock(_FOLD_CACHE_PRECOMPUTED_LOCK) do
        get(_FOLD_CACHE_PRECOMPUTED, cache, nothing)
    end
    Q_obs = pre === nothing ? predict_outcome(cache.outcome_model, cache.df) : copy(pre.Q_obs)
    Q1 = predict_outcome(cache.outcome_model, cache.df; treatment_values = a1)
    Q0 = a0 == a ? copy(Q_obs) : predict_outcome(cache.outcome_model, cache.df; treatment_values = a0)
    has_pre_exposure = pre !== nothing && !isempty(pre.μ_te)
    pre_density_matches = has_pre_exposure && (pre.density_a === density_a || pre.density_a == density_a)

    H1 = ones(n)
    H0 = ones(n)
    W = cache.W

    for (fold_i, test_idx) in enumerate(cache.outcome_model.fold_test_idx)
        train_idx = pre === nothing ? _complement_indices(n, test_idx) : pre.train_idx[fold_i]
        if density_ratio == :gaussian || density_ratio == :hybrid
            if has_pre_exposure
                μ_tr = pre.μ_tr[fold_i]
                μ_te = pre.μ_te[fold_i]
                σ_fold = pre_density_matches ? pre.σ_fold[fold_i] :
                    robust_residual_sd(density_a[train_idx] .- μ_tr)
            else
                sl_a = cache.exposure_model.fold_models[fold_i]
                μ_tr = Vector{Float64}(predict_super_learner(
                    sl_a,
                    cache.exposure_model.W[train_idx, :],
                ))
                μ_te = Vector{Float64}(predict_super_learner(
                    sl_a,
                    cache.exposure_model.W[test_idx, :],
                ))
                σ_fold = robust_residual_sd(density_a[train_idx] .- μ_tr)
            end
            selected = policy_mask === nothing ? nothing : policy_mask[test_idx]
            Hg1, Hg0 = _policy_density_ratio(
                resolved_policy, policy_param,
                a[test_idx], density_a[test_idx], a1[test_idx], a0[test_idx],
                μ_te, σ_fold, selected;
                weight_truncation, clamp_aware, shift_amount, shift_reference, L, U,
            )
            if density_ratio == :hybrid
                W_tr = W[train_idx, :]
                fold_seed = cache.rng_seed + UInt(fold_i) * 0x9e3779b97f4a7c15
                clf1 = _fit_density_ratio_classifier(a[train_idx], a1[train_idx], W_tr; rng = StableRNG(fold_seed))
                clf0 = _fit_density_ratio_classifier(a[train_idx], a0[train_idx], W_tr; rng = StableRNG(fold_seed + UInt(1)))
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
            rng = StableRNG(cache.rng_seed + UInt(fold_i) * 0x9e3779b97f4a7c15)
            clf1 = _fit_density_ratio_classifier(a[train_idx], a1[train_idx], W_tr; rng = rng)
            clf0 = _fit_density_ratio_classifier(a[train_idx], a0[train_idx], W_tr; rng = rng)
            H1[test_idx] = _ratio_from_classifier(clf1, a[test_idx], W[test_idx, :]; trunc = trunc)
            H0[test_idx] = _ratio_from_classifier(clf0, a[test_idx], W[test_idx, :]; trunc = trunc)
        end
    end

    trunc_used = weight_truncation ? Float64(trunc) : Inf
    stabilize = !clamp_aware && !_policy_has_structural_zeros(resolved_policy)
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
