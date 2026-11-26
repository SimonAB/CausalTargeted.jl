# ---------------------------------------------------------------------------
# δ-invariant grid cache & main cross-fitted exponential-tilting TMLE estimator
# ---------------------------------------------------------------------------

struct _StandardisedFoldBasis
    Psi_tr::Matrix{Float64}
    Psi_te::Matrix{Float64}
    active_cols::Vector{Int}
    col_means::Vector{Float64}
    col_sds::Vector{Float64}
    n_raw_cols::Int
end

function _build_fold_bases(
    W::Matrix{Float64},
    train_sets::Vector{Vector{Int}},
    fold_sets::Vector{Vector{Int}},
)
    d = size(W, 2)
    bases = Vector{_StandardisedFoldBasis}(undef, length(fold_sets))
    for k in eachindex(fold_sets)
        Psi_tr, active_cols, col_means, col_sds = _fit_standardised_basis(@view W[train_sets[k], :])
        Psi_te = _apply_standardised_basis(@view(W[fold_sets[k], :]), active_cols, col_means, col_sds, d)
        bases[k] = _StandardisedFoldBasis(Psi_tr, Psi_te, active_cols, col_means, col_sds, d)
    end
    return bases
end

struct _TiltGridCache
    n::Int
    trt::Symbol
    outcome::Symbol
    normalizer_covariates::Vector{Symbol}
    outcome_covariates::Vector{Symbol}
    family_outcome::Symbol
    outcome_method::Symbol
    folds::Int
    independent_folds::Bool
    a::Vector{Float64}
    y::Vector{Float64}
    fold_sets_m0::Vector{Vector{Int}}
    train_sets_m0::Vector{Vector{Int}}
    basis_m0::Vector{_StandardisedFoldBasis}
    fold_sets_y::Vector{Vector{Int}}
    train_sets_y::Vector{Vector{Int}}
    basis_y::Vector{_StandardisedFoldBasis}
    basis_a::Vector{_StandardisedFoldBasis}
    mu_hat_folds::Union{Nothing, Vector{Vector{Float64}}}
end

"""
    _build_tilt_grid_cache(df, trt, outcome, covariates; kwargs...) -> _TiltGridCache

Precompute the δ-invariant components shared across a grid of exponential-tilt
strengths: numeric `a`/`y` vectors, `CovariateSchema`-encoded design matrices,
cross-fitting fold partitions and standardised fold bases, and (for `:plugin` /
`:plugin_poisson`) the δ-invariant Stage-1 outcome regression `μ̂(A, W)` on each
training fold. Thread-safe (read-only after construction).
"""
function _build_tilt_grid_cache(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::AbstractVector{Symbol};
    family_outcome::Symbol = :gaussian,
    folds::Int = 5,
    independent_folds::Bool = true,
    outcome_method::Symbol = :plugin,
    normalizer_covariates::AbstractVector{Symbol} = covariates,
    outcome_covariates::AbstractVector{Symbol} = covariates,
    rng::AbstractRNG = StableRNG(2026),
)
    family_outcome in (:gaussian, :binomial, :poisson, :negbin) || throw(ArgumentError(
        "family_outcome must be :gaussian, :binomial, :poisson, or :negbin; got $family_outcome",
    ))
    outcome_method in (:plugin, :plugin_poisson, :direct) || throw(ArgumentError(
        "unsupported outcome_method=$outcome_method; expected :plugin, :plugin_poisson, or :direct",
    ))
    n = nrow(df)
    n >= 4 || throw(ArgumentError("estimate_exponential_tilt requires at least 4 observations"))

    a = collect(Float64, df[!, trt])::Vector{Float64}
    y = collect(Float64, df[!, outcome])::Vector{Float64}
    all(isfinite, a) || throw(ArgumentError("treatment :$trt contains non-finite values"))
    all(isfinite, y) || throw(ArgumentError("outcome :$outcome contains non-finite values"))

    norm_covs = collect(Symbol, normalizer_covariates)
    out_covs = collect(Symbol, outcome_covariates)
    schema_m0 = fit_covariate_schema(df, norm_covs)
    W_m0 = transform_covariates(schema_m0, df)::Matrix{Float64}
    same_covs = (out_covs == norm_covs)
    W_y = if same_covs
        W_m0
    else
        schema_y = fit_covariate_schema(df, out_covs)
        transform_covariates(schema_y, df)::Matrix{Float64}
    end

    eff_folds = clamp(folds, 2, max(2, n ÷ 2))
    fold_sets_m0 = crossfit_indices(n, eff_folds, rng)
    fold_sets_y = independent_folds ?
        crossfit_indices(n, eff_folds, rng) : fold_sets_m0
    train_sets_m0 = [_complement_indices(n, test_idx) for test_idx in fold_sets_m0]
    train_sets_y = independent_folds ?
        [_complement_indices(n, test_idx) for test_idx in fold_sets_y] : train_sets_m0

    basis_m0 = _build_fold_bases(W_m0, train_sets_m0, fold_sets_m0)
    basis_a = independent_folds ? _build_fold_bases(W_m0, train_sets_y, fold_sets_y) : basis_m0
    basis_y = same_covs ? basis_a : _build_fold_bases(W_y, train_sets_y, fold_sets_y)

    mu_hat_folds = if outcome_method in (:plugin, :plugin_poisson)
        Vector{Float64}[
            _fit_stage1_mu_hat(
                basis_y[k].Psi_tr,
                @view(a[train_sets_y[k]]),
                @view(y[train_sets_y[k]]),
                family_outcome,
            )
            for k in eachindex(fold_sets_y)
        ]
    else
        nothing
    end

    return _TiltGridCache(
        n, trt, outcome, norm_covs, out_covs, family_outcome, outcome_method,
        eff_folds, independent_folds, a, y,
        fold_sets_m0, train_sets_m0, basis_m0,
        fold_sets_y, train_sets_y, basis_y, basis_a, mu_hat_folds,
    )
end

function _tilt_eif_summary(
    obs::Vector{Float64},
    theta_star::Vector{Float64},
    H::Vector{Float64};
    alpha::Real,
    compute_bootstrap::Bool,
    bootstrap_B::Int,
    bootstrap_weights::Symbol,
    rng::AbstractRNG,
)
    n = length(obs)
    psi = mean(theta_star)
    psi0 = mean(obs)
    est = psi - psi0
    tmle_score = mean(H .* (obs .- theta_star))
    ic = H .* (obs .- theta_star) .+ theta_star .- obs .- est
    se = std(ic) / sqrt(n)
    lower, upper = wald_ci(est, se; alpha = alpha)
    if compute_bootstrap
        boot = studentised_multiplier_ci(
            est, ic; alpha = alpha, B = bootstrap_B, weights = bootstrap_weights, rng = rng,
        )
        boot_lower, boot_upper = boot.lower, boot.upper
    else
        boot_lower, boot_upper = NaN, NaN
    end
    return (
        psi = psi, psi0 = psi0, estimate = est, tmle_score = tmle_score,
        ic = ic, se = se, lower = lower, upper = upper,
        boot_lower = boot_lower, boot_upper = boot_upper,
    )
end

"""
    estimate_exponential_tilt(df, trt, outcome, covariates, delta, policy; kwargs...) -> NamedTuple

Cross-fitted calibration-balanced TMLE for an exponential-tilting stochastic
intervention ``g_\\delta(a \\mid W) \\propto \\exp(\\delta \\phi(a)) g(a \\mid W)``.

Estimates both:
- Outcome contrast ``\\tau(\\delta) = \\psi(\\delta) - \\mathbb{E}[Y]`` with canonical-link
  TMLE targeting via `_weighted_intercept_tilt` and exact EIF
  ``\\varphi_\\delta(O) = H_\\delta(A, W)(Y - \\theta^*_\\delta(W)) + \\theta^*_\\delta(W) - Y - \\hat{\\tau}(\\delta)``.
- Realised mean exposure shift ``\\Delta(\\delta) = \\mathbb{E}_{g_\\delta}[A] - \\mathbb{E}[A]``
  with the matching Gaussian EIF.

Returns both Wald CIs (`lower`, `upper`) and studentised multiplier bootstrap CIs
(`boot_lower`, `boot_upper`), alongside weight ESS and TMLE score diagnostics.

# Keyword arguments
- `family_outcome::Symbol = :gaussian`: `:gaussian`, `:binomial`, `:poisson`, or `:negbin`.
- `folds::Int = 5`: number of cross-fitting folds (automatically capped to `n ÷ 2` if needed).
- `independent_folds::Bool = true`: use separate cross-fit partitions for ``\\hat{m}_0(W)`` and
  ``\\hat{\\theta}_\\delta(W)`` to attenuate finite-sample correlation in the second-order DR
  remainder (Newey & Robins 2018); set `false` for a single shared partition.
- `lambda::Real = 0.0`: optional L2 ridge penalty on standardised non-intercept normaliser coefficients.
- `outcome_method::Symbol = :plugin`: `:plugin` (two-stage Rao-Blackwellised), `:plugin_poisson`, or `:direct`.
- `enforce_bounds::Bool = true`: project bounded-policy ``\\log \\hat{m}_0(W)`` onto ``[-|\\delta|M, |\\delta|M]``.
- `normalizer_covariates = covariates`: override covariate columns used for ``\\hat{m}_0(W)`` (e.g. for DR tests).
- `outcome_covariates = covariates`: override covariate columns used for ``\\hat{\\theta}_\\delta(W)`` (e.g. for DR tests).
- `alpha::Real = 0.05`: two-sided significance level.
- `bootstrap_B::Int = 999`: replicates for studentised multiplier bootstrap.
- `bootstrap_weights::Symbol = :rademacher`: `:rademacher` (default) or `:mammen`.
- `compute_bootstrap::Bool = true`: compute studentised multiplier bootstrap CIs; when `false`,
  skips `studentised_multiplier_ci` and returns `NaN` for `boot_lower`, `boot_upper`,
  `shift_boot_lower`, and `shift_boot_upper`.
- `tilt_cache = nothing`: optional internal `_TiltGridCache` from `_build_tilt_grid_cache`
  to reuse δ-invariant covariate encodings, folds, and Stage-1 outcome fits across a δ grid.
- `rng::AbstractRNG = StableRNG(2026)`: RNG for cross-fitting and multiplier bootstrap.
"""
function estimate_exponential_tilt(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::AbstractVector{Symbol},
    delta::Real,
    policy::ExponentialTiltPolicy = exponential_tilt_policy();
    family_outcome::Symbol = :gaussian,
    folds::Int = 5,
    independent_folds::Bool = true,
    lambda::Real = 0.0,
    outcome_method::Symbol = :plugin,
    enforce_bounds::Bool = true,
    normalizer_covariates::AbstractVector{Symbol} = covariates,
    outcome_covariates::AbstractVector{Symbol} = covariates,
    alpha::Real = 0.05,
    bootstrap_B::Int = 999,
    bootstrap_weights::Symbol = :rademacher,
    compute_bootstrap::Bool = true,
    tilt_cache::Union{Nothing, _TiltGridCache} = nothing,
    rng::AbstractRNG = StableRNG(2026),
)
    family_outcome in (:gaussian, :binomial, :poisson, :negbin) || throw(ArgumentError(
        "family_outcome must be :gaussian, :binomial, :poisson, or :negbin; got $family_outcome",
    ))
    δ = Float64(delta)
    isfinite(δ) || throw(ArgumentError("delta must be finite; got $delta"))
    λ = Float64(lambda)
    (isfinite(λ) && λ >= 0.0) || throw(ArgumentError(
        "ridge penalty lambda must be finite and >= 0; got $lambda",
    ))
    n = nrow(df)
    n >= 4 || throw(ArgumentError("estimate_exponential_tilt requires at least 4 observations"))

    if tilt_cache !== nothing
        tilt_cache.n == n || throw(ArgumentError(
            "tilt_cache row count $(tilt_cache.n) does not match data nrow $n",
        ))
        (tilt_cache.trt === trt && tilt_cache.outcome === outcome) || throw(ArgumentError(
            "tilt_cache columns (:$(tilt_cache.trt), :$(tilt_cache.outcome)) do not match (:$trt, :$outcome)",
        ))
        (tilt_cache.family_outcome === family_outcome && tilt_cache.outcome_method === outcome_method) || throw(ArgumentError(
            "tilt_cache family/method ($(tilt_cache.family_outcome), $(tilt_cache.outcome_method)) does not match ($family_outcome, $outcome_method)",
        ))
        (tilt_cache.normalizer_covariates == normalizer_covariates && tilt_cache.outcome_covariates == outcome_covariates) || throw(ArgumentError(
            "tilt_cache covariates do not match requested normalizer/outcome covariates",
        ))
        a = tilt_cache.a
        y = tilt_cache.y
    else
        a = collect(Float64, df[!, trt])::Vector{Float64}
        y = collect(Float64, df[!, outcome])::Vector{Float64}
        all(isfinite, a) || throw(ArgumentError("treatment :$trt contains non-finite values"))
        all(isfinite, y) || throw(ArgumentError("outcome :$outcome contains non-finite values"))
    end

    w_bounds = tilt_weight_bounds(δ, policy)

    # Trivial zero-tilt shortcut: g_0 = g, τ(0) = 0, Δ(0) = 0
    if iszero(δ)
        if tilt_cache === nothing
            norm_covs = collect(Symbol, normalizer_covariates)
            out_covs = collect(Symbol, outcome_covariates)
            schema_m0 = fit_covariate_schema(df, norm_covs)
            transform_covariates(schema_m0, df)::Matrix{Float64}
            if out_covs != norm_covs
                schema_y = fit_covariate_schema(df, out_covs)
                transform_covariates(schema_y, df)::Matrix{Float64}
            end
        end
        H = ones(Float64, n)
        zeros_n = zeros(Float64, n)
        boot_val = compute_bootstrap ? 0.0 : NaN
        return (
            delta = 0.0,
            policy = policy,
            family_outcome = family_outcome,
            outcome_method = outcome_method,
            estimate = 0.0,
            psi = mean(y),
            psi0 = mean(y),
            se = 0.0,
            lower = 0.0,
            upper = 0.0,
            boot_lower = boot_val,
            boot_upper = boot_val,
            ic = zeros_n,
            tmle_score = 0.0,
            shift_estimate = 0.0,
            shift_se = 0.0,
            shift_lower = 0.0,
            shift_upper = 0.0,
            shift_boot_lower = boot_val,
            shift_boot_upper = boot_val,
            shift_ic = copy(zeros_n),
            shift_tmle_score = 0.0,
            H = H,
            theta_init = fill(mean(y), n),
            theta_star = fill(mean(y), n),
            theta_a_star = fill(mean(a), n),
            ess = Float64(n),
            ess_ratio = 1.0,
            mean_H = 1.0,
            min_H = 1.0,
            max_H = 1.0,
            weight_bounds = w_bounds,
            fold_fits = TiltNormalizerFit[],
            inference_status = :trivial,
        )
    end

    H = Vector{Float64}(undef, n)
    theta_init = Vector{Float64}(undef, n)
    theta_a_init = Vector{Float64}(undef, n)
    fold_fits = TiltNormalizerFit[]
    log_w_all = tilt_log_weight(a, δ, policy)

    if tilt_cache !== nothing
        # 1. Cross-fitted convex calibration-balanced normaliser m̂_0(W) using cached fold bases
        for k in eachindex(tilt_cache.fold_sets_m0)
            test_idx = tilt_cache.fold_sets_m0[k]
            train_idx = tilt_cache.train_sets_m0[k]
            b = tilt_cache.basis_m0[k]
            fit_m0 = _fit_tilt_normalizer_from_basis(
                b.Psi_tr, b.active_cols, b.col_means, b.col_sds, b.n_raw_cols,
                @view(log_w_all[train_idx]), δ, policy;
                lambda = λ,
            )
            push!(fold_fits, fit_m0)
            H[test_idx] = _predict_tilt_weights_from_basis(
                fit_m0, b.Psi_te, @view(log_w_all[test_idx]);
                enforce_bounds = enforce_bounds,
            )
        end

        # 2. Cross-fitted outcome nuisance θ̂_δ(W) and exposure nuisance θ̂^A_δ(W) using cached bases & Stage-1 μ̂
        for k in eachindex(tilt_cache.fold_sets_y)
            test_idx = tilt_cache.fold_sets_y[k]
            train_idx = tilt_cache.train_sets_y[k]
            log_w_tr = @view log_w_all[train_idx]
            max_lw = maximum(log_w_tr)
            w_tr = Vector{Float64}(undef, length(train_idx))
            @inbounds for i in eachindex(log_w_tr)
                w_tr[i] = exp(log_w_tr[i] - max_lw)
            end

            by = tilt_cache.basis_y[k]
            ba = tilt_cache.basis_a[k]
            mu_hat_k = tilt_cache.mu_hat_folds === nothing ? nothing : tilt_cache.mu_hat_folds[k]
            theta_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                by.Psi_tr, by.Psi_te, @view(a[train_idx]), @view(y[train_idx]), w_tr;
                family = family_outcome, outcome_method = outcome_method, mu_hat = mu_hat_k,
            )
            theta_a_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                ba.Psi_tr, ba.Psi_te, @view(a[train_idx]), @view(a[train_idx]), w_tr;
                family = :gaussian, outcome_method = :direct,
            )
        end
    else
        norm_covs = collect(Symbol, normalizer_covariates)
        out_covs = collect(Symbol, outcome_covariates)
        schema_m0 = fit_covariate_schema(df, norm_covs)
        W_m0 = transform_covariates(schema_m0, df)::Matrix{Float64}
        same_covs = (out_covs == norm_covs)
        W_y = if same_covs
            W_m0
        else
            schema_y = fit_covariate_schema(df, out_covs)
            transform_covariates(schema_y, df)::Matrix{Float64}
        end

        eff_folds = clamp(folds, 2, max(2, n ÷ 2))
        fold_sets_m0 = crossfit_indices(n, eff_folds, rng)
        fold_sets_y = independent_folds ?
            crossfit_indices(n, eff_folds, rng) : fold_sets_m0

        # 1. Cross-fitted convex calibration-balanced normaliser m̂_0(W)
        for test_idx in fold_sets_m0
            train_idx = _complement_indices(n, test_idx)
            Psi_tr, active_cols, col_means, col_sds = _fit_standardised_basis(@view W_m0[train_idx, :])
            Psi_te = _apply_standardised_basis(@view(W_m0[test_idx, :]), active_cols, col_means, col_sds, size(W_m0, 2))
            fit_m0 = _fit_tilt_normalizer_from_basis(
                Psi_tr, active_cols, col_means, col_sds, size(W_m0, 2),
                @view(log_w_all[train_idx]), δ, policy;
                lambda = λ,
            )
            push!(fold_fits, fit_m0)
            H[test_idx] = _predict_tilt_weights_from_basis(
                fit_m0, Psi_te, @view(log_w_all[test_idx]);
                enforce_bounds = enforce_bounds,
            )
        end

        # 2. Cross-fitted outcome nuisance θ̂_δ(W) and exposure nuisance θ̂^A_δ(W).
        # Shift log_w by its fold maximum before exponentiating so unbounded linear
        # tilts with large `center` never overflow or underflow; weighted GLMs only
        # depend on relative weights `w / mean(w)`.
        for test_idx in fold_sets_y
            train_idx = _complement_indices(n, test_idx)
            log_w_tr = @view log_w_all[train_idx]
            max_lw = maximum(log_w_tr)
            w_tr = Vector{Float64}(undef, length(train_idx))
            @inbounds for i in eachindex(log_w_tr)
                w_tr[i] = exp(log_w_tr[i] - max_lw)
            end

            Psi_y_tr, ac_y, cm_y, cs_y = _fit_standardised_basis(@view W_y[train_idx, :])
            Psi_y_te = _apply_standardised_basis(@view(W_y[test_idx, :]), ac_y, cm_y, cs_y, size(W_y, 2))
            theta_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                Psi_y_tr, Psi_y_te, @view(a[train_idx]), @view(y[train_idx]), w_tr;
                family = family_outcome, outcome_method = outcome_method,
            )

            if same_covs
                theta_a_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                    Psi_y_tr, Psi_y_te, @view(a[train_idx]), @view(a[train_idx]), w_tr;
                    family = :gaussian, outcome_method = :direct,
                )
            else
                Psi_a_tr, ac_a, cm_a, cs_a = _fit_standardised_basis(@view W_m0[train_idx, :])
                Psi_a_te = _apply_standardised_basis(@view(W_m0[test_idx, :]), ac_a, cm_a, cs_a, size(W_m0, 2))
                theta_a_init[test_idx] = _fit_predict_tilt_outcome_from_basis(
                    Psi_a_tr, Psi_a_te, @view(a[train_idx]), @view(a[train_idx]), w_tr;
                    family = :gaussian, outcome_method = :direct,
                )
            end
        end
    end

    # 3. Canonical-link one-step TMLE targeting via _weighted_intercept_tilt
    _, theta_star, _, boundary_y = _weighted_intercept_tilt(
        y, theta_init, theta_init, H, family_outcome, 1.0,
    )
    _, theta_a_star, _, boundary_a = _weighted_intercept_tilt(
        a, theta_a_init, theta_a_init, H, :gaussian, 1.0,
    )

    # 4. Outcome contrast τ(δ) = ψ(δ) - E[Y] and exact EIF
    out_sum = _tilt_eif_summary(
        y, theta_star, H;
        alpha = alpha, compute_bootstrap = compute_bootstrap,
        bootstrap_B = bootstrap_B, bootstrap_weights = bootstrap_weights, rng = rng,
    )

    # 5. Realised exposure shift Δ(δ) = E_{g_δ}[A] - E[A] and exact EIF
    shift_sum = _tilt_eif_summary(
        a, theta_a_star, H;
        alpha = alpha, compute_bootstrap = compute_bootstrap,
        bootstrap_B = bootstrap_B, bootstrap_weights = bootstrap_weights, rng = rng,
    )

    sum_H = sum(H)
    sum_H2 = sum(abs2, H)
    ess = sum_H2 > 0.0 ? (sum_H^2) / sum_H2 : 0.0
    inference_status = (boundary_y || boundary_a) ? :boundary_outcome : :available

    return (
        delta = δ,
        policy = policy,
        family_outcome = family_outcome,
        outcome_method = outcome_method,
        estimate = out_sum.estimate,
        psi = out_sum.psi,
        psi0 = out_sum.psi0,
        se = out_sum.se,
        lower = out_sum.lower,
        upper = out_sum.upper,
        boot_lower = out_sum.boot_lower,
        boot_upper = out_sum.boot_upper,
        ic = out_sum.ic,
        tmle_score = out_sum.tmle_score,
        shift_estimate = shift_sum.estimate,
        shift_se = shift_sum.se,
        shift_lower = shift_sum.lower,
        shift_upper = shift_sum.upper,
        shift_boot_lower = shift_sum.boot_lower,
        shift_boot_upper = shift_sum.boot_upper,
        shift_ic = shift_sum.ic,
        shift_tmle_score = shift_sum.tmle_score,
        H = H,
        theta_init = theta_init,
        theta_star = theta_star,
        theta_a_star = theta_a_star,
        ess = ess,
        ess_ratio = ess / n,
        mean_H = mean(H),
        min_H = minimum(H),
        max_H = maximum(H),
        weight_bounds = w_bounds,
        fold_fits = fold_fits,
        inference_status = inference_status,
    )
end
