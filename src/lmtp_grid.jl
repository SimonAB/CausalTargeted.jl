"""LMTP modified treatment policy grid (TE contrasts vs natural policy).

# References

- Díaz, Williams, Hoffman & Schenck (2023), *JASA* — LMTP identification and estimators
- Williams & Díaz (2023), *Observational Studies* — R `lmtp` software companion
- van der Laan & Rose (2011) — TMLE / Super Learner practice
"""

using DataFrames
using Statistics
using Random
using StableRNGs
using Base.Threads

# Policy traits for ExponentialTiltPolicy (defined in exponential_tilt.jl)
_policy_support_status(::ExponentialTiltPolicy) = :exponential_tilt_policy
_policy_requested_shift(::ExponentialTiltPolicy, delta::Real) = Float64(delta)
_policy_uses_fold_cache(::ExponentialTiltPolicy) = false

function _policy_validate_grid_options(
    ::ExponentialTiltPolicy,
    deltas;
    density_ratio::Symbol,
    estimator::Symbol,
    family_outcome::Symbol,
    cv_trunc::Bool,
    weight_truncation::Bool,
    positivity::Bool,
    imputation,
)
    _validate_explicit_policy_grid_common(; positivity, imputation)
    cv_trunc && throw(ArgumentError(
        "exponential tilt policy does not use weight truncation or cv_trunc=true",
    ))
    estimator in (:tmle, :itmle) || throw(ArgumentError(
        "exponential tilt policy in run_lmtp_grid supports canonical-link TMLE (estimator=:tmle or :itmle)",
    ))
    family_outcome in (:gaussian, :binomial, :poisson, :negbin) || throw(ArgumentError(
        "exponential tilt policy supports family_outcome in (:gaussian, :binomial, :poisson, :negbin); got $family_outcome",
    ))
    all(isfinite, deltas) || throw(ArgumentError(
        "exponential tilt policy requires finite deltas",
    ))
    return nothing
end

"""
    run_lmtp_grid(data, trt, outcome; baseline, kwargs...) -> DataFrame

Cross-fitted SuperLearner LMTP TMLE grid matching R `run_lmtp_grid()` (`shift_scale = "z"`).

Uses clamp-aware hybrid targeting: when many observations hit clamp bounds,
the TMLE fluctuation is down-weighted toward g-computation.

Defaults:
- `density_ratio = :gaussian` (stable at sheep *n*; use `:hybrid` / `:classification` for large *n*)
- `cv_trunc = false` (prespecified `trunc`; opt-in variance heuristic is not cross-validation)
- `estimator = :tmle` (score-solving; pass `:sdr` / `:eif` / `:itmle` as needed)
- `epochs = 1` (do not inherit mediation-grid `epochs=30`)
- `simultaneous = true` adds multiplier-bootstrap simultaneous bands
- `parallel = true` when `Threads.nthreads() > 1`
- `cache_nuisances = true` reuses fold outcome / exposure models across δ

Quantile clamping creates boundary masses under a continuous treatment law, so
ordinary nonparametric EIF intervals are unavailable for nonzero hard-clamped
shifts. Use `smooth_bounded_logit_shift_policy(L, U)` for a distinct smooth,
invertible policy with EIF intervals; bounds must be fixed in advance and
uncapped Gaussian ratios are required for those intervals.

Use `feasible_additive_shift_policy(L, U; sd_scale)` to interpret the grid
values in fixed SD units. This policy retains scores whose proposed shift
falls outside the bounds. It uses Gaussian density ratios, with both
preimages included where the shifted and retained branches overlap.

Use `proportional_headroom_policy(L, U)` for a deterministic, range-adaptive
contraction on prespecified bounds `[L, U]` with grid values `delta` in
`(-1, 1)`. Positive shifts move each exposure in `[L, U]` by a proportion
`delta` of its remaining headroom toward `U`, and negative shifts move by
`abs(delta)` of remaining headroom toward `L`, without boundary clamping.
It uses exact single-branch Gaussian density ratios and provides uncapped EIF
intervals with `weight_truncation=false`.

Use `exponential_tilt_policy(; bound = 2.0, center = 0.0)` for a stochastic
exponential-tilting intervention ``g_\\delta(a \\mid W) \\propto \\exp(\\delta \\phi(a)) g(a \\mid W)``
with calibration-balanced TMLE (Díaz & Hejazi 2020; Schindl, Shen & Kennedy 2024).
When `bound = M > 0`, ``\\phi(a) = M \\tanh((a - c) / M)`` automatically bounds the
density ratio in ``[\\exp(-2|\\delta|M), \\exp(2|\\delta|M)]`` without ad-hoc weight
truncation; `effective_shift` reports the estimated realised shift ``\\hat{\\Delta}(\\delta)``.
"""
function run_lmtp_grid(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    baseline::Vector{Symbol},
    deltas = default_deltas(),
    lower_q = mtp_settings().lower_q,
    upper_q = mtp_settings().upper_q,
    folds = mtp_settings().folds,
    epochs::Int = 3,
    stratify_by = resolved_stratify_by(),
    shift_scale = mtp_settings().shift_scale,
    learners_outcome = DEFAULT_SL_LEARNERS,
    learners_trt = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    simultaneous::Bool = true,
    n_boot_sim::Int = 999,
    alpha_sim::Real = 0.05,
    rng = StableRNG(42),
    parallel::Bool = nthreads() > 1,
    cache_nuisances::Bool = true,
    positivity::Bool = false,
    handle_missing::Symbol = :drop,
    imputation::Union{Nothing, ImputationDraws} = nothing,
    weight_truncation::Bool = true,
    policy::Union{Nothing, AbstractFixedBoundContinuousPolicy, ExponentialTiltPolicy} = nothing,
)
    validate_contrast_learners(learners_outcome; context = "run_lmtp_grid outcome")
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "unsupported LMTP estimator=$estimator; expected :tmle, :itmle, :eif, :aipw, or :sdr",
    ))
    if policy === nothing && any(d -> !isapprox(d, 0; atol = 1e-12), deltas)
        @warn "Hard-clamped continuous LMTP shifts map ranges of A to the bounds, creating boundary point masses. Ordinary nonparametric EIF intervals are unavailable; use smooth_bounded_logit_shift_policy with fixed bounds and no weight truncation for regular EIF inference." maxlog=1
    end
    _policy_validate_grid_options(
        policy, deltas;
        density_ratio, estimator, family_outcome, cv_trunc,
        weight_truncation, positivity, imputation,
    )
    if imputation !== nothing
        return _run_lmtp_grid_imputed(
            data, trt, outcome, imputation;
            baseline = baseline,
            deltas = deltas,
            lower_q = lower_q,
            upper_q = upper_q,
            folds = folds,
            epochs = epochs,
            stratify_by = stratify_by,
            shift_scale = shift_scale,
            learners_outcome = learners_outcome,
            learners_trt = learners_trt,
            family_outcome = family_outcome,
            density_ratio = density_ratio,
            estimator = estimator,
            trunc = trunc,
            cv_trunc = cv_trunc,
            simultaneous = simultaneous,
            n_boot_sim = n_boot_sim,
            alpha_sim = alpha_sim,
            rng = rng,
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            positivity = positivity,
            weight_truncation = weight_truncation,
        )
    end
    all_cols = unique(vcat(baseline, [trt]))
    miss = handle_missing_data(
        data, outcome, all_cols, handle_missing;
        rng = rng, rung = :L2, time_indexed = false,
    )
    data_clean, ipcw_w, extra_cols = miss
    if !isempty(extra_cols)
        baseline = unique(vcat(baseline, extra_cols))
    end
    validate_family_outcome(data_clean[!, outcome], family_outcome)
    df = make_analysis_strata(data_clean, stratify_by)
    pooled = stratify_by !== nothing
    adjust = columns_present(df, unique(vcat(baseline, pooled ? [stratify_by] : Symbol[])))
    adjust = [c for c in adjust if c != trt]
    a = try
        Float64.(df[!, trt])
    catch
        throw(ArgumentError(
            "treatment :$trt must be numeric; categorical treatment is not supported",
        ))
    end
    sd_a = std(a)
    L, U = exposure_bounds(a, lower_q, upper_q)
    density_values = _policy_density_values(policy, a)
    a_nat = policy === nothing ? apply_shift_policy(a, 0.0, L, U) : copy(a)
    nat_ref = copy(a_nat)

    diag_exp = sparse_exposure_diagnostic(a)
    if diag_exp.sparse && Base.get_extension(@__MODULE__, :CausalTargetedEvoTreesExt) !== nothing
        learners_outcome = (:evotree, :mean)
        learners_trt = (:evotree, :mean)
    end

    all_trivial = all(d -> isapprox(d, 0; atol = 1e-12), deltas)
    fold_cache = cache_nuisances && !all_trivial && _policy_uses_fold_cache(policy) ? build_lmtp_fold_cache(
        df, trt, outcome, adjust, folds, rng;
        learners_outcome = learners_outcome,
        learners_trt = learners_trt,
        family_outcome = family_outcome,
        weight_truncation = weight_truncation,
        exposure_values = density_values,
        fit_exposure = density_ratio !== :classification,
    ) : nothing

    strata = get_target_strata(df)
    tilt_caches = if cache_nuisances && !all_trivial && policy isa ExponentialTiltPolicy
        caches = Dict{String, _TiltGridCache}()
        for stratum in strata
            sub = pooled ? df[string.(df.STRAT) .== stratum, :] : df
            adjust_s = pooled ? [c for c in adjust if c != stratify_by] : adjust
            caches[stratum] = _build_tilt_grid_cache(
                sub, trt, outcome, adjust_s;
                family_outcome = family_outcome,
                folds = folds,
                rng = rng,
            )
        end
        caches
    else
        nothing
    end
    jobs = _parallel_delta_jobs(deltas, strata)
    base_seed = _rng_base_seed(rng)
    rows = NamedTuple[]
    stratum_ics = Dict{String, Vector{Tuple{Int, Vector{Float64}}}}()

    _run_job = function(j)
        d, stratum = jobs[j]
        local_rng = _job_rng(base_seed, j, stratum, d)
        _lmtp_delta_job(
            d, stratum, df, trt, outcome, adjust, a, nat_ref, sd_a,
            L, U, lower_q, upper_q, shift_scale, stratify_by, pooled,
            folds, local_rng, fold_cache;
            learners_outcome = learners_outcome,
            learners_trt = learners_trt,
            family_outcome = family_outcome,
            density_ratio = density_ratio,
            estimator = estimator,
            trunc = trunc,
            cv_trunc = cv_trunc,
            epochs = epochs,
            weight_truncation = weight_truncation,
            policy = policy,
            tilt_cache = tilt_caches === nothing ? nothing : get(tilt_caches, stratum, nothing),
        )
    end

    job_out = Vector{Tuple{
        NamedTuple, Union{Nothing, Vector{Float64}}, Union{Nothing, Vector{Float64}},
    }}(undef, length(jobs))
    if parallel && nthreads() > 1
        @threads for j in eachindex(jobs)
            job_out[j] = _run_job(j)
        end
    else
        for j in eachindex(jobs)
            job_out[j] = _run_job(j)
        end
    end
    for (j, (row, ic, point_score)) in enumerate(job_out)
        row, ic = _apply_ipcw_to_lmtp_row(
            row, ic, ipcw_w;
            point_score = point_score,
            inference_status = row.inference_status,
        )
        push!(rows, row)
        ic === nothing && continue
        stratum = string(jobs[j][2])
        push!(get!(stratum_ics, stratum, Tuple{Int, Vector{Float64}}[]), (length(rows), ic))
    end

    out = DataFrame(rows)
    out.lwr_sim = copy(out.lwr)
    out.upr_sim = copy(out.upr)
    out.crit_sim = fill(NaN, nrow(out))
    out.truncation_selection = fill(cv_trunc ? :variance_heuristic : :prespecified, nrow(out))

    if simultaneous
        for (stratum, entries) in stratum_ics
            length(entries) < 2 && continue
            lens = unique(length(ic) for (_, ic) in entries)
            length(lens) != 1 && continue
            ic_mat = hcat([ic for (_, ic) in entries]...)
            crit = multiplier_simultaneous_critical(
                ic_mat; n_boot = n_boot_sim, alpha = alpha_sim, rng = rng,
            )
            for (row_idx, _) in entries
                est = out.est[row_idx]
                se = out.se[row_idx]
                out.crit_sim[row_idx] = crit
                out.lwr_sim[row_idx] = est - crit * se
                out.upr_sim[row_idx] = est + crit * se
            end
        end
    end

    imputed_covariate = handle_missing in (:impute, :ipcw_impute) &&
        any(c -> get(miss.meta.miss_rates, c, 0.0) > 0, all_cols)
    if imputed_covariate
        nontrivial = .!isapprox.(out.delta, 0.0; atol = 1e-12)
        out.inference_status[nontrivial] .= :unsupported_covariate_imputation
        for col in (:se, :lwr, :upr, :lwr_sim, :upr_sim, :crit_sim)
            out[nontrivial, col] .= NaN
        end
    end

    sort!(out, [:stratum, :delta])
    if positivity
        rep = positivity_report(
            data, trt;
            deltas = deltas, stratify_by = stratify_by,
            lower_q = lower_q, upper_q = upper_q, shift_scale = shift_scale,
        )
        attach_positivity_summary!(out, rep)
    end
    attach_missingness_metadata!(out, miss.meta)
    return out
end

function _lmtp_delta_job(
    d::Float64,
    stratum::String,
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    adjust::Vector{Symbol},
    a::Vector{Float64},
    nat_ref::Vector{Float64},
    sd_a::Float64,
    L::Real,
    U::Real,
    lower_q::Real,
    upper_q::Real,
    shift_scale::String,
    stratify_by,
    pooled::Bool,
    folds::Int,
    rng,
    fold_cache;
    kwargs...,
)
    stratum_mask = BitVector(string.(df.STRAT) .== stratum)
    scale_by = pooled ? mean(stratum_mask) : 1.0
    policy = kwargs[:policy]
    if policy isa ExponentialTiltPolicy
        diag = (
            requested_shift = d,
            stratum_n = count(stratum_mask),
            stratum_clamp_prop = 0.0,
            global_clamp_prop = 0.0,
            effective_shift_mean = 0.0,
            shift_retention = NaN,
            support_status = :exponential_tilt_policy,
        )
        if isapprox(d, 0; atol = 1e-12)
            return _lmtp_row(d, 0.0, 0.0, 0.0, 0.0, diag, NaN, NaN, sd_a;
                severity = 0.0, stratum = stratum, weight_truncation = false), nothing, nothing
        end
        sub = pooled ? df[stratum_mask, :] : df
        adjust_s = pooled ? [c for c in adjust if c != stratify_by] : adjust
        tilt_cache = get(kwargs, :tilt_cache, nothing)
        try
            res = estimate_exponential_tilt(
                sub, trt, outcome, adjust_s, d, policy;
                family_outcome = kwargs[:family_outcome],
                folds = folds,
                compute_bootstrap = false,
                tilt_cache = tilt_cache,
                rng = rng,
            )
            diag = merge(diag, (effective_shift_mean = res.shift_estimate,))
            available = res.inference_status === :available
            se = available ? res.se : NaN
            lwr = available ? res.lower : NaN
            upr = available ? res.upper : NaN
            row = _lmtp_row(
                d, res.estimate, se, lwr, upr, diag, NaN, NaN, sd_a;
                severity = 0.0, stratum = stratum,
                targeting_weight = 1.0,
                targeting_kind = :weighted_intercept,
                weight_truncation = false,
                inference_status = res.inference_status,
            )
            full_ic = if pooled
                v = zeros(Float64, nrow(df))
                v[stratum_mask] .= res.ic ./ scale_by
                v
            else
                res.ic
            end
            point_score = full_ic .+ res.estimate
            return row, (available ? full_ic : nothing), point_score
        catch error
            error isa ArgumentError && rethrow()
            return _lmtp_row(d, NaN, NaN, NaN, NaN, diag, NaN, NaN, sd_a;
                stratum = stratum, targeting_weight = 1.0,
                targeting_kind = :weighted_intercept,
                weight_truncation = false,
                inference_status = :fit_failed), nothing, nothing
        end
    end
    fixed_policy = policy !== nothing
    diag = fixed_policy ? (
        requested_shift = _policy_requested_shift(policy, d),
        stratum_n = count(stratum_mask),
        stratum_clamp_prop = 0.0,
        global_clamp_prop = 0.0,
        effective_shift_mean = 0.0,
        shift_retention = NaN,
        support_status = _policy_support_status(policy),
    ) : support_diagnostics(
        df, trt, stratum, stratify_by, lower_q, upper_q, d, shift_scale;
        min_stratum_n = mtp_settings().min_stratum_n,
        max_stratum_clamp_prop = mtp_settings().max_stratum_clamp_prop,
        min_shift_retention = mtp_settings().min_shift_retention,
    )
    if isapprox(d, 0; atol = 1e-12)
        return _lmtp_row(d, 0.0, 0.0, 0.0, 0.0, diag,
            fixed_policy ? NaN : lower_q, fixed_policy ? NaN : upper_q, sd_a;
            severity = 0.0, stratum = stratum,
            weight_truncation = kwargs[:weight_truncation]), nothing, nothing
    end
    req = diag.requested_shift
    if !isfinite(req)
        return _lmtp_row(d, NaN, NaN, NaN, NaN, diag, lower_q, upper_q, sd_a;
            stratum = stratum,
            targeting_kind = _lmtp_targeting_kind(kwargs[:estimator]),
            weight_truncation = kwargs[:weight_truncation]), nothing, nothing
    end
    a_shift = fixed_policy ? apply_policy_values(
        a, d, policy; stratum_mask = pooled ? stratum_mask : nothing,
    ) : apply_shift_policy(a, req, L, U; stratum_mask = pooled ? stratum_mask : nothing)
    if fixed_policy
        active = pooled ? stratum_mask : trues(length(a))
        diag = merge(diag, (effective_shift_mean = mean(a_shift[active] .- a[active]),))
        if _policy_reports_retention(policy)
            diag = merge(diag, (shift_retention = count(a_shift[active] .!= a[active]) / count(active),))
        end
    end
    add_diag = fixed_policy ? (clamp = 0.0, severity = 0.0) :
        additive_clamp_diagnostics(pooled ? a[stratum_mask] : a, req, L, U)
    tw = fixed_policy ? 1.0 : targeting_weight_from_clamp(add_diag.clamp)
    try
        out = if fold_cache !== nothing
            comp = lmtp_components_from_cache(
                fold_cache, a_shift, nat_ref;
                density_ratio = kwargs[:density_ratio],
                trunc = kwargs[:trunc],
                cv_trunc = kwargs[:cv_trunc],
                weight_truncation = kwargs[:weight_truncation],
                L = fixed_policy ? nothing : L, U = fixed_policy ? nothing : U,
                shift_amount = fixed_policy ? nothing : req,
                shift_reference = 0.0,
                policy = policy,
                policy_delta = fixed_policy ? d : nothing,
                policy_mask = pooled ? stratum_mask : nothing,
            )
            lmtp_tmle_from_components(
                comp;
                estimator = kwargs[:estimator],
                targeting_weight = tw,
                epochs = kwargs[:epochs],
            )
        else
            lmtp_tmle_contrast(
                df, trt, outcome, adjust, a_shift, nat_ref, folds, rng;
                learners_outcome = kwargs[:learners_outcome],
                family_outcome = kwargs[:family_outcome],
                learners_trt = kwargs[:learners_trt],
                density_ratio = kwargs[:density_ratio],
                estimator = kwargs[:estimator],
                trunc = kwargs[:trunc],
                cv_trunc = kwargs[:cv_trunc],
                targeting_weight = tw,
                epochs = kwargs[:epochs],
                L = fixed_policy ? nothing : L, U = fixed_policy ? nothing : U,
                shift_amount = fixed_policy ? nothing : req,
                shift_reference = 0.0,
                policy = policy,
                policy_delta = fixed_policy ? d : nothing,
                policy_mask = pooled ? stratum_mask : nothing,
                weight_truncation = kwargs[:weight_truncation],
            )
        end
        est = out.estimate / scale_by
        targeting_kind = _lmtp_targeting_kind(kwargs[:estimator])
        available = out.inference_status === :available
        se = available ? out.se / scale_by : NaN
        lwr = available ? out.lower / scale_by : NaN
        upr = available ? out.upper / scale_by : NaN
        row = _lmtp_row(
            d, est, se, lwr, upr, diag, fixed_policy ? NaN : lower_q,
            fixed_policy ? NaN : upper_q, sd_a;
            severity = add_diag.severity, stratum = stratum,
            targeting_weight = tw,
            targeting_kind = targeting_kind,
            weight_truncation = kwargs[:weight_truncation],
            inference_status = out.inference_status,
        )
        point_score = hasproperty(out, :point_score) ? out.point_score ./ scale_by : nothing
        ic = available && hasproperty(out, :ic) ? out.ic ./ scale_by : nothing
        return row, ic, point_score
    catch error
        # Input and configuration errors must agree with the cached pathway.
        # Numerical/model failures remain tolerated as per-delta failure rows.
        error isa ArgumentError && rethrow()
        return _lmtp_row(d, NaN, NaN, NaN, NaN, diag,
            fixed_policy ? NaN : lower_q, fixed_policy ? NaN : upper_q, sd_a;
            stratum = stratum, targeting_weight = tw,
            targeting_kind = _lmtp_targeting_kind(kwargs[:estimator]),
            weight_truncation = kwargs[:weight_truncation],
            inference_status = fixed_policy ? :fit_failed : :unsupported_clamped_policy), nothing, nothing
    end
end

"""
    _lmtp_row(...) -> NamedTuple

Typed LMTP grid row (column names match the returned `DataFrame`).
"""
function _lmtp_row(
    d, est, se, lwr, upr, diag, lower_q, upper_q, sd_a;
    severity = 0.0,
    stratum = "full_population",
    targeting_weight = 0.0,
    targeting_kind = isapprox(d, 0; atol = 1e-12) ? :trivial : :weighted_intercept,
    weight_truncation = true,
    inference_status = isapprox(d, 0; atol = 1e-12) ? :trivial : :unsupported_clamped_policy,
)
    clamp_v = Float64(coalesce(diag.stratum_clamp_prop, diag.global_clamp_prop, 0.0))
    return (
        delta = Float64(d),
        estimand = "TE",
        est = Float64(est),
        se = Float64(se),
        lwr = Float64(lwr),
        upr = Float64(upr),
        clamp = clamp_v,
        severity = Float64(severity),
        effective_shift = Float64(diag.effective_shift_mean),
        shift_retention = Float64(diag.shift_retention),
        lower_q = Float64(lower_q),
        upper_q = Float64(upper_q),
        sd_exposure = Float64(sd_a),
        support_status = string(diag.support_status),
        stratum = string(stratum),
        targeting_weight = Float64(targeting_weight),
        targeting_kind = Symbol(targeting_kind),
        weight_truncation = Bool(weight_truncation),
        inference_status = Symbol(inference_status),
    )
end

"""Describe the estimator route used for a nonzero LMTP grid contrast."""
function _lmtp_targeting_kind(estimator::Symbol)
    return estimator in (:eif, :aipw, :sdr) ? :one_step : :weighted_intercept
end

export run_lmtp_grid

"""
    _run_lmtp_grid_imputed(data, trt, outcome, imputation; kwargs...) -> DataFrame

Run [`run_lmtp_grid`](@ref) on each completed draw (`handle_missing=:drop`) and
pool with Rubin's rule. Opt-in only via `imputation=`.
"""
function _run_lmtp_grid_imputed(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    imputation::ImputationDraws;
    kwargs...,
)
    imputation.outcome === outcome || throw(ArgumentError(
        "imputation outcome :$(imputation.outcome) does not match grid outcome :$outcome",
    ))
    grids = DataFrame[
        run_lmtp_grid(
            draw, trt, outcome;
            handle_missing = :drop,
            imputation = nothing,
            kwargs...,
        )
        for draw in imputation.draws
    ]
    out = pool_lmtp_grids(grids; rubin = true)
    out.targeting_weight = Float64.(grids[1].targeting_weight)
    out.targeting_kind = Symbol.(grids[1].targeting_kind)
    out.weight_truncation = Bool.(grids[1].weight_truncation)
    out.inference_status = Symbol.(grids[1].inference_status)
    nontrivial = .!isapprox.(out.delta, 0.0; atol = 1e-12)
    out.inference_status[nontrivial] .= :unsupported_clamped_policy
    for column in (:se, :lwr, :upr, :lwr_sim, :upr_sim, :crit_sim)
        out[nontrivial, column] .= NaN
    end
    meta = merge(imputation.meta, (
        strategy = :posterior_gaussian_mar,
        n_draws = imputation.n_draws,
        mar_set = imputation.mar_set,
    ))
    attach_missingness_metadata!(out, meta)
    return out
end

function _apply_ipcw_to_lmtp_row(
    row::NamedTuple,
    ic,
    ipcw_w::AbstractVector{<:Real};
    point_score::Union{Nothing, AbstractVector{<:Real}} = nothing,
    inference_status::Symbol = :available,
)
    ic === nothing && point_score === nothing && return row, ic
    _uses_ipcw_weights(ipcw_w) || return row, ic
    score = point_score === nothing ? ic .+ row.est : point_score
    length(score) == length(ipcw_w) || throw(ArgumentError(
        "IPCW weights length $(length(ipcw_w)) does not match point score $(length(score))",
    ))
    s = weighted_influence_summary(score, ipcw_w)
    if inference_status !== :available
        row = merge(row, (est = s.estimate, se = NaN, lwr = NaN, upr = NaN))
        return row, nothing
    end
    lwr, upr = wald_ci(s.estimate, s.se)
    row = merge(row, (est = s.estimate, se = s.se, lwr = lwr, upr = upr))
    return row, s.ic
end
