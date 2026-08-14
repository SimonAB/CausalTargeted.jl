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

"""
    run_lmtp_grid(data, trt, outcome; baseline, kwargs...) -> DataFrame

Cross-fitted SuperLearner LMTP TMLE grid matching R `run_lmtp_grid()` (`shift_scale = "z"`).

Uses clamp-aware hybrid targeting: when many observations hit clamp bounds,
the TMLE fluctuation is down-weighted toward g-computation.

Defaults:
- `density_ratio = :gaussian` (stable at sheep *n*; use `:hybrid` / `:classification` for large *n*)
- `cv_trunc = true` (select hard truncation among candidates)
- `estimator = :tmle` (score-solving; pass `:sdr` / `:eif` / `:itmle` as needed)
- `epochs = 1` (do not inherit mediation-grid `epochs=30`)
- `simultaneous = true` adds multiplier-bootstrap simultaneous bands
- `parallel = true` when `Threads.nthreads() > 1`
- `cache_nuisances = true` reuses fold outcome / exposure models across δ
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
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = true,
    simultaneous::Bool = true,
    n_boot_sim::Int = 999,
    alpha_sim::Real = 0.05,
    rng = StableRNG(42),
    parallel::Bool = nthreads() > 1,
    cache_nuisances::Bool = true,
    positivity::Bool = false,
    handle_missing::Symbol = :drop,
)
    validate_contrast_learners(learners_outcome; context = "run_lmtp_grid outcome")
    all_cols = unique(vcat(baseline, [trt]))
    data_clean, ipcw_w, extra_cols = handle_missing_data(data, outcome, all_cols, handle_missing; rng = rng)
    if !isempty(extra_cols)
        baseline = unique(vcat(baseline, extra_cols))
    end
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
    a_nat = apply_shift_policy(a, 0.0, L, U)
    nat_ref = copy(a_nat)

    diag_exp = sparse_exposure_diagnostic(a)
    if diag_exp.sparse && Base.get_extension(@__MODULE__, :CausalTargetedEvoTreesExt) !== nothing
        learners_outcome = (:evotree, :mean)
        learners_trt = (:evotree, :mean)
    end

    fold_cache = cache_nuisances ? build_lmtp_fold_cache(
        df, trt, outcome, adjust, folds, rng;
        learners_outcome = learners_outcome,
        learners_trt = learners_trt,
    ) : nothing

    strata = get_target_strata(df)
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
            density_ratio = density_ratio,
            estimator = estimator,
            trunc = trunc,
            cv_trunc = cv_trunc,
            epochs = epochs,
        )
    end

    if parallel && nthreads() > 1
        job_out = Vector{Tuple{NamedTuple, Union{Nothing, Vector{Float64}}}}(undef, length(jobs))
        @threads for j in eachindex(jobs)
            job_out[j] = _run_job(j)
        end
        for (j, (row, ic)) in enumerate(job_out)
            row, ic = _apply_ipcw_to_lmtp_row(row, ic, ipcw_w)
            push!(rows, row)
            ic === nothing && continue
            stratum = string(jobs[j][2])
            push!(get!(stratum_ics, stratum, Tuple{Int, Vector{Float64}}[]), (length(rows), ic))
        end
    else
        for j in eachindex(jobs)
            row, ic = _run_job(j)
            row, ic = _apply_ipcw_to_lmtp_row(row, ic, ipcw_w)
            push!(rows, row)
            ic === nothing && continue
            stratum = string(jobs[j][2])
            push!(get!(stratum_ics, stratum, Tuple{Int, Vector{Float64}}[]), (length(rows), ic))
        end
    end

    out = DataFrame(rows)
    out.lwr_sim = copy(out.lwr)
    out.upr_sim = copy(out.upr)
    out.crit_sim = fill(NaN, nrow(out))

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

    sort!(out, [:stratum, :delta])
    if positivity
        rep = positivity_report(
            data, trt;
            deltas = deltas, stratify_by = stratify_by,
            lower_q = lower_q, upper_q = upper_q, shift_scale = shift_scale,
        )
        attach_positivity_summary!(out, rep)
    end
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
    diag = support_diagnostics(
        df, trt, stratum, stratify_by, lower_q, upper_q, d, shift_scale;
        min_stratum_n = mtp_settings().min_stratum_n,
        max_stratum_clamp_prop = mtp_settings().max_stratum_clamp_prop,
        min_shift_retention = mtp_settings().min_shift_retention,
    )
    if isapprox(d, 0; atol = 1e-12)
        return _lmtp_row(d, 0.0, 0.0, 0.0, 0.0, diag, lower_q, upper_q, sd_a;
            severity = 0.0, stratum = stratum), nothing
    end
    req = diag.requested_shift
    if !isfinite(req)
        return _lmtp_row(d, NaN, NaN, NaN, NaN, diag, lower_q, upper_q, sd_a; stratum = stratum), nothing
    end
    a_shift = apply_shift_policy(a, req, L, U; stratum_mask = pooled ? stratum_mask : nothing)
    add_diag = additive_clamp_diagnostics(pooled ? a[stratum_mask] : a, req, L, U)
    tw = targeting_weight_from_clamp(add_diag.clamp)
    try
        out = if fold_cache !== nothing
            comp = lmtp_components_from_cache(
                fold_cache, a_shift, nat_ref;
                density_ratio = kwargs[:density_ratio],
                trunc = kwargs[:trunc],
                cv_trunc = kwargs[:cv_trunc],
                L = L, U = U,
                shift_policy = req,
                shift_reference = 0.0,
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
                learners_trt = kwargs[:learners_trt],
                density_ratio = kwargs[:density_ratio],
                estimator = kwargs[:estimator],
                trunc = kwargs[:trunc],
                cv_trunc = kwargs[:cv_trunc],
                targeting_weight = tw,
                epochs = kwargs[:epochs],
                L = L, U = U,
                shift_policy = req,
                shift_reference = 0.0,
            )
        end
        est = out.estimate / scale_by
        se = out.se / scale_by
        lwr, upr = wald_ci(est, se)
        row = _lmtp_row(
            d, est, se, lwr, upr, diag, lower_q, upper_q, sd_a;
            severity = add_diag.severity, stratum = stratum,
        )
        ic_entry = nothing
        if all(isfinite, out.ic)
            ic_entry = out.ic ./ scale_by
        end
        return row, ic_entry
    catch error
        # Input and configuration errors must agree with the cached pathway.
        # Numerical/model failures remain tolerated as per-delta failure rows.
        error isa ArgumentError && rethrow()
        return _lmtp_row(d, NaN, NaN, NaN, NaN, diag, lower_q, upper_q, sd_a; stratum = stratum), nothing
    end
end

"""
    _lmtp_row(...) -> NamedTuple

Typed LMTP grid row (column names match the returned `DataFrame`).
"""
function _lmtp_row(d, est, se, lwr, upr, diag, lower_q, upper_q, sd_a; severity = 0.0, stratum = "full_population")
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
    )
end

export run_lmtp_grid

function _apply_ipcw_to_lmtp_row(row::NamedTuple, ic, ipcw_w::AbstractVector{<:Real})
    ic === nothing && return row, ic
    _uses_ipcw_weights(ipcw_w) || return row, ic
    length(ic) == length(ipcw_w) || throw(ArgumentError(
        "IPCW weights length $(length(ipcw_w)) does not match influence curve $(length(ic))",
    ))
    # `lmtp_tmle_contrast` returns a mean-zero IC; restore the uncentred curve
    ic_uncent = ic .+ row.est
    s = weighted_influence_summary(ic_uncent, ipcw_w)
    lwr, upr = wald_ci(s.estimate, s.se)
    row = merge(row, (est = s.estimate, se = s.se, lwr = lwr, upr = upr))
    return row, s.ic
end
