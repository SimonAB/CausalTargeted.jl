"""LMTP modified treatment policy grid (TE contrasts vs natural policy)."""

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
- `epochs = 1` (do not inherit crumble's `epochs=30`)
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
    epochs::Int = 1,
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
)
    df = make_analysis_strata(data, stratify_by)
    pooled = stratify_by !== nothing
    adjust = columns_present(df, unique(vcat(baseline, pooled ? [stratify_by] : Symbol[])))
    adjust = [c for c in adjust if c != trt]
    a = Float64.(df[!, trt])
    sd_a = std(a)
    L, U = exposure_bounds(a, lower_q, upper_q)
    a_nat = apply_shift_policy(a, 0.0, L, U)
    nat_ref = copy(a_nat)

    diag_exp = sparse_exposure_diagnostic(a)
    if diag_exp.sparse
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
    rows = Dict{String, Any}[]
    stratum_ics = Dict{String, Vector{Tuple{Int, Vector{Float64}}}}()

    _run_job = function(j)
        d, stratum = jobs[j]
        _lmtp_delta_job(
            d, stratum, df, trt, outcome, adjust, a, nat_ref, sd_a,
            L, U, lower_q, upper_q, shift_scale, stratify_by, pooled,
            folds, rng, fold_cache;
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
        job_out = Vector{Tuple{Dict{String, Any}, Union{Nothing, Vector{Float64}}}}(undef, length(jobs))
        @threads for j in eachindex(jobs)
            job_out[j] = _run_job(j)
        end
        for (j, (row, ic)) in enumerate(job_out)
            push!(rows, row)
            ic === nothing && continue
            stratum = string(jobs[j][2])
            push!(get!(stratum_ics, stratum, Tuple{Int, Vector{Float64}}[]), (length(rows), ic))
        end
    else
        for j in eachindex(jobs)
            row, ic = _run_job(j)
            push!(rows, row)
            ic === nothing && continue
            stratum = string(jobs[j][2])
            push!(get!(stratum_ics, stratum, Tuple{Int, Vector{Float64}}[]), (length(rows), ic))
        end
    end

    for r in rows
        r["lwr_sim"] = r["lwr"]
        r["upr_sim"] = r["upr"]
        r["crit_sim"] = NaN
    end

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
                est = rows[row_idx]["est"]
                se = rows[row_idx]["se"]
                rows[row_idx]["crit_sim"] = crit
                rows[row_idx]["lwr_sim"] = est - crit * se
                rows[row_idx]["upr_sim"] = est + crit * se
            end
        end
    end

    out = DataFrame(rows)
    sort!(out, [:stratum, :delta])
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
    local_rng = StableRNG(42)
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
                df, trt, outcome, adjust, a_shift, nat_ref, folds, local_rng;
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
    catch
        return _lmtp_row(d, NaN, NaN, NaN, NaN, diag, lower_q, upper_q, sd_a; stratum = stratum), nothing
    end
end

function _lmtp_row(d, est, se, lwr, upr, diag, lower_q, upper_q, sd_a; severity = 0.0, stratum = "full_population")
    return Dict(
        "delta" => d,
        "estimand" => "TE",
        "est" => est,
        "se" => se,
        "lwr" => lwr,
        "upr" => upr,
        "clamp" => coalesce(diag.stratum_clamp_prop, diag.global_clamp_prop, 0.0),
        "severity" => severity,
        "effective_shift" => diag.effective_shift_mean,
        "shift_retention" => diag.shift_retention,
        "lower_q" => lower_q,
        "upper_q" => upper_q,
        "sd_exposure" => sd_a,
        "support_status" => diag.support_status,
        "stratum" => string(stratum),
    )
end

export run_lmtp_grid
