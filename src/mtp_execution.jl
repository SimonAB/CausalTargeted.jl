"""Parallel δ-grid execution and typed estimand dispatch."""

using DataFrames
using Base.Threads
using CausalDynamics
using Random
using StableRNGs

"""
    _lmtp_grid_kwargs(kwargs) -> NamedTuple

Return a `NamedTuple` containing only keyword arguments that [`run_lmtp_grid`](@ref)
accepts, so a wider keyword list (e.g. from [`execute_estimand`](@ref)) can be
splatted without passing unknown options.
"""
function _lmtp_grid_kwargs(kwargs)
    allowed = (
        :deltas, :lower_q, :upper_q, :folds, :epochs, :stratify_by, :shift_scale,
        :learners_outcome, :learners_trt, :family_outcome, :density_ratio, :estimator, :trunc,
        :cv_trunc, :weight_truncation, :simultaneous, :n_boot_sim, :alpha_sim, :rng,
        :parallel, :cache_nuisances, :positivity, :handle_missing, :imputation,
    )
    return (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
end

"""
    _mediation_grid_kwargs(kwargs, n_mc::Int) -> NamedTuple

Like [`_lmtp_grid_kwargs`](@ref), but for [`run_mediation_grid`](@ref): filters
`kwargs`, sets `learners` from `:learners` or `:learners_outcome`, and merges
`n_mc` (defaulting to the second argument when `:n_mc` is absent from `kwargs`).
"""
function _mediation_grid_kwargs(kwargs, n_mc::Int)
    allowed = (
        :deltas, :lower_q, :upper_q, :folds, :epochs, :stratify_by, :shift_scale,
        :trunc, :weight_truncation, :rng, :parallel, :cache_nuisances, :positivity, :moc, :estimator, :effect,
    )
    base = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
    learners = get(kwargs, :learners, get(kwargs, :learners_outcome, DEFAULT_SL_LEARNERS))
    return merge(base, (; learners = learners, n_mc = get(kwargs, :n_mc, n_mc)))
end
const _crumble_grid_kwargs = _mediation_grid_kwargs  # legacy

"""
    _sequential_lmtp_kwargs(kwargs) -> NamedTuple

Keyword subset for [`run_sequential_lmtp`](@ref). Maps the last element of
`:deltas` to `:delta` when `:delta` is missing, and `:learners_outcome` to
`:learners` when `:learners` is missing.
"""
function _sequential_lmtp_kwargs(kwargs)
    allowed = (
        :delta, :folds, :learners, :learners_trt, :lower_q, :upper_q, :shift,
        :policies, :rng, :baseline, :time_vary, :handle_missing, :trunc,
    )
    base = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
    if !haskey(base, :delta) && haskey(kwargs, :deltas)
        ds = kwargs[:deltas]
        d = ds isa AbstractVector ? Float64(last(ds)) : Float64(ds)
        base = merge(base, (; delta = d))
    end
    if !haskey(base, :learners) && haskey(kwargs, :learners_outcome)
        base = merge(base, (; learners = kwargs[:learners_outcome]))
    end
    return base
end

"""
    _survival_lmtp_kwargs(kwargs) -> NamedTuple

Keyword subset for [`run_survival_lmtp`](@ref), with the same `:deltas` → `:delta`
and `:learners_outcome` → `:learners` fallbacks as [`_sequential_lmtp_kwargs`](@ref).
"""
function _survival_lmtp_kwargs(kwargs)
    allowed = (
        :delta, :folds, :learners, :lower_q, :upper_q, :shift, :rng,
        :baseline, :time_vary, :censor, :horizon,
    )
    base = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
    if !haskey(base, :delta) && haskey(kwargs, :deltas)
        ds = kwargs[:deltas]
        d = ds isa AbstractVector ? Float64(last(ds)) : Float64(ds)
        base = merge(base, (; delta = d))
    end
    if !haskey(base, :learners) && haskey(kwargs, :learners_outcome)
        base = merge(base, (; learners = kwargs[:learners_outcome]))
    end
    return base
end

"""
    execute_estimand(estimand, data; kwargs...) -> DataFrame

Dispatch a structural [`Estimand`](@ref) to the appropriate targeted estimator and
return a `DataFrame` of point estimates, (where available) standard errors, and
intervals, with one or more rows depending on the estimand (e.g. δ grid,
mediation effects, MSM coefficients).

Plans nuisances via [`plan_mtp`](@ref) when `plan` is `nothing`. Sequential policies
may be updated from `id_result` via [`plan_sequential`](@ref). Keyword arguments
are filtered per branch and forwarded to the underlying `run_*` routine
(`run_lmtp_grid`, `run_mediation_grid`, `run_sequential_lmtp`, `run_survival_lmtp`,
`run_discrete_lmtp`, `run_two_part_discrete_lmtp`, `run_repeated_outcome_msm`, …).

# Main arguments
- `estimand`: instance of a concrete `Estimand` type (continuous or discrete LMTP,
  mediation, sequential, survival, two-part, MSM, …).
- `data`: `DataFrame` containing the columns named in `estimand`.

# Common keywords
- `id_result`: optional identification result for sequential policies.
- `plan`: optional pre-built `MTPPlan`; if omitted, built from `estimand` and `data`.
- `parallel`: parallelise grid or stratified jobs when `true` (default: `true` if
  `Base.Threads.nthreads() > 1`).
- `cache_nuisances`: passed through to nuisance estimation.
- `metadata`: if `true`, attach fold/learner/estimator/density-ratio metadata to the frame.
- `nuisance_source`, `temporal_lags`, `n_mc`: passed to [`plan_mtp`](@ref) and
  mediation drivers as applicable.

Remaining `kwargs` are passed only insofar as each branch allows (e.g. `folds`, `rng`,
`learners_outcome`, `estimator`, `trunc`).

!!! note "Continuous LMTP contrasts"
    Pointwise continuous interventions may return rows with limited or no inferential
    uncertainty; see the specific `run_*` documentation.
"""
function execute_estimand(
    estimand::Estimand,
    data::DataFrame;
    id_result::Union{Nothing, IdentificationResult} = nothing,
    plan::Union{Nothing, MTPPlan} = nothing,
    parallel::Bool = (nthreads() > 1),
    cache_nuisances::Bool = true,
    metadata::Bool = true,
    nuisance_source::Symbol = :graph,
    temporal_lags::Union{Nothing, NamedTuple} = nothing,
    n_mc::Int = 32,
    kwargs...,
)
    if estimand isa SequentialPolicy && id_result !== nothing
        estimand = plan_sequential(estimand, id_result)
    end

    plan === nothing && (plan = plan_mtp(
        estimand, data; id_result = id_result, nuisance_source = nuisance_source,
        temporal_lags = temporal_lags, n_mc = n_mc, kwargs...,
    ))

    df = if estimand isa InterventionalMean
        run_lmtp_grid(
            data, estimand.trt, estimand.outcome;
            baseline = estimand.adjustment,
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            lower_q = estimand.shift.lower_q,
            upper_q = estimand.shift.upper_q,
            shift_scale = estimand.shift.scale,
            _lmtp_grid_kwargs(kwargs)...,
        )
    elseif estimand isa MediationContrast
        run_mediation_grid(
            data, estimand.trt, estimand.outcome;
            covar = estimand.adjustment,
            mediators = estimand.mediators,
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            lower_q = estimand.shift.lower_q,
            upper_q = estimand.shift.upper_q,
            shift_scale = estimand.shift.scale,
            _mediation_grid_kwargs(kwargs, n_mc)...,
        )
    elseif estimand isa LongitudinalPolicy
        run_lmtp_grid(
            data, estimand.trt, estimand.outcome;
            baseline = estimand.adjustment,
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            lower_q = estimand.shift.lower_q,
            upper_q = estimand.shift.upper_q,
            shift_scale = estimand.shift.scale,
            _lmtp_grid_kwargs(kwargs)...,
        )
    elseif estimand isa ScalarMediation
        run_mediation_scalar(
            data, estimand.trt, estimand.outcome;
            covar = estimand.adjustment,
            mediators = estimand.mediators,
            kwargs...,
        )
    elseif estimand isa SequentialPolicy
        res = run_sequential_lmtp(data, estimand; _sequential_lmtp_kwargs(kwargs)...)
        DataFrame([(
            delta = res.delta,
            estimand = "TE",
            est = res.estimate,
            se = res.se,
            lwr = res.estimate - 1.96 * res.se,
            upr = res.estimate + 1.96 * res.se,
            times = res.times,
            stratum = "full_population",
            inference_status = res.inference_status,
        )])
    elseif estimand isa SurvivalPolicy
        res = run_survival_lmtp(data, estimand; _survival_lmtp_kwargs(kwargs)...)
        DataFrame([(
            delta = res.delta,
            estimand = "survival",
            est = res.estimate,
            se = res.se,
            lwr = res.estimate - 1.96 * res.se,
            upr = res.estimate + 1.96 * res.se,
            times = res.times,
            horizon = res.horizon,
            stratum = "full_population",
            inference_status = res.inference_status,
        )])
    elseif estimand isa DiscreteInterventionalMean
        allowed = (
            :folds, :rng, :learners_outcome, :learners_trt, :density_ratio,
            :estimator, :trunc, :epochs, :handle_missing, :family_outcome,
            :weight_truncation, :cluster, :fold_groups,
        )
        disc_kw = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
        res = run_discrete_lmtp(
            data, estimand.trt, estimand.outcome;
            policy = estimand.policy,
            baseline = estimand.adjustment,
            disc_kw...,
        )
        DataFrame([(
            delta = NaN,
            estimand = "TE",
            est = res.estimate,
            se = res.se,
            lwr = res.lower,
            upr = res.upper,
            n_changed = res.n_changed,
            positivity_ok = res.positivity.ok,
            inference_status = res.inference_status,
            stratum = "full_population",
        )])
    elseif estimand isa TwoPartInterventionalMean
        allowed = (
            :folds, :rng, :learners_outcome, :learners_trt, :density_ratio,
            :estimator, :trunc, :epochs, :handle_missing, :cluster,
            :family_presence, :family_intensity, :weight_truncation, :fold_groups,
        )
        tp_kw = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
        res = run_two_part_discrete_lmtp(
            data, estimand.trt;
            presence = estimand.presence,
            intensity = estimand.intensity,
            policy = estimand.policy,
            baseline = estimand.adjustment,
            tp_kw...,
        )
        DataFrame([
            (
                delta = NaN,
                estimand = "presence",
                est = res.presence.estimate,
                se = res.presence.se,
                lwr = res.presence.lower,
                upr = res.presence.upper,
                n_changed = res.presence.n_changed,
                positivity_ok = res.presence.positivity.ok,
                inference_status = res.presence.inference_status,
                stratum = "full_population",
            ),
            (
                delta = NaN,
                estimand = "intensity",
                est = res.intensity.estimate,
                se = res.intensity.se,
                lwr = res.intensity.lower,
                upr = res.intensity.upper,
                n_changed = res.intensity.n_changed,
                positivity_ok = res.intensity.positivity.ok,
                inference_status = res.intensity.inference_status,
                stratum = "full_population",
            ),
        ])
    elseif estimand isa RepeatedOutcomeMSM
        allowed = (
            :folds, :rng, :learners, :learners_outcome, :learners_trt,
            :handle_missing, :estimator, :cluster, :strata, :propensity,
        )
        msm_kw = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
        learners = get(msm_kw, :learners,
            get(msm_kw, :learners_outcome, DEFAULT_SL_LEARNERS))
        res = run_repeated_outcome_msm(
            data, estimand.trt, estimand.outcomes;
            baseline = estimand.adjustment,
            learners = learners,
            (; (p.first => p.second for p in pairs(msm_kw) if p.first in (
                :folds, :rng, :learners_trt, :handle_missing, :estimator, :cluster,
                :strata, :propensity,
            ))...)...,
        )
        z = 1.96
        index = res.parameter_index
        df_out = DataFrame(
            delta = fill(NaN, length(res.estimates)),
            estimand = fill("TE", length(res.estimates)),
            outcome = String.([x.outcome for x in index]),
            est = res.estimates,
            se = res.se,
            lwr = res.estimates .- z .* res.se,
            upr = res.estimates .+ z .* res.se,
            positivity_ok = fill(res.positivity.ok, length(res.estimates)),
            stratum = [x.stratum === nothing ? "full_population" : string(x.stratum) for x in index],
        )
        metadata!(df_out, "causal_targeted_msm_covariance", res.covariance; style = :note)
        attach_missingness_metadata!(df_out, res.missingness)
        df_out
    elseif estimand isa ParametricRepeatedOutcomeMSM
        allowed = (
            :folds, :rng, :learners, :learners_outcome, :learners_trt,
            :handle_missing, :estimator, :cluster, :strata, :propensity,
        )
        msm_kw = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
        learners = get(msm_kw, :learners,
            get(msm_kw, :learners_outcome, DEFAULT_SL_LEARNERS))
        res = run_parametric_repeated_msm(
            data, estimand.trt, estimand.outcomes;
            baseline = estimand.adjustment,
            design = estimand.design,
            target = estimand.target,
            learners = learners,
            (; (p.first => p.second for p in pairs(msm_kw) if p.first in (
                :folds, :rng, :learners_trt, :handle_missing, :estimator, :cluster,
                :strata, :propensity,
            ))...)...,
        )
        z = 1.96
        p = length(res.coefficients)
        df_out = DataFrame(
            delta = fill(NaN, p),
            estimand = fill("MSM", p),
            coefficient = String.(res.coef_names),
            est = res.coefficients,
            se = res.se,
            lwr = res.coefficients .- z .* res.se,
            upr = res.coefficients .+ z .* res.se,
            positivity_ok = fill(res.positivity.ok, p),
            stratum = fill("full_population", p),
        )
        metadata!(df_out, "causal_targeted_msm_covariance", res.covariance; style = :note)
        metadata!(df_out, "causal_targeted_fitted_tau", res.fitted_tau; style = :note)
        attach_missingness_metadata!(df_out, res.missingness)
        df_out
    else
        error("Unsupported estimand type $(typeof(estimand))")
    end

    if metadata
        seq_factor = estimand isa SequentialPolicy && (
            !isempty(estimand.policies) ||
            let p = get(kwargs, :policies, DiscreteTreatmentPolicy[])
                p isa DiscreteTreatmentPolicy || !isempty(p)
            end
        )
        density_ratio_meta = if estimand isa RepeatedOutcomeMSM ||
                estimand isa ParametricRepeatedOutcomeMSM
            :propensity
        elseif estimand isa DiscreteInterventionalMean || seq_factor
            get(kwargs, :density_ratio, :classification)
        else
            get(kwargs, :density_ratio, :gaussian)
        end
        meta = build_run_metadata(
            estimand, plan.certificate;
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            folds = get(kwargs, :folds, mtp_settings().folds),
            epochs = get(kwargs, :epochs, 1),
            estimator = get(kwargs, :estimator, :tmle),
            density_ratio = density_ratio_meta,
            learners_outcome = get(kwargs, :learners_outcome, DEFAULT_SL_LEARNERS),
            learners_trt = get(kwargs, :learners_trt, DEFAULT_SL_LEARNERS),
            n_mc = get(kwargs, :n_mc, n_mc),
        )
        attach_run_metadata!(df, meta)
    end
    return df
end

"""
    _parallel_delta_jobs(deltas, strata) -> Vector{Tuple}
"""
function _parallel_delta_jobs(deltas, strata)
    jobs = Tuple{Float64, String}[]
    for stratum in strata
        for d in deltas
            push!(jobs, (d, stratum))
        end
    end
    return jobs
end

"""
    _rng_base_seed(rng) -> UInt

Stable base seed derived from an `AbstractRNG` for forking per-job streams.
"""
function _rng_base_seed(rng::AbstractRNG)
    return CausalDynamics.stable_rng_seed(rng)
end

"""
    _job_rng(base_seed, job_index, stratum, delta) -> StableRNG

Deterministic per-job RNG so parallel δ-jobs do not share one seed stream.
"""
function _job_rng(base_seed::UInt, job_index::Integer, stratum, delta)
    return StableRNG(CausalDynamics.stable_seed(
        base_seed, Int(job_index), string(stratum), Float64(delta),
    ))
end

export execute_estimand
