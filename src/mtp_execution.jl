"""Parallel δ-grid execution and typed estimand dispatch."""

using DataFrames
using Base.Threads
using CausalDynamics
using Random
using StableRNGs

"""Keyword subsets for grid drivers (avoid splatting incompatible options)."""
function _lmtp_grid_kwargs(kwargs)
    allowed = (
        :deltas, :lower_q, :upper_q, :folds, :epochs, :stratify_by, :shift_scale,
        :learners_outcome, :learners_trt, :density_ratio, :estimator, :trunc,
        :cv_trunc, :simultaneous, :n_boot_sim, :alpha_sim, :rng,
        :parallel, :cache_nuisances, :positivity,
    )
    return (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
end

function _mediation_grid_kwargs(kwargs, n_mc::Int)
    allowed = (
        :deltas, :lower_q, :upper_q, :folds, :epochs, :stratify_by, :shift_scale,
        :trunc, :rng, :parallel, :cache_nuisances, :positivity, :moc, :estimator, :effect,
    )
    base = (; (p.first => p.second for p in pairs(kwargs) if p.first in allowed)...)
    learners = get(kwargs, :learners, get(kwargs, :learners_outcome, DEFAULT_SL_LEARNERS))
    return merge(base, (; learners = learners, n_mc = get(kwargs, :n_mc, n_mc)))
end
const _crumble_grid_kwargs = _mediation_grid_kwargs  # legacy

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
        )])
    elseif estimand isa DiscreteInterventionalMean
        allowed = (
            :folds, :rng, :learners_outcome, :learners_trt, :density_ratio,
            :estimator, :trunc, :epochs, :handle_missing,
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
            stratum = "full_population",
        )])
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
        density_ratio_meta = if estimand isa DiscreteInterventionalMean || seq_factor
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
    return UInt(mod(hash(rng), typemax(UInt)))
end

"""
    _job_rng(base_seed, job_index, stratum, delta) -> StableRNG

Deterministic per-job RNG so parallel δ-jobs do not share one seed stream.
"""
function _job_rng(base_seed::UInt, job_index::Integer, stratum, delta)
    return StableRNG(hash((base_seed, Int(job_index), string(stratum), Float64(delta))))
end

export execute_estimand
