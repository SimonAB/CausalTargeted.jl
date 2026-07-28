"""Parallel δ-grid execution and typed estimand dispatch."""

using DataFrames
using Base.Threads
using CausalDynamics

"""
    execute_estimand(estimand, data; id_result, plan, parallel, cache_nuisances, metadata, kwargs...) -> DataFrame
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
    kwargs...,
)
    plan === nothing && (plan = plan_mtp(
        estimand, data; id_result = id_result, nuisance_source = nuisance_source, kwargs...,
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
            kwargs...,
        )
    elseif estimand isa MediationContrast
        run_crumble_grid(
            data, estimand.trt, estimand.outcome;
            covar = estimand.adjustment,
            mediators = estimand.mediators,
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            lower_q = estimand.shift.lower_q,
            upper_q = estimand.shift.upper_q,
            shift_scale = estimand.shift.scale,
            kwargs...,
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
            kwargs...,
        )
    elseif estimand isa ScalarMediation
        run_crumble_scalar(
            data, estimand.trt, estimand.outcome;
            covar = estimand.adjustment,
            mediators = estimand.mediators,
            kwargs...,
        )
    else
        error("Unsupported estimand type $(typeof(estimand))")
    end

    if metadata
        meta = build_run_metadata(
            estimand, plan.certificate;
            parallel = parallel,
            cache_nuisances = cache_nuisances,
            folds = get(kwargs, :folds, mtp_settings().folds),
            epochs = get(kwargs, :epochs, 1),
            estimator = get(kwargs, :estimator, :tmle),
            density_ratio = get(kwargs, :density_ratio, :gaussian),
            learners_outcome = get(kwargs, :learners_outcome, DEFAULT_SL_LEARNERS),
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

export execute_estimand, _parallel_delta_jobs
