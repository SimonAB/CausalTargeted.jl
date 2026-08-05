"""MTP execution planning: cost estimates and validation."""

using DataFrames
using CausalDynamics

"""
    MTPPlan

Validated execution plan for one estimand grid.
"""
struct MTPPlan
    estimand::Estimand
    certificate::IdentificationCertificate
    deltas::Vector{Float64}
    strata::Vector{String}
    n_obs::Int
    estimated_fits::Int
    estimated_seconds::Float64
end

"""
    plan_mtp(estimand, data; id_result, deltas, ...) -> MTPPlan

When `id_result` is provided (from [`identify`](@ref)), it is wrapped into the
certificate; otherwise a minimal certificate is built from the estimand fields.
"""
function plan_mtp(
    estimand::Estimand,
    data::DataFrame;
    id_result::Union{Nothing, IdentificationResult} = nothing,
    deltas = default_deltas(),
    stratify_by = resolved_stratify_by(),
    folds::Int = mtp_settings().folds,
    epochs::Int = 1,
    n_mc::Int = 32,
    nuisance_source::Symbol = :graph,
    temporal_lags::Union{Nothing, NamedTuple} = nothing,
    kwargs...,
)
    df = make_analysis_strata(data, stratify_by)
    strata = get_target_strata(df)
    n = nrow(df)
    warn_if_folds_too_large(n, folds)

    trt, out, adj, meds = _estimand_fields(estimand)
    if temporal_lags === nothing && estimand isa LongitudinalPolicy
        temporal_lags = (treat_lag = estimand.treat_lag, outcome_lag = estimand.outcome_lag)
    elseif temporal_lags === nothing && estimand isa SequentialPolicy
        temporal_lags = (
            treat_lag = 1,
            outcome_lag = length(estimand.treatments),
        )
    end

    cert = if id_result !== nothing
        identification_certificate(
            id_result, trt, out;
            adjustment = adj,
            mediators = meds,
            nuisance_source = nuisance_source,
            temporal_lags = temporal_lags,
        )
    else
        stub_query = if estimand isa SequentialPolicy
            TemporalEffectQuery(trt, out, 1, length(estimand.treatments))
        else
            TotalEffectQuery(trt, out)
        end
        stub = IdentificationResult(
            query = stub_query,
            graph_hash = UInt64(0),
            adjustment = adj,
            mediators = meds,
            moc = Symbol[],
            strategy = estimand isa SequentialPolicy ? :temporal_sequential : :unspecified,
            identifiable = true,
            assumptions = Symbol[],
            temporal_nodes = Tuple{Symbol, Int}[],
        )
        identification_certificate(
            stub, trt, out;
            adjustment = adj, mediators = meds,
            nuisance_source = nuisance_source,
            temporal_lags = temporal_lags,
        )
    end

    !cert.result.identifiable && @warn "Effect may not be identifiable" trt out

    n_delta = count(d -> !isapprox(d, 0; atol = 1e-12), deltas)
    engine = normalize_engine(estimand_engine(estimand))

    fits_per_delta = if engine == :lmtp
        folds * 2
    elseif engine == :mediation
        folds * (3 + length(cert.mediators)) * max(epochs, 1)
    else
        folds * (3 + length(cert.mediators))
    end

    n_effects = engine == :mediation ? 3 : 1
    estimated_fits = length(strata) * n_delta * fits_per_delta * n_effects
    engine == :mediation && (estimated_fits *= n_mc)

    sec_per_fit = n > 150 ? 0.15 : 0.05
    estimated_seconds = estimated_fits * sec_per_fit

    return MTPPlan(estimand, cert, collect(deltas), strata, n, estimated_fits, estimated_seconds)
end

function _estimand_fields(estimand::Estimand)
    if estimand isa InterventionalMean
        return estimand.trt, estimand.outcome, estimand.adjustment, Symbol[]
    elseif estimand isa MediationContrast
        return estimand.trt, estimand.outcome, estimand.adjustment, estimand.mediators
    elseif estimand isa LongitudinalPolicy
        return estimand.trt, estimand.outcome, estimand.adjustment, Symbol[]
    elseif estimand isa ScalarMediation
        return estimand.trt, estimand.outcome, estimand.adjustment, estimand.mediators
    elseif estimand isa SequentialPolicy
        return first(estimand.treatments), estimand.outcome, estimand.baseline, Symbol[]
    end
    error("Unknown estimand $(typeof(estimand))")
end

"""
    summarise_plan(plan) -> String
"""
function summarise_plan(plan::MTPPlan)
    eng = estimand_engine(plan.estimand)
    trt, out, _, _ = _estimand_fields(plan.estimand)
    return """
    MTP plan ($trt → $out, engine=$eng)
      n=$(plan.n_obs), strata=$(length(plan.strata)), δ-grid=$(length(plan.deltas))
      estimated nuisance fits ≈ $(plan.estimated_fits)
      estimated runtime ≈ $(round(plan.estimated_seconds; digits=1))s
      identifiable=$(plan.certificate.result.identifiable), strategy=$(plan.certificate.result.strategy)
    """
end

export MTPPlan, plan_mtp, summarise_plan
