"""Run provenance metadata bridging CausalDynamics identification."""

using DataFrames
using Dates
using CausalDynamics

"""
    IdentificationCertificate

Application-layer certificate wrapping [`IdentificationResult`](@ref) with column symbols.
"""
struct IdentificationCertificate
    trt::Symbol
    outcome::Symbol
    result::IdentificationResult
    adjustment::Vector{Symbol}
    mediators::Vector{Symbol}
    nuisance_source::Symbol
    temporal_lags::Union{Nothing, NamedTuple}
end

"""
    identification_certificate(result, trt, outcome; adjustment, mediators, source, temporal_lags) -> IdentificationCertificate
"""
function identification_certificate(
    result::IdentificationResult,
    trt::Symbol,
    outcome::Symbol;
    adjustment::Vector{Symbol} = Symbol.(result.adjustment),
    mediators::Vector{Symbol} = Symbol.(result.mediators),
    nuisance_source::Symbol = :graph,
    temporal_lags::Union{Nothing, NamedTuple} = nothing,
)
    return IdentificationCertificate(
        trt, outcome, result,
        adjustment, mediators,
        nuisance_source, temporal_lags,
    )
end

"""
    certificate_dict(cert) -> Dict{String, Any}
"""
function certificate_dict(cert::IdentificationCertificate)
    r = cert.result
    base = CausalDynamics.certificate_dict(r)
    return Dict{String, Any}(
        "id_trt" => string(cert.trt),
        "id_outcome" => string(cert.outcome),
        "id_adjustment" => join(string.(cert.adjustment), ","),
        "id_mediators" => join(string.(cert.mediators), ","),
        "id_identifiable" => r.identifiable,
        "id_strategy" => string(r.strategy),
        "id_graph_hash" => string(r.graph_hash, base = 16),
        "id_nuisance_source" => string(cert.nuisance_source),
        "id_temporal_treat_lag" => cert.temporal_lags === nothing ? missing : cert.temporal_lags.treat_lag,
        "id_temporal_outcome_lag" => cert.temporal_lags === nothing ? missing : cert.temporal_lags.outcome_lag,
    )
end

"""
    RunMetadata

Provenance for a single MTP grid execution.
"""
struct RunMetadata
    estimand_type::String
    engine::Symbol
    estimator::String
    density_ratio::String
    learners_outcome::String
    folds::Int
    epochs::Int
    parallel::Bool
    cache_nuisances::Bool
    package_version::String
    run_utc::String
    certificate::IdentificationCertificate
end

function pkg_version_string()
    path = joinpath(pkgdir(@__MODULE__), "Project.toml")
    isfile(path) || return "0.1.0"
    for line in eachline(path)
        m = match(r"^version\s*=\s*\"([^\"]+)\"", line)
        m !== nothing && return m.captures[1]
    end
    return "0.1.0"
end

"""
    build_run_metadata(estimand, cert; kwargs...) -> RunMetadata
"""
function build_run_metadata(
    estimand::Estimand,
    cert::IdentificationCertificate;
    estimator::Symbol = :tmle,
    density_ratio::Symbol = :gaussian,
    learners_outcome = DEFAULT_SL_LEARNERS,
    folds::Int = mtp_settings().folds,
    epochs::Int = 1,
    parallel::Bool = false,
    cache_nuisances::Bool = false,
)
    return RunMetadata(
        string(typeof(estimand)),
        estimand_engine(estimand),
        string(estimator),
        string(density_ratio),
        join(string.(collect(learners_outcome)), ","),
        folds, epochs, parallel, cache_nuisances,
        pkg_version_string(),
        string(Dates.now(UTC)),
        cert,
    )
end

"""
    metadata_dict(meta) -> Dict{String, Any}
"""
function metadata_dict(meta::RunMetadata)
    d = certificate_dict(meta.certificate)
    merge!(d, Dict{String, Any}(
        "meta_estimand_type" => meta.estimand_type,
        "meta_engine" => string(meta.engine),
        "meta_estimator" => meta.estimator,
        "meta_density_ratio" => meta.density_ratio,
        "meta_learners" => meta.learners_outcome,
        "meta_folds" => meta.folds,
        "meta_epochs" => meta.epochs,
        "meta_parallel" => meta.parallel,
        "meta_cache" => meta.cache_nuisances,
        "meta_package_version" => meta.package_version,
        "meta_run_utc" => meta.run_utc,
    ))
    return d
end

"""
    attach_run_metadata!(df, meta) -> DataFrame
"""
function attach_run_metadata!(df::DataFrame, meta::RunMetadata)
    cols = metadata_dict(meta)
    for (k, v) in cols
        df[!, k] = fill(v, nrow(df))
    end
    return df
end

export IdentificationCertificate, RunMetadata
export identification_certificate, certificate_dict
export build_run_metadata, metadata_dict, attach_run_metadata!
