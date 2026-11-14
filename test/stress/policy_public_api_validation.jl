"""Finite-draw public policy validation, conditional on the selected Gaussian law."""
module PolicyPublicAPIValidation

using CausalTargeted, RxInfer, GraphPPL
using DataFrames, CSV, SHA, TOML, LinearAlgebra, Statistics, Random, StableRNGs
include("policy_prior_factorial_support.jl")
const Support = PolicyPriorFactorialSupport
include(joinpath(@__DIR__, "..", "logistic_tail_reference.jl"))
const Reference = LogisticTailReference

export parse_options, assess_endpoint_rules, checked_gaussian_endpoints,
    evaluate_public_dataset, public_api_summaries, main

function parse_options(arguments)
    options = Dict{Symbol,Any}(:profile=>"pilot", :replications=>nothing, :streams=>20,
        :draws=>[500, 2000, 8000], :seed=>20261115, :out=>nothing, :cells=>String[])
    i = 1
    while i <= length(arguments)
        key = startswith(arguments[i], "--") ? Symbol(arguments[i][3:end]) : :unknown
        haskey(options, key) || throw(ArgumentError("unknown option $(arguments[i])"))
        i < length(arguments) || throw(ArgumentError("missing value for $(arguments[i])"))
        value = arguments[i+1]
        options[key] = key in (:replications, :streams, :seed) ? parse(Int, value) :
            key == :draws ? parse.(Int, split(value, ',')) : key == :cells ? String.(split(value, ',')) : value
        i += 2
    end
    options[:profile] in ("pilot", "main") || throw(ArgumentError("profile must be pilot or main"))
    isnothing(options[:replications]) && (options[:replications] = options[:profile] == "pilot" ? 2 : 100)
    options[:replications] > 0 && options[:streams] > 0 || throw(ArgumentError("replications and streams must be positive"))
    options[:seed] >= 0 || throw(ArgumentError("seed must be nonnegative"))
    !isempty(options[:draws]) && all(>=(2), options[:draws]) && allunique(options[:draws]) ||
        throw(ArgumentError("draw counts must be distinct integers ≥ 2"))
    isnothing(options[:out]) && (options[:out] = joinpath(@__DIR__, "public_api_" * options[:profile]))
    return (; options...)
end

"""Check Gaussian-law quantile refinement and independent tail/normalisation diagnostics."""
function assess_endpoint_rules(working, tight; scale)
    endpoint_change = max(abs(working.lower-tight.lower), abs(working.upper-tight.upper))
    mass_change = abs(expm1(working.log_mass-tight.log_mass))
    error = max(working.error, tight.error, working.residual, tight.residual, working.closure, tight.closure)
    finite = all(isfinite, (working.lower, working.upper, tight.lower, tight.upper, error, endpoint_change, mass_change))
    status = !finite ? "invalid_reference" : error > 5e-6 ? "integration_error" :
        mass_change > 5e-6 ? "mass_disagreement" : endpoint_change > 1e-4 ? "endpoint_disagreement" : "ok"
    return (; available=status == "ok", status, lower=scale*tight.lower, upper=scale*tight.upper,
        error, endpoint_change, mass_change)
end

function endpoint_probe(mean, covariance; rtol, atol)
    rule = Reference.bivariate_policy_reference(zeros(0, 2), Float64[], mean, covariance; rtol, atol)
    tails = Reference.policy_tails(rule)
    closure = maximum(abs(Reference.policy_tail_details(rule, x).probability +
        Reference.policy_tail_details(rule, x; upper_tail=true).probability - 1) for x in (tails.lower, tails.upper))
    return (; lower=tails.lower, upper=tails.upper, error=tails.maximum_integration_error,
        residual=tails.maximum_tail_residual, closure, log_mass=log(rule.mass)+rule.log_peak)
end

"""Integrate the ideal policy endpoints of a fitted Gaussian, not the full posterior."""
function checked_gaussian_endpoints(fit, scale)
    working = endpoint_probe(fit.posterior_mean, fit.posterior_covariance; rtol=1e-7, atol=1e-10)
    tight = endpoint_probe(fit.posterior_mean, fit.posterior_covariance; rtol=1e-9, atol=1e-12)
    return assess_endpoint_rules(working, tight; scale)
end

fit_public(data, prior, method) = fit_bayesian_logistic(data, :Y, :A, Symbol[];
    prior=GaussianCoefficientPrior(; mean=prior.mean, covariance=prior.covariance),
    covariance=method, maxiter=1000, tol=1e-10)
contrast_public(fit, data; draws, rng) = bayesian_policy_contrast(fit, data,
    discrete_static_policy(1); deltas=[0], n_draws=draws, mc_level=0.95, rng)

reference_failure(status) = (; available=false, status, lower=NaN, upper=NaN,
    error=NaN, endpoint_change=NaN, mass_change=NaN)
interrupted(exception) = exception isa InterruptException && throw(exception)

"""Fit each public covariance mode once, then vary only independent draw streams.

Current public APIs refit for each covariance mode: two public fits per
dataset/prior. The LRVB fit includes its correction; this driver does not claim
to time the correction separately. Reference failure never suppresses a public
evaluation. Fresh identical stream seeds couple draw counts, priors and methods.
"""
function evaluate_public_dataset(cell, replication; seed=20261115, streams=20,
        draws=[500, 2000, 8000], fitter=fit_public,
        ideal_reference=checked_gaussian_endpoints, public_contrast=contrast_public,
        methods=(:variational, :linear_response), priors=nothing)
    streams > 0 && all(>=(2), draws) && !isempty(draws) || throw(ArgumentError("invalid streams or draw counts"))
    dataset = Support.simulate_factorial_dataset(cell, replication; seed)
    data = DataFrame(A=dataset.A, Y=dataset.y)
    priors = isnothing(priors) ? filter(p -> p.deployable, Support.prior_specs(cell, dataset.beta)) : priors
    all(p -> p.deployable, priors) || throw(ArgumentError("public validation permits deployable priors only"))
    rows = NamedTuple[]
    for prior in priors, method in methods
        fit, fit_exception = nothing, ""
        fit_timing = @timed try
            fit = fitter(data, prior, method)
        catch exception
            interrupted(exception)
            fit_exception = sprint(showerror, exception)
        end
        ideal = reference_failure(fit === nothing ? "fit_exception" : "fit_unavailable")
        reference_exception = ""
        reference_timing = @timed if fit !== nothing && fit.metadata.uncertainty_available
            try
                ideal = ideal_reference(fit, dataset.policy_scale)
            catch exception
                interrupted(exception)
                ideal = reference_failure("reference_exception")
                reference_exception = sprint(showerror, exception)
            end
        end
        for stream in 1:streams, n_draws in draws
            # The stream ID extends the dataset ID but includes no fitted prior/method/draw count.
            draw_seed = Support.PolicyCalibrationSupport.replication_seed(dataset.dataset_id, stream, seed, 1)
            row = (; cell_id=String(cell.id), dataset_id=dataset.dataset_id, replication,
                prior_id=String(prior.id), prior_scale=prior.scale, method=string(method),
                stream, draws=n_draws, draw_seed, seed, n=cell.n, control_n=cell.control_n,
                policy_truth=dataset.policy_truth, fit_calls_per_dataset_prior=length(methods),
                fit_status=fit === nothing ? "fit_exception" : string(fit.status),
                fit_seconds=fit_timing.time, fit_bytes=fit_timing.bytes,
                fit_compilation_seconds=fit_timing.compile_time,
                reference_available=ideal.available, reference_status=string(ideal.status),
                ideal_lower=ideal.lower, ideal_upper=ideal.upper, reference_error=ideal.error,
                reference_endpoint_change=ideal.endpoint_change, reference_mass_change=ideal.mass_change,
                reference_seconds=reference_timing.time, reference_bytes=reference_timing.bytes,
                reference_compilation_seconds=reference_timing.compile_time,
                available=false, status=fit === nothing ? "fit_exception" : "not_run",
                lower=NaN, upper=NaN, policy_mean=NaN, mean_mcse=NaN, width=NaN,
                lower_endpoint_error=NaN, upper_endpoint_error=NaN,
                lower_endpoint_squared_error=NaN, upper_endpoint_squared_error=NaN,
                lower_mc_lower=NaN, lower_mc_upper=NaN, upper_mc_lower=NaN, upper_mc_upper=NaN,
                lower_mc_contains_ideal=missing, upper_mc_contains_ideal=missing,
                draw_seconds=0.0, draw_bytes=0, draw_compilation_seconds=0.0,
                endpoint_mc_scope="conditional_on_gaussian_moments", endpoint_mc_simultaneous=false,
                exception=fit_exception, reference_exception)
            draw_timing = @timed if fit !== nothing
                try
                    result = public_contrast(fit, data; draws=n_draws, rng=StableRNG(draw_seed))
                    available = result.metadata.uncertainty != :unavailable &&
                        all(isfinite, (only(result.lower), only(result.upper), only(result.mcse)))
                    lower, upper = only(result.lower), only(result.upper)
                    lb, ub = result.endpoint_mc_bounds.lower_endpoint, result.endpoint_mc_bounds.upper_endpoint
                    lower_error = available && ideal.available ? lower-ideal.lower : NaN
                    upper_error = available && ideal.available ? upper-ideal.upper : NaN
                    row = merge(row, (; available, status=available ? "ok" : "uncertainty_unavailable",
                        lower, upper, policy_mean=only(result.mean), mean_mcse=only(result.mcse), width=upper-lower,
                        lower_endpoint_error=lower_error, upper_endpoint_error=upper_error,
                        lower_endpoint_squared_error=lower_error^2, upper_endpoint_squared_error=upper_error^2,
                        lower_mc_lower=only(lb.lower), lower_mc_upper=only(lb.upper),
                        upper_mc_lower=only(ub.lower), upper_mc_upper=only(ub.upper),
                        lower_mc_contains_ideal=available && ideal.available ? only(lb.lower) <= ideal.lower <= only(lb.upper) : missing,
                        upper_mc_contains_ideal=available && ideal.available ? only(ub.lower) <= ideal.upper <= only(ub.upper) : missing,
                        endpoint_mc_scope=string(result.metadata.endpoint_mc_scope),
                        endpoint_mc_simultaneous=result.metadata.endpoint_mc_simultaneous))
                catch exception
                    interrupted(exception)
                    row = merge(row, (; status="public_exception", exception=sprint(showerror, exception)))
                end
            end
            push!(rows, merge(row, (; draw_seconds=draw_timing.time, draw_bytes=draw_timing.bytes,
                draw_compilation_seconds=draw_timing.compile_time)))
        end
    end
    return rows
end

const SUMMARY_METRICS = (:lower_endpoint_error, :upper_endpoint_error,
    :lower_endpoint_squared_error, :upper_endpoint_squared_error,
    :lower_mc_contains_ideal, :upper_mc_contains_ideal, :width, :mean_mcse, :draw_seconds)
finite_real(x) = x isa Real && isfinite(x)

"""Average streams within datasets before computing sample-SD Monte Carlo errors.

Counts on descriptive metrics are independent datasets, never stream rows.
Endpoint errors/bound inclusion require an available ideal-law reference as well
as public uncertainty; public availability has its own all-attempt denominator.
"""
function public_api_summaries(rows)
    groups = Dict{Tuple{String,String,String,Int},Vector{Any}}()
    seen = Set{Tuple{String,String,String,Int,String,Int}}()
    for row in rows
        key = (String(row.cell_id), String(row.prior_id), string(row.method), Int(row.draws))
        unique_key = (key..., String(row.dataset_id), Int(row.stream))
        unique_key in seen && throw(ArgumentError("duplicate public validation stream"))
        push!(seen, unique_key)
        push!(get!(groups, key, Any[]), row)
    end
    summaries = NamedTuple[]
    for key in sort!(collect(keys(groups)))
        group = groups[key]
        dataset_ids = sort!(unique(String(r.dataset_id) for r in group))
        summary = (; cell_id=key[1], prior_id=key[2], method=key[3], draws=key[4],
            datasets=length(dataset_ids), attempted_streams=length(group),
            available_streams=count(r -> r.available, group),
            reference_available_streams=count(r -> r.reference_available, group),
            scope="conditional_on_gaussian_moments", simultaneous=false)
        for (name, field) in ((:availability, :available), (:reference_availability, :reference_available))
            dataset_means = [mean(Float64(getproperty(r, field)) for r in group if r.dataset_id == id) for id in dataset_ids]
            n = length(dataset_means)
            summary = merge(summary, NamedTuple{(Symbol(name, :_mean), Symbol(name, :_mcse), Symbol(name, :_count))}(
                (mean(dataset_means), n < 2 ? NaN : std(dataset_means)/sqrt(n), n)))
        end
        for name in SUMMARY_METRICS
            dataset_means = Float64[]
            for id in dataset_ids
                values = Float64[getproperty(r, name) for r in group if r.dataset_id == id &&
                    (name == :draw_seconds || r.available) &&
                    (!(name in (:lower_mc_contains_ideal, :upper_mc_contains_ideal,
                        :lower_endpoint_error, :upper_endpoint_error,
                        :lower_endpoint_squared_error, :upper_endpoint_squared_error)) || r.reference_available) &&
                    hasproperty(r, name) && finite_real(getproperty(r, name))]
                isempty(values) || push!(dataset_means, mean(values))
            end
            n = length(dataset_means)
            statistics = (n == 0 ? NaN : mean(dataset_means), n < 2 ? NaN : std(dataset_means)/sqrt(n), n)
            summary = merge(summary, NamedTuple{(Symbol(name, :_mean), Symbol(name, :_mcse), Symbol(name, :_count))}(statistics))
        end
        push!(summaries, summary)
    end
    return summaries
end

function source_hashes()
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = [@__FILE__, joinpath(@__DIR__, "policy_prior_factorial_support.jl"),
        joinpath(@__DIR__, "policy_calibration_support.jl"), joinpath(@__DIR__, "..", "logistic_tail_reference.jl"),
        joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "Manifest.toml"), joinpath(package_root, "Project.toml")]
    for directory in ("src", "ext"), (root, _, files) in walkdir(joinpath(package_root, directory))
        append!(paths, [joinpath(root, file) for file in files if endswith(file, ".jl")])
    end
    return Dict(relpath(path, package_root)=>bytes2hex(sha256(read(path))) for path in paths if isfile(path))
end

"""Run an explicit pilot/main subset; output directories are never overwritten.

Interrupted runs retain completed raw dataset blocks. Resume is deliberately
unsupported here; the factorial driver owns the validated resumable archive.
"""
function main(arguments=ARGS)
    options = parse_options(arguments)
    cells = Support.factorial_cells()
    if !isempty(options.cells)
        all(id -> any(c -> c.id == id, cells), options.cells) || throw(ArgumentError("unknown selected cell"))
        cells = filter(c -> c.id in options.cells, cells)
    end
    isdir(options.out) && !isempty(readdir(options.out)) && throw(ArgumentError("output directory must be empty"))
    mkpath(options.out)
    BLAS.set_num_threads(1)
    metadata = Dict{String,Any}("schema"=>"public-api-v2", "scope"=>"conditional_on_gaussian_moments",
        "simultaneous"=>false, "replications"=>options.replications, "streams"=>options.streams,
        "draws"=>options.draws, "seed"=>options.seed, "cells"=>[c.id for c in cells],
        "source_hashes"=>source_hashes(), "julia"=>string(VERSION), "threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(), "cpu"=>Sys.CPU_NAME,
        "RxInfer"=>string(Base.pkgversion(RxInfer)), "GraphPPL"=>string(Base.pkgversion(GraphPPL)),
        "CausalTargeted"=>string(Base.pkgversion(CausalTargeted)), "status"=>"running",
        "public_fits_per_dataset_prior"=>2, "fit_timing_includes_lrvb_correction"=>true,
        "resume_supported"=>false, "started_unix"=>time())
    write_metadata() = open(io -> TOML.print(io, metadata; sorted=true), joinpath(options.out, "metadata.toml"), "w")
    write_metadata()
    rows = NamedTuple[]
    for cell in cells, replication in 1:options.replications
        block = evaluate_public_dataset(cell, replication; seed=options.seed,
            streams=options.streams, draws=options.draws)
        CSV.write(joinpath(options.out, "raw.tsv"), block; delim='\t', append=!isempty(rows), missingstring="NA")
        append!(rows, block)
        println("public validation ", cell.id, " replication ", replication, "/", options.replications,
            ": ", count(r -> r.available, block), "/", length(block), " available")
        flush(stdout)
    end
    CSV.write(joinpath(options.out, "summary.tsv"), public_api_summaries(rows); delim='\t', missingstring="NA")
    metadata["status"] = "complete"
    metadata["elapsed_seconds"] = time()-metadata["started_unix"]
    metadata["rows"] = length(rows)
    metadata["raw_sha256"] = bytes2hex(sha256(read(joinpath(options.out, "raw.tsv"))))
    write_metadata()
    return rows
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PolicyPublicAPIValidation.main()
end
