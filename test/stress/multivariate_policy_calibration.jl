"""Repeated-sampling validation of the assignment-to-one policy contrast.

Run `julia --project=test/stress --threads=auto test/stress/multivariate_policy_calibration.jl --profile screen`.
Checkpoint tables are data only: resume parses TSV/TOML and never evaluates code.
"""
module MultivariatePolicyCalibration
using CSV, SHA, TOML, LinearAlgebra, Statistics, Random, StableRNGs

const POLICY_METHODS = (:adaptive_reference, :variational, :linear_response)
const ROW_FIELDS = (:functional, :cell_id, :scenario, :experiment, :n, :control_n, :control_successes,
    :treated_n, :treated_successes, :identification_status, :replication,
    :method, :seed, :draw_seed, :beta0, :beta1, :arm_truth, :policy_truth, :policy_scale,
    :separation, :prior_mean0, :prior_mean1, :prior_cov00, :prior_cov01, :prior_cov11,
    :status, :available, :covered, :pit, :width, :policy_error, :policy_mean,
    :mean_mcse, :draws, :cdf_change, :tail_error, :mass_change, :fititers, :seconds,
    :covariance_trace_ratio, :covariance_trace_ratio_formula, :backend, :exception)
const STRING_FIELDS = Set((:functional, :cell_id, :scenario, :experiment, :method,
    :separation, :identification_status, :status, :covariance_trace_ratio_formula, :backend, :exception))
const INT_FIELDS = Set((:n, :control_n, :control_successes, :treated_n, :treated_successes,
    :replication, :seed, :draw_seed, :draws, :fititers))

"""A complete, typed failed row; unavailable coverage is always `missing`."""
function empty_policy_row(; cell_id, replication, method, seed, scenario, experiment, n,
        draw_seed = 0, status = "not_run")
    (; functional = "policy_vs_observed", cell_id = string(cell_id),
        scenario = string(scenario), experiment = string(experiment), n = Int(n),
        control_n = -1, control_successes = -1, treated_n = -1, treated_successes = -1,
        identification_status = "user_assumed", replication = Int(replication),
        method = string(method), seed = Int(seed),
        draw_seed = UInt64(draw_seed), beta0 = NaN, beta1 = NaN, arm_truth = NaN,
        policy_truth = NaN, policy_scale = NaN, separation = "unknown",
        prior_mean0 = NaN, prior_mean1 = NaN, prior_cov00 = NaN, prior_cov01 = NaN,
        prior_cov11 = NaN, status = string(status), available = false,
        covered = missing, pit = NaN, width = NaN, policy_error = NaN,
        policy_mean = NaN, mean_mcse = NaN, draws = 0, cdf_change = NaN,
        tail_error = NaN, mass_change = NaN, fititers = 0, seconds = 0.0,
        covariance_trace_ratio = NaN,
        covariance_trace_ratio_formula = "tr(posterior_covariance)/tr(prior_covariance)",
        backend = "none", exception = "")
end

# Sorting dictionary keys makes the fingerprint independent of TOML/dictionary order.
function canonical_configuration(value)
    value isa NamedTuple && return canonical_configuration(Dict(string(k) => v for (k, v) in pairs(value)))
    if value isa AbstractDict
        return "{" * join([repr(string(k)) * ":" * canonical_configuration(value[k])
            for k in sort!(collect(keys(value)); by = string)], ",") * "}"
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join(canonical_configuration.(collect(value)), ",") * "]"
    end
    return repr(value)
end
configuration_fingerprint(config) = bytes2hex(sha256(canonical_configuration(config)))

function validate_checkpoint_rows(rows, allowed_cells)
    groups = Dict{Tuple{String, Int}, Set{String}}()
    for row in rows
        propertynames(row) == ROW_FIELDS || throw(ArgumentError("checkpoint columns differ from the declared schema"))
        for field in ROW_FIELDS
            value = getproperty(row, field)
            field == :covered && continue
            valid = field in STRING_FIELDS ? value isa AbstractString :
                field in INT_FIELDS ? value isa Integer : field == :available ? value isa Bool : value isa Real
            valid || throw(ArgumentError("invalid or missing checkpoint field $field"))
        end
        key = (string(row.cell_id), Int(row.replication))
        key[1] in allowed_cells || throw(ArgumentError("checkpoint contains an unselected cell $(key[1])"))
        key[2] > 0 || throw(ArgumentError("checkpoint replication must be positive"))
        string(row.functional) == "policy_vs_observed" || throw(ArgumentError("checkpoint has an unknown functional"))
        method = string(row.method)
        method in string.(POLICY_METHODS) || throw(ArgumentError("checkpoint has an unknown method"))
        seen = get!(groups, key, Set{String}())
        method in seen && throw(ArgumentError("duplicate checkpoint key $key/$method"))
        push!(seen, method)
        row.available isa Bool || throw(ArgumentError("availability must be Boolean"))
        if row.available
            row.covered isa Bool || throw(ArgumentError("available rows require Boolean coverage"))
            isfinite(row.pit) && 0 <= row.pit <= 1 || throw(ArgumentError("available rows require a valid PIT"))
        else
            ismissing(row.covered) || throw(ArgumentError("unavailable coverage must be missing"))
        end
    end
    all(length(methods) == 3 for methods in values(groups)) ||
        throw(ArgumentError("checkpoint contains an incomplete method triple"))
    return Set(keys(groups))
end

function row_bytes(rows)
    buffer = IOBuffer()
    CSV.write(buffer, rows; delim = '\t', writeheader = false, missingstring = "NA")
    take!(buffer)
end
advance_digest(digest, rows) = bytes2hex(sha256(vcat(codeunits(digest), row_bytes(rows))))

mutable struct PolicyCheckpoint
    directory::String
    fingerprint::String
    rows::Vector{NamedTuple}
    completed::Set{Tuple{String, Int}}
    metadata::Dict{String, Any}
end

function write_metadata!(checkpoint)
    temporary = joinpath(checkpoint.directory, "metadata.toml.tmp")
    open(temporary, "w") do io
        TOML.print(io, checkpoint.metadata; sorted = true)
        flush(io)
    end
    mv(temporary, joinpath(checkpoint.directory, "metadata.toml"); force = true)
end

function parsed_checkpoint_rows(path)
    types = Dict(field => field in STRING_FIELDS ? String : field == :draw_seed ? UInt64 : field in INT_FIELDS ? Int :
        field in (:available, :covered) ? Bool : Float64 for field in ROW_FIELDS)
    try
        table = CSV.File(path; delim = '\t', missingstring = "NA", types,
            strict = true, silencewarnings = true, stringtype = String)
        Tuple(Symbol.(propertynames(table))) == ROW_FIELDS ||
            throw(ArgumentError("checkpoint columns differ from the declared schema"))
        return NamedTuple[NamedTuple(row) for row in table]
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("invalid raw.tsv: " * sprint(showerror, exception)))
    end
end

"""Create or verify a checkpoint. Target replication count is deliberately outside `config`."""
function open_checkpoint(directory, config; resume = false, provenance = Dict{String, Any}())
    fingerprint = configuration_fingerprint(config)
    rawpath, metapath = joinpath(directory, "raw.tsv"), joinpath(directory, "metadata.toml")
    if resume
        isfile(rawpath) && isfile(metapath) || throw(ArgumentError("resume requires raw.tsv and metadata.toml"))
        isfile(metapath * ".tmp") && throw(ArgumentError("unfinished metadata checkpoint exists"))
        metadata = try TOML.parsefile(metapath) catch exception
            throw(ArgumentError("invalid metadata.toml: " * sprint(showerror, exception)))
        end
        get(metadata, "fingerprint", nothing) == fingerprint || throw(ArgumentError("checkpoint source/configuration fingerprint differs"))
        configuration_fingerprint(get(metadata, "configuration", Dict())) == fingerprint ||
            throw(ArgumentError("checkpoint embedded configuration differs"))
        get(metadata, "schema_version", 0) == 1 || throw(ArgumentError("unknown checkpoint schema"))
        rows = parsed_checkpoint_rows(rawpath)
        completed = validate_checkpoint_rows(rows, config.cells)
        digest = fingerprint
        for firstrow in 1:3:length(rows)
            triple = rows[firstrow:firstrow+2]
            length(unique((row.cell_id, row.replication) for row in triple)) == 1 ||
                throw(ArgumentError("checkpoint method triples are not contiguous"))
            digest = advance_digest(digest, triple)
        end
        digest == get(metadata, "checkpoint_digest", nothing) || throw(ArgumentError("checkpoint content digest differs"))
        length(completed) == get(metadata, "completed_datasets", -1) || throw(ArgumentError("checkpoint dataset count differs"))
        return PolicyCheckpoint(abspath(directory), fingerprint, rows, completed, metadata)
    end
    if isdir(directory) && !isempty(readdir(directory))
        throw(ArgumentError("output directory is not empty; choose another directory or use --resume"))
    end
    mkpath(directory)
    metadata = Dict{String, Any}("schema_version" => 1, "fingerprint" => fingerprint,
        "configuration" => Dict(string(k) => v for (k, v) in pairs(config)),
        "checkpoint_digest" => fingerprint, "completed_datasets" => 0,
        "provenance" => provenance)
    checkpoint = PolicyCheckpoint(abspath(directory), fingerprint, NamedTuple[], Set{Tuple{String, Int}}(), metadata)
    # A zero-row typed table supplies a header without a fabricated dataset.
    exemplar = empty_policy_row(; cell_id = "", replication = 1, method = :adaptive_reference,
        seed = 0, scenario = "", experiment = "", n = 0)
    open(rawpath, "w") do io
        CSV.write(io, [exemplar]; delim = '\t', missingstring = "NA")
    end
    # Retain just the schema header.
    header = first(readlines(rawpath))
    write(rawpath, header * "\n")
    write_metadata!(checkpoint)
    return checkpoint
end

"""Flush one complete dataset and its integrity checkpoint before continuing."""
function append_dataset!(checkpoint, rows)
    newkeys = validate_checkpoint_rows(rows, checkpoint.metadata["configuration"]["cells"])
    length(rows) == 3 && length(newkeys) == 1 || throw(ArgumentError("append requires one complete method triple"))
    isempty(intersect(newkeys, checkpoint.completed)) || throw(ArgumentError("dataset already completed"))
    open(joinpath(checkpoint.directory, "raw.tsv"), "a") do io
        write(io, row_bytes(rows))
        flush(io)
    end
    append!(checkpoint.rows, rows)
    union!(checkpoint.completed, newkeys)
    checkpoint.metadata["checkpoint_digest"] = advance_digest(checkpoint.metadata["checkpoint_digest"], rows)
    checkpoint.metadata["completed_datasets"] = length(checkpoint.completed)
    write_metadata!(checkpoint)
    return checkpoint
end

function parse_options(arguments)
    options = Dict{String, Any}("profile" => "screen", "replications" => nothing,
        "seed" => 913, "draws" => 4096, "out" => nothing, "resume" => false, "cells" => String[])
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        argument == "--resume" && (options["resume"] = true; index += 1; continue)
        key = startswith(argument, "--") ? argument[3:end] : ""
        haskey(options, key) && key != "resume" || throw(ArgumentError("unknown option $argument"))
        index < length(arguments) || throw(ArgumentError("missing value for $argument"))
        value = arguments[index + 1]
        options[key] = key in ("replications", "seed", "draws") ? parse(Int, value) :
            key == "cells" ? split(value, ',') : value
        index += 2
    end
    options["profile"] in ("screen", "validation") || throw(ArgumentError("profile must be screen or validation"))
    isnothing(options["replications"]) && (options["replications"] = options["profile"] == "screen" ? 50 : 500)
    options["replications"] > 0 || throw(ArgumentError("replications must be positive"))
    options["draws"] >= 2 || throw(ArgumentError("draws must be at least two"))
    options["seed"] >= 0 || throw(ArgumentError("seed must be nonnegative"))
    isnothing(options["out"]) && (options["out"] = joinpath(@__DIR__, "policy_calibration_" * options["profile"]))
    (; (Symbol(key) => value for (key, value) in options)...)
end

function source_hashes()
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = [joinpath(package_root, "Project.toml"), joinpath(@__DIR__, "Project.toml"),
        @__FILE__, joinpath(@__DIR__, "policy_calibration_support.jl"),
        joinpath(@__DIR__, "policy_calibration_reference.jl"),
        joinpath(@__DIR__, "..", "logistic_tail_reference.jl")]
    for directory in ("src", "ext")
        for (root, _, files) in walkdir(joinpath(package_root, directory))
            append!(paths, [joinpath(root, file) for file in files if endswith(file, ".jl")])
        end
    end
    Dict(relpath(path, package_root) => bytes2hex(sha256(read(path))) for path in sort!(unique(paths)))
end


"""Installed numerical backends that must remain identical when a run is resumed."""
function environment_contract()
    Dict{String, Any}("julia" => string(VERSION),
        "CausalTargeted" => string(Base.pkgversion(CausalTargeted)),
        "RxInfer" => string(Base.pkgversion(RxInfer)),
        "GraphPPL" => string(Base.pkgversion(GraphPPL)),
        "QuadGK" => string(Base.pkgversion(QuadGK)),
        "Distributions" => string(Base.pkgversion(Distributions)),
        "StableRNGs" => string(Base.pkgversion(StableRNGs)),
        "CSV" => string(Base.pkgversion(CSV)),
        "blas" => string(BLAS.get_config()))
end

function backend_provenance()
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    git_read(arguments) = try strip(read(Cmd(vcat(["git", "-C", package_root], arguments)), String)) catch; "unavailable" end
    Dict{String, Any}("julia" => string(VERSION), "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(), "blas" => string(BLAS.get_config()),
        "source_head" => git_read(["rev-parse", "HEAD"]),
        "source_dirty" => !isempty(git_read(["status", "--porcelain"])),
        "CausalTargeted" => string(Base.pkgversion(CausalTargeted)),
        "GraphPPL" => string(Base.pkgversion(GraphPPL)),
        "RxInfer" => string(Base.pkgversion(RxInfer)),
        "QuadGK" => string(Base.pkgversion(QuadGK)),
        "Distributions" => string(Base.pkgversion(Distributions)),
        "CSV" => string(Base.pkgversion(CSV)),
        "StableRNGs" => string(Base.pkgversion(StableRNGs)))
end

function fit_policy_engine(X, y, prior_mean, prior_covariance)
    extension = Base.get_extension(CausalTargeted, :CausalTargetedRxInferExt)
    isnothing(extension) && error("the optional GraphPPL/RxInfer extension did not load")
    extension._fit_logistic_engine(Matrix{Float64}(X), Float64.(y), Float64.(prior_mean),
        Matrix{Float64}(prior_covariance); maxiter = 1000, tol = 1e-10,
        covariance = :linear_response)
end

logistic_policy_probability(value) = value >= 0 ? inv(1 + exp(-value)) : exp(value) / (1 + exp(value))

"""Monte Carlo descriptive moments/widths; these never determine interval coverage."""
function gaussian_policy_description(mean, covariance, standard_draws, scale, truth)
    coefficients = mean .+ cholesky(Symmetric(covariance)).L * standard_draws
    values = scale .* (logistic_policy_probability.(coefficients[1, :] .+ coefficients[2, :]) .-
        logistic_policy_probability.(coefficients[1, :]))
    policy_mean = Statistics.mean(values)
    (; policy_mean, policy_error = policy_mean - truth,
        mean_mcse = std(values) / sqrt(length(values)),
        width = quantile(values, 0.975) - quantile(values, 0.025), draws = length(values))
end

function checked_fields(result)
    (; status = string(result.status), available = Bool(result.available),
        covered = result.available ? result.covered : missing,
        pit = Float64(result.pit), cdf_change = Float64(result.cdf_change),
        tail_error = Float64(result.tail_error), mass_change = Float64(result.mass_change))
end

function covariance_is_usable(covariance)
    covariance isa AbstractMatrix && size(covariance) == (2, 2) &&
        all(isfinite, covariance) && issymmetric(covariance) && isposdef(Symmetric(covariance))
end


"""Validate the fixed binary cohort and its positive policy multiplier."""
function dataset_scientific_trace(dataset, n)
    length(dataset.A) == n && length(dataset.y) == n ||
        throw(ArgumentError("dataset lengths must match the declared cohort size"))
    all(value in (0, 1) for value in dataset.A) || throw(ArgumentError("exposure must be binary"))
    all(value in (0, 1) for value in dataset.y) || throw(ArgumentError("outcome must be binary"))
    control_n = count(==(0), dataset.A)
    treated_n = count(==(1), dataset.A)
    control_n > 0 && treated_n > 0 || throw(ArgumentError("both exposure arms must be present"))
    isfinite(dataset.policy_scale) && 0 < dataset.policy_scale <= 1 &&
        isapprox(dataset.policy_scale, control_n / n; rtol = 1e-12, atol = 1e-14) ||
        throw(ArgumentError("policy scale must equal the observed untreated fraction"))
    isfinite(dataset.arm_truth) && isfinite(dataset.policy_truth) &&
        isapprox(dataset.policy_truth, dataset.policy_scale * dataset.arm_truth; rtol = 1e-12, atol = 1e-14) ||
        throw(ArgumentError("policy truth must equal scale times the arm contrast"))
    (; control_n, treated_n,
        control_successes = count(index -> dataset.A[index] == 0 && dataset.y[index] == 1, eachindex(dataset.A)),
        treated_successes = count(index -> dataset.A[index] == 1 && dataset.y[index] == 1, eachindex(dataset.A)))
end

"""Return exactly three rows even when simulation, fitting or a reference fails."""
function evaluate_policy_dataset(cell, replication; seed, draws = 4096,
        simulator = PolicyCalibrationSupport.simulate_policy_replication,
        fit_engine = fit_policy_engine, checked_cdf = checked_policy_cdf)
    base = empty_policy_row(; cell_id = cell.id, replication, method = :adaptive_reference,
        seed, scenario = cell.scenario, experiment = cell.experiment, n = cell.n)
    rows = NamedTuple[merge(base, (; method = string(method))) for method in POLICY_METHODS]
    dataset = try
        simulator(cell, replication; seed)
    catch exception
        exception isa InterruptException && rethrow()
        return [merge(row, (; status = "simulation_failure", exception = sprint(showerror, exception))) for row in rows]
    end
    scientific_trace = dataset_scientific_trace(dataset, cell.n)
    metadata = (; draw_seed = UInt64(dataset.draw_seed), beta0 = Float64(dataset.beta[1]),
        beta1 = Float64(dataset.beta[2]), arm_truth = Float64(dataset.arm_truth),
        policy_truth = Float64(dataset.policy_truth), policy_scale = Float64(dataset.policy_scale),
        separation = string(dataset.separation), prior_mean0 = Float64(cell.prior_mean[1]),
        prior_mean1 = Float64(cell.prior_mean[2]), prior_cov00 = Float64(cell.prior_covariance[1, 1]),
        prior_cov01 = Float64(cell.prior_covariance[1, 2]), prior_cov11 = Float64(cell.prior_covariance[2, 2]))
    rows = NamedTuple[merge(row, metadata, scientific_trace) for row in rows]
    reference_start = time_ns()
    rows[1] = try
        compressed = PolicyCalibrationSupport.compress_binary_likelihood(dataset.A, dataset.y)
        result = checked_cdf(compressed.X, compressed.y, cell.prior_mean,
            cell.prior_covariance, dataset.arm_truth; weights = compressed.weights)
        merge(rows[1], checked_fields(result), (; backend = "adaptive_bivariate_posterior"))
    catch exception
        exception isa InterruptException && rethrow()
        merge(rows[1], (; status = "reference_exception", exception = sprint(showerror, exception)))
    end
    rows[1] = merge(rows[1], (; seconds = (time_ns() - reference_start) / 1e9))
    fit_start = time_ns()
    fit, fit_exception = try
        (fit_engine(dataset.X, dataset.y, cell.prior_mean, cell.prior_covariance), "")
    catch exception
        exception isa InterruptException && rethrow()
        (nothing, sprint(showerror, exception))
    end
    fit_seconds = (time_ns() - fit_start) / 1e9
    fit_usable = fit !== nothing && fit.converged && length(fit.mean) == 2 && all(isfinite, fit.mean)
    # The same standard draws couple the two Gaussian descriptive summaries.
    standard_draws = randn(StableRNG(dataset.draw_seed), 2, draws)
    for (index, method) in ((2, :variational), (3, :linear_response))
        row = merge(rows[index], (; seconds = fit_seconds,
            fititers = fit === nothing ? 0 : Int(fit.n_iterations),
            backend = "RxInfer_" * string(method)))
        if !fit_usable
            rows[index] = merge(row, (; status = fit === nothing ? "fit_exception" : "fit_" * string(fit.status),
                exception = fit_exception))
            continue
        end
        covariance = method == :variational ? fit.variational_covariance : fit.corrected_covariance
        if !covariance_is_usable(covariance)
            rows[index] = merge(row, (; status = string(method) * "_covariance_failure"))
            continue
        end
        start = time_ns()
        row = merge(row, (; covariance_trace_ratio = tr(covariance) / tr(cell.prior_covariance)))
        # A reference failure above cannot suppress these known-truth evaluations.
        row = try
            result = checked_cdf(zeros(0, 2), Float64[], fit.mean, covariance,
                dataset.arm_truth; gaussian = true)
            merge(row, checked_fields(result))
        catch exception
            exception isa InterruptException && rethrow()
            merge(row, (; status = "gaussian_reference_exception", exception = sprint(showerror, exception)))
        end
        row = try
            merge(row, gaussian_policy_description(fit.mean, covariance, standard_draws,
                dataset.policy_scale, dataset.policy_truth))
        catch exception
            exception isa InterruptException && rethrow()
            merge(row, (; exception = isempty(row.exception) ? "descriptive_MC: " * sprint(showerror, exception) :
                row.exception * "; descriptive_MC: " * sprint(showerror, exception)))
        end
        rows[index] = merge(row, (; seconds = fit_seconds + (time_ns() - start) / 1e9))
    end
    rows
end


"""Contain unexpected per-dataset exceptions without discarding the completed checkpoint."""
function run_policy_dataset(cell, replication; seed, kwargs...)
    try
        evaluate_policy_dataset(cell, replication; seed, kwargs...)
    catch exception
        exception isa InterruptException && rethrow()
        [merge(empty_policy_row(; cell_id = cell.id, replication, method, seed,
            scenario = cell.scenario, experiment = cell.experiment, n = cell.n),
            (; status = "dataset_exception", exception = sprint(showerror, exception)))
            for method in POLICY_METHODS]
    end
end

"""Return only uncompleted dataset keys; extending the target preserves previous work."""
function pending_datasets(cells, replications, completed)
    any(key[2] > replications for key in completed) &&
        throw(ArgumentError("requested replication target is below a completed replication"))
    [(cell, replication) for cell in cells for replication in 1:replications
        if !((string(cell.id), replication) in completed)]
end

function write_reports!(checkpoint)
    summary = PolicyCalibrationSupport.summarise_policy_rows(checkpoint.rows)
    paired = PolicyCalibrationSupport.paired_policy_summary(checkpoint.rows)
    flattened = [merge(row, (; pit_deciles = join(row.pit_deciles, ','))) for row in summary]
    CSV.write(joinpath(checkpoint.directory, "summary.tsv"), flattened; delim = '\t', missingstring = "NA")
    CSV.write(joinpath(checkpoint.directory, "paired.tsv"), paired; delim = '\t', missingstring = "NA")
    open(joinpath(checkpoint.directory, "report.md"), "w") do io
        println(io, "## Multivariate policy calibration\n")
        println(io, "Completed ", length(checkpoint.completed), " datasets. Functional: assignment-to-one versus observed treatment in the supplied cohort.\n")
        println(io, "Coverage uses an independently tightened adaptive CDF at the known truth. Gaussian Monte Carlo means, mean MCSEs and equal-tailed widths are descriptive only. Reference widths and errors are unavailable.\n")
        println(io, "| Cell | Method | Available / attempts | Conditional coverage (MCSE) | All-attempt inclusion (MCSE) |")
        println(io, "|---|---|---:|---:|---:|")
        for row in summary
            println(io, "| ", row.cell_id, " | ", row.method, " | ", row.available, " / ", row.attempted,
                " | ", round(row.coverage_available; digits = 4), " (", round(row.mcse_available; digits = 4),
                ") | ", round(row.coverage_all_attempts; digits = 4), " (", round(row.mcse_all_attempts; digits = 4), ") |")
        end
        println(io, "\nAll-attempt inclusion counts unavailable intervals as noncoverage. Failure reasons and separate denominators appear in summary.tsv. Paired LRVB minus raw VI coverage differences and paired MCSEs appear in paired.tsv.\n")
        println(io, "Prior-predictive PIT deciles have an SBC interpretation; fixed-parameter CDF diagnostics do not. Fixed-parameter Bayesian coverage need not equal 95%.\n")
        println(io, "Posterior/prior covariance trace ratios use tr(posterior_covariance)/tr(prior_covariance). Source hashes, backend versions, source head/dirty state and first-call compilation timing appear in metadata.toml.")
    end
end


"""Preserve the original first-call timing and retain later warm-ups separately."""
function record_warmup_timing!(checkpoint, seconds; resume = false)
    initial_key = "first_call_including_compilation_seconds"
    if !haskey(checkpoint.metadata, initial_key)
        checkpoint.metadata[initial_key] = Float64(seconds)
    else
        key = resume ? "resume_warmup_seconds" : "subsequent_warmup_seconds"
        push!(get!(checkpoint.metadata, key, Float64[]), Float64(seconds))
    end
    write_metadata!(checkpoint)
    return checkpoint
end

function run_calibration(options)
    cells = PolicyCalibrationSupport.calibration_cells(Symbol(options.profile))
    if !isempty(options.cells)
        requested = Set(options.cells)
        all(id in string.(getproperty.(cells, :id)) for id in requested) || throw(ArgumentError("unknown selected cell"))
        cells = filter(cell -> string(cell.id) in requested, cells)
    end
    config = (; profile = options.profile, seed = options.seed, draws = options.draws,
        cells = string.(getproperty.(cells, :id)), source_hashes = source_hashes(),
        environment_contract = environment_contract(),
        numerical_contract = "posterior:1e-5/1e-8,1e-7/1e-10,gates5e-5;gaussian:1e-7/1e-10,1e-9/1e-12,gates5e-6;engine:maxiter1000,tol1e-10")
    BLAS.set_num_threads(1)
    checkpoint = open_checkpoint(options.out, config; resume = options.resume,
        provenance = backend_provenance())
    remaining = pending_datasets(cells, options.replications, checkpoint.completed)
    if !isempty(remaining)
        firstcell, firstreplication = first(remaining)
        compilation_seconds = @elapsed run_policy_dataset(firstcell, firstreplication;
            seed = options.seed, draws = options.draws)
        record_warmup_timing!(checkpoint, compilation_seconds; resume = options.resume)
        for (cell, replication) in remaining
            rows = run_policy_dataset(cell, replication; seed = options.seed, draws = options.draws)
            append_dataset!(checkpoint, rows)
            println((; cell = cell.id, replication, completed = length(checkpoint.completed),
                statuses = getproperty.(rows, :status)))
            flush(stdout)
        end
    end
    write_reports!(checkpoint)
    println("Calibration records: ", checkpoint.directory)
    checkpoint
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    @eval using CausalTargeted, GraphPPL, RxInfer, QuadGK, Distributions
    include(joinpath(@__DIR__, "policy_calibration_support.jl"))
    include(joinpath(@__DIR__, "policy_calibration_reference.jl"))
    Base.invokelatest(run_calibration, options)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    MultivariatePolicyCalibration.main(ARGS)
end
