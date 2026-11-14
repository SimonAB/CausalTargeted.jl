"""Paired prior-factorial validation. Schema v2 never reads or changes v1 archives."""
module PolicyPriorFactorial
using CSV, SHA, TOML, LinearAlgebra, Statistics, Random, StableRNGs
include("multivariate_policy_calibration.jl")
const Legacy = MultivariatePolicyCalibration
@eval Legacy include(joinpath(@__DIR__, "policy_calibration_support.jl"))
const METHODS = Legacy.POLICY_METHODS
const EXTRA_FIELDS = (:cdf_decision_lower, :cdf_decision_upper, :dataset_id, :prior_id, :prior_scale, :centring, :deployable,
    :fidelity_selected, :fidelity_available, :fidelity_status, :reference_content,
    :endpoint_error, :endpoint_error_arm, :lower_endpoint_error, :upper_endpoint_error,
    :mean_error_sd, :coefficient_mean_error_sd, :covariance_error, :endpoint_available,
    :moments_available, :reference_content_available, :endpoint_status, :moments_status,
    :content_status, :reference_seconds, :fit_seconds,
    :gaussian_seconds, :fidelity_seconds)
const ROW_FIELDS = (Legacy.ROW_FIELDS..., EXTRA_FIELDS...)
const STRING_FIELDS = union(Legacy.STRING_FIELDS, Set((:dataset_id, :prior_id, :centring, :fidelity_status, :endpoint_status, :moments_status, :content_status)))
const BOOL_FIELDS = Set((:available, :covered, :deployable, :fidelity_selected, :fidelity_available, :endpoint_available, :moments_available, :reference_content_available))
configuration_fingerprint(config) = Legacy.configuration_fingerprint(config)

function factorial_row(row; dataset_id, prior, fidelity_selected = false)
    merge(row, (; cdf_decision_lower = NaN, cdf_decision_upper = NaN, dataset_id = string(dataset_id), prior_id = string(prior.id),
        prior_scale = Float64(prior.scale), centring = string(prior.centring),
        deployable = Bool(prior.deployable), fidelity_selected, fidelity_available = false,
        fidelity_status = fidelity_selected ? "not_run" : "not_selected", reference_content = NaN,
        endpoint_error = NaN, endpoint_error_arm = NaN, lower_endpoint_error = NaN,
        upper_endpoint_error = NaN, mean_error_sd = NaN, coefficient_mean_error_sd = NaN, covariance_error = NaN,
        endpoint_available = false, moments_available = false, reference_content_available = false,
        endpoint_status = "not_run", moments_status = "not_run", content_status = "not_run",
        reference_seconds = 0.0, fit_seconds = 0.0, gaussian_seconds = 0.0, fidelity_seconds = 0.0))
end

"""Validate complete datasets, including every declared prior and method triple."""
function validate_checkpoint_rows(rows, config)
    groups = Dict{Tuple{String, Int}, Vector{NamedTuple}}()
    ids = Dict{String, Tuple{String, Int}}()
    for row in rows
        propertynames(row) == ROW_FIELDS || throw(ArgumentError("checkpoint columns differ from v2 schema"))
        for field in ROW_FIELDS
            value = getproperty(row, field)
            field == :covered && continue
            valid = field in STRING_FIELDS ? value isa AbstractString : field in BOOL_FIELDS ? value isa Bool :
                field in Legacy.INT_FIELDS ? value isa Integer : value isa Real
            valid || throw(ArgumentError("invalid checkpoint field $field"))
        end
        key = (string(row.cell_id), Int(row.replication))
        key[1] in config.cells && key[2] > 0 || throw(ArgumentError("invalid dataset key"))
        isempty(row.dataset_id) && throw(ArgumentError("empty dataset ID"))
        row.seed == config.seed || throw(ArgumentError("dataset master seed differs"))
        expected_draw_seed = Legacy.PolicyCalibrationSupport.replication_seed(key[1], key[2], config.seed, 1)
        row.status == "simulation_failure" || row.draw_seed == expected_draw_seed ||
            throw(ArgumentError("dataset draw seed differs from the deterministic stream"))
        get(ids, row.dataset_id, key) == key || throw(ArgumentError("dataset ID is reused"))
        ids[row.dataset_id] = key
        row.functional == "policy_vs_observed" || throw(ArgumentError("unknown functional"))
        row.available ? (row.covered isa Bool && isfinite(row.pit) && 0 <= row.pit <= 1 ||
            throw(ArgumentError("available rows require coverage and PIT"))) :
            (ismissing(row.covered) || throw(ArgumentError("unavailable coverage must be missing")))
        push!(get!(groups, key, NamedTuple[]), row)
    end
    identity_fields = (:dataset_id, :seed, :draw_seed, :n, :control_n, :control_successes,
        :treated_n, :treated_successes, :beta0, :beta1, :arm_truth, :policy_truth,
        :policy_scale, :separation, :experiment, :identification_status)
    for (key, dataset_rows) in groups
        expected = Set((string(prior), string(method)) for prior in config.prior_ids[key[1]] for method in METHODS)
        actual = [(row.prior_id, row.method) for row in dataset_rows]
        length(actual) == length(expected) && Set(actual) == expected ||
            throw(ArgumentError("dataset requires every exact prior/method triple"))
        firstrow = first(dataset_rows)
        all(all(isequal(getproperty(row, field), getproperty(firstrow, field)) for field in identity_fields)
            for row in dataset_rows) || throw(ArgumentError("dataset truth/count/seed identity differs across priors"))
        for prior in config.prior_ids[key[1]]
            triple = filter(row -> row.prior_id == prior, dataset_rows)
            prior_fields = (:prior_scale, :centring, :deployable, :prior_mean0, :prior_mean1,
                :prior_cov00, :prior_cov01, :prior_cov11)
            all(all(isequal(getproperty(row, field), getproperty(first(triple), field)) for field in prior_fields)
                for row in triple) || throw(ArgumentError("prior metadata differs within method triple"))
        end
    end
    Set(keys(groups))
end

mutable struct FactorialCheckpoint
    directory::String
    fingerprint::String
    rows::Vector{NamedTuple}
    completed::Set{Tuple{String, Int}}
    metadata::Dict{String, Any}
end
write_metadata!(checkpoint) = Legacy.write_metadata!(checkpoint)

function parsed_rows(path)
    types = Dict(field => field in STRING_FIELDS ? String : field == :draw_seed ? UInt64 :
        field in Legacy.INT_FIELDS ? Int : field in BOOL_FIELDS ? Bool : Float64 for field in ROW_FIELDS)
    try
        table = CSV.File(path; delim = '\t', missingstring = "NA", types, strict = true,
            silencewarnings = true, stringtype = String)
        Tuple(Symbol.(propertynames(table))) == ROW_FIELDS || throw(ArgumentError("not a v2 table"))
        NamedTuple[NamedTuple(row) for row in table]
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("invalid raw.tsv: " * sprint(showerror, exception)))
    end
end
configuration(metadata) = (; cells = metadata["configuration"]["cells"],
    prior_ids = metadata["configuration"]["prior_ids"], seed = metadata["configuration"]["seed"])

function checkpoint_digest(rows, config, fingerprint)
    digest = fingerprint
    index = 1
    seen = Set{Tuple{String, Int}}()
    while index <= length(rows)
        key = (rows[index].cell_id, rows[index].replication)
        key in seen && throw(ArgumentError("dataset records are not contiguous"))
        push!(seen, key)
        count = 3length(config.prior_ids[key[1]])
        last = index + count - 1
        last <= length(rows) || throw(ArgumentError("partial dataset checkpoint"))
        block = rows[index:last]
        all((row.cell_id, row.replication) == key for row in block) || throw(ArgumentError("partial dataset checkpoint"))
        digest = Legacy.advance_digest(digest, block)
        index = last + 1
    end
    digest
end

function open_checkpoint(directory, config; resume = false, provenance = Dict{String, Any}())
    fingerprint = configuration_fingerprint(config)
    rawpath = joinpath(directory, "raw.tsv"); metapath = joinpath(directory, "metadata.toml")
    if resume
        isfile(rawpath) && isfile(metapath) || throw(ArgumentError("resume requires raw.tsv and metadata.toml"))
        isfile(metapath * ".tmp") && throw(ArgumentError("unfinished metadata checkpoint exists"))
        metadata = TOML.parsefile(metapath)
        get(metadata, "schema_version", 0) == 2 || throw(ArgumentError("v2 rejects other checkpoint schemas"))
        get(metadata, "fingerprint", nothing) == fingerprint || throw(ArgumentError("source/configuration/backend differs"))
        configuration_fingerprint(get(metadata, "configuration", Dict())) == fingerprint ||
            throw(ArgumentError("embedded configuration differs"))
        rows = parsed_rows(rawpath)
        completed = validate_checkpoint_rows(rows, config)
        checkpoint_digest(rows, config, fingerprint) == get(metadata, "checkpoint_digest", nothing) ||
            throw(ArgumentError("checkpoint content digest differs"))
        length(completed) == get(metadata, "completed_datasets", -1) || throw(ArgumentError("dataset count differs"))
        return FactorialCheckpoint(abspath(directory), fingerprint, rows, completed, metadata)
    end
    isdir(directory) && !isempty(readdir(directory)) && throw(ArgumentError("output directory must be empty"))
    mkpath(directory)
    metadata = Dict{String, Any}("schema_version" => 2, "fingerprint" => fingerprint,
        "configuration" => Dict(string(k) => v for (k, v) in pairs(config)),
        "checkpoint_digest" => fingerprint, "completed_datasets" => 0, "provenance" => provenance)
    checkpoint = FactorialCheckpoint(abspath(directory), fingerprint, NamedTuple[], Set{Tuple{String, Int}}(), metadata)
    write(rawpath, join(string.(ROW_FIELDS), '\t') * "\n")
    write_metadata!(checkpoint)
    checkpoint
end

function append_dataset!(checkpoint, rows)
    config = configuration(checkpoint.metadata)
    keys = validate_checkpoint_rows(rows, config)
    length(keys) == 1 || throw(ArgumentError("append requires one complete dataset"))
    isempty(intersect(keys, checkpoint.completed)) || throw(ArgumentError("dataset already completed"))
    open(joinpath(checkpoint.directory, "raw.tsv"), "a") do io
        write(io, Legacy.row_bytes(rows)); flush(io)
    end
    append!(checkpoint.rows, rows); union!(checkpoint.completed, keys)
    checkpoint.metadata["checkpoint_digest"] = Legacy.advance_digest(checkpoint.metadata["checkpoint_digest"], rows)
    checkpoint.metadata["completed_datasets"] = length(checkpoint.completed)
    write_metadata!(checkpoint)
    checkpoint
end

function parse_options(arguments)
    options = Dict{String, Any}("profile" => "pilot", "experiment" => "fixed_parameter",
        "replications" => nothing, "seed" => 20261115, "draws" => 4096, "out" => nothing,
        "resume" => false, "cells" => String[], "fidelity_replications" => nothing)
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        argument == "--resume" && (options["resume"] = true; index += 1; continue)
        key = startswith(argument, "--") ? replace(argument[3:end], '-' => '_') : ""
        haskey(options, key) && key != "resume" || throw(ArgumentError("unknown option $argument"))
        index < length(arguments) || throw(ArgumentError("missing value for $argument"))
        value = arguments[index + 1]
        options[key] = key in ("replications", "seed", "draws", "fidelity_replications") ? parse(Int, value) :
            key == "cells" ? split(value, ',') : value
        index += 2
    end
    options["profile"] in ("pilot", "validation") || throw(ArgumentError("profile must be pilot or validation"))
    options["experiment"] in ("fixed_parameter", "prior_predictive") || throw(ArgumentError("invalid experiment"))
    pilot = options["profile"] == "pilot"
    isnothing(options["replications"]) && (options["replications"] = pilot ? 50 : 1000)
    isnothing(options["fidelity_replications"]) && (options["fidelity_replications"] = pilot ? 0 : 100)
    options["replications"] > 0 && options["draws"] >= 2 && options["seed"] >= 0 && options["fidelity_replications"] >= 0 ||
        throw(ArgumentError("invalid numeric option"))
    isnothing(options["out"]) && (options["out"] = joinpath(@__DIR__, "policy_prior_factorial_" * options["experiment"] * "_" * options["profile"]))
    (; (Symbol(key) => value for (key, value) in options)...)
end

"""Evaluate a dataset once and preserve all prior/method attempts independently."""
function evaluate_factorial_dataset(cell, replication; seed, draws = 4096, fidelity_replications = 0,
        simulator = PolicyPriorFactorialSupport.simulate_factorial_dataset,
        prior_builder = PolicyPriorFactorialSupport.prior_specs,
        fit_engine = Legacy.fit_policy_engine, checked_cdf = (args...; kwargs...) -> Legacy.checked_policy_cdf(args...; kwargs...),
        fidelity_engine = evaluate_fidelity)
    dataset = try
        simulated = simulator(cell, replication; seed)
        Legacy.dataset_scientific_trace(simulated, cell.n)
        simulated
    catch exception
        exception isa InterruptException && rethrow()
        return failed_dataset_rows(cell, replication; seed, prior_builder,
            fidelity_selected = replication <= fidelity_replications, exception)
    end
    priors = prior_builder(cell, dataset.beta)
    rows = NamedTuple[]
    selected = replication <= fidelity_replications
    for prior in priors
        descriptor = (; cell..., scenario = :prior_factorial, prior_mean = prior.mean,
            prior_covariance = prior.covariance)
        captured_fit = Ref{Any}(nothing)
        fit_seconds = Ref(0.0)
        fitting = function(args...)
            start = time_ns()
            try
                fit = fit_engine(args...)
                captured_fit[] = fit
                return fit
            finally
                fit_seconds[] = (time_ns() - start) / 1e9
            end
        end
        checks = NamedTuple[]
        checked = function(X, y, mean, covariance, truth; gaussian = false, kwargs...)
            result = checked_cdf(X, y, mean, covariance, truth; gaussian, kwargs...)
            push!(checks, (; gaussian, covariance, result))
            result
        end
        prior_rows = Legacy.evaluate_policy_dataset(descriptor, replication; seed, draws,
            simulator = (args...; kwargs...) -> dataset, fit_engine = fitting, checked_cdf = checked)
        decorated = [factorial_row(row; dataset_id = dataset.dataset_id, prior,
            fidelity_selected = selected) for row in prior_rows]
        decorated = [merge(row, captured_cdf_bounds(checks, row.method, captured_fit[])) for row in decorated]
        reference_seconds = decorated[1].seconds
        decorated = [merge(row, (; reference_seconds, fit_seconds = fit_seconds[],
            gaussian_seconds = row.method == "adaptive_reference" ? 0.0 : max(0.0, row.seconds - fit_seconds[])))
            for row in decorated]
        if selected
            start = time_ns()
            decorated = try
                fidelity_engine(decorated, dataset, prior, captured_fit[])
            catch exception
                exception isa InterruptException && rethrow()
                [merge(row, (; fidelity_status = "fidelity_exception: " * sprint(showerror, exception))) for row in decorated]
            end
            seconds = (time_ns() - start) / 1e9
            decorated = [merge(row, (; fidelity_seconds = seconds)) for row in decorated]
        end
        append!(rows, decorated)
    end
    rows
end

"""Retain numerical CDF decision bounds without changing 95% coverage status.

Gaussian evaluations are matched by covariance so skipping an ineligible raw
covariance cannot accidentally attach the LRVB bounds to the raw row.
"""
function captured_cdf_bounds(checks, method, fit)
    candidates = if method == "adaptive_reference"
        filter(check -> !check.gaussian, checks)
    elseif fit === nothing
        NamedTuple[]
    else
        covariance = getproperty(fit, method == "variational" ? :variational_covariance : :corrected_covariance)
        filter(check -> check.gaussian && isequal(check.covariance, covariance), checks)
    end
    isempty(candidates) && return (; cdf_decision_lower = NaN, cdf_decision_upper = NaN)
    result = first(candidates).result
    (; cdf_decision_lower = hasproperty(result, :decision_lower) ? Float64(result.decision_lower) : NaN,
        cdf_decision_upper = hasproperty(result, :decision_upper) ? Float64(result.decision_upper) : NaN)
end

function failed_dataset_rows(cell, replication; seed, prior_builder, fidelity_selected, exception)
    # Priors do not require the random generating beta in the zero-centred
    # prior-predictive experiment. No outcomes or truths are fabricated.
    priors = prior_builder(cell, cell.beta_fixed)
    dataset_id = "$(cell.id)|seed=$(seed)|rep=$(replication)"
    rows = NamedTuple[]
    for prior in priors, method in METHODS
        base = Legacy.empty_policy_row(; cell_id = cell.id, replication, method, seed,
            scenario = :prior_factorial, experiment = cell.experiment, n = cell.n)
        row = merge(base, (; status = "simulation_failure", exception = sprint(showerror, exception),
            prior_mean0 = Float64(prior.mean[1]), prior_mean1 = Float64(prior.mean[2]),
            prior_cov00 = Float64(prior.covariance[1, 1]), prior_cov01 = Float64(prior.covariance[1, 2]),
            prior_cov11 = Float64(prior.covariance[2, 2])))
        push!(rows, factorial_row(row; dataset_id, prior, fidelity_selected))
    end
    rows
end

function run_factorial_dataset(cell, replication; seed, kwargs...)
    # Unexpected exceptions are reported explicitly; an incomplete dataset is never
    # committed and therefore cannot silently become a completed checkpoint.
    evaluate_factorial_dataset(cell, replication; seed, kwargs...)
end

# A1 keeps mass, first moments and second moments on a common dimensionless
# scale. Named integration barriers avoid recompiling the full nested rule for
# every numeric centre or scale value.
moment_vector_norm(values) = norm(view(values, 1:6))

function standardised_inner_moments(z₁, z₀, rule, centre::NTuple{2, Float64}, scales::NTuple{2, Float64})
    isfinite(z₁) || return zeros(6)
    β₀ = rule.μ[1] + rule.L[1, 1] * z₀
    β₁ = rule.μ[2] + rule.L[2, 1] * z₀ + rule.L[2, 2] * z₁
    u₀, u₁ = (β₀ - centre[1]) / scales[1], (β₁ - centre[2]) / scales[2]
    weight = exp(Legacy.LogisticTailReference.bivariate_log_kernel([z₀, z₁],
        rule.X, rule.y, rule.μ, rule.L, rule.weights) - rule.log_peak)
    weight .* [1.0, u₀, u₁, u₀^2, u₀ * u₁, u₁^2]
end

function standardised_outer_moments(z₀, rule, centre::NTuple{2, Float64}, scales::NTuple{2, Float64})
    isfinite(z₀) || return zeros(7)
    value, error = Legacy.LogisticTailReference.quadgk(
        z₁ -> standardised_inner_moments(z₁, z₀, rule, centre, scales),
        -Inf, rule.mode[2], Inf; rtol = rule.rtol, atol = rule.atol)
    vcat(value, error)
end

"""Integrate centred, scaled moments under one rule and a fixed common scale."""
function standardised_moments(rule, centre, scales)
    length(centre) == 2 && all(isfinite, centre) || throw(ArgumentError("moment centre must contain two finite values"))
    length(scales) == 2 && all(s -> isfinite(s) && s > 0, scales) ||
        throw(ArgumentError("moment scales must contain two finite positive values"))
    c = (Float64(centre[1]), Float64(centre[2]))
    d = (Float64(scales[1]), Float64(scales[2]))
    raw, outer_error = Legacy.LogisticTailReference.quadgk(
        z₀ -> standardised_outer_moments(z₀, rule, c, d),
        -Inf, rule.mode[1], Inf; rtol = rule.rtol, atol = rule.atol, norm = moment_vector_norm)
    mass = raw[1]
    isfinite(mass) && mass > 0 || error("standardised moment normalisation failed")
    normalised_raw = raw[1:6] ./ mass
    mean = normalised_raw[2:3]
    second = [normalised_raw[4] normalised_raw[5]; normalised_raw[5] normalised_raw[6]]
    covariance = Matrix(Symmetric(second - mean * transpose(mean)))
    (; mass, mean, covariance, normalised_raw,
        integration_error = outer_error + abs(raw[7]), outer_error, inner_error = abs(raw[7]),
        mass_disagreement = abs(mass - rule.mass) / rule.mass)
end

"""Propagate a combined quadrature estimate through normalisation and covariance.

These are conservative algebraic bounds conditional on the heuristic adaptive
quadrature error estimate, rather than rigorous integration enclosures.
"""
function moment_error_bounds(normalised_raw, mass, error)
    length(normalised_raw) == 6 && all(isfinite, normalised_raw) ||
        throw(ArgumentError("six finite normalised raw moments are required"))
    isfinite(mass) && isfinite(error) && 0 <= error < mass ||
        throw(ArgumentError("moment integration requires 0 ≤ error < mass"))
    delta = (error / (mass - error)) .* (1 .+ abs.(normalised_raw))
    mean = normalised_raw[2:3]
    dmean = delta[2:3]
    dsecond = [delta[4] delta[5]; delta[5] delta[6]]
    covariance_bounds = [dsecond[i, j] + abs(mean[i]) * dmean[j] +
        abs(mean[j]) * dmean[i] + dmean[i] * dmean[j] for i in 1:2, j in 1:2]
    second = [normalised_raw[4] normalised_raw[5]; normalised_raw[5] normalised_raw[6]]
    covariance = second - mean * transpose(mean)
    covariance_relative_bound = norm(covariance_bounds) / norm(covariance)
    (; mean_bound = maximum(dmean), covariance_bounds, covariance_relative_bound, normalised_error_bounds = delta)
end

"""A1: refine moments and estimated numerical errors in a shared dimensionless basis."""
function refined_standardised_moments(working, tight)
    preliminary = Legacy.LogisticTailReference.bivariate_moments(tight)
    centre = preliminary.mean
    scales = sqrt.(diag(preliminary.covariance))
    coarse = standardised_moments(working, centre, scales)
    fine = standardised_moments(tight, centre, scales)
    coarse_bounds = moment_error_bounds(coarse.normalised_raw, coarse.mass, coarse.integration_error)
    fine_bounds = moment_error_bounds(fine.normalised_raw, fine.mass, fine.integration_error)
    mean_change = maximum(abs.(coarse.mean - fine.mean) ./ sqrt.(diag(fine.covariance)))
    covariance_change = norm(coarse.covariance - fine.covariance) / norm(fine.covariance)
    failures = String[]
    mean_change <= 5e-4 && covariance_change <= 5e-4 || push!(failures, "moment_refinement")
    max(coarse.mass_disagreement, fine.mass_disagreement) <= 5e-5 || push!(failures, "moment_mass")
    max(coarse_bounds.mean_bound, fine_bounds.mean_bound) <= 5e-4 || push!(failures, "moment_mean_error")
    max(coarse_bounds.covariance_relative_bound, fine_bounds.covariance_relative_bound) <= 5e-4 ||
        push!(failures, "moment_covariance_error")
    moments = (; mean = centre + scales .* fine.mean,
        covariance = Matrix(Diagonal(scales) * fine.covariance * Diagonal(scales)))
    (; available = isempty(failures), failures, moments, coarse, fine, centre, scales,
        coarse_bounds, fine_bounds, mean_change, covariance_change)
end

"""Independently tightened ideal-law endpoints and coefficient moments."""
function refined_law(X, y, mean, covariance; weights = nothing, gaussian = false)
    ref = Legacy.LogisticTailReference
    rtol, atol, gate = gaussian ? (1e-7, 1e-10, 5e-6) : (1e-5, 1e-8, 5e-5)
    working = ref.bivariate_policy_reference(X, y, mean, covariance; weights, rtol, atol)
    tight = ref.bivariate_policy_reference(X, y, mean, covariance; weights, rtol = rtol / 100, atol = atol / 100)
    left = ref.policy_tails(working); right = ref.policy_tails(tight)
    failures = String[]
    for (endpoint, probability) in ((right.lower, 0.025), (right.upper, 0.975))
        probe = Legacy.policy_cdf_probe(tight, endpoint)
        check = Legacy.assess_policy_cdf(Legacy.policy_cdf_probe(working, endpoint), probe;
            cdf_tol = gate, integration_tol = gate, mass_tol = gate)
        # Endpoint checks assess numerical quality, not truth inclusion; the
        # coverage-boundary ambiguity label at a quantile is expected here.
        check.status in (:ok, :boundary_ambiguous) || push!(failures, "endpoint_" * string(check.status))
        abs(probe.cdf - probability) <= gate || push!(failures, "endpoint_residual")
    end
    max(abs(left.lower - right.lower), abs(left.upper - right.upper)) <= 1e-4 || push!(failures, "endpoint_refinement")
    endpoint_failures = unique(failures)
    moment_failures = String[]
    fine = if gaussian
        (; mean, covariance)
    else
        try
            result = refined_standardised_moments(working, tight)
            append!(moment_failures, result.failures)
            result.moments
        catch exception
            exception isa InterruptException && rethrow()
            push!(moment_failures, "moment_exception: " * sprint(showerror, exception))
            nothing
        end
    end
    (; working, tight, tails = right, moments = fine,
        endpoints_available = isempty(endpoint_failures), moments_available = isempty(moment_failures),
        endpoint_failures, moment_failures)
end

function fidelity_status(row)
    row.endpoint_available && row.moments_available && row.reference_content_available ? "ok" :
        join(["endpoints=" * row.endpoint_status, "moments=" * row.moments_status,
            "content=" * row.content_status], ";")
end

"""Compute ideal Gaussian-law fidelity separately from known-truth coverage."""
function evaluate_fidelity(rows, dataset, prior, fit)
    ref = Legacy.LogisticTailReference
    compressed = Legacy.PolicyCalibrationSupport.compress_binary_likelihood(dataset.A, dataset.y)
    posterior = refined_law(compressed.X, compressed.y, prior.mean, prior.covariance; weights = compressed.weights)
    for index in 1:3
        row = merge(rows[index], (; endpoint_status = isempty(posterior.endpoint_failures) ? "ok" : join(posterior.endpoint_failures, ','),
            moments_status = isempty(posterior.moment_failures) ? "ok" : join(posterior.moment_failures, ',')))
        covariance = nothing
        mean = nothing
        if index == 1
            law = posterior
        else
            fit_usable = fit !== nothing && fit.converged && length(fit.mean) == 2 && all(isfinite, fit.mean)
            if !fit_usable
                rows[index] = merge(row, (; fidelity_status = "fit_unavailable",
                    endpoint_status = "fit_unavailable", moments_status = "fit_unavailable", content_status = "fit_unavailable"))
                continue
            end
            covariance = getproperty(fit, index == 2 ? :variational_covariance : :corrected_covariance)
            if !Legacy.covariance_is_usable(covariance)
                rows[index] = merge(row, (; fidelity_status = "covariance_unavailable",
                    endpoint_status = "covariance_unavailable", moments_status = "covariance_unavailable", content_status = "covariance_unavailable"))
                continue
            end
            mean = fit.mean
            law = refined_law(zeros(0, 2), Float64[], mean, covariance; gaussian = true)
        end
        if posterior.moments_available
            error_sd = index == 1 ? 0.0 : maximum(abs.(mean - posterior.moments.mean) ./ sqrt.(diag(posterior.moments.covariance)))
            covariance_error = index == 1 ? 0.0 : norm(covariance - posterior.moments.covariance) / norm(posterior.moments.covariance)
            row = merge(row, (; moments_available = true, moments_status = "ok", mean_error_sd = error_sd,
                coefficient_mean_error_sd = error_sd, covariance_error))
        end
        if posterior.endpoints_available && law.endpoints_available
            lower_error = law.tails.lower - posterior.tails.lower
            upper_error = law.tails.upper - posterior.tails.upper
            arm_error = max(abs(lower_error), abs(upper_error))
            row = merge(row, (; endpoint_available = true, endpoint_status = "ok",
                endpoint_error = dataset.policy_scale * arm_error, endpoint_error_arm = arm_error,
                lower_endpoint_error = dataset.policy_scale * lower_error,
                upper_endpoint_error = dataset.policy_scale * upper_error,
                width = dataset.policy_scale * (law.tails.upper - law.tails.lower)))
        else
            row = merge(row, (; endpoint_status = join(vcat(posterior.endpoint_failures, law.endpoint_failures), ',')))
        end
        if law.endpoints_available
            content = ref.refine_policy_interval(posterior.working, posterior.tight, law.tails.lower,
                law.tails.upper; content_tol = 5e-5, tail_tol = 5e-5, integration_tol = 5e-5)
            row = merge(row, (; reference_content_available = content.converged,
                reference_content = content.converged ? content.content : NaN,
                content_status = content.converged ? "ok" : "reference_interval_content"))
        else
            row = merge(row, (; content_status = "gaussian_endpoints_unavailable"))
        end
        status = fidelity_status(row)
        rows[index] = merge(row, (; fidelity_status = status, fidelity_available = status == "ok"))
    end
    rows
end

function source_hashes()
    hashes = Legacy.source_hashes()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    for file in (@__FILE__, joinpath(@__DIR__, "policy_prior_factorial_support.jl"))
        hashes[relpath(file, root)] = bytes2hex(sha256(read(file)))
    end
    for file in ("Manifest.toml", "test/stress/Manifest.toml", "docs/BAYESIAN_VALIDATION_PROTOCOL_V2.md")
        path = joinpath(root, file)
        isfile(path) && (hashes[file] = bytes2hex(sha256(read(path))))
    end
    hashes
end

function write_reports!(checkpoint)
    reports = PolicyPriorFactorialSupport.factorial_summaries(checkpoint.rows)
    for (filename, table) in (("summary.tsv", reports.summaries),
            ("paired_methods.tsv", reports.paired_methods), ("paired_priors.tsv", reports.paired_priors))
        CSV.write(joinpath(checkpoint.directory, filename), table; delim = '\t', missingstring = "NA")
    end
    open(joinpath(checkpoint.directory, "report.md"), "w") do io
        println(io, "## Paired policy prior factorial validation\n")
        println(io, "Completed ", length(checkpoint.completed), " independent datasets. Schema v2; rows across priors and methods are paired.\n")
        println(io, "Known-truth coverage uses tightened CDF integration for every attempt. A failed posterior reference does not suppress Gaussian-law coverage. All-attempt inclusion counts unavailable intervals as noncoverage.\n")
        println(io, "Fidelity uses A1 centred, SD-scaled moment quadrature with shared coarse/fine scaling and propagated dimensionless numerical-error gates (heuristic estimates, not rigorous enclosures), the prespecified replication subset and ideal-law endpoints, with separate availability and finite metric counts. Unselected fidelity metrics are NaN. Coefficient mean error is the maximum absolute mean discrepancy divided by its reference marginal SD; covariance error is relative Frobenius norm. Endpoint errors are maximum absolute errors in cohort units, with arm and signed endpoint errors retained.\n")
        println(io, "Monte Carlo descriptive means use common standard draws across all priors and methods. Widths outside the fidelity subset are finite-draw descriptions; selected successful fidelity rows use ideal-law widths. Fixed-truth inclusion is not SBC. Prior-predictive calibration applies only to matching zero-centred priors.\n")
        println(io, "See summary.tsv for separate availability and metric denominators, paired_methods.tsv for LRVB minus raw VI, and paired_priors.tsv for declared paired prior contrasts. Reference, fitting, Gaussian-law assessment and fidelity timings are separate; repeated method rows share the same fit time and must not be summed as separate fits.")
    end
end

function run_factorial(options)
    BLAS.set_num_threads(1)
    cells = PolicyPriorFactorialSupport.factorial_cells(; experiment = Symbol(options.experiment))
    if !isempty(options.cells)
        all(id in string.(getproperty.(cells, :id)) for id in options.cells) || throw(ArgumentError("unknown selected cell"))
        cells = filter(cell -> string(cell.id) in options.cells, cells)
    end
    prior_ids = Dict(string(cell.id) => string.(getproperty.(PolicyPriorFactorialSupport.prior_specs(cell,
        cell.experiment == :fixed_parameter ? cell.beta_fixed : zeros(2)), :id)) for cell in cells)
    config = (; profile = options.profile, experiment = options.experiment, seed = options.seed,
        draws = options.draws, fidelity_replications = options.fidelity_replications,
        cells = string.(getproperty.(cells, :id)), prior_ids, source_hashes = source_hashes(),
        environment_contract = Legacy.environment_contract(),
        metric_contract = "A1:dimensionless_centred_SD_scaled_moment_quadrature;propagated_mean_bound<=5e-4;propagated_covariance_relative_Frobenius_bound<=5e-4;mean_error_sd=max_j(abs(mean_j-reference_mean_j)/reference_sd_j);covariance_error=Frobenius(cov-reference_cov)/Frobenius(reference_cov);endpoint_error=scale*max(abs(lower_error),abs(upper_error))",
        numerical_contract = "A1:normalised_moments_common_fine_preliminary_centre_scale;posterior:1e-5/1e-8,1e-7/1e-10,gates5e-5;gaussian:1e-7/1e-10,1e-9/1e-12,gates5e-6;engine:maxiter1000,tol1e-10")
    checkpoint = open_checkpoint(options.out, config; resume = options.resume, provenance = Legacy.backend_provenance())
    remaining = Legacy.pending_datasets(cells, options.replications, checkpoint.completed)
    for (cell, replication) in remaining
        start = time_ns()
        rows = run_factorial_dataset(cell, replication; seed = options.seed, draws = options.draws,
            fidelity_replications = options.fidelity_replications)
        if !haskey(checkpoint.metadata, "first_dataset_including_compilation_seconds")
            checkpoint.metadata["first_dataset_including_compilation_seconds"] = (time_ns() - start) / 1e9
        end
        append_dataset!(checkpoint, rows)
        println((; cell = cell.id, replication, completed = length(checkpoint.completed),
            available = count(row -> row.available, rows), fidelity_available = count(row -> row.fidelity_available, rows)))
        flush(stdout)
    end
    write_reports!(checkpoint)
    checkpoint
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    @eval Legacy using CausalTargeted, GraphPPL, RxInfer, QuadGK, Distributions
    @eval Legacy include(joinpath(@__DIR__, "policy_calibration_reference.jl"))
    include(joinpath(@__DIR__, "policy_prior_factorial_support.jl"))
    Base.invokelatest(run_factorial, options)
end
end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PolicyPriorFactorial.main(ARGS)
end
