module PolicyPriorFactorialSupport

using LinearAlgebra, Random, Statistics
using StableRNGs: StableRNG
using Distributions: Bernoulli, MvNormal

if !isdefined(@__MODULE__, :PolicyCalibrationSupport)
    include("policy_calibration_support.jl")
end
using .PolicyCalibrationSupport: replication_seed, logistic, classify_separation

export factorial_cells, prior_specs, simulate_factorial_dataset,
    paired_metric_summary, factorial_summaries

scale_id(s) = replace(string(Float64(s)), "." => "p")

"""Return the eight fixed-truth cells or twelve separate prior-predictive cells."""
function factorial_cells(; experiment::Symbol=:fixed_parameter)
    experiment in (:fixed_parameter, :prior_predictive) ||
        throw(ArgumentError("unknown factorial experiment: $experiment"))
    cells = NamedTuple[]
    for n in (20, 50), control_n in (n ÷ 2, n ÷ 10)
        if experiment == :fixed_parameter
            for effect in (0.9, 3.0)
                push!(cells, (; id="fixed_n$(n)_c$(control_n)_b$(scale_id(effect))",
                    n, control_n, experiment, beta_fixed=[-0.3, effect]))
            end
        else
            for prior_scale in (0.5, 1.5, 3.0)
                push!(cells, (; id="predictive_n$(n)_c$(control_n)_s$(scale_id(prior_scale))",
                    n, control_n, experiment, beta_fixed=zeros(2), prior_scale))
            end
        end
    end
    return cells
end

function validate_cell(cell)
    cell.id isa AbstractString && !isempty(cell.id) || throw(ArgumentError("cell ID must be nonempty"))
    cell.n isa Integer && cell.n >= 2 || throw(ArgumentError("n must be an integer ≥ 2"))
    cell.control_n isa Integer && 0 < cell.control_n < cell.n ||
        throw(ArgumentError("exact control counts must leave both arms nonempty"))
    cell.experiment in (:fixed_parameter, :prior_predictive) || throw(ArgumentError("unknown experiment"))
    validate_beta(cell.beta_fixed)
    if cell.experiment == :prior_predictive
        hasproperty(cell, :prior_scale) && cell.prior_scale isa Real &&
            isfinite(cell.prior_scale) && cell.prior_scale > 0 ||
            throw(ArgumentError("prior-predictive cells require a finite positive scale"))
    end
    return nothing
end

function validate_beta(beta)
    length(beta) == 2 && all(x -> x isa Real && isfinite(x), beta) ||
        throw(ArgumentError("beta must contain two finite coefficients"))
    return nothing
end

"""Declare twelve coefficient priors, or the single exact generating prior.

Only zero-centred priors are deployable. Truth-centred and treatment-offset
means are explicitly labelled oracle diagnostics and are never defaults.
"""
function prior_specs(cell, beta)
    validate_cell(cell)
    validate_beta(beta)
    scales = cell.experiment == :prior_predictive ? (Float64(cell.prior_scale),) : (0.5, 1.5, 3.0)
    centrings = cell.experiment == :prior_predictive ? (:zero,) : (:zero, :oracle, :oracle_plus, :oracle_minus)
    specs = NamedTuple[]
    for scale in scales, centring in centrings
        prior_mean = centring == :zero ? zeros(2) : Float64.(beta)
        centring == :oracle_plus && (prior_mean[2] += 2scale)
        centring == :oracle_minus && (prior_mean[2] -= 2scale)
        push!(specs, (; id="$(centring)_s$(scale_id(scale))", mean=prior_mean,
            covariance=scale^2 * Matrix{Float64}(I, 2, 2), scale, centring,
            deployable=centring == :zero))
    end
    return specs
end

"""Generate one cohort independently of every fitted prior.

The stable FNV/StableRNG streams depend only on cell ID, replication and master
seed. Prior-predictive coefficients come from that cell's exact fitted prior.
The cohort target is the control fraction times the arm risk difference.
"""
function simulate_factorial_dataset(cell, replication::Integer; seed::Integer=20261115)
    validate_cell(cell)
    replication > 0 || throw(ArgumentError("replication must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    simulation_seed = replication_seed(cell.id, replication, seed, 0)
    rng = StableRNG(simulation_seed)
    A = vcat(zeros(Int, cell.control_n), ones(Int, cell.n - cell.control_n))
    X = hcat(ones(cell.n), A)
    beta = cell.experiment == :fixed_parameter ? copy(cell.beta_fixed) :
        rand(rng, MvNormal(zeros(2), cell.prior_scale^2 * Matrix{Float64}(I, 2, 2)))
    y = [Int(rand(rng, Bernoulli(logistic(eta)))) for eta in X * beta]
    arm_truth = logistic(sum(beta)) - logistic(beta[1])
    policy_scale = cell.control_n / cell.n
    dataset_id = "$(cell.id)|seed=$(seed)|rep=$(replication)"
    return (; A, X, y, beta, arm_truth, policy_truth=policy_scale * arm_truth,
        policy_scale, separation=classify_separation(A, y),
        draw_seed=replication_seed(cell.id, replication, seed, 1), dataset_id)
end

isfinite_real(x) = x isa Real && isfinite(x)
safe_mean(xs) = isempty(xs) ? NaN : mean(xs)
binomial_mcse(p, n) = n == 0 ? NaN : sqrt(p * (1 - p) / n)
field(row, name) = hasproperty(row, name) ? getproperty(row, name) : missing

function metric(row, name)
    if name == :policy_error
        estimate, truth = field(row, :policy_mean), field(row, :policy_truth)
        return isfinite_real(estimate) && isfinite_real(truth) ? estimate - truth : field(row, :policy_error)
    end
    return field(row, name)
end

function coverage(row)
    value = field(row, :covered)
    ismissing(value) && return missing
    value isa Real && value in (0, 1) || throw(ArgumentError("covered must be binary or missing"))
    return Float64(value)
end

"""Summarise finite paired values as b minus a, with sample-SD Monte Carlo error.

Missing/nonfinite entries are excluded as pairs; zero or one pair has undefined
MCSE. Neither missing intervals nor their widths are replaced with zeros.
"""
function paired_metric_summary(values_a, values_b)
    length(values_a) == length(values_b) || throw(ArgumentError("paired vectors must have equal lengths"))
    differences = Float64[b - a for (a, b) in zip(values_a, values_b) if isfinite_real(a) && isfinite_real(b)]
    count = length(differences)
    return (; count, delta=safe_mean(differences), mcse=count < 2 ? NaN : std(differences) / sqrt(count))
end

const METRICS = (:width, :policy_error, :reference_content, :endpoint_error,
    :mean_error_sd, :covariance_error, :seconds)

function metric_available(row, name)
    flag = name == :reference_content ? :reference_content_available :
        name == :endpoint_error ? :endpoint_available :
        name in (:mean_error_sd, :covariance_error) ? :moments_available : :available
    value = hasproperty(row, flag) ? getproperty(row, flag) : row.available
    value isa Bool || throw(ArgumentError("metric availability must be Boolean"))
    return value
end

function central_inclusion(rows, level)
    attempted = length(rows)
    alpha = (1 - level) / 2
    count = covered = ambiguous = 0
    if Symbol(first(rows).experiment) == :prior_predictive
        for row in rows
            lower, upper = field(row, :cdf_decision_lower), field(row, :cdf_decision_upper)
            string(field(row, :status)) in ("ok", "boundary_ambiguous") || continue
            isfinite_real(lower) && isfinite_real(upper) && lower <= upper || continue
            inside = lower >= alpha && upper <= 1 - alpha
            outside = upper < alpha || lower > 1 - alpha
            if inside || outside
                count += 1
                covered += inside
            else
                ambiguous += 1
            end
        end
    end
    p = count == 0 ? NaN : covered / count
    return (; count, ambiguous, unavailable=attempted-count,
        coverage_available=p, coverage_all_attempts=count == 0 ? NaN : covered/attempted,
        mcse_available=binomial_mcse(p,count))
end

function group_summary(key, rows)
    available_rows = filter(r -> r.available, rows)
    attempted, available = length(rows), length(available_rows)
    inclusions = Float64[coverage(r) for r in available_rows if !ismissing(coverage(r))]
    coverage_count = length(inclusions)
    covered = sum(inclusions)
    coverage_available = coverage_count == 0 ? NaN : covered / coverage_count
    coverage_all_attempts = covered / attempted
    experiment = Symbol(first(rows).experiment)
    pits = experiment == :prior_predictive ? Float64[field(r, :pit) for r in available_rows
        if isfinite_real(field(r, :pit)) && 0 <= field(r, :pit) <= 1] : Float64[]
    pit_deciles = zeros(Int, 10)
    for pit in pits
        pit_deciles[min(10, floor(Int, 10pit) + 1)] += 1
    end
    failures = Dict{String,Int}()
    for row in rows
        row.available && continue
        status = string(field(row, :status))
        failures[status] = get(failures, status, 0) + 1
    end
    separation_labels = (:none, :complete, :quasi, :single_class, :unknown)
    separation_counts = Dict(label => 0 for label in separation_labels)
    for row in rows
        value = field(row, :separation)
        label = value isa Union{Symbol,AbstractString} ? Symbol(value) : :unknown
        haskey(separation_counts, label) || (label = :unknown)
        separation_counts[label] += 1
    end
    summary = (; cell_id=key[1], prior_id=key[2], method=key[3], experiment,
        attempted, available, failed=attempted-available, coverage_count,
        coverage_ambiguous=available-coverage_count, coverage_available, coverage_all_attempts,
        mcse_available=binomial_mcse(coverage_available, coverage_count),
        mcse_all_attempts=binomial_mcse(coverage_all_attempts, attempted),
        pit_deciles, pit_count=length(pits),
        pit_label=experiment == :prior_predictive ? :prior_predictive : :fixed_parameter_not_sbc,
        failure_reasons=join(["$reason=$(failures[reason])" for reason in sort!(collect(keys(failures)))], ";"),
        separation_none=separation_counts[:none], separation_complete=separation_counts[:complete],
        separation_quasi=separation_counts[:quasi], separation_single_class=separation_counts[:single_class],
        separation_unknown=separation_counts[:unknown])
    for name in METRICS
        values = Float64[metric(row, name) for row in rows
            if metric_available(row, name) && isfinite_real(metric(row, name))]
        names = name == :policy_error ? (:policy_error_bias, :policy_error_rmse, :policy_error_count) :
            (Symbol(name, :_mean), Symbol(name, :_count))
        statistics = name == :policy_error ? (safe_mean(values), isempty(values) ? NaN : sqrt(mean(abs2, values)), length(values)) :
            (safe_mean(values), length(values))
        summary = merge(summary, NamedTuple{names}(statistics))
    end
    for level in (0.5, 0.8)
        result = central_inclusion(rows, level)
        names = Tuple(Symbol("central", round(Int,100level), "_", key) for key in keys(result))
        summary = merge(summary, NamedTuple{names}(Base.values(result)))
    end
    return summary
end

function compare_groups(rows_a, rows_b)
    by_id_a = Dict(String(r.dataset_id) => r for r in rows_a)
    by_id_b = Dict(String(r.dataset_id) => r for r in rows_b)
    ids = sort!(collect(intersect(keys(by_id_a), keys(by_id_b))))
    a, b = [by_id_a[id] for id in ids], [by_id_b[id] for id in ids]
    jointly_available = [(ra, rb) for (ra, rb) in zip(a, b) if ra.available && rb.available]
    ca = [coverage(ra) for (ra, rb) in jointly_available]
    cb = [coverage(rb) for (ra, rb) in jointly_available]
    cov = paired_metric_summary(ca, cb)
    availability = paired_metric_summary([r.available for r in a], [r.available for r in b])
    inclusion(r) = r.available && !ismissing(coverage(r)) ? coverage(r) : 0.0
    all_inclusion = paired_metric_summary(inclusion.(a), inclusion.(b))
    result = (; attempts=length(ids), attempted_a=length(rows_a), attempted_b=length(rows_b),
        unmatched_a=length(rows_a)-length(ids), unmatched_b=length(rows_b)-length(ids),
        available_a=count(r -> r.available, a), available_b=count(r -> r.available, b),
        paired=length(jointly_available), coverage_count=cov.count,
        coverage_delta=cov.delta, coverage_delta_mcse=cov.mcse,
        availability_delta=availability.delta, availability_delta_mcse=availability.mcse,
        all_attempts_inclusion_delta=all_inclusion.delta,
        all_attempts_inclusion_delta_mcse=all_inclusion.mcse)
    for name in METRICS
        pairs = [(ra,rb) for (ra,rb) in zip(a,b) if metric_available(ra,name) && metric_available(rb,name)]
        s = paired_metric_summary([metric(ra, name) for (ra, rb) in pairs],
            [metric(rb, name) for (ra, rb) in pairs])
        result = merge(result, NamedTuple{(Symbol(name, :_count), Symbol(name, :_delta), Symbol(name, :_delta_mcse))}((s.count, s.delta, s.mcse)))
    end
    return result
end

"""Return group summaries and paired method/prior contrasts on common dataset IDs.

Groups are cell/prior/method specific. Fixed-truth PITs are never labelled or
pooled as calibration. Optional metrics retain explicit finite-value counts.
Method contrasts are LRVB minus raw VI; prior contrasts are exploratory and use
lexical prior-ID order (b minus a). Availability and all-attempt inclusion have
separate paired summaries, while widths/fidelity require common availability.
"""
function factorial_summaries(rows)
    groups = Dict{Tuple{String,String,Symbol},Vector{Any}}()
    seen = Set{Tuple{String,String,Symbol,String}}()
    experiments = Dict{String,Symbol}()
    for row in rows
        row.available isa Bool || throw(ArgumentError("availability must be Boolean"))
        key = (String(row.cell_id), String(row.prior_id), Symbol(row.method))
        experiment = Symbol(row.experiment)
        experiment in (:fixed_parameter, :prior_predictive) || throw(ArgumentError("unknown experiment"))
        get!(experiments, key[1], experiment) == experiment || throw(ArgumentError("experiments must not be pooled"))
        unique_key = (key..., String(row.dataset_id))
        unique_key in seen && throw(ArgumentError("duplicate factorial row: $unique_key"))
        push!(seen, unique_key)
        group = get!(groups, key, Any[])
        if row.available
            coverage(row)
        end
        push!(group, row)
    end
    keys_sorted = sort!(collect(keys(groups)); by=x -> (x[1], x[2], string(x[3])))
    summaries = [group_summary(key, groups[key]) for key in keys_sorted]
    paired_methods, paired_priors = NamedTuple[], NamedTuple[]
    for key in keys_sorted
        cell_id, prior_id, method = key
        if method == :variational && haskey(groups, (cell_id, prior_id, :linear_response))
            push!(paired_methods, merge((; cell_id, prior_id, method_a=:variational,
                method_b=:linear_response, comparison_label=:method_fidelity),
                compare_groups(groups[key], groups[(cell_id, prior_id, :linear_response)])))
        end
        for other in keys_sorted
            other[1] == cell_id && other[3] == method && other[2] > prior_id || continue
            push!(paired_priors, merge((; cell_id, method, prior_a=prior_id,
                prior_b=other[2], comparison_label=:exploratory), compare_groups(groups[key], groups[other])))
        end
    end
    return (; summaries, paired_methods, paired_priors)
end

end # module
