module PolicyCalibrationSupport

using LinearAlgebra, Random, Statistics
using StableRNGs: StableRNG
using Distributions: Bernoulli, MvNormal

export calibration_cells, simulate_policy_replication, compress_binary_likelihood,
    summarise_policy_rows, paired_policy_summary

"""Return the prespecified screen (2) or validation (14) calibration cells.

Prior-scale cells use independent simulation streams. Their summaries compare
separate calibration experiments, not paired effects attributable to the prior.
"""
function calibration_cells(profile::Symbol)
    profile in (:screen, :validation) || throw(ArgumentError("unknown calibration profile: $profile"))
    cells = NamedTuple[]
    sizes = profile == :screen ? (20,) : (20, 50)
    scenarios = profile == :screen ? (:balanced,) : (:balanced, :rare_control, :strong_prior, :separated)
    for n in sizes, scenario in scenarios, experiment in (:fixed_parameter, :prior_predictive)
        scenario == :separated && experiment == :prior_predictive && continue
        variance = scenario == :strong_prior ? 0.25 : 2.25
        push!(cells, (; id="$(scenario)_n$(n)_$(experiment)", scenario, n, experiment,
            prior_mean=zeros(2), prior_covariance=variance * Matrix{Float64}(I, 2, 2),
            beta_fixed=scenario == :separated ? [-3.0, 6.0] : [-0.3, 0.9]))
    end
    return cells
end

# FNV-1a over an explicit encoding avoids Julia's process-dependent hash.
# A reserved low bit separates simulation and Gaussian-draw channels.
function replication_seed(cell_id, replication, seed, channel)
    value = UInt64(0xcbf29ce484222325)
    for byte in codeunits("$(seed)|$(ncodeunits(cell_id)):$(cell_id)|$(replication)")
        value = (value ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return (value << 1) | UInt64(channel)
end

logistic(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))

function classify_separation(A, y)
    all(==(first(y)), y) && return :single_class
    control = y[A .== 0]
    treated = y[A .== 1]
    pure_control = all(==(first(control)), control)
    pure_treated = all(==(first(treated)), treated)
    pure_control && pure_treated && return :complete
    (pure_control || pure_treated) && return :quasi
    return :none
end

"""Simulate a fixed binary cohort, coefficient truth and Bernoulli outcomes.

The policy truth is the untreated cohort fraction times the arm risk difference.
The draw seed is a separate deterministic stream for shared Gaussian draws.
"""
function simulate_policy_replication(cell, replication::Integer; seed::Integer=20261114)
    replication > 0 || throw(ArgumentError("replication must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    cell.n >= 2 || throw(ArgumentError("both exposure levels require n ≥ 2"))
    rng = StableRNG(replication_seed(cell.id, replication, seed, 0))
    ncontrol = cell.scenario == :rare_control ? 1 : cell.n ÷ 2
    A = vcat(zeros(Int, ncontrol), ones(Int, cell.n - ncontrol))
    X = hcat(ones(cell.n), A)
    beta = if cell.experiment == :prior_predictive
        rand(rng, MvNormal(cell.prior_mean, cell.prior_covariance))
    elseif cell.experiment == :fixed_parameter
        copy(cell.beta_fixed)
    else
        throw(ArgumentError("unknown experiment: $(cell.experiment)"))
    end
    y = [Int(rand(rng, Bernoulli(logistic(eta)))) for eta in X * beta]
    arm_truth = logistic(sum(beta)) - logistic(beta[1])
    policy_scale = ncontrol / cell.n
    return (; A, X, y, beta, arm_truth, policy_scale,
        policy_truth=policy_scale * arm_truth, separation=classify_separation(A, y),
        draw_seed=replication_seed(cell.id, replication, seed, 1))
end

"""Collapse identical binary exposure/outcome rows to positive integer frequencies."""
function compress_binary_likelihood(A, y)
    length(A) == length(y) && !isempty(A) || throw(ArgumentError("A and y must have equal, nonzero lengths"))
    for values in (A, y)
        all(x -> x isa Real && isfinite(x) && (x == 0 || x == 1), values) ||
            throw(ArgumentError("A and y must contain only nonmissing binary values"))
    end
    counts = zeros(Int, 2, 2)
    for (a, outcome) in zip(A, y)
        counts[Int(a) + 1, Int(outcome) + 1] += 1
    end
    exposures, outcomes, weights = Int[], Int[], Int[]
    for a in 0:1, outcome in 0:1
        frequency = counts[a + 1, outcome + 1]
        frequency == 0 && continue
        push!(exposures, a)
        push!(outcomes, outcome)
        push!(weights, frequency)
    end
    return (; X=hcat(ones(length(weights)), exposures), y=outcomes, weights)
end

normalise_label(value) = Symbol(value)
finite_values(rows, field) = Float64[getproperty(row, field) for row in rows
    if getproperty(row, field) isa Real && isfinite(getproperty(row, field))]
safe_mean(values) = isempty(values) ? NaN : mean(values)
binomial_mcse(p, n) = n == 0 ? NaN : sqrt(p * (1 - p) / n)

function covered_value(row)
    value = row.covered
    value isa Real && (value == 0 || value == 1) ||
        throw(ArgumentError("available rows require binary covered values"))
    return Int(value)
end

"""Summarise each cell/method without dropping failures from attempt denominators.

Finite descriptive metrics use available rows only. Empty scalar summaries are
NaN; explicit counts distinguish unavailable quantities from observed zeros.
PIT deciles include 1 in the last bin, and only prior-predictive PITs are SBC.
Realised separation counts include every attempt, including unavailable methods;
rows without a recorded separation category contribute to `separation_unknown`.
"""
function summarise_policy_rows(rows)
    groups = Dict{Tuple{String,Symbol},Vector{Any}}()
    for row in rows
        key = (String(row.cell_id), normalise_label(row.method))
        push!(get!(groups, key, Any[]), row)
    end
    summaries = NamedTuple[]
    for key in sort!(collect(keys(groups)); by=x -> (x[1], String(x[2])))
        group = groups[key]
        successes = filter(row -> row.available, group)
        attempted, available = length(group), length(successes)
        covered = sum(covered_value(row) for row in successes; init=0)
        coverage_available = available == 0 ? NaN : covered / available
        coverage_all_attempts = covered / attempted
        pits = filter(x -> 0 <= x <= 1, finite_values(successes, :pit))
        pit_deciles = zeros(Int, 10)
        for pit in pits
            pit_deciles[min(10, floor(Int, 10 * pit) + 1)] += 1
        end
        widths = finite_values(successes, :width)
        errors = finite_values(successes, :policy_error)
        failures = Dict{String,Int}()
        for row in group
            row.available && continue
            status = String(normalise_label(row.status))
            failures[status] = get(failures, status, 0) + 1
        end
        failure_reasons = join(["$reason=$(failures[reason])" for reason in sort!(collect(keys(failures)))], ";")
        separation_counts = Dict(label => 0 for label in (:none, :complete, :quasi, :single_class, :unknown))
        for row in group
            recorded = hasproperty(row, :separation) ? row.separation : :unknown
            label = recorded isa Union{Symbol,AbstractString} ? normalise_label(recorded) : :unknown
            haskey(separation_counts, label) || (label = :unknown)
            separation_counts[label] += 1
        end
        firstrow = first(group)
        experiment = normalise_label(firstrow.experiment)
        push!(summaries, (; cell_id=key[1], method=key[2], n=firstrow.n,
            scenario=normalise_label(firstrow.scenario), experiment, attempted, available,
            failed=attempted-available, coverage_available, coverage_all_attempts,
            mcse_available=binomial_mcse(coverage_available, available),
            mcse_all_attempts=binomial_mcse(coverage_all_attempts, attempted),
            pit_deciles, pit_count=length(pits),
            pit_label=experiment == :prior_predictive ? :sbc : :fixed_parameter_not_sbc,
            width_mean=safe_mean(widths), width_count=length(widths),
            policy_error_bias=safe_mean(errors),
            policy_error_rmse=isempty(errors) ? NaN : sqrt(mean(abs2, errors)),
            policy_error_count=length(errors), failure_reasons,
            separation_none=separation_counts[:none],
            separation_complete=separation_counts[:complete],
            separation_quasi=separation_counts[:quasi],
            separation_single_class=separation_counts[:single_class],
            separation_unknown=separation_counts[:unknown]))
    end
    return summaries
end

"""Compare LRVB and raw VI coverage on jointly available replications.

The MCSE uses the sample standard deviation of within-replication coverage
differences. It is undefined for fewer than two pairs. Duplicate method keys
are rejected, including duplicate reference rows in resumed checkpoints.
The separate all-attempt inclusion comparison counts unavailable intervals as
noncoverage; both method records must exist for every attempted replication.
"""
function paired_policy_summary(rows)
    seen = Set{Tuple{String,Int,Symbol}}()
    groups = Dict{String,Dict{Int,Dict{Symbol,Any}}}()
    for row in rows
        method = normalise_label(row.method)
        key = (String(row.cell_id), Int(row.replication), method)
        key in seen && throw(ArgumentError("duplicate calibration row: $key"))
        push!(seen, key)
        method in (:variational, :linear_response) || continue
        cell = get!(groups, key[1], Dict{Int,Dict{Symbol,Any}}())
        replication = get!(cell, key[2], Dict{Symbol,Any}())
        replication[method] = row
    end
    summaries = NamedTuple[]
    for cell_id in sort!(collect(keys(groups)))
        replications = groups[cell_id]
        differences = Float64[]
        inclusion_differences = Float64[]
        variational_available, linear_response_available = 0, 0
        for replication in values(replications)
            vi = get(replication, :variational, nothing)
            lr = get(replication, :linear_response, nothing)
            vi !== nothing && lr !== nothing ||
                throw(ArgumentError("incomplete variational/linear-response pair in cell $cell_id"))
            vi_available = vi.available
            lr_available = lr.available
            variational_available += vi_available
            linear_response_available += lr_available
            vi_inclusion = vi_available ? covered_value(vi) : 0
            lr_inclusion = lr_available ? covered_value(lr) : 0
            push!(inclusion_differences, lr_inclusion - vi_inclusion)
            vi_available && lr_available || continue
            push!(differences, lr_inclusion - vi_inclusion)
        end
        paired = length(differences)
        all_attempts_paired = length(inclusion_differences)
        push!(summaries, (; cell_id, attempts=length(replications), paired,
            coverage_delta=safe_mean(differences),
            coverage_delta_mcse=paired < 2 ? NaN : std(differences) / sqrt(paired),
            all_attempts_paired,
            all_attempts_inclusion_delta=safe_mean(inclusion_differences),
            all_attempts_inclusion_delta_mcse=all_attempts_paired < 2 ? NaN :
                std(inclusion_differences) / sqrt(all_attempts_paired),
            variational_available, linear_response_available,
            availability_difference=linear_response_available-variational_available))
    end
    return summaries
end

end # module
