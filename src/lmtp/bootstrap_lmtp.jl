"""Nonparametric bootstrap uncertainty for discrete LMTP contrasts."""

"""
    _bootstrap_row_indices(n, rng; cluster_ids=nothing) -> Vector{Int}

Row indices for one bootstrap replicate (i.i.d. or cluster resample).
"""
function _bootstrap_row_indices(
    n::Int,
    rng;
    cluster_ids::Union{Nothing, AbstractVector} = nothing,
)
    return first(_bootstrap_resample(n, rng; cluster_ids = cluster_ids))
end

function _expand_cluster_sample(cluster_ids::AbstractVector, sampled::AbstractVector)
    idx = Int[]
    copy_ids = Int[]
    for (copy_id, cid) in enumerate(sampled)
        for i in eachindex(cluster_ids)
            if isequal(cluster_ids[i], cid)
                push!(idx, i)
                push!(copy_ids, copy_id)
            end
        end
    end
    isempty(idx) && throw(ArgumentError("cluster resampling produced no rows"))
    return idx, copy_ids
end

function _bootstrap_resample(
    n::Int,
    rng;
    cluster_ids::Union{Nothing, AbstractVector} = nothing,
)
    if cluster_ids === nothing
        return rand(rng, 1:n, n), nothing
    end
    length(cluster_ids) == n || throw(ArgumentError("cluster_ids length must match n"))
    uniq = unique(cluster_ids)
    isempty(uniq) && throw(ArgumentError("cluster_ids must be non-empty"))
    sampled = rand(rng, uniq, length(uniq))
    return _expand_cluster_sample(cluster_ids, sampled)
end

function _bootstrap_cluster_argument(df_b, cluster, copy_ids)
    cluster === nothing && return nothing
    if cluster isa Symbol
        boot_column = Symbol("_bootstrap_cluster")
        while hasproperty(df_b, boot_column)
            boot_column = Symbol(boot_column, "_")
        end
        df_b[!, boot_column] = copy_ids
        return boot_column
    end
    return copy_ids
end

"""Group bootstrap copies by source identity, independently of covariance copy IDs."""
function _bootstrap_fold_groups(indices::AbstractVector; cluster_ids = nothing)
    return cluster_ids === nothing ? collect(indices) : collect(cluster_ids[indices])
end

function _bootstrap_gate(n_success::Int, B::Int, min_success_fraction::Real)
    0 < min_success_fraction <= 1 || throw(ArgumentError(
        "min_success_fraction must lie in (0, 1]",
    ))
    required = max(2, ceil(Int, min_success_fraction * B))
    return n_success >= required, required
end

_record_bootstrap_failure!(failures::Dict{String, Int}, error) =
    (failures[sprint(showerror, error)] = get(failures, sprint(showerror, error), 0) + 1)

function _bootstrap_hurdle_positivity_ok(result)
    return all((
        result.presence.hi.positivity.ok,
        result.presence.ref.positivity.ok,
        result.intensity.hi.positivity.ok,
        result.intensity.ref.positivity.ok,
    ))
end

"""
    _percentile_ci(samples; alpha=0.05) -> Tuple

Two-sided percentile interval from bootstrap draws.
"""
function _percentile_ci(samples::AbstractVector{<:Real}; alpha::Real = 0.05)
    xs = sort(collect(Float64, samples))
    isempty(xs) && return (NaN, NaN)
    lo = xs[max(1, ceil(Int, (alpha / 2) * length(xs)))]
    hi = xs[min(length(xs), floor(Int, (1 - alpha / 2) * length(xs)))]
    return (lo, hi)
end

"""
    bootstrap_discrete_lmtp_contrast(df, trt, outcome; B, rng, cluster, ...) -> NamedTuple

Percentile bootstrap CI for [`run_discrete_lmtp_contrast`](@ref). Uses reduced
`folds` inside replicates by default. Cluster resampling gives every sampled
copy a fresh cluster ID. Only finite replicates with positivity supported both
at the point fit and in the replicate count as successful. If fewer than
`max(2, ceil(min_success_fraction*B))`
replicates succeed, the interval and bootstrap SE are `NaN`; the point estimate
is retained and failures are reported.
"""
function bootstrap_discrete_lmtp_contrast(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    arm_hi,
    arm_ref,
    levels,
    B::Int = 200,
    rng = StableRNG(1),
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
    min_success_fraction::Real = 0.8,
    folds::Int = 2,
    kwargs...,
)
    B < 1 && throw(ArgumentError("B must be ≥ 1"))
    0 < min_success_fraction <= 1 || throw(ArgumentError(
        "min_success_fraction must lie in (0, 1]",
    ))
    point = run_discrete_lmtp_contrast(
        df, trt, outcome;
        arm_hi = arm_hi,
        arm_ref = arm_ref,
        levels = levels,
        folds = folds,
        rng = rng,
        cluster = cluster,
        kwargs...,
    )
    n = nrow(df)
    cluster_ids = resolve_cluster_ids(df, cluster, n)
    boots = Float64[]
    failures = Dict{String, Int}()
    n_positivity_failures = 0
    for b in 1:B
        brng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(rng), b))
        idx, copy_ids = _bootstrap_resample(n, brng; cluster_ids = cluster_ids)
        df_b = df[idx, :]
        cluster_b = _bootstrap_cluster_argument(df_b, cluster, copy_ids)
        try
            res = run_discrete_lmtp_contrast(
                df_b, trt, outcome;
                arm_hi = arm_hi,
                arm_ref = arm_ref,
                levels = levels,
                folds = folds,
                rng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(brng), 1)),
                cluster = cluster_b,
                fold_groups = _bootstrap_fold_groups(idx; cluster_ids = cluster_ids),
                kwargs...,
            )
            if !point.positivity.ok || !res.positivity.ok
                n_positivity_failures += 1
                _record_bootstrap_failure!(failures, "point or replicate positivity violation")
            elseif isfinite(res.estimate)
                push!(boots, res.estimate)
            else
                _record_bootstrap_failure!(failures, "non-finite bootstrap estimate")
            end
        catch error
            _record_bootstrap_failure!(failures, error)
        end
    end
    n_success = length(boots)
    gate, required_success = _bootstrap_gate(n_success, B, min_success_fraction)
    if !gate
        return (;
            estimate = point.estimate,
            se = NaN,
            lower = NaN,
            upper = NaN,
            boot_samples = boots,
            n_success = n_success,
            n_attempted = B,
            n_failed = B - n_success,
            n_positivity_failures = n_positivity_failures,
            required_success = required_success,
            status = :insufficient_success,
            failures = failures,
            point = point,
        )
    end
    se = std(boots; corrected = true)
    lwr, upr = _percentile_ci(boots)
    return (;
        estimate = point.estimate,
        se = se,
        lower = lwr,
        upper = upr,
        boot_samples = boots,
        n_success = n_success,
        n_attempted = B,
        n_failed = B - n_success,
        n_positivity_failures = n_positivity_failures,
        required_success = required_success,
        status = :ok,
        failures = failures,
        point = point,
    )
end

"""
    bootstrap_two_part_discrete_lmtp_contrast(df, trt; presence, intensity, ...) -> NamedTuple

Bootstrap summaries for presence and intensity arms of
[`run_two_part_discrete_lmtp_contrast`](@ref). Both parts use the same
successful replicates; if the minimum success fraction is not met, returned
bootstrap SEs and limits are `NaN` and the failure summary is retained.
Replicates count as successful only when the point fit and both arms of both
hurdle components satisfy positivity.
"""
function bootstrap_two_part_discrete_lmtp_contrast(
    df::DataFrame,
    trt::Symbol;
    presence::Symbol,
    intensity::Symbol,
    arm_hi,
    arm_ref,
    levels,
    B::Int = 200,
    rng = StableRNG(1),
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
    min_success_fraction::Real = 0.8,
    folds::Int = 2,
    kwargs...,
)
    B >= 1 || throw(ArgumentError("B must be ≥ 1"))
    0 < min_success_fraction <= 1 || throw(ArgumentError(
        "min_success_fraction must lie in (0, 1]",
    ))
    point = run_two_part_discrete_lmtp_contrast(
        df, trt;
        presence = presence,
        intensity = intensity,
        arm_hi = arm_hi,
        arm_ref = arm_ref,
        levels = levels,
        folds = folds,
        rng = rng,
        cluster = cluster,
        kwargs...,
    )
    n = nrow(df)
    cluster_ids = resolve_cluster_ids(df, cluster, n)
    pres_boots = Float64[]
    int_boots = Float64[]
    failures = Dict{String, Int}()
    n_positivity_failures = 0
    for b in 1:B
        brng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(rng), b))
        idx, copy_ids = _bootstrap_resample(n, brng; cluster_ids = cluster_ids)
        df_b = df[idx, :]
        cluster_b = _bootstrap_cluster_argument(df_b, cluster, copy_ids)
        try
            res = run_two_part_discrete_lmtp_contrast(
                df_b, trt;
                presence = presence,
                intensity = intensity,
                arm_hi = arm_hi,
                arm_ref = arm_ref,
                levels = levels,
                folds = folds,
                rng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(brng), 1)),
                cluster = cluster_b,
                fold_groups = _bootstrap_fold_groups(idx; cluster_ids = cluster_ids),
                kwargs...,
            )
            if !_bootstrap_hurdle_positivity_ok(point) ||
                !_bootstrap_hurdle_positivity_ok(res)
                n_positivity_failures += 1
                _record_bootstrap_failure!(failures, "point or replicate hurdle positivity violation")
            elseif isfinite(res.presence.estimate) && isfinite(res.intensity.estimate)
                push!(pres_boots, res.presence.estimate)
                push!(int_boots, res.intensity.estimate)
            else
                _record_bootstrap_failure!(failures, "non-finite paired hurdle estimate")
            end
        catch error
            _record_bootstrap_failure!(failures, error)
        end
    end
    n_success = length(pres_boots)
    gate, required_success = _bootstrap_gate(n_success, B, min_success_fraction)
    pres_se = gate ? std(pres_boots; corrected = true) : NaN
    int_se = gate ? std(int_boots; corrected = true) : NaN
    pres_lwr, pres_upr = gate ? _percentile_ci(pres_boots) : (NaN, NaN)
    int_lwr, int_upr = gate ? _percentile_ci(int_boots) : (NaN, NaN)
    return (;
        presence = (;
            estimate = point.presence.estimate,
            se = pres_se,
            lower = pres_lwr,
            upper = pres_upr,
            boot_samples = pres_boots,
            n_success = n_success,
            n_attempted = B,
            n_failed = B - n_success,
            n_positivity_failures = n_positivity_failures,
            required_success = required_success,
            status = gate ? :ok : :insufficient_success,
            failures = failures,
        ),
        intensity = (;
            estimate = point.intensity.estimate,
            se = int_se,
            lower = int_lwr,
            upper = int_upr,
            boot_samples = int_boots,
            n_success = n_success,
            n_attempted = B,
            n_failed = B - n_success,
            n_positivity_failures = n_positivity_failures,
            required_success = required_success,
            status = gate ? :ok : :insufficient_success,
            failures = failures,
        ),
        point = point,
    )
end

export bootstrap_discrete_lmtp_contrast, bootstrap_two_part_discrete_lmtp_contrast
