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
    if cluster_ids === nothing
        return rand(rng, 1:n, n)
    end
    length(cluster_ids) == n || throw(ArgumentError("cluster_ids length must match n"))
    uniq = unique(cluster_ids)
    sampled = rand(rng, uniq, length(uniq))
    idx = Int[]
    for cid in sampled
        for i in 1:n
            cluster_ids[i] == cid && push!(idx, i)
        end
    end
    isempty(idx) && return rand(rng, 1:n, n)
    return idx
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
`folds` inside replicates by default. Returns point estimate, bootstrap SE,
interval, draws, and success count.
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
    folds::Int = 2,
    kwargs...,
)
    B < 1 && throw(ArgumentError("B must be ≥ 1"))
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
    for b in 1:B
        brng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(rng), b))
        idx = _bootstrap_row_indices(n, brng; cluster_ids = cluster_ids)
        df_b = df[idx, :]
        try
            res = run_discrete_lmtp_contrast(
                df_b, trt, outcome;
                arm_hi = arm_hi,
                arm_ref = arm_ref,
                levels = levels,
                folds = folds,
                rng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(brng), 1)),
                cluster = cluster,
                kwargs...,
            )
            isfinite(res.estimate) && push!(boots, res.estimate)
        catch
        end
    end
    n_success = length(boots)
    if n_success == 0
        return (;
            estimate = point.estimate,
            se = point.se,
            lower = point.lower,
            upper = point.upper,
            boot_samples = boots,
            n_success = 0,
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
        point = point,
    )
end

"""
    bootstrap_two_part_discrete_lmtp_contrast(df, trt; presence, intensity, ...) -> NamedTuple

Bootstrap summaries for presence and intensity arms of
[`run_two_part_discrete_lmtp_contrast`](@ref).
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
    folds::Int = 2,
    kwargs...,
)
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
    for b in 1:B
        brng = StableRNG(CausalDynamics.stable_seed(CausalDynamics.stable_rng_seed(rng), b))
        idx = _bootstrap_row_indices(n, brng; cluster_ids = cluster_ids)
        df_b = df[idx, :]
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
                cluster = cluster,
                kwargs...,
            )
            isfinite(res.presence.estimate) && push!(pres_boots, res.presence.estimate)
            isfinite(res.intensity.estimate) && push!(int_boots, res.intensity.estimate)
        catch
        end
    end
    pres_se = isempty(pres_boots) ? point.presence.se : std(pres_boots; corrected = true)
    int_se = isempty(int_boots) ? point.intensity.se : std(int_boots; corrected = true)
    pres_lwr, pres_upr = isempty(pres_boots) ?
        (point.presence.lower, point.presence.upper) :
        _percentile_ci(pres_boots)
    int_lwr, int_upr = isempty(int_boots) ?
        (point.intensity.lower, point.intensity.upper) :
        _percentile_ci(int_boots)
    return (;
        presence = (;
            estimate = point.presence.estimate,
            se = pres_se,
            lower = pres_lwr,
            upper = pres_upr,
            boot_samples = pres_boots,
            n_success = length(pres_boots),
        ),
        intensity = (;
            estimate = point.intensity.estimate,
            se = int_se,
            lower = int_lwr,
            upper = int_upr,
            boot_samples = int_boots,
            n_success = length(int_boots),
        ),
        point = point,
    )
end

export bootstrap_discrete_lmtp_contrast, bootstrap_two_part_discrete_lmtp_contrast
