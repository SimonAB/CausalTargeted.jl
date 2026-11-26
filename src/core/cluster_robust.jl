"""Cluster-robust variance for cross-fitted influence functions."""

"""
    resolve_cluster_ids(df, cluster, n) -> Union{Nothing, Vector}

Normalise `cluster` to a length-`n` id vector, or `nothing`. When `cluster` is a
vector aligned to the input rows, pass `input_n=nrow(df)` and `input_rows` indexing
into the analysis frame after missingness handling.
"""
function resolve_cluster_ids(
    df::DataFrame,
    cluster,
    n::Int;
    input_n::Int = n,
    input_rows = nothing,
)
    if cluster === nothing
        return nothing
    elseif cluster isa Symbol
        hasproperty(df, cluster) || throw(ArgumentError(
            "cluster column :$cluster not found in analysis frame",
        ))
        return collect(df[!, cluster])
    else
        if length(cluster) == n
            return collect(cluster)
        elseif input_rows !== nothing && length(cluster) == input_n
            return collect(cluster[input_rows])
        else
            throw(ArgumentError(
                "cluster vector length $(length(cluster)) must match input n=$input_n or analysis n=$n",
            ))
        end
    end
end

"""
    cluster_robust_variance(ic; cluster=nothing) -> Float64

Sampling variance of the mean of a centred influence vector `ic`.
Unit-level: ``\\mathrm{Var}=\\|ic\\|^2/n^2``; cluster level: sum within clusters first.
"""
function cluster_robust_variance(
    ic::AbstractVector{<:Real};
    cluster::Union{Nothing, AbstractVector} = nothing,
)
    n = length(ic)
    n < 1 && throw(ArgumentError("ic must be non-empty"))
    ic64 = Float64.(ic)
    if cluster === nothing
        return sum(abs2, ic64) / n^2
    end
    length(cluster) == n || throw(ArgumentError("cluster length must match ic"))
    groups = Dict{Any, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, cluster[i], Int[]), i)
    end
    s2 = 0.0
    for idxs in values(groups)
        s = sum(ic64[idxs])
        s2 += s * s
    end
    return s2 / n^2
end

export resolve_cluster_ids, cluster_robust_variance
