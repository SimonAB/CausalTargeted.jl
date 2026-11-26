"""Domain transport weights for estimation under selection / environment shift."""

using DataFrames
using Statistics

"""
    domain_transport_weights(data, domain; target, source=nothing, trunc=0.01) -> Vector{Float64}

Stabilised IPTW-style weights for transporting from observed domains to `target`.

When `domain` is binary / two-level and `source` is omitted, uses all non-target
rows as source. Weights are ``w_i = P(D=target) / P(D=d_i)`` clipped away from
zero (simple marginal transport; no outcome model).
"""
function domain_transport_weights(
    data::DataFrame,
    domain::Symbol;
    target,
    source = nothing,
    trunc::Float64 = 0.01,
)
    hasproperty(data, domain) || throw(ArgumentError("missing domain column :$domain"))
    d = data[!, domain]
    n = length(d)
    n == 0 && return Float64[]
    levels = unique(d)
    target in levels || throw(ArgumentError("target $target not in domain levels=$levels"))
    if source !== nothing
        source in levels || throw(ArgumentError("source $source not in domain levels=$levels"))
    end
    # Empirical domain probabilities
    p = Dict{Any, Float64}()
    for lev in levels
        p[lev] = count(==(lev), d) / n
    end
    p_t = max(p[target], trunc)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        di = d[i]
        if source !== nothing && di != source && di != target
            w[i] = 0.0
            continue
        end
        p_d = max(get(p, di, trunc), trunc)
        w[i] = p_t / p_d
    end
    # Normalise to mean 1 on positive weights
    pos = w .> 0
    if any(pos)
        w[pos] ./= mean(w[pos])
    end
    return w
end

"""
    transport_weighted_mean(values, weights) -> Float64

Weighted mean ``∑ w_i y_i / ∑ w_i``.
"""
function transport_weighted_mean(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    length(values) == length(weights) || throw(ArgumentError("length mismatch"))
    sw = sum(weights)
    sw ≈ 0 && throw(ArgumentError("weights sum to zero"))
    return sum(weights .* values) / sw
end

export domain_transport_weights, transport_weighted_mean
