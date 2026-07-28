"""Compare user DAG adjustment sets to an alternative (e.g. discovery) set.

Discovery algorithms (PC, OCE, …) are useful for *sensitivity*, never as silent
oracles that replace a user-specified DAG in production defaults.

# References

- Pearl (2009); Spirtes, Glymour & Scheines (2000) — structural search vs confirmation
- CausalDynamics Associations.jl bridge — optional discovery → `IdentificationResult`
"""

using CausalDynamics

"""
    adjustment_set_disagreement(user_adj, alt_adj) -> NamedTuple

Set differences between a user-specified adjustment set and an alternative
(discovery or sensitivity) set. Never replaces the user DAG silently.
"""
function adjustment_set_disagreement(
    user_adj::AbstractVector{Symbol},
    alt_adj::AbstractVector{Symbol},
)
    u = Set(Symbol.(user_adj))
    a = Set(Symbol.(alt_adj))
    only_user = sort!(collect(setdiff(u, a)))
    only_alt = sort!(collect(setdiff(a, u)))
    shared = sort!(collect(intersect(u, a)))
    return (
        only_user = only_user,
        only_alt = only_alt,
        shared = shared,
        jaccard = isempty(u) && isempty(a) ? 1.0 :
            length(shared) / max(length(union(u, a)), 1),
        agree = isempty(only_user) && isempty(only_alt),
    )
end

"""
    discovery_adjustment_sensitivity(user_adj, alt_adj; note) -> Dict{String,Any}

Columns suitable for merging into an identification certificate dict under
`id_sensitivity_*` keys.
"""
function discovery_adjustment_sensitivity(
    user_adj::AbstractVector{Symbol},
    alt_adj::AbstractVector{Symbol};
    note::AbstractString = "alternative_adjustment",
)
    d = adjustment_set_disagreement(user_adj, alt_adj)
    return Dict{String, Any}(
        "id_sensitivity_note" => string(note),
        "id_sensitivity_agree" => d.agree,
        "id_sensitivity_jaccard" => d.jaccard,
        "id_sensitivity_only_user" => join(string.(d.only_user), ","),
        "id_sensitivity_only_alt" => join(string.(d.only_alt), ","),
        "id_sensitivity_shared" => join(string.(d.shared), ","),
    )
end

"""
    merge_discovery_sensitivity!(cert_dict, user_adj, alt_adj; note) -> Dict
"""
function merge_discovery_sensitivity!(
    cert_dict::AbstractDict,
    user_adj::AbstractVector{Symbol},
    alt_adj::AbstractVector{Symbol};
    note::AbstractString = "discovery_vs_user_dag",
)
    merge!(cert_dict, discovery_adjustment_sensitivity(user_adj, alt_adj; note = note))
    return cert_dict
end

export adjustment_set_disagreement, discovery_adjustment_sensitivity, merge_discovery_sensitivity!
