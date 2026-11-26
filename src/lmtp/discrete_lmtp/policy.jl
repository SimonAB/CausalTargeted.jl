# ---------------------------------------------------------------------------
# DiscreteTreatmentPolicy, DiscreteInterventionalMean, support & positivity
# ---------------------------------------------------------------------------

"""
    DiscreteTreatmentPolicy

Modified treatment policy on a finite exposure. Recode maps, a static
`do(A = a*)`, or a user function `(a, wrow) -> a'` (R `lmtp` `shift`).
`mtp=true` when the rule depends on the natural value of `A`.
"""
struct DiscreteTreatmentPolicy <: AbstractMTPPolicy
    levels::Vector{Any}
    recode::Dict{Any, Any}
    static::Any
    shift::Union{Nothing, Function}
    mtp::Bool
end

function DiscreteTreatmentPolicy(;
    levels = Any[],
    recode = Dict{Any, Any}(),
    static = nothing,
    shift = nothing,
    mtp::Bool = true,
)
    return DiscreteTreatmentPolicy(
        collect(Any, levels), Dict{Any, Any}(recode), static, shift, mtp,
    )
end

"""
    discrete_recode_policy(recode; levels=nothing, mtp=true) -> DiscreteTreatmentPolicy

Map observed levels onto new levels (e.g. `Dict("high" => "low")`). Unmapped
values are left unchanged.
"""
function discrete_recode_policy(recode; levels = nothing, mtp::Bool = true)
    rec = Dict{Any, Any}(recode)
    lev = levels === nothing ? unique(vcat(collect(keys(rec)), collect(values(rec)))) :
        collect(Any, levels)
    return DiscreteTreatmentPolicy(; levels = lev, recode = rec, mtp = mtp)
end

"""
    discrete_static_policy(value; levels=nothing) -> DiscreteTreatmentPolicy

Static assignment `do(A = value)` (R `lmtp` with `mtp=false`).
"""
function discrete_static_policy(value; levels = nothing)
    lev = levels === nothing ? Any[value] : collect(Any, levels)
    return DiscreteTreatmentPolicy(; levels = lev, static = value, mtp = false)
end

"""
    discrete_shift_policy(f; levels=nothing, mtp=true) -> DiscreteTreatmentPolicy

User function `f(a, wrow) -> a′` analogous to R `shift(data, trt)`.
"""
function discrete_shift_policy(f::Function; levels = nothing, mtp::Bool = true)
    lev = levels === nothing ? Any[] : collect(Any, levels)
    return DiscreteTreatmentPolicy(; levels = lev, shift = f, mtp = mtp)
end

"""
    DiscreteInterventionalMean(trt, outcome, adjustment, policy)

L2 mean under a [`DiscreteTreatmentPolicy`](@ref), contrasted with the natural
policy.
"""
struct DiscreteInterventionalMean <: Estimand
    trt::Symbol
    outcome::Symbol
    adjustment::Vector{Symbol}
    policy::DiscreteTreatmentPolicy
end

estimand_engine(::DiscreteInterventionalMean) = :discrete_lmtp

"""
    _canonical_treatment_levels(a, policy) -> Vector

Union of observed treatment values in `a`, policy-declared `levels`, any `static` level, and keys/values of
`recode`, with duplicates removed and order preserved (sorted when `sort` succeeds). Used for dummy coding and
2n density-ratio training so every policy arm appears in the schema.
"""
function _canonical_treatment_levels(a::AbstractVector, policy::DiscreteTreatmentPolicy)
    observed = collect(unique(a))
    extra = copy(policy.levels)
    if policy.static !== nothing
        push!(extra, policy.static)
    end
    append!(extra, collect(keys(policy.recode)))
    append!(extra, collect(values(policy.recode)))
    merged = Any[]
    seen = Set{Any}()
    for v in vcat(observed, extra)
        v in seen && continue
        push!(merged, v)
        push!(seen, v)
    end
    try
        return sort(merged)
    catch
        return merged
    end
end

"""
    _as_policy_level(value, levels)

Return `value` if it matches an element of `levels` by equality or by `string` equality (so `"1"` and `1`
align); otherwise return `value` unchanged. Used when building treatment frames for consistent factor levels.
"""
function _as_policy_level(value, levels::AbstractVector)
    value in levels && return value
    # Allow integer/string interchange ("1" vs 1) without silent Float64 coercion.
    for lev in levels
        string(lev) == string(value) && return lev
    end
    return value
end

"""Map `a` through `recode` by key equality or by string-matched keys; if no match, return `a`."""
function _recode_lookup(recode::AbstractDict, a)
    haskey(recode, a) && return recode[a]
    sa = string(a)
    for (k, v) in recode
        string(k) == sa && return v
    end
    return a
end

"""
    apply_discrete_policy(a, W, policy) -> Vector

Intervened exposure under `policy`. `W` is an `n × p` numeric matrix whose
rows are passed to a user `shift` function.
"""
function apply_discrete_policy(
    a::AbstractVector,
    W::AbstractMatrix,
    policy::DiscreteTreatmentPolicy,
)
    n = length(a)
    size(W, 1) == n || throw(DimensionMismatch(
        "covariate rows $(size(W, 1)) do not match treatment length $n",
    ))
    out = Vector{Any}(undef, n)
    if policy.shift !== nothing
        @inbounds for i in 1:n
            out[i] = policy.shift(a[i], view(W, i, :))
        end
        return out
    end
    if policy.static !== nothing
        fill!(out, policy.static)
        return out
    end
    @inbounds for i in 1:n
        out[i] = _recode_lookup(policy.recode, a[i])
    end
    return out
end

function apply_discrete_policy(a::AbstractVector, policy::DiscreteTreatmentPolicy)
    n = length(a)
    return apply_discrete_policy(a, zeros(n, 0), policy)
end

"""Row-wise support of a deterministic policy's treatment distribution."""
function _discrete_policy_support(
    a_observed::AbstractVector,
    W::AbstractMatrix{<:Real},
    policy::DiscreteTreatmentPolicy;
    natural_levels::Union{Nothing, AbstractVector} = nothing,
)
    n = length(a_observed)
    size(W, 1) == n || throw(DimensionMismatch(
        "covariate rows $(size(W, 1)) do not match treatment length $n",
    ))
    possible_natural_levels = natural_levels === nothing ?
        _canonical_treatment_levels(a_observed, policy) :
        unique(vcat(collect(natural_levels), collect(unique(a_observed))))
    support = falses(n)
    for i in 1:n
        wrow = view(W, i:i, :)
        for natural_level in possible_natural_levels
            assigned = only(apply_discrete_policy([natural_level], wrow, policy))
            if string(assigned) == string(a_observed[i])
                support[i] = true
                break
            end
        end
    end
    return support
end

"""
    discrete_positivity(a_obs, a_policy) -> NamedTuple

Levels assigned by the policy that never occur in the observed treatment.
"""
function discrete_positivity(a_obs::AbstractVector, a_policy::AbstractVector)
    observed = Set(string.(a_obs))
    assigned = unique(string.(a_policy))
    missing_support = [lev for lev in assigned if !(lev in observed)]
    return (
        n_observed_levels = length(observed),
        n_policy_levels = length(assigned),
        empty_support = missing_support,
        ok = isempty(missing_support),
    )
end

"""
    _factorise_treatment(a) -> Vector

Coerce treatment `a` to `String` codes: pass through `AbstractString`; for `Integer` or `Bool`, or at most 12
distinct values, stringify; otherwise raise (likely continuous exposure). Used before discrete LMTP and mediation.
"""
function _factorise_treatment(a::AbstractVector)
    values = collect(a)
    T = Base.nonmissingtype(eltype(values))
    T <: AbstractString && return values
    unique_values = unique(values)
    if T <: Integer || length(unique_values) <= 12
        return string.(values)
    end
    throw(ArgumentError(
        "treatment looks continuous ($(length(unique_values)) unique values); " *
        "use ShiftPolicy / run_lmtp_grid, or pass a String / categorical exposure",
    ))
end
