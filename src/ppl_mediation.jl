"""PPL / bootstrap scalar mediation (graph-agnostic)."""

using DataFrames
using Statistics
using Distributions
using GLM
using Random
using CausalDynamics

"""
    prepare_ppl_mediation_spec(g, treatment, outcome, data, covar, mediators; node_names) -> NamedTuple

Build a PPL-ready specification using [`prepare_for_rxinfer`](@ref) when the graph
contains the treatment and outcome nodes.
"""
function prepare_ppl_mediation_spec(
    g,
    treatment::Symbol,
    outcome::Symbol,
    data::DataFrame,
    covar::Vector{Symbol},
    mediators::Vector{Symbol};
    node_names::Union{Nothing, Dict{Int, Symbol}} = nothing,
)
    if g !== nothing && node_names !== nothing
        idx = Dict(s => i for (i, s) in node_names)
        (treatment in keys(idx) && outcome in keys(idx)) || return _generic_ppl_spec(
            treatment, outcome, data, covar, mediators,
        )
        return prepare_for_rxinfer(g, idx[treatment], idx[outcome]; node_names = node_names, data = data)
    end
    return _generic_ppl_spec(treatment, outcome, data, covar, mediators)
end

"""
    prepare_ppl_mediation_spec(treatment, outcome, data, covar, mediators; kwargs...) -> NamedTuple

Graph-agnostic convenience wrapper (no DAG).
"""
function prepare_ppl_mediation_spec(
    treatment::Symbol,
    outcome::Symbol,
    data::DataFrame,
    covar::Vector{Symbol},
    mediators::Vector{Symbol};
    kwargs...,
)
    return prepare_ppl_mediation_spec(
        nothing, treatment, outcome, data, covar, mediators; kwargs...,
    )
end

function _generic_ppl_spec(treatment, outcome, data, covar, mediators)
    return (
        graph = nothing,
        treatment = treatment,
        outcome = outcome,
        confounders = covar,
        mediators = mediators,
        data = data,
        is_identifiable = true,
    )
end

"""
    conjugate_mediation_bootstrap(df, trt, outcome, covar, mediators; n_boot, rng) -> DataFrame
"""
function conjugate_mediation_bootstrap(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covar::Vector{Symbol},
    mediators::Vector{Symbol};
    n_boot::Int = 500,
    rng::AbstractRNG = StableRNG(42),
)
    cols = unique(vcat([trt, outcome], covar, mediators))
    sub = dropmissing(df[:, cols])
    n = nrow(sub)
    nde_s = Float64[]
    nie_s = Float64[]

    for _ in 1:n_boot
        idx = rand(rng, 1:n, n)
        boot = sub[idx, :]
        a_b = Float64.(boot[!, trt])
        y_b = Float64.(boot[!, outcome])
        X_b = hcat(ones(n), [Float64.(boot[!, c]) for c in covar]...)

        Xm = hcat(a_b, [Float64.(boot[!, m]) for m in mediators]..., X_b)
        β = GLM.coef(lm(Xm, y_b))
        p_w = size(X_b, 2)
        β_a = β[1]
        β_m = β[2:(1 + length(mediators))]
        β_w = β[(2 + length(mediators)):end]

        μ_m_a0 = zeros(length(mediators), n)
        μ_m_a1 = zeros(length(mediators), n)
        for (j, m) in enumerate(mediators)
            Xm_j = hcat(a_b, X_b)
            βj = GLM.coef(lm(Xm_j, Float64.(boot[!, m])))
            βj_w = βj[2:end]
            βj_a = βj[1]
            μ_m_a0[j, :] .= X_b * βj_w
            μ_m_a1[j, :] .= X_b * βj_w .+ βj_a
        end

        m0 = vec(mean(μ_m_a0; dims = 2))
        m1 = vec(mean(μ_m_a1; dims = 2))
        y_a0m0 = mean(β_a * 0.0 .+ sum(β_m[k] * m0[k] for k in eachindex(mediators)) .+ X_b * β_w)
        y_a1m0 = mean(β_a * 1.0 .+ sum(β_m[k] * m0[k] for k in eachindex(mediators)) .+ X_b * β_w)
        y_a1m1 = mean(β_a * 1.0 .+ sum(β_m[k] * m1[k] for k in eachindex(mediators)) .+ X_b * β_w)

        push!(nde_s, y_a1m0 - y_a0m0)
        push!(nie_s, y_a1m1 - y_a1m0)
    end

    rows = Dict{String, Any}[]
    for (lab, samples) in (("NDE", nde_s), ("NIE", nie_s), ("TE", nde_s .+ nie_s))
        est = mean(samples)
        se = std(samples)
        lwr, upr = quantile(samples, (0.025, 0.975))
        push!(rows, Dict(
            "effect" => lab, "estimate" => est, "se" => se,
            "lower" => lwr, "upper" => upr, "method" => "conjugate_bootstrap",
        ))
    end
    return DataFrame(rows)
end

"""
    run_crumble_scalar_ppl(data, trt, outcome; mediators, covar, method, kwargs...) -> DataFrame
"""
function run_crumble_scalar_ppl(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    mediators::Vector{Symbol},
    covar::Vector{Symbol},
    method::Symbol = :eif,
    kwargs...,
)
    method == :conjugate_bootstrap &&
        return conjugate_mediation_bootstrap(data, trt, outcome, covar, mediators; kwargs...)
    return run_mediation_scalar(data, trt, outcome; mediators = mediators, covar = covar, kwargs...)
end

const run_mediation_scalar_ppl = run_crumble_scalar_ppl  # preferred name once PPL path matures

export prepare_ppl_mediation_spec, conjugate_mediation_bootstrap, run_crumble_scalar_ppl
