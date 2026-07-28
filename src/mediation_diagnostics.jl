"""Mediation grid diagnostics: MC stability across nested draw counts.

Continuous-exposure mediation estimators that integrate over mediator draws
inherit Monte Carlo error; at small *n* this can dominate sampling variability.

# References

- Liu et al. (2024), arXiv:2408.14620 — modern mediation + MTP estimation
- Díaz & Hejazi (2020), *JRSS-B* — stochastic intervention mediation
- Hejazi et al. (2023), *Biostatistics* — stochastic interventional (in)direct effects
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    mediation_n_mc_sweep(data, trt, outcome; covar, mediators, n_mc_values, delta, kwargs...) -> DataFrame

Re-fit mediation NDE/NIE/TE at a single `delta` for each `n_mc` in `n_mc_values`.
Useful for checking whether nested-MC noise dominates inference.
"""
function mediation_n_mc_sweep(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    covar::Vector{Symbol},
    mediators::Vector{Symbol},
    n_mc_values::AbstractVector{<:Integer} = [32, 64, 128],
    delta::Float64 = 1.0,
    folds = mtp_settings().folds,
    learners = DEFAULT_SL_LEARNERS,
    rng::AbstractRNG = StableRNG(42),
    kwargs...,
)
    rows = Dict{String, Any}[]
    for n_mc in n_mc_values
        grid = run_mediation_grid(
            data, trt, outcome;
            covar = covar,
            mediators = mediators,
            deltas = [delta],
            folds = folds,
            learners = learners,
            n_mc = n_mc,
            parallel = false,
            cache_nuisances = false,
            rng = rng,
            kwargs...,
        )
        for estimand in ("TE", "NDE", "NIE")
            sub = grid[(string.(grid.estimand) .== estimand), :]
            isempty(sub) && continue
            r = sub[1, :]
            push!(rows, Dict(
                "n_mc" => n_mc,
                "delta" => delta,
                "estimand" => estimand,
                "est" => r.est,
                "se" => r.se,
                "lwr" => r.lwr,
                "upr" => r.upr,
            ))
        end
    end
    return DataFrame(rows)
end

"""
    mediation_stability_summary(sweep::DataFrame; central_estimands=("TE", "NDE", "NIE")) -> NamedTuple

Summarise a [`mediation_n_mc_sweep`](@ref) table: SE vs `n_mc`, and whether
signs of central estimates are stable across Monte Carlo depths.
"""
function mediation_stability_summary(
    sweep::DataFrame;
    central_estimands = ("TE", "NDE", "NIE"),
)
    isempty(sweep) && return (
        n_mc_values = Int[],
        se_by_n_mc = Dict{Int, Float64}(),
        sign_stable = Dict{String, Bool}(),
        est_by_n_mc = Dict{String, Vector{Float64}}(),
    )
    n_mcs = sort(unique(Int.(sweep.n_mc)))
    se_by = Dict{Int, Float64}()
    for n_mc in n_mcs
        sub = sweep[Int.(sweep.n_mc) .== n_mc, :]
        te = sub[string.(sub.estimand) .== "TE", :]
        se_by[n_mc] = isempty(te) ? NaN : Float64(te.se[1])
    end
    sign_stable = Dict{String, Bool}()
    est_by = Dict{String, Vector{Float64}}()
    for lab in central_estimands
        ests = Float64[]
        for n_mc in n_mcs
            sub = sweep[(Int.(sweep.n_mc) .== n_mc) .& (string.(sweep.estimand) .== lab), :]
            push!(ests, isempty(sub) ? NaN : Float64(sub.est[1]))
        end
        est_by[lab] = ests
        finite = filter(isfinite, ests)
        if length(finite) < 2
            sign_stable[lab] = true
        else
            signs = sign.(finite)
            # Treat near-zero as null (stable with either sign)
            signs = [abs(e) < 0.02 ? 0.0 : s for (e, s) in zip(finite, signs)]
            nz = filter(!=(0.0), signs)
            sign_stable[lab] = isempty(nz) || all(==(first(nz)), nz)
        end
    end
    return (
        n_mc_values = n_mcs,
        se_by_n_mc = se_by,
        sign_stable = sign_stable,
        est_by_n_mc = est_by,
    )
end

"""
    mediation_stability_markdown(sweep; title) -> String
"""
function mediation_stability_markdown(
    sweep::DataFrame;
    title::AbstractString = "Mediation nested-MC stability",
)
    s = mediation_stability_summary(sweep)
    io = IOBuffer()
    println(io, "# $title\n")
    println(io, "| n_mc | TE SE |")
    println(io, "|------|-------|")
    for n_mc in s.n_mc_values
        println(io, "| ", n_mc, " | ", round(get(s.se_by_n_mc, n_mc, NaN); digits = 4), " |")
    end
    println(io)
    println(io, "| estimand | sign-stable across n_mc |")
    println(io, "|----------|------------------------|")
    for (lab, ok) in s.sign_stable
        println(io, "| ", lab, " | ", ok, " |")
    end
    return String(take!(io))
end

export mediation_n_mc_sweep, mediation_stability_summary, mediation_stability_markdown
