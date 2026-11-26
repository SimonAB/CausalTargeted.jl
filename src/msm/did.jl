"""Difference-in-differences estimators: canonical 2×2 and staggered adoption.

# References

- Abadie (2005) — semiparametric DiD
- Sant'Anna & Zhao (2020) — doubly robust DiD (DR-DiD)
- Callaway & Sant'Anna (2021) — staggered DiD with group-time ATTs
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    run_did_2x2(df; unit, time, treatment, outcome) -> NamedTuple

Classic 2×2 difference-in-differences estimator for panel data with two periods.

Assumes:
- `time` column has exactly two values (sorted: pre, post)
- `treatment` is time-invariant group indicator (0/1)
- Parallel trends hold

Returns `(att, se, ci_lower, ci_upper, n_treated, n_control)`.
"""
function run_did_2x2(
    df::DataFrame;
    unit::Symbol = :unit,
    time::Symbol = :time,
    treatment::Symbol = :treat,
    outcome::Symbol = :Y,
)
    times = sort(unique(df[!, time]))
    length(times) == 2 || error("run_did_2x2 requires exactly 2 time periods, got $(length(times))")
    t_pre, t_post = times

    pre = df[df[!, time] .== t_pre, :]
    post = df[df[!, time] .== t_post, :]

    treated_units = unique(pre[pre[!, treatment] .== 1.0, unit])
    control_units = unique(pre[pre[!, treatment] .== 0.0, unit])

    y_pre_t = mean(pre[in.(pre[!, unit], Ref(Set(treated_units))), outcome])
    y_post_t = mean(post[in.(post[!, unit], Ref(Set(treated_units))), outcome])
    y_pre_c = mean(pre[in.(pre[!, unit], Ref(Set(control_units))), outcome])
    y_post_c = mean(post[in.(post[!, unit], Ref(Set(control_units))), outcome])

    att = (y_post_t - y_pre_t) - (y_post_c - y_pre_c)

    # Cluster-robust SE (unit-level)
    n_t = length(treated_units)
    n_c = length(control_units)
    ΔY_t = Float64[]
    for u in treated_units
        y_pre_u = only(pre[pre[!, unit] .== u, outcome])
        y_post_u = only(post[post[!, unit] .== u, outcome])
        push!(ΔY_t, y_post_u - y_pre_u)
    end
    ΔY_c = Float64[]
    for u in control_units
        y_pre_u = only(pre[pre[!, unit] .== u, outcome])
        y_post_u = only(post[post[!, unit] .== u, outcome])
        push!(ΔY_c, y_post_u - y_pre_u)
    end
    var_t = n_t > 1 ? var(ΔY_t) / n_t : 0.0
    var_c = n_c > 1 ? var(ΔY_c) / n_c : 0.0
    se = sqrt(var_t + var_c)
    ci_lower = att - 1.96 * se
    ci_upper = att + 1.96 * se

    return (att = att, se = se, ci_lower = ci_lower, ci_upper = ci_upper,
            n_treated = n_t, n_control = n_c)
end

"""
    run_did_staggered(df; unit, time, treatment, outcome, cohort) -> DataFrame

Staggered difference-in-differences using cohort-time group ATTs.

For each treatment cohort, estimates ATT(g, t) for post-treatment periods using
never-treated units as controls (Callaway–Sant'Anna style).

Returns a DataFrame with columns: `cohort`, `time`, `att`, `se`, `n_treated`, `n_control`.
"""
function run_did_staggered(
    df::DataFrame;
    unit::Symbol = :unit,
    time::Symbol = :time,
    treatment::Symbol = :treat,
    outcome::Symbol = :Y,
    cohort::Symbol = :cohort,
)
    cohorts = sort(unique(df[df[!, cohort] .!= "never", cohort]))
    times = sort(unique(df[!, time]))
    never_units = unique(df[df[!, cohort] .== "never", unit])
    unit_ids = unique(df[!, unit])
    unit_index = Dict(value => i for (i, value) in enumerate(unit_ids))
    cell_influence = Matrix{Float64}(undef, length(unit_ids), 0)

    results = Dict{String, Any}[]
    for g in cohorts
        g_units = unique(df[df[!, cohort] .== g, unit])
        # Find first treatment period for this cohort
        g_data = df[in.(df[!, unit], Ref(Set(g_units))), :]
        first_treat_t = minimum(g_data[g_data[!, treatment] .== 1.0, time])
        # Pre-period: last period before treatment
        pre_t = maximum(t for t in times if t < first_treat_t)

        for t in times
            t >= first_treat_t || continue
            # ATT(g, t) = (Ȳ_g,t - Ȳ_g,pre) - (Ȳ_never,t - Ȳ_never,pre)
            y_g_t = mean(df[(in.(df[!, unit], Ref(Set(g_units)))) .& (df[!, time] .== t), outcome])
            y_g_pre = mean(df[(in.(df[!, unit], Ref(Set(g_units)))) .& (df[!, time] .== pre_t), outcome])
            y_n_t = mean(df[(in.(df[!, unit], Ref(Set(never_units)))) .& (df[!, time] .== t), outcome])
            y_n_pre = mean(df[(in.(df[!, unit], Ref(Set(never_units)))) .& (df[!, time] .== pre_t), outcome])

            att_gt = (y_g_t - y_g_pre) - (y_n_t - y_n_pre)

            # Simple SE (unit-level ΔY variance)
            n_g = length(g_units)
            n_n = length(never_units)
            ΔY_g = [mean(df[(df[!, unit] .== u) .& (df[!, time] .== t), outcome]) -
                     mean(df[(df[!, unit] .== u) .& (df[!, time] .== pre_t), outcome])
                     for u in g_units]
            ΔY_n = [mean(df[(df[!, unit] .== u) .& (df[!, time] .== t), outcome]) -
                     mean(df[(df[!, unit] .== u) .& (df[!, time] .== pre_t), outcome])
                     for u in never_units]
            v_g = n_g > 1 ? var(ΔY_g) / n_g : 0.0
            v_n = n_n > 1 ? var(ΔY_n) / n_n : 0.0
            se = sqrt(v_g + v_n)

            # Unit-level contributions retain covariance from shared controls
            # and repeated units across cohort-time estimates.
            ψ = zeros(length(unit_ids))
            mean_g = mean(ΔY_g)
            mean_n = mean(ΔY_n)
            correction_g = n_g > 1 ? sqrt(n_g / (n_g - 1)) : 1.0
            correction_n = n_n > 1 ? sqrt(n_n / (n_n - 1)) : 1.0
            for (i, u) in enumerate(g_units)
                ψ[unit_index[u]] = correction_g * (ΔY_g[i] - mean_g) / n_g
            end
            for (i, u) in enumerate(never_units)
                ψ[unit_index[u]] = -correction_n * (ΔY_n[i] - mean_n) / n_n
            end
            cell_influence = hcat(cell_influence, ψ)

            push!(results, Dict{String, Any}(
                "cohort" => g, "time" => t, "att" => att_gt, "se" => se,
                "n_treated" => n_g, "n_control" => n_n,
            ))
        end
    end

    result = DataFrame(results)
    metadata!(result, "causal_targeted_did_unit_influence", (
        unit_ids = unit_ids,
        influence = cell_influence,
        cell_keys = collect(zip(result.cohort, result.time)),
    ); style = :note)
    return result
end

"""
    aggregate_did(res::DataFrame) -> NamedTuple

Simple equal-weight ATT from staggered DiD results. When produced by
[`run_did_staggered`](@ref), the standard error uses its full unit-level
influence covariance, including shared never-treated controls. Cell-only tables
report unavailable uncertainty. `independence_se` retains the historical
independence approximation as a diagnostic only.
"""
function aggregate_did(res::DataFrame)
    nrow(res) > 0 || throw(ArgumentError("cannot aggregate an empty DiD result"))
    att = mean(res.att)
    influence_meta = metadata(res, "causal_targeted_did_unit_influence", nothing)
    if influence_meta === nothing
        independence_se = sqrt(mean(res.se .^ 2) / nrow(res))
        return (
            att = att, se = NaN, independence_se = independence_se, n_cells = nrow(res),
            covariance = nothing, covariance_status = :unavailable,
        )
    end
    indices = Int[]
    for key in zip(res.cohort, res.time)
        index = findfirst(isequal(key), influence_meta.cell_keys)
        index === nothing && throw(ArgumentError("DiD cell is absent from its influence metadata"))
        push!(indices, index)
    end
    contributions = influence_meta.influence[:, indices]
    covariance = transpose(contributions) * contributions
    weights = fill(1 / nrow(res), nrow(res))
    se = sqrt(max(dot(weights, covariance * weights), 0.0))
    return (
        att = att, se = se, n_cells = nrow(res),
        covariance = covariance, covariance_status = :available,
    )
end

export run_did_2x2, run_did_staggered, aggregate_did
