"""Sensitivity helpers for continuous MTP estimates at small *n*.

Provides tipping-point bias factors and a partial-*R*² style omitted-confounder
calibration (Cinelli–Hazlett-inspired scalars). These are diagnostic, not a
substitute for design or identification.

# References

- Cinelli & Hazlett (2020), *JRSS-B* — partial *R*² / robustness-value OVB framework
- VanderWeele & Ding (2017) — E-value (complementary reporting tool)
- Rosenbaum (2002) — classical observational sensitivity analysis
"""

using DataFrames
using Statistics
using Distributions

"""
    tipping_point_bias(est, se; alpha=0.05) -> NamedTuple

Smallest additive bias that would move a Wald CI to include zero (sign flip /
nullification). Returns `bias_needed`, `est`, `se`, `z_crit`, `ci_excludes_zero`.
"""
function tipping_point_bias(est::Real, se::Real; alpha::Real = 0.05)
    e = Float64(est)
    s = Float64(se)
    (!isfinite(e) || !isfinite(s) || s <= 0) && return (
        bias_needed = NaN, est = e, se = s, z_crit = NaN, ci_excludes_zero = false,
    )
    z = quantile(Normal(), 1 - alpha / 2)
    lwr, upr = e - z * s, e + z * s
    excludes = (lwr > 0 && upr > 0) || (lwr < 0 && upr < 0)
    # Distance from estimate to nearest CI edge that hits 0, or to 0 if already includes 0
    if !excludes
        bias_needed = 0.0
    else
        # Bias that shifts the nearer CI bound onto 0
        bias_needed = e > 0 ? lwr : -upr
        bias_needed = abs(bias_needed)
    end
    return (
        bias_needed = bias_needed,
        est = e,
        se = s,
        z_crit = z,
        ci_excludes_zero = excludes,
    )
end

"""
    partial_r2_calibration(est, se; r2_du=0.01, r2yu=0.01, n, df_adjust=1) -> NamedTuple

Approximate relative bias from an omitted confounder with partial *R*² with
treatment (`r2_du`) and outcome (`r2yu`), following the spirit of Cinelli &
Hazlett (2020). Returns `adjusted_est`, `bias`, `rel_bias`.

Assumptions are strong (linear / residualised); treat as a calibration dial.
"""
function partial_r2_calibration(
    est::Real,
    se::Real;
    r2_du::Real = 0.01,
    r2_yu::Real = 0.01,
    n::Integer = 100,
    df_adjust::Integer = 1,
)
    e = Float64(est)
    s = Float64(se)
    (!isfinite(e) || !isfinite(s) || s <= 0) && return (
        adjusted_est = NaN, bias = NaN, rel_bias = NaN, r2_du = Float64(r2_du), r2_yu = Float64(r2_yu),
    )
    # BF = sqrt(r2_yu * r2_du / ((1-r2_yu)*(1-r2_du))) * se * sqrt(df)
    df = max(Int(n) - Int(df_adjust) - 1, 1)
    r_d = clamp(Float64(r2_du), 0.0, 0.999)
    r_y = clamp(Float64(r2_yu), 0.0, 0.999)
    bf = sqrt((r_y * r_d) / ((1 - r_y) * (1 - r_d) + 1e-15))
    bias = sign(e) * bf * s * sqrt(df)
    adj = e - bias
    return (
        adjusted_est = adj,
        bias = bias,
        rel_bias = abs(e) > 1e-12 ? abs(bias / e) : NaN,
        r2_du = r_d,
        r2_yu = r_y,
    )
end

"""
    sensitivity_report(est, se; n, alpha, r2_grid) -> DataFrame

One-row tipping point plus a small grid of partial-*R*² calibrations.
"""
function sensitivity_report(
    est::Real,
    se::Real;
    n::Integer = 100,
    alpha::Real = 0.05,
    r2_grid = [0.01, 0.05, 0.10],
)
    tip = tipping_point_bias(est, se; alpha = alpha)
    rows = Dict{String, Any}[]
    push!(rows, Dict(
        "kind" => "tipping_point",
        "est" => tip.est,
        "se" => tip.se,
        "bias_needed" => tip.bias_needed,
        "ci_excludes_zero" => tip.ci_excludes_zero,
        "adjusted_est" => tip.est,
        "r2_du" => missing,
        "r2_yu" => missing,
        "rel_bias" => missing,
    ))
    for r2 in r2_grid
        cal = partial_r2_calibration(est, se; r2_du = r2, r2_yu = r2, n = n)
        push!(rows, Dict(
            "kind" => "partial_r2",
            "est" => Float64(est),
            "se" => Float64(se),
            "bias_needed" => missing,
            "ci_excludes_zero" => tip.ci_excludes_zero,
            "adjusted_est" => cal.adjusted_est,
            "r2_du" => cal.r2_du,
            "r2_yu" => cal.r2_yu,
            "rel_bias" => cal.rel_bias,
        ))
    end
    return DataFrame(rows)
end

"""
    sensitivity_markdown(report::DataFrame; title) -> String
"""
function sensitivity_markdown(
    report::DataFrame;
    title::AbstractString = "Sensitivity (tipping point / partial R²)",
)
    io = IOBuffer()
    println(io, "# $title\n")
    tip = report[string.(report.kind) .== "tipping_point", :]
    if nrow(tip) > 0
        r = tip[1, :]
        println(io, "- Estimate: ", round(Float64(r.est); digits = 4))
        println(io, "- SE: ", round(Float64(r.se); digits = 4))
        println(io, "- CI excludes zero: ", r.ci_excludes_zero)
        println(io, "- Bias needed to nullify: ", round(Float64(coalesce(r.bias_needed, NaN)); digits = 4))
        println(io)
    end
    cal = report[string.(report.kind) .== "partial_r2", :]
    if nrow(cal) > 0
        println(io, "| R² (du=yu) | adjusted est | |bias|/|est| |")
        println(io, "|------------|--------------|-------------|")
        for r in eachrow(cal)
            println(io, "| ", r.r2_du, " | ",
                round(Float64(r.adjusted_est); digits = 4), " | ",
                round(Float64(coalesce(r.rel_bias, NaN)); digits = 3), " |")
        end
    end
    return String(take!(io))
end

export tipping_point_bias, partial_r2_calibration, sensitivity_report, sensitivity_markdown
