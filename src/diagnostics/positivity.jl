"""Positivity and clamp diagnostics for continuous MTP grids.

# References

- Petersen et al. (2012), *Stat Methods Med Res* — diagnosing positivity violations
- Díaz et al. (2023), *JASA* — MTPs designed to respect support / positivity
- Hernán & Robins (2020) — textbook treatment of overlap
"""

using DataFrames
using Statistics

"""
    positivity_report(data, trt; deltas, stratify_by, lower_q, upper_q, shift_scale, kwargs...) -> DataFrame

Tidy atlas of support / clamp diagnostics across the δ-grid and strata.

Uses `support_diagnostics` and `additive_clamp_diagnostics`.
Statuses: `ok`, `weak_support`, `unsupported_shift`.
"""
function positivity_report(
    data::DataFrame,
    trt::Symbol;
    deltas = default_deltas(),
    stratify_by = resolved_stratify_by(),
    lower_q = mtp_settings().lower_q,
    upper_q = mtp_settings().upper_q,
    shift_scale = mtp_settings().shift_scale,
    min_stratum_n::Int = mtp_settings().min_stratum_n,
    max_stratum_clamp_prop::Real = mtp_settings().max_stratum_clamp_prop,
    min_shift_retention::Real = mtp_settings().min_shift_retention,
)
    df = make_analysis_strata(data, stratify_by)
    pooled = stratify_by !== nothing
    a = Float64.(df[!, trt])
    L, U = exposure_bounds(a, lower_q, upper_q)
    rows = Dict{String, Any}[]
    for stratum in get_target_strata(df)
        stratum_mask = BitVector(string.(df.STRAT) .== stratum)
        for d in deltas
            diag = support_diagnostics(
                df, trt, stratum, stratify_by, lower_q, upper_q, d, shift_scale;
                min_stratum_n = min_stratum_n,
                max_stratum_clamp_prop = max_stratum_clamp_prop,
                min_shift_retention = min_shift_retention,
            )
            req = diag.requested_shift
            add = if isfinite(req)
                additive_clamp_diagnostics(pooled ? a[stratum_mask] : a, req, L, U)
            else
                (clamp = NaN, severity = NaN, effective_shift = NaN, shift_retention = NaN)
            end
            push!(rows, Dict(
                "trt" => string(trt),
                "stratum" => string(stratum),
                "delta" => Float64(d),
                "requested_shift" => req,
                "stratum_n" => diag.stratum_n,
                "stratum_clamp_prop" => diag.stratum_clamp_prop,
                "global_clamp_prop" => diag.global_clamp_prop,
                "effective_shift_mean" => diag.effective_shift_mean,
                "shift_retention" => diag.shift_retention,
                "additive_clamp" => add.clamp,
                "additive_severity" => add.severity,
                "support_status" => diag.support_status,
                "L" => diag.L,
                "U" => diag.U,
            ))
        end
    end
    return DataFrame(rows)
end

"""
    positivity_markdown(report::DataFrame; title) -> String

Compact markdown summary of a [`positivity_report`](@ref) table.
"""
function positivity_markdown(report::DataFrame; title::AbstractString = "Positivity / clamp atlas")
    isempty(report) && return "# $title\n\n(no rows)\n"
    n_ok = count(==("ok"), string.(report.support_status))
    n_weak = count(==("weak_support"), string.(report.support_status))
    n_bad = count(==("unsupported_shift"), string.(report.support_status))
    io = IOBuffer()
    println(io, "# $title\n")
    println(io, "| status | n |")
    println(io, "|--------|---|")
    println(io, "| ok | $n_ok |")
    println(io, "| weak_support | $n_weak |")
    println(io, "| unsupported_shift | $n_bad |")
    println(io)
    weak = report[string.(report.support_status) .!= "ok", :]
    if nrow(weak) > 0
        println(io, "## Non-ok cells (sample)\n")
        println(io, "| delta | stratum | status | clamp | retention |")
        println(io, "|-------|---------|--------|-------|-----------|")
        for r in eachrow(first(weak, min(20, nrow(weak))))
            println(io, "| ", r.delta, " | ", r.stratum, " | ", r.support_status, " | ",
                round(Float64(coalesce(r.stratum_clamp_prop, NaN)); digits = 3), " | ",
                round(Float64(coalesce(r.shift_retention, NaN)); digits = 3), " |")
        end
    end
    return String(take!(io))
end

"""
    attach_positivity_summary!(grid, report) -> DataFrame

Left-join key positivity columns onto an MTP grid by `delta` and `stratum`.
"""
function attach_positivity_summary!(grid::DataFrame, report::DataFrame)
    isempty(report) && return grid
    keep = [
        "support_status", "stratum_clamp_prop", "shift_retention",
        "requested_shift", "stratum_n",
    ]
    for c in keep
        c in names(report) || continue
        lookup = Dict{Tuple{Float64, String}, Any}()
        for r in eachrow(report)
            lookup[(Float64(r.delta), string(r.stratum))] = r[c]
        end
        grid[!, Symbol(c)] = [
            get(lookup, (Float64(d), string(s)), missing)
            for (d, s) in zip(grid.delta, grid.stratum)
        ]
    end
    return grid
end

export positivity_report, positivity_markdown, attach_positivity_summary!
