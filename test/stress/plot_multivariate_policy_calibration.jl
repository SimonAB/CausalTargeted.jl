# Render a completed study with the package's existing plotting environment:
# julia --project=docs --threads=auto test/stress/plot_multivariate_policy_calibration.jl DIRECTORY
using CairoMakie, CSV
CairoMakie.activate!(type = "svg")

"""Pointwise Wilson score interval for a binomial Monte Carlo proportion."""
function wilson_interval(successes, attempted; z = 1.959963984540054)
    0 <= successes <= attempted && attempted > 0 || throw(ArgumentError("invalid binomial counts"))
    p = successes / attempted
    denominator = 1 + z^2 / attempted
    centre = (p + z^2 / (2attempted)) / denominator
    radius = z * sqrt(p * (1 - p) / attempted + z^2 / (4attempted^2)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)
end

function plot_policy_calibration(directory)
    rows = collect(CSV.File(joinpath(directory, "summary.tsv"); delim = '\t'))
    records = Dict((Symbol(r.experiment), Int(r.n), Symbol(r.scenario), Symbol(r.method)) => r for r in rows)
    length(records) == length(rows) || throw(ArgumentError("duplicate summary entries"))
    scenarios = (:balanced, :rare_control, :strong_prior, :separated)
    labels = ["Balanced", "One control", "Strong prior", "Separation-prone"]
    methods = (:adaptive_reference, :variational, :linear_response)
    colors = (:black, :gray55, :steelblue4)
    markers = (:circle, :rect, :diamond)
    offsets = (-0.19, 0.0, 0.19)
    fig = Figure(size = (1080, 760), fontsize = 14, backgroundcolor = :white)
    Label(fig[0, 1:2], "Inclusion of central 95% policy credible intervals"; fontsize = 19)
    for (ri, experiment) in enumerate((:prior_predictive, :fixed_parameter)), (ci, n) in enumerate((20, 50))
        title = (experiment == :prior_predictive ? "Prior predictive" : "Fixed parameters") * " · n = $n"
        ax = Axis(fig[ri, ci]; title, xlabel = ri == 2 ? "Repeated-sampling inclusion" : "",
            yticks = (1:4, labels), yreversed = true, xticks = 0:0.25:1,
            xgridvisible = false, ygridvisible = false, topspinevisible = false,
            rightspinevisible = false, yticklabelsvisible = ci == 1)
        xlims!(ax, -0.025, 1.025)
        ylims!(ax, 0.5, 4.5)
        ri == 1 && hidexdecorations!(ax; grid = false)
        vlines!(ax, [0.95]; color = :gray70, linestyle = :dash, linewidth = 1)
        for (si, scenario) in enumerate(scenarios), (mi, method) in enumerate(methods)
            key = (experiment, n, scenario, method)
            haskey(records, key) || continue
            record = records[key]
            available = Int(record.available)
            available == 0 && continue
            p = Float64(record.coverage_available)
            hits = round(Int, p * available)
            isapprox(hits / available, p; atol = 1e-12) || error("summary proportion is not a count ratio")
            lower, upper = wilson_interval(hits, available)
            y = si + offsets[mi]
            lines!(ax, [lower, upper], [y, y]; color = colors[mi], linewidth = 1.2)
            scatter!(ax, [p], [y]; color = colors[mi], marker = markers[mi], markersize = 8)
        end
        if experiment == :prior_predictive
            text!(ax, 0.5, 4; text = "not specified", color = :gray60,
                align = (:center, :center), fontsize = 12)
        end
    end
    colgap!(fig.layout, 35)
    elements = [MarkerElement(; marker = markers[i], color = colors[i], markersize = 9) for i in 1:3]
    Legend(fig[3, 1:2], elements, ["Adaptive posterior reference", "Raw VI", "LRVB"];
        orientation = :horizontal, framevisible = false)
    Label(fig[4, 1:2],
        "Bars: pointwise 95% Wilson Monte Carlo intervals using available evaluations; see summary.tsv for failure denominators.\n" *
        "Dashed line: 0.95. Fixed-parameter Bayesian coverage need not be nominal. Separation-prone cells are fixed-parameter only.";
        fontsize = 11, halign = :left, justification = :left, tellwidth = false)
    save(joinpath(directory, "coverage.svg"), fig)
    save(joinpath(directory, "coverage.png"), fig; px_per_unit = 1.5)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("provide the completed study directory")
    plot_policy_calibration(only(ARGS))
end
