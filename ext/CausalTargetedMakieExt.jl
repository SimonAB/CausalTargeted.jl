"""Makie extension for plotting already-estimated MTP effects."""
module CausalTargetedMakieExt

import CausalTargeted
using DataFrames: AbstractDataFrame
using Makie

const UNITS_PER_INCH = 120.0
const POINT_SCALE = UNITS_PER_INCH / 72.0

const ESTIMAND_ORDER = (:TE, :NDE, :NIE)
const ESTIMAND_COLORS = Dict(
    :TE => RGBf(0 / 255, 114 / 255, 178 / 255),
    :NDE => RGBf(230 / 255, 159 / 255, 0 / 255),
    :NIE => RGBf(0 / 255, 158 / 255, 115 / 255),
)
const ESTIMAND_MARKERS = Dict(:TE => :circle, :NDE => :utriangle, :NIE => :rect)
const LONG_DASH = Linestyle([0.0, 7.0, 10.0])
const ESTIMAND_LINESTYLES = Dict(:TE => :solid, :NDE => LONG_DASH, :NIE => :dashdot)
const CLAMP_COLORS = RGBf[
    RGBf(42 / 255, 157 / 255, 143 / 255),
    RGBf(168 / 255, 218 / 255, 220 / 255),
    RGBf(244 / 255, 241 / 255, 222 / 255),
    RGBf(244 / 255, 162 / 255, 97 / 255),
    RGBf(231 / 255, 111 / 255, 81 / 255),
]
const GRID_MAJOR = RGBf(0.92, 0.92, 0.92)
const GRID_MINOR = RGBf(0.96, 0.96, 0.96)
const ZERO_GREY = RGBf(0.40, 0.40, 0.40)

_pt(x::Real) = Float64(x) * POINT_SCALE
_mm(x::Real) = Float64(x) * (72.0 / 25.4) * POINT_SCALE

function _canvas_size(size_inches)
    length(size_inches) == 2 ||
        throw(ArgumentError("figure_size must contain width and height in inches"))
    width, height = Float64.(size_inches)
    all(isfinite, (width, height)) && width > 0 && height > 0 ||
        throw(ArgumentError("figure_size values must be finite and positive"))
    return (round(Int, width * UNITS_PER_INCH), round(Int, height * UNITS_PER_INCH))
end

function _float_vector(values::AbstractVector, name::AbstractString)
    isempty(values) && throw(ArgumentError("$name must not be empty"))
    output = Vector{Float64}(undef, length(values))
    for i in eachindex(values)
        value = values[i]
        value isa Real || throw(ArgumentError(
            "$name[$i] must be a finite real number; got $(repr(value))",
        ))
        output[i] = Float64(value)
        isfinite(output[i]) ||
            throw(ArgumentError("$name[$i] must be finite; got $(repr(value))"))
    end
    return output
end

function _canonical_estimand(value, index::Int)
    ismissing(value) &&
        throw(ArgumentError("estimand[$index] is missing; expected TE, NDE, or NIE"))
    label = Symbol(uppercase(strip(string(value))))
    label in ESTIMAND_ORDER || throw(ArgumentError(
        "unsupported estimand $(repr(value)) at index $index; expected TE, NDE, or NIE",
    ))
    return label
end

function _estimand_vector(estimand, n::Int)
    estimand === nothing && return fill(:TE, n)
    if estimand isa AbstractVector
        length(estimand) == n || throw(DimensionMismatch(
            "estimand has length $(length(estimand)); expected $n",
        ))
        return [_canonical_estimand(estimand[i], i) for i in eachindex(estimand)]
    end
    label = _canonical_estimand(estimand, 1)
    return fill(label, n)
end

function _clamp_vector(clamp_values, n::Int)
    clamp_values === nothing && return fill(nothing, n)
    if !(clamp_values isa AbstractVector)
        clamp_values = fill(clamp_values, n)
    end
    length(clamp_values) == n || throw(DimensionMismatch(
        "clamp has length $(length(clamp_values)); expected $n",
    ))
    output = Vector{Union{Nothing, Float64}}(undef, n)
    for i in eachindex(clamp_values)
        value = clamp_values[i]
        if ismissing(value) || value === nothing
            output[i] = nothing
        elseif value isa Real
            numeric = Float64(value)
            output[i] = isfinite(numeric) ? Base.clamp(numeric, 0.0, 1.0) : nothing
        else
            throw(ArgumentError(
                "clamp[$i] must be a real number, missing, or nothing; got $(repr(value))",
            ))
        end
    end
    return output
end

function _validate_curve_data(shift, estimate, lower, upper; clamp = nothing, estimand = nothing)
    n = length(shift)
    for (name, values) in (("estimate", estimate), ("lower", lower), ("upper", upper))
        length(values) == n || throw(DimensionMismatch(
            "$name has length $(length(values)); expected $n to match shift",
        ))
    end
    x = _float_vector(shift, "shift")
    y = _float_vector(estimate, "estimate")
    lo = _float_vector(lower, "lower")
    hi = _float_vector(upper, "upper")
    for i in eachindex(lo)
        lo[i] <= hi[i] || throw(ArgumentError(
            "lower[$i] ($(lo[i])) exceeds upper[$i] ($(hi[i]))",
        ))
    end
    labels = _estimand_vector(estimand, n)
    clamps = _clamp_vector(clamp, n)
    return (; shift = x, estimate = y, lower = lo, upper = hi,
            estimand = labels, clamp = clamps)
end

function _median(values::AbstractVector{<:Real})
    isempty(values) && throw(ArgumentError("cannot compute the median of an empty vector"))
    ordered = sort(Float64.(values))
    middle = (length(ordered) + 1) ÷ 2
    return isodd(length(ordered)) ?
           ordered[middle] : (ordered[middle] + ordered[middle + 1]) / 2
end

function _effect_geometry(data; strip_gap_fraction::Real, strip_height_fraction::Real)
    strip_gap_fraction >= 0 || throw(ArgumentError("strip_gap_fraction must be nonnegative"))
    strip_height_fraction >= 0 ||
        throw(ArgumentError("strip_height_fraction must be nonnegative"))
    y_min = minimum(vcat(data.estimate, data.lower, data.upper, [0.0]))
    y_max = maximum(vcat(data.estimate, data.lower, data.upper, [0.0]))
    y_span = y_max - y_min
    if !(isfinite(y_span) && y_span > 0)
        y_span = max(abs(y_min), abs(y_max), 1.0)
    end
    strip_ymax = y_min - Float64(strip_gap_fraction) * y_span
    strip_ymin = strip_ymax - Float64(strip_height_fraction) * y_span
    unique_x = sort(unique(data.shift))
    x_step = length(unique_x) > 1 ? _median(diff(unique_x)) : 0.1
    x_step > 0 || throw(ArgumentError("distinct shift values must have positive spacing"))
    return (; y_min, y_max, y_span, strip_ymin, strip_ymax, x_step)
end

function _clamp_geometry(data, geometry)
    grouped = Dict{Float64, Vector{Float64}}()
    for i in eachindex(data.shift)
        value = data.clamp[i]
        value === nothing && continue
        push!(get!(grouped, data.shift[i], Float64[]), value)
    end

    rectangles = NamedTuple[]
    for x in sort(collect(keys(grouped)))
        values = grouped[x]
        reference = first(values)
        if any(value -> !isapprox(value, reference; atol = 1e-12, rtol = 1e-8), values)
            throw(ArgumentError(
                "clamp values must be consistent across estimands for shift $x; " *
                "found $(sort(unique(values))) after squishing to [0, 1]",
            ))
        end
        push!(rectangles, (
            shift = x,
            value = reference,
            xmin = x - geometry.x_step / 2,
            xmax = x + geometry.x_step / 2,
            ymin = geometry.strip_ymin,
            ymax = geometry.strip_ymax,
        ))
    end
    return rectangles
end

function _clamp_color(value::Real)
    v = Base.clamp(Float64(value), 0.0, 1.0)
    v == 1.0 && return CLAMP_COLORS[end]
    segment = floor(Int, 4v) + 1
    fraction = (v - (segment - 1) / 4) * 4
    lo, hi = CLAMP_COLORS[segment], CLAMP_COLORS[segment + 1]
    return RGBf(
        (1 - fraction) * lo.r + fraction * hi.r,
        (1 - fraction) * lo.g + fraction * hi.g,
        (1 - fraction) * lo.b + fraction * hi.b,
    )
end

function _axis(parent; title, xlabel, ylabel, base_size::Real)
    ax = Axis(
        parent;
        backgroundcolor = :white,
        title = title === nothing ? "" : string(title),
        titlealign = :left,
        titlefont = :regular,
        titlesize = _pt(1.2 * base_size),
        xlabel = string(xlabel),
        ylabel = string(ylabel),
        xlabelsize = _pt(base_size),
        ylabelsize = _pt(base_size),
        xticklabelsize = _pt(0.8 * base_size),
        yticklabelsize = _pt(0.8 * base_size),
        xminorticks = IntervalsBetween(2),
        yminorticks = IntervalsBetween(2),
        xgridvisible = true,
        ygridvisible = true,
        xgridcolor = GRID_MAJOR,
        ygridcolor = GRID_MAJOR,
        xgridwidth = _mm(0.20),
        ygridwidth = _mm(0.20),
        xminorgridvisible = true,
        yminorgridvisible = true,
        xminorgridcolor = GRID_MINOR,
        yminorgridcolor = GRID_MINOR,
        xminorgridwidth = _mm(0.15),
        yminorgridwidth = _mm(0.15),
    )
    hidespines!(ax)
    return ax
end

function _present_estimands(labels)
    return [estimand for estimand in ESTIMAND_ORDER if estimand in labels]
end

function _estimand_legend(parent, estimands; base_size::Real)
    line_width = _mm(0.9)
    marker_size = _mm(1.5)
    elements = [[
        LineElement(
            color = ESTIMAND_COLORS[estimand],
            linestyle = ESTIMAND_LINESTYLES[estimand],
            linewidth = line_width,
        ),
        MarkerElement(
            color = ESTIMAND_COLORS[estimand],
            marker = ESTIMAND_MARKERS[estimand],
            markersize = marker_size,
            strokewidth = _mm(0.28),
            strokecolor = :white,
        ),
    ] for estimand in estimands]
    return Legend(
        parent,
        elements,
        string.(estimands);
        orientation = :horizontal,
        framevisible = false,
        labelsize = _pt(0.82 * base_size),
        patchsize = (_pt(22), _pt(10)),
        padding = (0, 0, 0, 0),
    )
end

function _clamp_key(parent; base_size::Real)
    key = GridLayout()
    parent[] = key
    Label(
        key[1, 1],
        "Proportion clamped";
        fontsize = _pt(0.82 * base_size),
        halign = :center,
        tellwidth = false,
    )
    colorbar = Colorbar(
        key[2, 1];
        colormap = CLAMP_COLORS,
        limits = (0.0, 1.0),
        ticks = ([0.0, 0.5, 1.0], ["0%", "50%", "100%"]),
        vertical = false,
        width = 5 / 2.54 * UNITS_PER_INCH,
        height = 0.30 / 2.54 * UNITS_PER_INCH,
        ticklabelsize = _pt(0.75 * base_size),
    )
    return (; layout = key, colorbar)
end

function mtp_curve!(
    ax,
    shift::AbstractVector,
    estimate::AbstractVector,
    lower::AbstractVector,
    upper::AbstractVector;
    clamp = nothing,
    estimand = nothing,
    panel_mode::Bool = false,
    ribbon_alpha::Real = 0.18,
    strip_gap_fraction::Real = 0.020,
    strip_height_fraction::Real = 0.055,
)
    0 <= ribbon_alpha <= 1 || throw(ArgumentError("ribbon_alpha must lie in [0, 1]"))
    data = _validate_curve_data(shift, estimate, lower, upper; clamp, estimand)
    geometry = _effect_geometry(data; strip_gap_fraction, strip_height_fraction)
    clamp_geometry = _clamp_geometry(data, geometry)

    clamp_plots = Any[]
    for rectangle in clamp_geometry
        points = Point2f[
            (rectangle.xmin, rectangle.ymin),
            (rectangle.xmax, rectangle.ymin),
            (rectangle.xmax, rectangle.ymax),
            (rectangle.xmin, rectangle.ymax),
        ]
        push!(clamp_plots, poly!(
            ax,
            points;
            color = _clamp_color(rectangle.value),
            strokewidth = 0,
        ))
    end

    series = NamedTuple[]
    for label in _present_estimands(data.estimand)
        indices = findall(==(label), data.estimand)
        sort!(indices; by = i -> data.shift[i])
        x = data.shift[indices]
        y = data.estimate[indices]
        lo = data.lower[indices]
        hi = data.upper[indices]
        color = ESTIMAND_COLORS[label]

        ribbon = band!(ax, x, lo, hi; color = (color, Float64(ribbon_alpha)))
        estimate_line = lines!(
            ax,
            x,
            y;
            color,
            linestyle = ESTIMAND_LINESTYLES[label],
            linewidth = _mm(panel_mode ? 0.7 : 0.9),
        )
        lower_edge = lines!(
            ax,
            x,
            lo;
            color = (color, 0.70),
            linestyle = LONG_DASH,
            linewidth = _mm(panel_mode ? 0.45 : 0.6),
        )
        upper_edge = lines!(
            ax,
            x,
            hi;
            color = (color, 0.70),
            linestyle = LONG_DASH,
            linewidth = _mm(panel_mode ? 0.45 : 0.6),
        )
        points = scatter!(
            ax,
            x,
            y;
            color,
            marker = ESTIMAND_MARKERS[label],
            markersize = _mm(panel_mode ? 0.8 : 2.0),
            strokewidth = _mm(panel_mode ? 0.15 : 0.28),
            strokecolor = :white,
        )
        push!(series, (
            estimand = label,
            shift = x,
            estimate = y,
            lower = lo,
            upper = hi,
            ribbon = ribbon,
            estimate_line = estimate_line,
            lower_edge = lower_edge,
            upper_edge = upper_edge,
            points = points,
        ))
    end

    zero_line = hlines!(
        ax,
        [0.0];
        color = ZERO_GREY,
        linestyle = :dash,
        linewidth = _mm(panel_mode ? 0.4 : 0.5),
    )

    x_min, x_max = extrema(data.shift)
    x_padding = 0.05 * (x_max - x_min + geometry.x_step)
    xlims!(
        ax,
        x_min - geometry.x_step / 2 - x_padding,
        x_max + geometry.x_step / 2 + x_padding,
    )
    plotted_ymin = isempty(clamp_geometry) ? geometry.y_min : geometry.strip_ymin
    ylims!(
        ax,
        plotted_ymin - 0.05 * geometry.y_span,
        geometry.y_max + 0.05 * geometry.y_span,
    )

    return (;
        series,
        zero_line,
        clamp_plots,
        clamp_geometry,
        geometry,
        estimands = _present_estimands(data.estimand),
    )
end

function plot_mtp_curve(
    shift::AbstractVector,
    estimate::AbstractVector,
    lower::AbstractVector,
    upper::AbstractVector;
    clamp = nothing,
    estimand = nothing,
    title = nothing,
    xlabel = "Shift (SD units)",
    ylabel = "Estimated effect (95% CI)",
    figure_size = (8.5, 4.5),
    base_size::Real = 12,
    ribbon_alpha::Real = 0.18,
    strip_gap_fraction::Real = 0.020,
    strip_height_fraction::Real = 0.055,
)
    data = _validate_curve_data(shift, estimate, lower, upper; clamp, estimand)
    has_clamp = any(value -> value !== nothing, data.clamp)
    estimands = _present_estimands(data.estimand)
    fig = Figure(
        size = _canvas_size(figure_size),
        backgroundcolor = :white,
        fonts = (; regular = "DejaVu Sans", bold = "DejaVu Sans Bold"),
    )

    row = 1
    if title !== nothing
        Label(
            fig[row, 1],
            string(title);
            fontsize = _pt(1.2 * base_size),
            halign = :left,
            tellwidth = false,
        )
        row += 1
    end

    header = GridLayout()
    fig[row, 1] = header
    if has_clamp
        _clamp_key(header[1, 1]; base_size)
        _estimand_legend(header[1, 2], estimands; base_size)
        colgap!(header, _pt(22))
    else
        _estimand_legend(header[1, 1], estimands; base_size)
    end
    row += 1

    ax = _axis(fig[row, 1]; title = nothing, xlabel, ylabel, base_size)
    mtp_curve!(
        ax,
        shift,
        estimate,
        lower,
        upper;
        clamp,
        estimand,
        panel_mode = false,
        ribbon_alpha,
        strip_gap_fraction,
        strip_height_fraction,
    )
    return fig, ax
end

function _column_name(selector, keyword::Symbol)
    selector isa Symbol && return selector
    selector isa AbstractString && return Symbol(selector)
    throw(ArgumentError(
        "$keyword must select one column by Symbol or string; got $(repr(selector))",
    ))
end

function _required_columns(df::AbstractDataFrame, selectors)
    available = propertynames(df)
    normalized = [(keyword, _column_name(selector, keyword)) for (keyword, selector) in selectors]
    missing_columns = [column for (_, column) in normalized if !(column in available)]
    isempty(missing_columns) || throw(ArgumentError(
        "missing required columns: " * join(string.(missing_columns), ", "),
    ))
    return Dict(normalized)
end

function _optional_column(df::AbstractDataFrame, selector, keyword::Symbol)
    selector === nothing && return nothing
    column = _column_name(selector, keyword)
    return column in propertynames(df) ? df[!, column] : nothing
end

function plot_mtp_curve(
    df::AbstractDataFrame;
    shift = :delta,
    estimate = :est,
    lower = :lwr,
    upper = :upr,
    clamp = :clamp,
    estimand = :estimand,
    kwargs...,
)
    columns = _required_columns(df, (
        :shift => shift,
        :estimate => estimate,
        :lower => lower,
        :upper => upper,
    ))
    clamp_values = _optional_column(df, clamp, :clamp)
    estimand_values = _optional_column(df, estimand, :estimand)
    return CausalTargeted.plot_mtp_curve(
        df[!, columns[:shift]],
        df[!, columns[:estimate]],
        df[!, columns[:lower]],
        df[!, columns[:upper]];
        clamp = clamp_values,
        estimand = estimand_values,
        kwargs...,
    )
end

end # module
