"""
    has_makie() -> Bool

Return `true` when the optional `CausalTargetedMakieExt` plotting extension is
loaded. Loading a Makie backend such as CairoMakie loads Makie and activates
the extension.
"""
function has_makie()
    return !isnothing(Base.get_extension(@__MODULE__, :CausalTargetedMakieExt))
end

function _require_makie!(f::Symbol)
    ext = Base.get_extension(@__MODULE__, :CausalTargetedMakieExt)
    ext === nothing && error(
        "Makie.jl is required for MTP plotting. Load a Makie backend first, " *
        "for example: using CairoMakie\nThen call CausalTargeted.$f(...).",
    )
    return ext
end

"""
    mtp_curve!(ax, shift, estimate, lower, upper;
        clamp=nothing, estimand=nothing, panel_mode=false,
        ribbon_alpha=0.18, strip_gap_fraction=0.020,
        strip_height_fraction=0.055)

Draw already-estimated modified-treatment-policy (MTP) effects into an existing
Makie `Axis`. This function performs no causal estimation.

The four required vectors contain shift values, point estimates, and lower and
upper interval limits. `estimand` may contain `TE`, `NDE`, and `NIE` labels and
defaults to `TE` for every row. `clamp` contains optional proportions clamped by
the treatment policy; finite values are squished to `[0, 1]`. Missing or
non-finite clamp values are omitted. Required numeric values must be finite,
`lower <= upper`, and all vectors must have equal lengths.

Rows are copied and sorted by shift within the stable estimand order TE, NDE,
NIE. Clamp values repeated across estimands must agree at each shift. The
returned named tuple contains plot handles plus `series` and `clamp_geometry`
metadata, making the result composable and inspectable without mutating inputs.

# Keywords
- `panel_mode=false`: use compact line and marker sizes suitable for a manually
  composed faceted layout when `true`.
- `ribbon_alpha=0.18`: interval-ribbon opacity.
- `strip_gap_fraction=0.020`: gap below the effect range as a fraction of its
  y span.
- `strip_height_fraction=0.055`: clamp-strip height as a fraction of its y span.
"""
function mtp_curve!(
    ax,
    shift::AbstractVector,
    estimate::AbstractVector,
    lower::AbstractVector,
    upper::AbstractVector;
    kwargs...,
)
    ext = _require_makie!(:mtp_curve!)
    return ext.mtp_curve!(ax, shift, estimate, lower, upper; kwargs...)
end

"""
    plot_mtp_curve(shift, estimate, lower, upper;
        clamp=nothing, estimand=nothing, title=nothing,
        xlabel="Shift (SD units)", ylabel="Estimated effect (95% CI)",
        figure_size=(8.5, 4.5), base_size=12,
        ribbon_alpha=0.18, strip_gap_fraction=0.020,
        strip_height_fraction=0.055) -> Figure, Axis

Create one complete Makie MTP effect-curve figure and return `(figure, axis)`.
The function visualises supplied estimates and intervals; it does not estimate
causal effects. Intervals may be pointwise or simultaneous—the configurable
`ylabel` should describe whichever interval the caller supplied.

`figure_size` is measured in inches. The default physical canvas is 8.5 × 4.5
inches, represented internally at 120 Makie layout units per inch. `title` is
left aligned. The remaining data and strip keywords have the same meaning and
validation as [`mtp_curve!`](@ref). If no finite clamp values are supplied, the
clamp strip and colourbar are omitted. Only estimands present in the data appear
in the top legend, ordered TE, NDE, NIE.

# Export at exact physical size

For a returned `fig`, PDF uses `pt_per_unit = 72 / 120`; 320-dpi PNG uses
`px_per_unit = 320 / 120`:

```julia
save("mtp_curve.pdf", fig; pt_per_unit = 72 / 120)
save("mtp_curve.png", fig; px_per_unit = 320 / 120)
```
"""
function plot_mtp_curve(
    shift::AbstractVector,
    estimate::AbstractVector,
    lower::AbstractVector,
    upper::AbstractVector;
    kwargs...,
)
    ext = _require_makie!(:plot_mtp_curve)
    return ext.plot_mtp_curve(shift, estimate, lower, upper; kwargs...)
end

"""
    plot_mtp_curve(df;
        shift=:delta, estimate=:est, lower=:lwr, upper=:upr,
        clamp=:clamp, estimand=:estimand, kwargs...) -> Figure, Axis

DataFrame convenience interface for [`plot_mtp_curve`](@ref). Its defaults
match the output of [`run_lmtp_grid`](@ref), so `plot_mtp_curve(grid)` works
directly. Required column selectors are `shift`, `estimate`, `lower`, and
`upper`. The `estimand` column is optional and defaults to TE when absent or
when `estimand=nothing`; the optional `clamp` column is similarly ignored when
absent or disabled. A single error lists every missing required column. Column
vectors are never mutated.

Load a Makie backend such as CairoMakie to activate this method. Other keywords,
including `title`, `xlabel`, `ylabel`, and `figure_size`, are forwarded to the
vector API. To plot simultaneous bands returned by `run_lmtp_grid`, pass
`lower=:lwr_sim`, `upper=:upr_sim`, and an appropriate `ylabel`.
"""
function plot_mtp_curve(table; kwargs...)
    ext = _require_makie!(:plot_mtp_curve)
    return ext.plot_mtp_curve(table; kwargs...)
end
