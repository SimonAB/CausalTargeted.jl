# Planning, certificates & plotting

```@meta
CurrentModule = CausalTargeted
```

Temporal panel estimation plans, MTP execution planning, policy selection,
identification certificates, run metadata, MTP effect curves, and synthetic
data-generating processes (DGPs).

## Temporal panel plans & sequential bridge

Use the qualified planner and plan type below when working with temporal
identification certificates from `CausalDynamics.jl`. The planner and type remain
unexported during the compatibility period; `run_estimation_plan` also accepts
plans returned by the legacy `CausalDynamics` planner and its session wrapper.
See [Methods and estimators](../methods.md) for a panel example.

```@docs
CausalTargeted.EstimationPlan
CausalTargeted.plan_targeted_estimation
run_estimation_plan
plan_sequential
sequential_spec_from_identification
```

## MTP planning, execution & settings

```@docs
MTPSettings
mtp_settings
default_deltas
resolved_stratify_by
MTPPlan
plan_mtp
summarise_plan
execute_estimand
```

## Policy choice

```@docs
PolicyChoice
choose_policy
```

## Identification certificates & run metadata

`identification_certificate(...; provenance=CDMProvenance(...))` carries
declared graph, mechanism, observation, policy, exogenous, numerical, scale,
and coupling identifiers into `certificate_dict`. Function implementations are
not fingerprinted: supply explicit stable identifiers in `CDMProvenance`.

The returned metadata includes `certificate_schema_version = "cdcs.certificate.v2"`.
Semantic identifiers remain distinct from run-time fields such as package
version, Julia version, and execution time.

```@docs
IdentificationCertificate
identification_certificate
certificate_dict
RunMetadata
build_run_metadata
attach_run_metadata!
metadata_dict
```

## MTP effect curves

These functions visualise effects that have already been estimated; they do not
fit an MTP estimator or identify a causal effect. Interval columns can be
pointwise or simultaneous bands when the estimator reports available limits;
unavailable `NaN` limits produce a point-only curve. Set `ylabel` to describe the
intervals passed to the function whenever the default
`"Estimated effect (95% CI)"` is not accurate.

Loading `CairoMakie` activates both the vector API and the `DataFrame`
convenience interface:

```@docs
has_makie
mtp_curve!
plot_mtp_curve
```

The `DataFrame` defaults match `run_lmtp_grid`, including the optional clamp
strip:

```julia
using CausalTargeted, CairoMakie

grid = run_lmtp_grid(data, :A, :Y; baseline = [:W])
fig, ax = plot_mtp_curve(grid; title = "Exposure → outcome")
```

For simultaneous intervals produced by the grid estimator, select those
columns and label them explicitly:

```julia
fig, ax = plot_mtp_curve(
    grid;
    lower = :lwr_sim,
    upper = :upr_sim,
    ylabel = "Estimated effect (simultaneous 95% band)",
)
```

### Synthetic TE/NDE/NIE example

```@example api-mtp
using CausalTargeted, CairoMakie, DataFrames

shifts = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
estimands = ["TE", "NDE", "NIE"]
df = DataFrame(
    delta = repeat(shifts, length(estimands)),
    estimand = repeat(estimands; inner = length(shifts)),
)
scale = Dict("TE" => 0.12, "NDE" => 0.08, "NIE" => 0.04)
df.est = [scale[e] * sinpi(x) for (x, e) in zip(df.delta, df.estimand)]
df.lwr = df.est .- 0.035
df.upr = df.est .+ 0.035
df.clamp = repeat([0.42, 0.28, 0.14, 0.03, 0.16, 0.30, 0.45], length(estimands))

fig, ax = plot_mtp_curve(df; title = "Synthetic exposure → outcome")
fig
```

For custom multi-panel layouts, create ordinary Makie axes and compose the
curve-level primitive directly:

```julia
panel = Figure()
ax = Axis(panel[1, 1]; title = "Panel A")
handles = mtp_curve!(ax, delta, estimate, lower, upper; clamp, estimand)
```

The default canvas uses 120 Makie layout units per inch. Save the exact
8.5 × 4.5 inch curve as PDF or 320-dpi PNG with:

```julia
save("mtp.pdf", fig; pt_per_unit = 72 / 120)
save("mtp.png", fig; px_per_unit = 320 / 120)
```

## Synthetic DGPs

```@docs
simulate_linear_mtp
simulate_discrete_survival_mtp
simulate_mixed_baseline_mtp
simulate_binomial_mtp
simulate_multinomial_outcome
simulate_categorical_treatment_mtp
simulate_sequential_factor_mtp
simulate_repeated_outcome_ate
```
