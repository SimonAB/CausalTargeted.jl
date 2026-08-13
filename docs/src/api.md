# API overview

```@meta
CurrentModule = CausalTargeted
```

Literature mapping lives in [Methods](methods.md); DOIs in [References](references.md).
Docstrings on the symbols below also carry `# References` sections.

## Small-*n* profile

- `recommend_folds` · `recommend_run_options` · `recommend_learners`
- `SMALL_N_SL_LEARNERS` · `adaptive_learners` · `warn_if_folds_too_large`

## LMTP and policies

- `run_lmtp_grid` · `lmtp_tmle_contrast` · `ShiftPolicy`
- `plot_mtp_curve` · `mtp_curve!` · `has_makie`
- `additive_shift_policy` · `multiplicative_shift_policy` · `threshold_shift_policy`
- `SequentialPolicy` · `run_sequential_lmtp` · `sequential_identification_certificate`
- `SurvivalPolicy` · `run_survival_lmtp` · `survival_identification_certificate`
- `build_lmtp_fold_cache`

## Mediation

- `run_mediation_grid` · `run_mediation_scalar`
- `mediation_n_mc_sweep` · `mediation_stability_summary` · `mediation_stability_markdown`
- `build_mediation_fold_cache` · `MediationFoldCache`
- Soft-deprecated aliases (emit `DeprecationWarning`): `run_crumble_grid`, `run_crumble_scalar`, `build_crumble_fold_cache`, `CrumbleFoldCache`, `run_crumble_scalar_ppl`
- Prefer: `run_mediation_grid` · `run_mediation_scalar` · `run_mediation_scalar_ppl` · `MediationFoldCache`

## Positivity and sensitivity

- `positivity_report` · `positivity_markdown` · `attach_positivity_summary!`
- `tipping_point_bias` · `partial_r2_calibration` · `sensitivity_report` · `sensitivity_markdown`
- `adjustment_set_disagreement` · `discovery_adjustment_sensitivity` · `merge_discovery_sensitivity!`

## Super Learner

- `fit_super_learner` · `predict_super_learner` · `SuperLearnerFit`
- Libraries: `DEFAULT_SL_LEARNERS`, `RICH_SL_LEARNERS` (includes `:randomforest`),
  `SMALL_N_SL_LEARNERS`; opt-in trees `:randomforest` / `:xgboost` after loading
  `MLJDecisionTreeInterface` / `MLJXGBoostInterface` (see [Methods](methods.md))
- Metalearners: `:discrete` (default, cross-fitted NNLS under squared error),
  `:invmse` (inverse training MSE), `:nnloglik` (`family=:binomial` only;
  logit-scale Bernoulli NLL, R `SuperLearner::method.NNloglik` parity)

## Planning and certificates

- `plan_mtp` · `summarise_plan` · `execute_estimand`
- `identification_certificate` · `certificate_dict`
- `build_run_metadata` · `attach_run_metadata!`

```@docs
CausalTargeted
recommend_run_options
run_lmtp_grid
run_mediation_grid
positivity_report
sensitivity_report
SequentialPolicy
```

## MTP effect curves

These functions visualise effects that have already been estimated; they do
not fit an MTP estimator or identify a causal effect. Interval columns can be
pointwise or simultaneous bands. Set `ylabel` to describe the intervals passed
to the function whenever the default `"Estimated effect (95% CI)"` is not
accurate.

Loading `CairoMakie` activates both the vector API and the DataFrame convenience
interface:

```@docs
has_makie
mtp_curve!
plot_mtp_curve
```

The DataFrame defaults match `run_lmtp_grid`, including the optional clamp
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
