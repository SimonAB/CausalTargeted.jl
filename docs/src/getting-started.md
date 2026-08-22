# Getting started

```@meta
CurrentModule = CausalTargeted
```

CausalTargeted estimates LMTP and related interventional targets once
[CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) has supplied a
graph and adjustment set. The walk-throughs below run end to end in Documenter
(synthetic data, small folds) so you can copy the pattern onto cohort tables.

Install from General, or develop a local clone:

```julia
using Pkg
Pkg.add("CausalTargeted")  # resolves CausalDynamics from General
# Pkg.develop(path="<path-to-local-clone>")  # e.g. after git clone
```

Load plotting backends when you want figures:

```julia
using CausalTargeted, CausalDynamics, Graphs, DAGMakie, CairoMakie
```

Mediation examples additionally need `using CausalMediation`.

Each walk-through below plots DAGs in the style of Cinelli, Forney & Pearl
(2022, [SMR](https://doi.org/10.1177/00491241221099552)): a **good-control
triangle** (`dagplot_confounding`) for identification, then the **estimand path**
(`dagplot_chain` or `dagplot_mediation`) with [`structural_edge_labels`](https://simonab.github.io/DAGMakie.jl/dev/) on edges where needed.

## 1. Identify, then estimate a δ-grid (LMTP)

**Graph.** Baseline confounding of a continuous exposure (`W → A → Y`, `W → Y`).

```@example lmtp-walk
using CausalTargeted, CausalDynamics, Graphs, StableRNGs

g = DiGraph(3)
add_edge!(g, 1, 2)  # W → A
add_edge!(g, 1, 3)  # W → Y
add_edge!(g, 2, 3)  # A → Y
names = Dict(1 => :W, 2 => :A, 3 => :Y)

id = identify(g, TotalEffectQuery(:A, :Y); node_names = names)
id.identifiable, id.adjustment, id.strategy
```

**Graph (identification).** Good-control triangle: `W` is adjusted; paths carry no estimates.

```@example lmtp-walk
using DAGMakie, CairoMakie

fig, _, _ = dagplot_confounding(["W", "A", "Y"];
    color_by = :adjustment,
    exposure = 2,
    outcome = 3,
    adjustment = Set([1]),  # good control W (Cinelli et al. 2022)
    title = "Good control W (identification)",
)
fig
```

**Data and defaults.** `recommend_run_options` picks folds and learners for the
sample size; pass the kwargs into `run_lmtp_grid`.

```@example lmtp-walk
df, truth = simulate_linear_mtp(200; rng = StableRNG(1))
opts = recommend_run_options(size(df, 1); engine = :lmtp)
δ = collect(-1.0:0.25:1.0)  # z-scale grid; default_deltas() uses -2:0.1:2

grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = δ,
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    learners_trt = opts.learners_trt,
    parallel = false,
    positivity = opts.positivity,
    simultaneous = false,
    rng = StableRNG(2),
)
grid[:, [:delta, :est, :se]]
```

**Certificate.** Attach identification provenance before archiving results:

```@example lmtp-walk
estimand = estimand_from_query(TotalEffectQuery(:A, :Y), [:W])
cert = identification_certificate(id, :A, :Y; adjustment = [:W])
meta = build_run_metadata(estimand, cert; folds = 2)
meta.certificate.result.identifiable
```

**Graph (total effect).** Estimand path `A → Y` only (confounder omitted from the TE diagram).

```@example lmtp-walk
using Graphs: edges, src, dst

δ_show = 0.5
te = only(grid[abs.(grid.delta .- δ_show) .< 1e-8, :est])
g_te, _ = chain_graph(["A", "Y"])
fig, _, _ = dagplot_chain(["A", "Y"];
    elabels = structural_edge_labels(g_te, ["TE\n$(round(te; digits = 2))"]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "Total effect (δ = $δ_show)",
)
fig
```

**Figure.** MTP curve with optional uncertainty band (`plot_mtp_curve` needs CairoMakie):

```@example lmtp-walk-fig
using CausalTargeted, CairoMakie, StableRNGs

df, _ = simulate_linear_mtp(200; rng = StableRNG(3))
δ = collect(-1.0:0.2:1.0)
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W], deltas = δ,
    folds = 2, parallel = false, simultaneous = false, rng = StableRNG(4),
)
fig, ax = plot_mtp_curve(grid; title = "LMTP δ-grid (synthetic)")
fig
```

Further detail: [Methods — LMTP](methods.md) · [Small-*n* checklist](small_n.md).

## 2. Typed plan (`execute_estimand`)

When you already hold an `IdentificationResult`, `plan_mtp` / `execute_estimand`
keep the certificate on the estimation path:

```@example plan-walk
using CausalTargeted, CausalDynamics, Graphs, StableRNGs

g = DiGraph(3)
add_edge!(g, 1, 2); add_edge!(g, 1, 3); add_edge!(g, 2, 3)
id = identify(g, TotalEffectQuery(:A, :Y); node_names = Dict(1 => :W, 2 => :A, 3 => :Y))

df, _ = simulate_linear_mtp(180; rng = StableRNG(5))
estimand = estimand_from_query(TotalEffectQuery(:A, :Y), [:W])
plan = plan_mtp(estimand, df; id_result = id, deltas = [0.5], folds = 2)
grid = execute_estimand(
    plan.estimand, df;
    id_result = id,
    deltas = [0.5],
    folds = 2,
    parallel = false,
    rng = StableRNG(6),
)
only(grid.est), plan.certificate.result.strategy
```

**Graph (identification).**

```@example plan-walk
using DAGMakie, CairoMakie

fig, _, _ = dagplot_confounding(["W", "A", "Y"];
    color_by = :adjustment,
    exposure = 2,
    outcome = 3,
    adjustment = Set([1]),
    title = "Good control W (identification)",
)
fig
```

**Graph (total effect).**

```@example plan-walk
using Graphs: edges, src, dst

te = only(grid.est)
g_te, _ = chain_graph(["A", "Y"])
fig, _, _ = dagplot_chain(["A", "Y"];
    elabels = structural_edge_labels(g_te, ["TE\n$(round(te; digits = 2))"]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "Typed plan TE (δ = 0.5)",
)
fig
```

## 3. Interventional mediation (CausalMediation)

Mediation TE / NDE / NIE under MTP shifts lives in
[CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl). CausalTargeted
supplies nuisances and optional façades once the weak dependency is loaded.

```@example mediation-walk
using CausalMediation, CausalTargeted, CausalDynamics, Graphs, StableRNGs

g = DiGraph(4)
add_edge!(g, 1, 2); add_edge!(g, 1, 3); add_edge!(g, 1, 4)
add_edge!(g, 2, 3); add_edge!(g, 2, 4); add_edge!(g, 3, 4)
names = Dict(1 => :W, 2 => :A, 3 => :M, 4 => :Y)

id = identify(
    g, MediationQuery(:A, :Y, [:M]; effect_kind = :interventional);
    node_names = names,
)
spec = spec_from_identification(id)

df, _ = simulate_continuous_mtp_mediation(200; rng = StableRNG(7))
res = run_mediation(
    spec, df;
    deltas = [0.5],
    folds = 2,
    n_mc = 16,
    estimator = :onestep,
    learners = DEFAULT_SL_LEARNERS,
    parallel = false,
    rng = StableRNG(8),
)
d = decompose(res)
d
```

**Graph (mediation paths).** Triangle for direct / indirect paths; `W` is a good control in estimation but not drawn here.

```@example mediation-walk
using DAGMakie, CairoMakie

fig, _, _ = dagplot_mediation(["A", "M", "Y"];
    title = "Mediation paths (A, M, Y)",
)
fig
```

**Graph (NDE / NIE).** Direct effect on `A → Y`, indirect via `A → M` (δ = 0.5).

```@example mediation-walk
using Graphs: edges, src, dst

g_med, _ = mediation_graph(["A", "M", "Y"])
elookup = Dict(
    (1, 3) => "NDE\n$(round(d.nde; digits = 2))",
    (1, 2) => "NIE\n$(round(d.nie; digits = 2))",
)
fig, _, _ = dagplot_mediation(["A", "M", "Y"];
    elabels = structural_edge_labels(g_med, [
        get(elookup, (src(e), dst(e)), "") for e in edges(g_med)
    ]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "Interventional mediation (δ = 0.5)",
)
fig
```

More mediation patterns (factor `A`, `moc`, nested MC): CausalMediation
[Getting started](https://simonab.github.io/CausalMediation.jl/dev/getting-started/).

## 4. Sequential LMTP

Two time points with a time-varying covariate (`L1` between `A1` and `A2`):

```@example sequential-walk
using CausalTargeted, CausalDynamics, StableRNGs, DataFrames

rng = StableRNG(9)
n = 80
W = randn(rng, n)
A1 = 0.5 .* W .+ randn(rng, n)
L1 = 0.3 .* A1 .+ randn(rng, n)
A2 = 0.4 .* L1 .+ 0.2 .* W .+ randn(rng, n)
Y = 0.5 .* A2 .+ 0.3 .* A1 .+ 0.2 .* W .+ randn(rng, n)
df = DataFrame(W = W, A1 = A1, L1 = L1, A2 = A2, Y = Y)

res = run_sequential_lmtp(
    df, [:A1, :A2], :Y;
    baseline = [:W],
    time_vary = [Symbol[], [:L1]],
    delta = 0.5,
    folds = 2,
    learners = SMALL_N_SL_LEARNERS,
    rng = rng,
)
res.estimate, res.times
```

With a temporal DAG in CausalDynamics, attach a certificate before estimating:

```@example sequential-walk
using CausalDynamics: TemporalDAGSpec, LaggedEdge, unroll_temporal_dag, TemporalEffectQuery

spec = TemporalDAGSpec([:w, :a, :l, :y], [
    LaggedEdge(:w, :a, 0),
    LaggedEdge(:a, :l, 0),
    LaggedEdge(:l, :a, 1),
    LaggedEdge(:a, :y, 0),
    LaggedEdge(:a, :y, 1),  # A₁ → Y (matches synthetic DGP)
    LaggedEdge(:w, :y, 0),
])
u = unroll_temporal_dag(spec, 2)
tq = TemporalEffectQuery(:a, :y, 1, 2)
cert = sequential_identification_certificate(u, tq)
cert.result.strategy, cert.temporal_lags
```

**Graph (identification).** Two-period unrolling; `w[1]` is the good control (baseline `W`).

```@example sequential-walk
using DAGMakie, CairoMakie
using CausalDynamics: temporal_node

w1 = temporal_node(u, :w, 1)
a2 = temporal_node(u, :a, 2)
y2 = temporal_node(u, :y, 2)
fig, _, _ = dagplot_temporal(u;
    dx = 2.4,
    dy = 1.7,
    figure_size = (720, 420),
    color_by = :adjustment,
    exposure = a2,
    outcome = y2,
    adjustment = Set([w1]),
    title = "Good control w[1] (identification)",
)
fig
```

**Graph (total effect).** Estimand path for terminal `Y` (confounder omitted); LMTP shifts both `A₁` and `A₂`.

```@example sequential-walk
using Graphs: edges, src, dst

g_te = DiGraph(4)
add_edge!(g_te, 1, 2)  # A₁ → L₁
add_edge!(g_te, 2, 3)  # L₁ → A₂
add_edge!(g_te, 1, 4)  # A₁ → Y
add_edge!(g_te, 3, 4)  # A₂ → Y
layout = [Point2f(0, 0), Point2f(1.1, 0), Point2f(2.2, 0), Point2f(2.2, -1.3)]
te_lbl = "TE\n$(round(res.estimate; digits = 2))"
elookup = Dict((3, 4) => te_lbl)
fig, _, _ = dagplot(g_te;
    layout = layout,
    labels = ["A₁", "L₁", "A₂", "Y"],
    elabels = structural_edge_labels(g_te, [
        get(elookup, (src(e), dst(e)), "") for e in edges(g_te)
    ]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "Sequential LMTP (δ = 0.5 on A₁, A₂)",
)
fig
```

## 5. Missing outcome (Observable policies)

Structural missingness claims live in CausalDynamics; estimators consume
Observable policies via `handle_missing`:

```@example missing-walk
using CausalTargeted, StableRNGs

df, _ = CausalTargeted.simulate_missing_outcome_mtp(200; rng = StableRNG(10))

grid_drop = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W], deltas = [0.5],
    folds = 2, parallel = false,
    handle_missing = :drop,
    rng = StableRNG(11),
)
grid_ipcw = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W], deltas = [0.5],
    folds = 2, parallel = false,
    handle_missing = :ipcw,
    rng = StableRNG(11),
)

(
    drop = only(grid_drop.est),
    ipcw = only(grid_ipcw.est),
    strategy_drop = missingness_metadata(grid_drop).strategy,
    strategy_ipcw = missingness_metadata(grid_ipcw).strategy,
)
```

**Graph (identification).** Good-control triangle (same backdoor pattern as §1).

```@example missing-walk
using DAGMakie, CairoMakie

fig, _, _ = dagplot_confounding(["W", "A", "Y"];
    color_by = :adjustment,
    exposure = 2,
    outcome = 3,
    adjustment = Set([1]),
    title = "MAR outcome — good control W",
)
fig
```

**Graph (total effect).** IPCW TE at δ = 0.5 on `A → Y`.

```@example missing-walk
using Graphs: edges, src, dst

te = only(grid_ipcw.est)
g_te, _ = chain_graph(["A", "Y"])
fig, _, _ = dagplot_chain(["A", "Y"];
    elabels = structural_edge_labels(g_te, ["TE\n$(round(te; digits = 2))"]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "IPCW TE (δ = 0.5)",
)
fig
```

Under MAR $Y$, `:drop` and `:ipcw` need not coincide. See [Missingness](missingness.md).

## 6. Positivity and sensitivity audit

Before interpreting a δ-curve, inspect positivity and optional omitted-confounder
sensitivity:

```@example audit-walk
using CausalTargeted, StableRNGs

df, _ = simulate_linear_mtp(200; rng = StableRNG(12))
δ = collect(-1.0:0.25:1.0)
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W], deltas = δ,
    folds = 2, parallel = false, positivity = true, rng = StableRNG(13),
)

te = grid.est[1]
se = grid.se[1]
rep = positivity_report(df, :A; deltas = grid.delta)
sens = sensitivity_report(te, se; n = size(df, 1))

(
    n_weak = count(==("weak_support"), string.(rep.support_status)),
    n_sens_rows = size(sens, 1),
    n_delta = size(grid, 1),
)
```

**Graph (identification).**

```@example audit-walk
using DAGMakie, CairoMakie

fig, _, _ = dagplot_confounding(["W", "A", "Y"];
    color_by = :adjustment,
    exposure = 2,
    outcome = 3,
    adjustment = Set([1]),
    title = "Positivity audit — good control W",
)
fig
```

**Graph (total effect).** TE at the first δ in the audit grid.

```@example audit-walk
using Graphs: edges, src, dst

δ₀ = grid.delta[1]
g_te, _ = chain_graph(["A", "Y"])
fig, _, _ = dagplot_chain(["A", "Y"];
    elabels = structural_edge_labels(g_te, ["TE\n$(round(te; digits = 2))"]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "TE at δ = $(round(δ₀; digits = 2))",
)
fig
```

## 7. G-computation plug-in (comparison)

When you want a plug-in contrast rather than the LMTP EIF path:

```@example gcomp-walk
using CausalTargeted, StableRNGs

df, truth = simulate_linear_mtp(200; rng = StableRNG(14))
res = run_gcomp(
    df, :A, :Y;
    covariates = [:W],
    folds = 2,
    learners = (:glm, :mean),
    rng = StableRNG(15),
)
res.estimate, res.se
```

**Graph (identification).**

```@example gcomp-walk
using DAGMakie, CairoMakie

fig, _, _ = dagplot_confounding(["W", "A", "Y"];
    color_by = :adjustment,
    exposure = 2,
    outcome = 3,
    adjustment = Set([1]),
    title = "G-computation — good control W",
)
fig
```

**Graph (total effect).** Plug-in contrast on `A → Y`.

```@example gcomp-walk
using Graphs: edges, src, dst

g_te, _ = chain_graph(["A", "Y"])
fig, _, _ = dagplot_chain(["A", "Y"];
    elabels = structural_edge_labels(g_te, ["TE\n$(round(res.estimate; digits = 2))"]),
    elabels_fontsize = 13,
    elabels_distance = 14,
    elabels_rotation = 0,
    title = "G-computation TE",
)
fig
```

## Next steps

| Topic | Page |
|-------|------|
| Literature ↔ API map | [Methods and literature](methods.md) |
| Missingness strata and certificates | [Missingness](missingness.md) |
| Conservation / ecology defaults | [Small-*n* checklist](small_n.md) |
| Real and semi-synthetic cohorts | [Stress validation](stress_validation.md) |
| Symbol index | [API overview](api.md) |
| Narrative book | [CDCS](https://simonab.github.io/causal-dynamics-book/) |
