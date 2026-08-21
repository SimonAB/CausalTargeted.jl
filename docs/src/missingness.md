# Missingness

CausalTargeted owns **Observable** numerical policies for incomplete tables once
Structural claims about response indicators $R$ are stated (or deliberately
omitted). CausalDynamics owns masks, certificates, and generative dropout; this
page covers what estimators do with Julia `missing`.

## Design rule

Same data gap; different object, different assumptions, different API.

| Layer | Owner | Object |
|-------|-------|--------|
| Structural | CausalDynamics | `MissingnessSpec` / `MissingnessCertificate` / `identify(...; missingness=)` |
| Dynamical | CausalDynamics + this package | Time-indexed gaps; sequential / survival runners |
| Observable | this package / CausalMediation | `handle_missing`, `impute_posterior`, metadata |

Do not coerce `Missing` to `Float64` without a documented strategy.
[`complete_numeric_column`](@ref) refuses silent promotion.

## Strategies (`handle_missing`)

[`handle_missing_data`](@ref) implements:

| Strategy | Outcome missingness | Covariate missingness |
|----------|---------------------|------------------------|
| `:drop` (default) | Complete-case rows | Drop incomplete rows |
| `:ipcw` | IPCW on $R_Y$, then analyse observed $Y$ | Drop incomplete covariates first |
| `:impute` | Drop incomplete $Y$ | Mean/mode impute + `*_miss` indicators |
| `:ipcw_impute` | IPCW after covariate imputation | Mean/mode impute + indicators |

Results are [`MissingDataResult`](@ref): cleaned frame, row weights, extra
indicator columns, and `meta` (`strategy`, miss rates, optional PCH `rung`,
`time_indexed`). Grid / NamedTuple runners attach the same meta via
[`missingness_metadata`](@ref) / [`with_missingness`](@ref).

For `:ipcw` and `:ipcw_impute`, fitted weights enter the influence-function
summary in `run_lmtp_grid`, `run_gcomp`, sequential LMTP, discrete LMTP, and
survival LMTP, so `:drop` and `:ipcw` need not coincide under MAR $Y$.

```julia
using CausalTargeted, StableRNGs

df, _ = simulate_missing_outcome_mtp(200; rng = StableRNG(1))
grid = run_lmtp_grid(
    df, :A, :Y; baseline = [:W], deltas = [0.5],
    folds = 2, learners_outcome = (:glm, :mean), learners_trt = (:glm, :mean),
    parallel = false, handle_missing = :ipcw, rng = StableRNG(1),
)
missingness_metadata(grid).strategy  # :ipcw
```

## Estimand families

| Family | API | Notes |
|--------|-----|-------|
| Point LMTP / MTP | `run_lmtp_grid(...; handle_missing=)` | Default `:drop` |
| G-computation | `run_gcomp(...; handle_missing=)` | Same four strategies |
| Sequential LMTP | `run_sequential_lmtp(...; handle_missing=)` | `time_indexed=true` in meta |
| Discrete LMTP | `run_discrete_lmtp(...; handle_missing=)` | Factor $A$ |
| Survival LMTP | `run_survival_lmtp(...; handle_missing=)` | MAR missing terminal $S_T$ ≠ censoring IPCW |
| Mediation | CausalMediation `handle_missing` | Forwards to `handle_missing_data` |

**Survival.** Censoring weights for discrete-time survival are a Dynamical
object. Do not reuse outcome-missingness IPCW as if it were right-censoring,
and do not treat missing terminal $S_T$ as a censoring covariate when weighting
MAR event indicators.

## Certificates and MAR sets

When identification attaches a missingness certificate,

```julia
using CausalDynamics, Graphs

g = DiGraph(3)
add_edge!(g, 1, 2); add_edge!(g, 1, 3); add_edge!(g, 2, 3)
id = identify(g, TotalEffectQuery(:A, :Y);
    node_names = Dict(1 => :W, 2 => :A, 3 => :Y),
    missingness = MissingnessSpec(:Y; regime = :mar, conditioning_set = [:W]),
)
mar_set(id)  # [:W]
```

[`mar_set`](@ref) returns `Symbol[]` when no certificate is present.

## Opt-in posterior imputation

[`impute_posterior`](@ref) draws completed continuous outcomes under a
**Gaussian MAR** model given predictors (or the certificate's `mar_set`).
Observed $Y$ are preserved; missing cells are drawn from
$N(\hat\mu(x),\hat\sigma^2)$. Unidentified MNAR certificates throw.

[`run_lmtp_grid`](@ref)`(...; imputation=draws)` runs the grid on each draw
(`handle_missing=:drop` per draw) and pools with Rubin's rule via
[`pool_lmtp_grids`](@ref). This path is opt-in; defaults remain `:drop` /
`:ipcw`. Turing / RxInfer backends are deferred.

## Stress notebooks

| Notebook | Content |
|----------|---------|
| [Missingness grid](stress_missingness.md) | Structural / Dynamical / Observable ledger |
| [Posterior](stress_posterior.md) | Gaussian MAR → pooled LMTP |
| [Deep SCM estimation](stress_deep_scm.md) | Raw assay missing → drop → encode → LMTP |

CDCS harness entry (book repo):

```bash
julia --project=. --threads=auto scripts/stress_harness/run_missingness_validation.jl
```
