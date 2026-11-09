# CausalTargeted.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21703329.svg)](https://doi.org/10.5281/zenodo.21703329)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://simonab.github.io/CausalTargeted.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package is part of the [CDCS Julia package set](https://github.com/SimonAB/causal-dynamics-book/tree/main/packages), which supports the [Causal Dynamics of Complex Systems book](https://simonab.github.io/causal-dynamics-book/).

CausalTargeted implements cross-fitted targeted estimators for continuous and
longitudinal exposures: longitudinal modified treatment policies (LMTP),
point-treatment repeated-outcome profiles with joint covariance, positivity
diagnostics, nested Monte Carlo stability checks, and omitted-confounder
sensitivity. The current development tip also provides parametric regression
standardisation. Interventional mediation (TE / NDE / NIE under MTP) is provided
by the companion CausalMediation package.
Defaults favour small-to-moderate sample sizes. Identification is delegated to
[CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl).

**Design principles:** [DESIGN.md](DESIGN.md) · [NAMING.md](NAMING.md) ·
[BOUNDARIES.md](BOUNDARIES.md) · [ecosystem](https://github.com/SimonAB/causal-dynamics-book/blob/main/packages/DESIGN_PRINCIPLES.md)

> On the Julia **General** registry (`Pkg.add("CausalTargeted")`). Requires Julia **1.12+**.
> Hard dependency [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl);
> optional [CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl) for mediation façades.
> Registry tracking: [REGISTRATION.md](REGISTRATION.md).

## Installation

```julia
using Pkg
Pkg.add("CausalTargeted")
using CausalTargeted
```

Development version **0.4.0** is not yet registered or tagged. Its API changes
are listed in [CHANGELOG.md](CHANGELOG.md). Install compatible development
versions of the CDCS packages together when following the current examples.

Development tip of `main`:

```julia
Pkg.add([
    PackageSpec(url="https://github.com/SimonAB/CausalDynamics.jl.git"),
    PackageSpec(url="https://github.com/SimonAB/CausalTargeted.jl.git"),
])
```

From a local clone:

```julia
Pkg.develop([
    PackageSpec(path="path/to/CausalDynamics.jl"),
    PackageSpec(path="path/to/CausalTargeted.jl"),
])
```

The registered release resolves CausalDynamics from General. Development
version 0.4.0 requires CausalDynamics 0.5 from a compatible checkout or URL.

## Quick start

```julia
using CausalTargeted, CausalDynamics

df, _ = simulate_linear_mtp(200)
opts = recommend_run_options(size(df, 1); engine = :lmtp)
grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = [-0.5, 0.0, 0.5],
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    learners_trt = opts.learners_trt,
    parallel = opts.parallel,
    positivity = opts.positivity,
)
```

Repeated outcomes under a static binary treatment (joint ``Σ`` for profile contrasts):

```julia
df, truth = simulate_repeated_outcome_ate(500)
res = run_repeated_outcome_msm(
    df, :A, [:Y1, :Y2, :Y3, :Y4];
    baseline = [:W], folds = 3, learners = (:glm, :mean),
)
# τ̂(t₃) − τ̂(t₂) with SE from the joint covariance
c = msm_contrast(res, 3, 2)
```

The parametric regression standardisation API below is in the development tip.
Fit an outcome model once and
standardise predictions over a target population under two treatment values:

```julia
using CausalTargeted, DataFrames, StatsModels

training = DataFrame(Y=[1.0, 2.0, 2.5, 3.5, 1.3, 2.2],
                     A=[0, 1, 0, 1, 0, 1], X=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
fit = fit_parametric_gcomp(@formula(Y ~ A + X), training; family=:gaussian)
target = select(training, Not(:Y))
effect = gcomp_contrast(fit, target; treatment=:A, reference=0, comparison=1)
```

This fitted parametric workflow is distinct from the cross-fitted `run_gcomp`
and LMTP estimators. It also supports selected GLM families and, through an
optional extension, supported mixed models; see the [methods guide](https://simonab.github.io/CausalTargeted.jl/dev/methods/).

## Ecosystem

CausalDynamics supplies graphs and identification certificates. CausalTargeted
fits nuisance models and estimates LMTP, repeated-outcome MSM, DiD and related
contrasts. [CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl)
provides mediation; [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl)
provides causal graph figures. Point-treatment CM/ATE/AIE workflows remain
complementary to [TMLE.jl](https://github.com/TARGENE/TMLE.jl).

Finite weight caps, clamped continuous policies and other unsupported score
contracts return point estimates with unavailable inferential SEs and limits.
Check `inference_status` and the [methods guide](https://simonab.github.io/CausalTargeted.jl/dev/methods/)
before interpreting intervals. Uncapped supported EIF routes still require
overlap, suitable nuisance fits and moment assumptions; small-sample nominal
coverage is not established by the regression tests.

The [ecosystem comparison](https://github.com/SimonAB/causal-dynamics-book/blob/main/packages/ECOSYSTEM_COMPARISON.md)
describes the boundaries between these packages and related R/Python tools.

## Testing and validation

CI develops tip CausalDynamics and CausalMediation so missingness and mediation APIs match the stack; `Pkg.test()` on Julia **1.13** is the merge gate. Julia **1.12** remains within the declared compatibility range. Default `Pkg.test` keeps API and invalid-input coverage; set `UNIT_STRESS=1` (alias `CT_UNIT_STRESS=1`) for multi-seed recovery, MixedModels/NB2 trajectory recovery, and the estimand × `handle_missing` strategy matrix. Quarto stress notebooks extend coverage to real and semi-synthetic cohorts (see [STRESS.md](STRESS.md)).

| Guardrail | What we exercise | Where |
|-----------|------------------|-------|
| **Unit / API** | Covariate schema, missing-data policies (`:drop`, IPCW, imputation), Super Learner / metalearners, LMTP and discrete LMTP grids, repeated-outcome MSM, sequential / survival policies, g-comp, DiD, sensitivity, transport, certificates, MTP plotting | `test/` |
| **Synthetic recovery** | Oracle TE / NDE / NIE under known DGPs; lean smoke in default CI; multi-seed / $n_\mathrm{mc}$ / learner grids under `UNIT_STRESS=1` | `test/test_recovery.jl`, `test/test_core.jl`, `test/test_metalearners.jl` |
| **Missingness matrix** | Edge-case policies always; full estimand × handle\_missing grid under `UNIT_STRESS=1` | `test/test_missingness_edge_cases.jl`, `test/test_missing_strategies_matrix.jl`, `test/test_posterior_imputation.jl` |
| **Integration / extensions** | CausalMediation weakdep façades, MLJ / EvoTrees / XGBoost / Flux learners, Makie MTP curves; MixedModels trajectory recovery under `UNIT_STRESS=1` | `test/test_mediation.jl`, `test/test_mlj_ext.jl`, `test/test_mtp_plotting.jl`, `test/test_mixedmodels.jl` |
| **Stress (pre-ship)** | Structural → Dynamical → Observable path; smoke-freeze matrices; timings and signed errors | [docs/stress/stress_validation.qmd](docs/stress/stress_validation.qmd) |
| **Deep SCM estimation** | Mediation / LMTP on encoded codes; missing $Y$ under certificates | [docs/stress/deep_scm_estimation_stress.qmd](docs/stress/deep_scm_estimation_stress.qmd) |
| **Missingness stress** | Structural certificates × Observable strategies × posterior pooling | [docs/stress/missingness_grid_stress.qmd](docs/stress/missingness_grid_stress.qmd), [missingness_posterior_stress.qmd](docs/stress/missingness_posterior_stress.qmd) |
| **Real / benchmark data** | CircVax sheep, tiny ecology tables, IHDP NPCI, LaLonde, airquality (incl. missing), JOBS II mediation | [docs/stress/stress_validation.qmd](docs/stress/stress_validation.qmd) · fixtures in [docs/data/](docs/data/) |
| **Harness** | Edge-unit smoke + optional Quarto render | [causal-dynamics-book/scripts/stress_harness](https://github.com/SimonAB/causal-dynamics-book/tree/main/scripts/stress_harness) |

If you have a scenario that should be harder to pass (tighter freeze bounds, heavier missingness, larger ecology or cohort LMTP), please open an issue — we welcome stress cases that expose gaps before users do.

## Optional Super Learner candidates

Default grid library is lean (`:glm`, `:mean`). Use `RICH_SL_LEARNERS` when you
want interactions / elastic-net / Random Forest / EvoTrees; load the matching
weakdeps first:

```julia
using CausalTargeted
using CausalMediation       # mediation façades (TE / NDE / NIE grids)
using MLJ, MLJLinearModels  # :glmnet_* and :mlj_*
using MLJDecisionTreeInterface  # :randomforest (in RICH_SL_LEARNERS)
using EvoTrees              # :evotree, :evotree_deep

fit_super_learner(X, y; learners = (:glm, :mlj_ridge, :mlj_lasso, :mean))

using MLJXGBoostInterface   # :xgboost (opt-in only; not in RICH_SL_LEARNERS)
fit_super_learner(X, y; learners = (:glm, :randomforest, :xgboost, :mean))

using MLJFlux  # activates CausalTargetedMLJFluxExt (also needs MLJ)
fit_super_learner(X, y; learners = (:glm, :mlj_mlp, :mean))
```

Linear MLJ fits column-standardise features (leading intercept of ones is
dropped). Tree learners (`:randomforest`, `:xgboost`) drop the intercept only
and leave predictors unscaled. Neural learners and `:xgboost` are never included
in `SMALL_N_SL_LEARNERS` / `adaptive_learners`; `:randomforest` is in
`RICH_SL_LEARNERS` but not in the adaptive path.

Binary nuisances default to `metalearner=:nnloglik` (R `SuperLearner::method.NNloglik`).
The ensemble NNLS metalearner is `:nnls`; the discrete Super Learner is
`:cv_selector`. Nest an ensemble Super Learner inside that selector with
`nested_sl_candidate` (Phillips eSL-inside-dSL; opt-in). Categorical outcomes use
`family=:multinomial`; categorical treatments in LMTP use `run_discrete_lmtp` at a
single time (classification density ratios). Multi-time factor recodes use
`SequentialPolicy` `policies` (`DiscreteTreatmentPolicy` per time, or one policy
broadcast). `estimand_from_query` maps a discrete `InterventionalPolicyQuery` to
`DiscreteInterventionalMean`; sequential factor recodes need `policies` plus wide
`treatments`.

```julia
fit_super_learner(X, y;
    learners = (:logistic, :mean),
    family = :binomial,
    metalearner = :nnloglik,
)

inner = nested_sl_candidate((:glm, :mean); name = :esl, metalearner = :nnls)
fit_super_learner(X, y; learners = (:mean, inner), metalearner = :cv_selector)
```

## LMTP inference and weight caps

Finite default ratio caps leave point estimation available but gate ordinary EIF uncertainty (`NaN` SEs, limits,
and influence curves; `inference_status=:truncated_weights`). The uncertainty gate leaves a computed point result
unchanged, while targeting or nuisance-cache repairs may change fitted values across releases. Set
`weight_truncation=false` for supported discrete
finite-support policies or an unclamped constant additive Gaussian shift. This opts out of package-level caps;
numerical guards inside the selected learner remain active. The uncapped route does not promise nominal coverage
in small samples, and clamped continuous policies remain outside the ordinary EIF contract.

## MTP effect curves (optional Makie)

After `run_lmtp_grid` or `run_mediation_grid`, visualise δ-indexed estimates with
`plot_mtp_curve` (load `CairoMakie` to activate the extension):

```julia
using CausalTargeted, CairoMakie

grid = run_lmtp_grid(data, :A, :Y; baseline = [:W])
fig, ax = plot_mtp_curve(grid; title = "Exposure → outcome")
```

Mediation grids pass TE / NDE / NIE on the same axes; optional clamp strips use
the `clamp` column when present.

## Related packages

This package covers **continuous and categorical-treatment LMTP** (including
sequential factor recodes) **and interventional mediation**. For
point-treatment CM / ATE / AIE, prefer
[TMLE.jl](https://github.com/TARGENE/TMLE.jl). Graphs and identification live
upstream in CausalDynamics (`prepare_for_tmle` bridges to TMLE.jl).

| Package | Role |
|---------|------|
| [TMLE.jl](https://github.com/TARGENE/TMLE.jl) | Point-treatment CM / ATE / AIE (TMLE, OSE, C-TMLE) |
| [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) | Graphs and identification certificates (required upstream) |
| [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) | DAG figures for causal diagrams |
| [CausalTables.jl](https://github.com/salbalkus/CausalTables.jl) | SCM-aware tables; often paired with TMLE.jl |
| [CausalInference.jl](https://github.com/mschauer/CausalInference.jl) | Structure learning and classical graphical criteria |

R analogues for the continuous / mediation slice (`lmtp`, `crumble`) are
conceptual parity, not API identity; see [NAMING.md](NAMING.md).

## Documentation

- [Documenter site](https://simonab.github.io/CausalTargeted.jl/dev/) (getting started, methods, small-*n* checklist, live figures)
- [Getting started](https://simonab.github.io/CausalTargeted.jl/dev/getting-started/) — identify → estimate walk-throughs
- [Methods and literature](docs/src/methods.md) — maps APIs to papers
- [Stress validation](STRESS.md) — Quarto notebook with dataset analyses, expected vs actual, DAGMakie / MTP figures ([`docs/stress/stress_validation.qmd`](docs/stress/stress_validation.qmd); [Documenter summary](https://simonab.github.io/CausalTargeted.jl/dev/stress_validation/); harness in [causal-dynamics-book](https://github.com/SimonAB/causal-dynamics-book/tree/main/scripts/stress_harness))
- [References](docs/src/references.md) — DOIs and BibTeX keys shared with the CDCS book
- [CDCS book](https://simonab.github.io/causal-dynamics-book/) — worked identify → estimate → display examples

Build Documenter pages locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

### Core citations

Full list in [References](docs/src/references.md). Highlights:

- Díaz, Williams, Hoffman & Schenck (2023). Nonparametric causal effects based on longitudinal modified treatment policies. *JASA*. [doi:10.1080/01621459.2021.1955691](https://doi.org/10.1080/01621459.2021.1955691)
- Díaz & Hejazi (2020). Causal mediation analysis for stochastic interventions. *JRSS-B*. [doi:10.1111/rssb.12362](https://doi.org/10.1111/rssb.12362)
- Liu, Williams, Rudolph & Díaz (2024). General targeted machine learning for modern causal mediation analysis. arXiv:2408.14620
- van der Laan & Rose (2011). *Targeted Learning*. Springer
- Cinelli & Hazlett (2020). Making sense of sensitivity. *JRSS-B*. [doi:10.1111/rssb.12348](https://doi.org/10.1111/rssb.12348)

## Acknowledgements

Part of the Causal Dynamics for Complex Systems (CDCS) project.
Maintainer: [Simon A. Babayan](https://orcid.org/0000-0002-4949-1117).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff) or:

```bibtex
@software{causaltargeted2026,
  author = {Babayan, Simon A.},
  title  = {CausalTargeted.jl: Cross-fitted LMTP and interventional mediation},
  year   = {2026},
  doi    = {10.5281/zenodo.21703329},
  url    = {https://github.com/SimonAB/CausalTargeted.jl}
}
```
