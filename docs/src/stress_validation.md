# Stress validation

Application-layer stress tests for the owned Julia causal stack. Package `test/`
suites stay lean; broader functionality, edge cases, recovery Monte Carlo, and
wall-clock checks live in the CDCS application harness.

The distinctive demonstration is **integration along the CDCS spine**: Structural
(identify / `GraphSCM` / $do$ / shared $\mathbf{u}$) → Dynamical (sequential /
survival) → Observable (LMTP, mediation, real cohorts) → audit, with Turing for
small $n$ and RxInfer for larger tables. Capability matrix:
[ECOSYSTEM_COMPARISON.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/ECOSYSTEM_COMPARISON.md).

**Methods notebook (Quarto):** [`docs/stress/stress_validation.qmd`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/stress/stress_validation.qmd)
runs dataset-by-dataset analyses with expected vs actual results, timings, and
DAGMakie / `plot_mtp_curve` / posterior figures. Fixtures: [`docs/data/`](https://github.com/SimonAB/CausalTargeted.jl/tree/main/docs/data).

**Canonical harness (data + runners):**
[SimonAB/causal-dynamics-book](https://github.com/SimonAB/causal-dynamics-book)
→ `scripts/stress_harness/` and `data/catalog.toml`.

**Julia↔R concordance** (known-truth LMTP / mediation vs `lmtp` / `crumble`)
remains under `scripts/synthetic_benchmark/` in the same repository. Do not
promote Super Learner defaults from a single Monte Carlo batch.

## Packages under test

| Package | Role in stress suite | Repository | Docs |
|---------|----------------------|------------|------|
| [CausalTargeted.jl](https://github.com/SimonAB/CausalTargeted.jl) | LMTP / g-comp / sequential / survival grids; Super Learner; missing-data strategies; positivity | this package | [Documenter](https://simonab.github.io/CausalTargeted.jl/dev/) |
| [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) | Graphs, `identify`, certificates, panel / CDM bridges | [GitHub](https://github.com/SimonAB/CausalDynamics.jl) | [Documenter](https://simonab.github.io/CausalDynamics.jl/dev/) |
| [CausalMediation.jl](https://github.com/SimonAB/CausalMediation.jl) | Interventional / natural mediation engines on JOBS II and synthetic MTP mediation | [GitHub](https://github.com/SimonAB/CausalMediation.jl) | [Documenter](https://simonab.github.io/CausalMediation.jl/dev/) |
| [DAGMakie.jl](https://github.com/SimonAB/DAGMakie.jl) | Adjustment / DAG display on toy identify scenarios | [GitHub](https://github.com/SimonAB/DAGMakie.jl) | [Documenter](https://simonab.github.io/DAGMakie.jl/dev/) |

### Related third-party packages

| Package | Use in this programme |
|---------|------------------------|
| [TMLE.jl](https://github.com/TARGENE/TMLE.jl) | Complementary point-treatment CM / ATE / AIE; CausalDynamics `prepare_for_tmle` |
| [CausalInference.jl](https://github.com/mschauer/CausalInference.jl) | Upstream graphical criteria / discovery smoke (e.g. NCI60) |
| [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) | DAG representation |
| [MLJ.jl](https://github.com/JuliaAI/MLJ.jl) (+ LinearModels / DecisionTree / EvoTrees, optional) | Super Learner candidates |
| [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl) | Optional MTP curve and DAG figures |
| R [`lmtp`](https://cran.r-project.org/package=lmtp) / [`crumble`](https://cran.r-project.org/package=crumble) | Dual-stack concordance only (not in the stress runners) |

## Design

| Profile | Intent |
|---------|--------|
| `smoke` | Minutes; every `smoke=true` catalog row |
| `nightly` | Hours; large \(n\) and rich libraries |
| `full` | Exhaustive pre-release |

Runners (`scripts/stress_harness/`):

1. **Functionality** — finite estimates or documented throws per catalog engine
2. **Recovery MC** — multi-seed absolute error and 95% coverage vs oracle (synthetic)
3. **Performance** — wall time for lean LMTP on mixed baselines and real microdata

Environment knobs: `STRESS_PROFILE`, `STRESS_N`, `STRESS_SEEDS`, `STRESS_FOLDS`, `STRESS_T`.

| `scale` | Typical \(n\) | Role |
|---------|---------------|------|
| `tiny` | \(\le 40\) | Conservation / ecology field trials |
| `small` | \(\le 200\) | Published sheep cohort, airquality |
| `medium` | \(\le 2\times 10^3\) | IHDP, CPS sample |
| `large` | \(\ge 2\times 10^3\) | Bird counts, Twins sample (nightly) |

| `domain` | Role |
|----------|------|
| `conservation_biology` | Liu et al. CircVax sheep; synthetic sheep / tiny panels |
| `ecology` | Lizards, Bt corn, bird counts |
| `ci_benchmark` | IHDP, Twins, Lalonde/CPS, JOBS II |
| `methods_synthetic` | Package DGPs (MTP, schema, missingness) |

## Methodological source papers

Full bibliographic list: [References](references.md). Stress design leans on:

### Estimation and policies

- Díaz, Williams, Hoffman & Schenck (2023). Nonparametric causal effects based on longitudinal modified treatment policies. *JASA*.
  [doi:10.1080/01621459.2021.1955691](https://doi.org/10.1080/01621459.2021.1955691)
- Williams & Díaz (2023). *lmtp*: An R package for estimating the causal effects of modified treatment policies. *Observational Studies*.
  [muse.jhu.edu/article/883479](https://muse.jhu.edu/article/883479)
- Díaz & Hejazi (2020). Causal mediation analysis for stochastic interventions. *JRSS-B*.
  [doi:10.1111/rssb.12362](https://doi.org/10.1111/rssb.12362)
- Liu, Williams, Rudolph & Díaz (2024). General targeted machine learning for modern causal mediation analysis.
  [arXiv:2408.14620](https://doi.org/10.48550/arXiv.2408.14620)
- van der Laan & Rose (2011). *Targeted Learning*. Springer — IPCW-TMLE and Super Learner practice

### Missing data and positivity

- Petersen et al. (2012). Diagnosing and responding to violations in the positivity assumption. *Stat Methods Med Res*.
  [doi:10.1177/0962280210386207](https://doi.org/10.1177/0962280210386207)
- Weberpals et al. (2024) — missing-data methods in TMLE (see harness notes / ISSUES)
- Berrevoets et al. (AISTATS 2023) — selective imputation for treatment-effect estimation

### Identification (CausalDynamics)

- Pearl (2009). *Causality* (2nd ed.)
- Shpitser & Pearl (2006). Identification of joint interventional distributions… *AAAI*

## Datasets and literature anchors

Catalog registry: [`data/catalog.toml`](https://github.com/SimonAB/causal-dynamics-book/blob/main/data/catalog.toml)
in the book repository. Fixtures under `data/fixtures/`; processed tables under
`data/processed/` (gitignored raw downloads).

### Conservation biology

| Catalog id | \(n\) | Source | Link |
|------------|-------|--------|------|
| `sheep_vaccine_liu2022` | 62 | Liu et al. (2022). Vaccine-induced time- and age-dependent mucosal immunity… *npj Vaccines*. | Paper [doi:10.1038/s41541-022-00501-0](https://doi.org/10.1038/s41541-022-00501-0); phenotype workbook [SimonAB/Liu2022](https://github.com/SimonAB/Liu2022) |
| `sheep_vaccine_synthetic_tiny` | 36 | Package / harness synthetic | — |
| `conservation_panel_tiny` | 24 × \(T{=}3\) | Harness sequential panel | — |

### Ecology (public microdata via [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/))

| Catalog id | \(n\) | Table | Link |
|------------|-------|-------|------|
| `ecology_lizards_tiny` | 24 | `aod::lizards` | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/aod/lizards.csv) |
| `ecology_bt_corn_tiny` | 16 | `agridat::gathmann.bt` | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/agridat/gathmann.bt.csv) |
| `ecology_bird_counts_large` | ~18k | `bayesrules::bird_counts` (nightly) | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/bayesrules/bird_counts.csv) |
| `continuous_exposure_micro` / `airquality_with_missing` | 153 | `datasets::airquality` | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/datasets/airquality.csv) |

### Causal-inference benchmarks

| Catalog id | Source | Link / note |
|------------|--------|-------------|
| `ihdp_npci_1` | Hill (2011) IHDP NPCI; CEVAE mirror | [ihdp_npci_1.csv](https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv) — Louizos et al. (2017) CEVAE |
| `twins_mortality_sample` | US twins / Almond; CEVAE mirror (nightly) | [TWINS](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS) |
| `lalonde_nsw` | LaLonde NSW (`MatchIt::lalonde`) | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv) |
| `cps_mixtape_sample` | Dehejia–Wahba / CPS mixtape lineage (nightly) | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/cps_mixtape.csv) |
| `mediation_jobs` | Imai et al. JOBS II (`mediation::jobs`) | [CSV](https://vincentarelbundock.github.io/Rdatasets/csv/mediation/jobs.csv) |

### Methods synthetics (in-package DGPs)

Exported or in-module generators exercised by the harness include
`simulate_linear_mtp`, `simulate_mixed_baseline_mtp`, `simulate_weak_positivity_mtp`,
`simulate_missing_outcome_mtp`, `simulate_missing_covariate_mtp`,
`simulate_discrete_survival_mtp`, `simulate_continuous_mtp_mediation`, and
harness factories for high-cardinality sites, wide baselines, and sequential panels.
See [`src/synthetic.jl`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/src/synthetic.jl).

## Smoke-profile freeze (2026-08-13)

Snapshots committed under
[`scripts/stress_harness/results/`](https://github.com/SimonAB/causal-dynamics-book/tree/main/scripts/stress_harness/results)
in the book repository:

| Runner | Artefact | Headline |
|--------|----------|----------|
| Functionality | `functionality_smoke_latest.tsv` | **31/31** engines `ok` (including documented expected gaps) |
| Recovery MC | `recovery_mc_smoke_latest.csv` | Lean SL, two seeds: linear / mixed / missing-outcome recover; weak positivity noisy; missing-covariate impute poor coverage |
| Performance | `perf_baseline_smoke.tsv` | Lean LMTP wall times (machine-dependent) |

Interpretation notes for recovery:

- `linear_mtp` / `mixed_baseline_mtp` / `missing_outcome_mtp` — small error on this freeze
- `weak_positivity_mtp` — noisy by design
- `missing_covariate_mtp` under `:impute` — soft failure mode until IPCW / imputation paths harden

## Open gaps (tracked issues)

| Gap | Issue |
|-----|-------|
| Sequential Monte Carlo oracles | Notebook audit (some dynamical rows still self-check) |
| MIRS spectra, full Twins \(X\), large ecology LMTP | [causal-dynamics-book#14](https://github.com/SimonAB/causal-dynamics-book/issues/14) |

**Closed in 2026-08-14 cycle:** IPCW in LMTP/g-comp ([CT#9](https://github.com/SimonAB/CausalTargeted.jl/issues/9)); sequential missing `Y` under `:drop` ([CT#10](https://github.com/SimonAB/CausalTargeted.jl/issues/10)); survival MAR $S_T$ IPCW wiring; mean-only contrast guard ([CT#11](https://github.com/SimonAB/CausalTargeted.jl/issues/11)); fold-stable mediation schema ([CT#8](https://github.com/SimonAB/CausalTargeted.jl/issues/8) / [CM#3](https://github.com/SimonAB/CausalMediation.jl/issues/3)); GraphSCM sorted parent order ([CD#8](https://github.com/SimonAB/CausalDynamics.jl/issues/8)); g-comp refitting bootstrap ([CT#13](https://github.com/SimonAB/CausalTargeted.jl/issues/13)); mediation PPL `handle_missing` ([CM#4](https://github.com/SimonAB/CausalMediation.jl/issues/4)).

Harness notes: [`ISSUES.md`](https://github.com/SimonAB/causal-dynamics-book/blob/main/scripts/stress_harness/ISSUES.md).

## Reproducing the suite

From a checkout of [causal-dynamics-book](https://github.com/SimonAB/causal-dynamics-book)
(with owned packages developed under `packages/`):

```bash
julia --project=. --threads=auto scripts/stress_harness/generate_synthetic.jl
julia --project=. --threads=auto scripts/stress_harness/fetch_real.jl --write-hash
STRESS_PROFILE=smoke julia --project=. --threads=auto scripts/stress_harness/run_functionality.jl
STRESS_PROFILE=smoke STRESS_SEEDS=5 julia --project=. --threads=auto scripts/stress_harness/run_recovery_mc.jl
STRESS_PROFILE=smoke julia --project=. --threads=auto scripts/stress_harness/run_performance.jl
```

Copy refreshed artefacts into `scripts/stress_harness/results/` when freezing a new methods snapshot.
Harness README: [`scripts/stress_harness/README.md`](https://github.com/SimonAB/causal-dynamics-book/blob/main/scripts/stress_harness/README.md).

## Minimal local checks (no full catalog)

Without the book tree, package unit tests remain the merge gate:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

For a hand smoke of mixed baselines and missing strategies:

```julia
using CausalTargeted, StableRNGs
df, t = simulate_mixed_baseline_mtp(120; rng = StableRNG(1))
run_lmtp_grid(df, :A, :Y; baseline = t.baseline, deltas = [0.0, 0.5],
              folds = 2, learners_outcome = DEFAULT_SL_LEARNERS, parallel = false)
```

See also [Small-*n* checklist](small_n.md) and [Methods and literature](methods.md).
