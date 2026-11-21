## Bayesian package improvement decisions

### Baseline and scope

CausalTargeted `cf28065`, version 0.4.2. The existing implementation already
exposes numerical convergence, raw/corrected covariance, LRVB response checks
and unavailable intervals. Do not duplicate these or interpret them as posterior
accuracy certificates. Existing public policy results expose mean MCSE but no
uncertainty diagnostic for estimated interval endpoints.

### D1: distinguish endpoint simulation error from posterior uncertainty

Status: implemented and tested; initial specification retained below. Affected operation:
`bayesian_policy_contrast(::BayesianLogisticFit, ...)`.

Cause: finite Gaussian draws estimate quantiles as well as means, but `mcse`
describes only posterior-mean simulation error. Increasing draws can stabilise
endpoints; it cannot repair an inaccurate Gaussian approximation or prior.

Candidate: additive `endpoint_mc_bounds` in the public result, controlled by
`mc_level=0.95`. Each policy gets bounds for each ideal-law interval endpoint
using binomial order statistics of its existing independent coefficient draws.
No extra inference, reference integration or outcome data is needed. Metadata
states pointwise Monte Carlo bounds conditional on the selected Gaussian law,
not simultaneous bounds, posterior credible limits, or an approximation check.
Unavailable uncertainty yields unavailable endpoint diagnostics too.

For probability p and N independent draws, K=#draws below the population
quantile has distribution Binomial(N,p) for a continuous law. Choose
l=quantile(Binomial(N,p),(1−mc_level)/2),
u=quantile(Binomial(N,p),(1+mc_level)/2)+1.
Then [draw_(l),draw_(u)] covers the population p-quantile with probability at
least mc_level. Indices 0 and N+1 use the known risk-difference support −1 and 1.
For atoms the interval is conservative; zero/duplicate policies require tests.
The two tail failure events each have probability at most (1−mc_level)/2.
The confidence statement applies to each endpoint separately.

Acceptance: exact binomial rank-probability check; independent analytic-law
simulation; public API identity/reproducibility/zero-policy/failed-fit contracts;
unchanged existing mean/covariance/quantile values at identical RNG state;
validated finite mc_level in (0,1). Report timing/allocation overhead, not a
speedup claim. Reuse sorted draws for quantile/rank work if numerical outputs
remain unchanged. No warnings or withholding based solely on arbitrary bound
width thresholds. Retain current draw defaults until separate evidence supports
changing them.

Evidence: 67 focused public/analytic checks, 378 additional independently
reviewed rank configurations, and the full package suite (3,498 assertions).
The public-API study used 800 datasets, 4,800 fits and 288,000 draw-stream
evaluations. Existing means, covariance, intervals and mean MCSE were identical
to the baseline at matched RNG states in the benchmark. No convergence or
covariance-selection rule changed. The empirical study contains a default-draw
lower-endpoint shortfall against its exact binomial expectation; retain that
finding and its independent reconstruction checks rather than describing every
cell as empirically calibrated. An independent fresh-seed replication on 19
November 2026 (`seed=20261119` across 200 datasets / 12,000 rows per method at
2,000 draws, 64 datasets at 8,000 draws, and 4,000 `Xoshiro` streams) did not
reproduce the shortfall (`variational` 95.98%, `linear_response` 95.49%,
`Xoshiro` 95.40% vs 95.54% exact expectation; symmetric ~2.0–2.4% tail misses),
confirming that the paired `seed=20261115` deviation reflected shared
`draw_seed` coupling across priors and methods rather than an order-statistic
or boundary defect. Original archives remain unmodified. This is a
simulation-precision diagnostic, not a remedy for the original posterior
coverage failures.

### D2: no automatic repair of prior conflict

Status: retain declared-prior semantics; add executable user guidance and the
paired refit/validation workflow. The November fixed-truth undercoverage
also occurs for the adaptive posterior. No justified covariance-only repair
follows. Implement paired sensitivity experiments and document a caller-declared
comparison workflow. Do not infer a scientifically correct prior from simulator
truth or select one by benchmark coverage.

### D3: bivariate approximation and efficiency audit (closed for Phases 1–3A)

Status: closed for the bivariate logistic regime on 21 November 2026 based on the
1,000-replication main archive
(`docs/validation/bayesian-main-1000-20261119/README.md`). Retain the existing
Jaakkola–Jordan VI + LRVB equations, eligibility gates, and public defaults;
no bivariate runtime remedy is justified.

Evidence (12,000 prior-predictive datasets + 8,000 fixed-parameter datasets /
96 cell×prior pairs):

- **Covariance fidelity:** `linear_response` (LRVB) reduced relative Frobenius
  covariance error over `variational` (Raw VI) in **12/12** prior-predictive
  cells (`1.4%–60.0%` → `0.3%–7.6%`, reductions of 1.0–52.4 percentage points)
  and **96/96** fixed-parameter cell×prior pairs (reductions of 0.6–58.1
  percentage points).
- **Posterior content and coverage:** Raw VI undercovered under diffuse priors
  (`s = 3.0`: prior-predictive coverage `86.1%–92.2%`, exact posterior content
  `86.4%–92.3%`). LRVB eliminated that undercoverage (prior-predictive coverage
  `94.2%–98.1%`, exact posterior content `95.0%–97.3%`).
- **Remaining approximation boundary (not a runtime defect):** Because LRVB
  corrects covariance while preserving a symmetric Gaussian law in logit space
  (identical mean discrepancy to Raw VI, `mean_error_sd_max = 0.217`), skewed
  small-sample / diffuse-prior posteriors (`n = 20`, `s = 3.0`) exhibit mild
  conservatism (`96.8%–98.1%` coverage) and larger worst-case endpoint error
  than Raw VI (`0.0637` vs `0.0210` in prior-predictive; `+0.0313 ± 0.0014` in
  `fixed_n20_c10_b3p0 / oracle_s3p0`, alongside a 47.6 pp covariance-error
  reduction).
- **Disposition:** Do not tune LRVB equations or defaults to trade away
  covariance accuracy for bivariate endpoint error. Document the small-*n* /
  diffuse-prior Gaussian-skewness conservatism as an explicit operating limit,
  and re-evaluate whether non-Gaussian importance refinement is warranted only
  after Phase 4 tests multi-covariate correlation, overlap, and misspecification.

### D4: correct the reference-moment gate and independent metric summaries

Status: implemented as explicitly declared pilot amendments A1/A2 in the v2
protocol. The original fidelity pilot kept 75/288 moment comparisons unavailable
under a mixed raw-moment error gate. A reproduced case had excellent moment
refinement but failed the aggregate absolute-error threshold because second
moments dominated its units. Reintegrate centred, marginal-SD-scaled moments
with propagated dimensionless error diagnostics; do not loosen probability or
endpoint gates. The original output remains immutable. The amended paired pilot
validated all 288 selected method comparisons, and 28 additional driver checks
cover analytic large-centre Gaussian moments, propagation and the reproduced
failure. These remain heuristic numerical estimates, not rigorous enclosures.

A separate six-assertion regression checks that ambiguous truth inclusion does
not discard independently validated endpoint/content/moment metrics. This is a
validation-summary correction, not a causal-estimator change. Generation source
snapshots distinguish original code from amended/postprocessing code.

### First-milestone boundary

The new public diagnostic, documentation and regression protection are ready
for local delivery. Full factorial 1,000-replication studies, higher-dimensional
causal mechanisms and untouched scenario confirmation remain future work.
Do not infer general posterior calibration from these pilots or claim the
Gaussian approximation itself has been improved in this milestone.

### D5: Phase 4 multi-covariate study, Phase 5 held-out confirmation, and hierarchical cluster boundaries (closed 21 November 2026)

Status: **closed** for Phases 4–5 and the hierarchical cluster baseline;
operating limits documented in `docs/src/bayesian_logistic.md` and
`docs/src/bayesian.md`.

- **Phase 4 multi-covariate logistic (`p ∈ {4, 5}`, `n ∈ {80, 200}`):**
  Across the 500-rep main archive (`12,000` paired datasets, `35,986 / 36,000 = 99.961%`
  availability; `docs/validation/bayesian-phase4-main-500-20261121/README.md`,
  fresh seed `2026112101`, verified against Turing NUTS in
  `scripts/stress_harness/run_phase4_nuts_spotcheck.jl`), `linear_response`
  (LRVB) strictly outperformed `variational` (Raw VI) in **24/24** cell×prior
  pairs on covariance error (`-8.4` to `-29.8 pp`; `mc_interact_spec_n80 / zero_s3p0`:
  `36.1% → 6.4%`), exact posterior content (`94.7%–95.3%` vs `93.2%–94.1%`),
  and max 95% endpoint error (`0.0026–0.0060` vs `0.0038–0.0083`).
- **Phase 5 untouched held-out confirmation (`1,000` reps × 3 reserved scenarios
  × 2 priors × 4 methods = `24,000` rows; `docs/validation/bayesian-phase5-heldout-20261121/README.md`):**
  `linear_response` (LRVB) outperformed both `variational` (Raw VI) and matched
  `laplace_map` (regularised logistic MAP + observed Hessian on the identical
  prior) in **6/6** untouched cell×prior pairs on covariance error (`3.1%–6.3%`
  vs `4.4%–11.8%` Laplace MAP and `8.3%–46.0%` Raw VI), exact posterior content
  (`94.7%–95.0%`), and 95% endpoint error. Prior-conflict under
  `heldout_n150_c30_b4p0 / zero_s1p5` reconfirmed Decision D2 (exact PSIS
  coverage `82.8%`, LRVB exact posterior content `94.8%`).
- **Hierarchical cluster baseline (`scripts/stress_harness/run_bayesian_cluster*_validation.jl`):**
  - *Clustered Gaussian & mediation*: Structured VI and PSIS closely track
    Turing NUTS (`K = 30`: PSIS mean error `0.080 SD`, SD rel error `3.1%`,
    `k̂ = 0.47`), scale to `K = 300, N = 4,495` in `0.84 s`, and achieve 90% PSIS
    gate pass rate in small-sample `K = 60` mediation.
  - *Clustered logistic*: At `K = 40`, PSIS cuts upper-CI error from `0.189 SD`
    (JJ-VI) to `0.044 SD` (`k̂ = 0.32`). **Operating boundary:** at `K = 100–300`,
    the Student-$t$ PSIS proposal on `(β, log σ_b²)` truthfully withholds
    (`k̂ = 0.73–1.18`, or `ESS = 341 < 400`). Document this scaling boundary
    explicitly rather than loosening PSIS gates.
  - *Clustered count & hurdle*: At `K = 35`, PSIS cuts Poisson SD relative error
    from `5.25%` (Laplace-VI) to `0.68%` and upper-CI error from `0.248 SD` to
    `0.044 SD` (`k̂ = 0.31`). Warmed runtime at `K = 300, N = 4,527` is `< 2.9 s`
    for Poisson/NB2 and `19.7 s` for Hurdle-NB2.
