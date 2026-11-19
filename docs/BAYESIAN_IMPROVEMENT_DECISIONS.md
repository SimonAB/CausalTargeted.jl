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

### D3: approximation and efficiency changes await diagnosis

Status: pending independent reference and paired comparisons. Do not modify the
LRVB equations, widen eligibility gates or change public defaults to improve
validation scores. Record any new reproducible defect here, with its own failing
test and before/after criterion before runtime edits.

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
