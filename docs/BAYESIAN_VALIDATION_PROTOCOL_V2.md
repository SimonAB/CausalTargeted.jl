## Bayesian logistic validation protocol v2

### Scope and baseline

Baseline: CausalTargeted `cf28065` (0.4.2). Preserve the November v1 archive.
This first tranche implements phases 0–3A of the
[package improvement plan](BAYESIAN_VALIDATION_NEXT_PHASE.md). It excludes book
changes, releases and inference changes in CausalDynamics/CausalMediation.

The estimand is assignment-to-one versus observed exposure on the supplied
cohort: ψ = (n₀/n)[logistic(β₀+β₁)−logistic(β₀)]. Identification and the logistic
likelihood remain user-assumed. No posterior variance enters EIF inference.

### Cells and pairing

Fixed-truth base cells: n = 20 or 50, n₀ = n/2 or n/10, β₀ = −0.3,
β₁ = 0.9 or 3.0 (eight cells). Simulate once per cell/replication, then fit
twelve declared priors: s²I for s = 0.5,1.5,3 and means zero, β_true,
β_true±2s[0,1]. Only zero-centred priors are deployable choices; truth-dependent
means are oracle diagnostics. No outcome-derived scaling or automatic prior
replacement is allowed.

Prior-predictive cells: each n/allocation/scale combination (twelve cells),
β drawn from the same zero-centred Gaussian prior used to fit that dataset.
Never label oracle fixed-truth experiments SBC.

Master seed 20261115, stable cell/replication streams. Fitted prior IDs cannot
enter fixed-truth data seeds. Common Gaussian draws across priors and methods
are independent of outcomes. Dataset IDs, truths and arm counts must match
across priors. Checkpoints use schema v2 and retain complete dataset blocks;
reject old schemas, duplicate/incomplete blocks, altered data/configuration,
source hashes or backend versions. Safe TSV/TOML parsing only.

Pilot: 50 datasets per base cell. Main: 1,000 per cell; fixed-truth 8,000
datasets/96,000 fits, prior predictive 12,000 datasets/12,000 fits. Independent
method evaluations on shared datasets are not independent replications.
Fidelity endpoints/moments use prespecified replication IDs 1:100; their
uncertainty uses that subset's denominator. Pilot endpoints may use a smaller
explicit subset for cost estimation, without changing confirmatory selection.

### References and gates

Reference A is the existing coefficient-space adaptive integrator. Reference B
is a separate probability-space implementation. For p₀,p₁ in (0,1), set
b = [logit(p₀),logit(p₁)−logit(p₀)]. Its unnormalised log density is

log N(b;μ,V) + y₀ log p₀ + (n₀−y₀)log(1−p₀)
+ y₁ log p₁ + (n₁−y₁)log(1−p₁)
− log[p₀(1−p₀)p₁(1−p₁)].

Binomial coefficient constants cancel in posterior normalisation. Compute the
CDF by integrating p₁≤p₀+u with clipped limits and split domain; compute the
survival probability directly too. Endpoint density limits are zero for proper
Gaussian priors. Reference B must not call A's density or integration helpers.

Both require tolerance refinement, normalisation/closure checks and integrated
inner/outer error diagnostics. Own error estimates ≤1e-5; cross-reference CDF
agreement ≤5e-5, quantile agreement ≤1e-4 in arm units. These estimates are not
rigorous enclosures. Unresolved reference comparisons remain unavailable.
Do not widen numerical gates after seeing results.

Primary engine settings remain maxiter=1000, tol=1e-10. Reuse v1 checked CDF
rules: posterior 1e-5/1e-8 versus 1e-7/1e-10, gates 5e-5; Gaussian rules
1e-7/1e-10 versus 1e-9/1e-12, gates 5e-6. Retain boundary ambiguity and failures.

### Metrics and decisions

Report conditional coverage, all-attempt inclusion, availability/failure reasons,
paired LRVB minus raw VI differences and paired prior comparisons. Width and
fidelity metrics exclude unavailable values with explicit finite counts.
Primary prior contrasts are each zero-centred scale against s=1.5 and each
oracle offset against the same-scale oracle-centred prior; other pairs are
exploratory. No omnibus superiority test or multiplicity-unadjusted declaration
of a winning prior. Fixed-truth coverage is descriptive, not a correctness gate.

Fidelity subset: reference content of ideal Gaussian policy intervals, endpoint
error, policy mean error and covariance discrepancy. Keep approximation error
relative to each prior's posterior separate from causal-truth error. Report
reference errors/availability separately from known-truth Gaussian-law coverage.
Prior predictive: PIT distributions and 50/80/95% central inclusion.

Public-API subset: 100 datasets per base cell, zero-centred priors, 20 independent
draw streams, draw counts 500/2000/8000 including current default 2000. Fit once
per dataset/prior where supported. Resample datasets rather than individual
streams for aggregate MC uncertainty. Separate endpoint simulation variability
from posterior uncertainty and model/approximation error.

Timing: BLAS=1; Julia project environment with threads. Report compilation,
fitting, LRVB correction, policy draws and reference integration separately,
with hardware/versions and allocations. Matched numerical accuracy/statuses
are prerequisites for any optimisation claim.

### Development and confirmation

The [decision record](BAYESIAN_IMPROVEMENT_DECISIONS.md) lists concrete package
remedies. A first candidate addresses absent public endpoint Monte Carlo
diagnostics, without claiming improved posterior approximation or changing priors.
Write a failing contract first, then prove/test the diagnostic independently.
Preserve existing statuses and unavailable-interval rules. No truth-dependent
runtime rule or silent covariance fallback is permitted.

Run an all-cell pilot and record separate coverage/endpoint budgets before the
main study. Record protocol/source hashes and amendments before confirmatory
execution. Changes made using these results are development changes; confirm
them later on untouched scenarios specified in the parent plan. A complete
factorial study is not a prerequisite for delivering a separately justified,
analytically checked diagnostic improvement, but unfinished studies must remain
explicitly unfinished.

### Pilot amendment A1: dimensionless moment diagnostics

Declared after the original fidelity pilot and before amended execution. The
pilot retains 75 unavailable moment comparisons out of 288 selected method rows.
Its raw-moment error gate compared mixed mass/first-/second-moment units with a
probability tolerance. In one reproducible case, mean and covariance refinements
were below 4e-7 while the coarse raw-moment aggregate estimate exceeded 5e-5.
This is not evidence of inaccurate posterior moments or a reason to relabel the
original archive. Generation sources have been copied and checked against all
recorded source hashes before this amendment.

Use preliminary fine coefficient moments only to choose a common centre and
positive marginal SDs. Hold that transformation constant in both quadrature
rules. Integrate the dimensionless vector [1,u₀,u₁,u₀²,u₀u₁,u₁²], where
u=(β−centre)./SD; retain integrated inner and outer error estimates. Reconstruct
reported coefficient moments from these centred integrals. Gaussian-law moments
remain analytic.

For combined integral error estimate e and mass M, require 0≤e<M. A normalised
raw component m has propagated estimate d=e/(M−e)(1+|m|). For covariance entry
ij, use d_second_ij+|mean_i|d_mean_j+|mean_j|d_mean_i+d_mean_i*d_mean_j.
Require maximum standardised mean estimate ≤5e-4 and relative Frobenius
covariance estimate ≤5e-4 for each rule; require mean and covariance refinement
≤5e-4 on the same standardised scale. These remain heuristic numerical error
estimates, not rigorous enclosures. Normalisation, CDF, tail and endpoint gates
are unchanged. Validate against analytic Gaussian moments with large nonzero
centres, then rerun the same pilot datasets into a new archive, not over the old
one. Do not infer population-wide approximation quality from the amended pilot.

### Pilot amendment A2: independent metric availability

A summary regression fixture shows that ambiguity in the 95% truth-inclusion
decision must not discard independently validated endpoint/content/moment
metrics. Use each metric's explicit availability flag when present; retain the
previous global-availability rule for legacy records without those flags.
Coverage and all-attempt inclusion definitions remain unchanged. Paired fidelity
comparisons require both metric flags, rather than both 95% coverage flags.
Archive postprocessing code separately from original generation code; disclose
any regenerated summaries and verify that this does not alter raw records.

### Main-run budget and execution lock (19 November 2026)

Locked prior to running the 1,000-replication studies:

- **Scope**:
  - Paired fixed-parameter factorial (`--experiment fixed_parameter --profile validation`): 8 cells × 1,000 replications × 12 priors × 2 methods (96,000 fits; 100 fidelity replications per cell; 4,096 common descriptive draws).
  - Prior-predictive calibration (`--experiment prior_predictive --profile validation`): 12 cells × 1,000 datasets (12,000 fits; 100 fidelity replications per cell; 4,096 draws).
- **Amendments & schema**: retain A1 (dimensionless centred SD-scaled moment quadrature) and A2 (independent metric availability) under schema v2 checkpoints (`--resume` enabled). Original `seed=20261115` pilot/public archives remain untouched; the 19 November 2026 fresh-seed replication (`seed=20261119` + `Xoshiro`) closes the default-draw lower-endpoint check without altering seeds or defaults.
- **Compute & storage guardrails**: run on an idle machine with `OPENBLAS_NUM_THREADS=1` (enforced via `BLAS.set_num_threads(1)` in the driver) and `--threads=auto`; write outputs to local gitignored directories (~200–400 MB uncompressed per study).
- **Execution commands**:
  - Fixed-parameter factorial:
    `OPENBLAS_NUM_THREADS=1 julia --project=test/stress --threads=auto test/stress/policy_prior_factorial.jl --experiment fixed_parameter --profile validation --resume --out /tmp/ct-validation-fixed-1000`
  - Prior-predictive calibration:
    `OPENBLAS_NUM_THREADS=1 julia --project=test/stress --threads=auto test/stress/policy_prior_factorial.jl --experiment prior_predictive --profile validation --resume --out /tmp/ct-validation-prior-1000`
