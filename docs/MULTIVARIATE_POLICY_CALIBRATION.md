## Multivariate policy calibration protocol

### Goal and scope

Measure repeated-sampling behaviour of the raw variational and LRVB Gaussian
policy laws against an independent adaptive bivariate posterior reference.
This is validation code; estimator equations and numerical acceptance gates
are held constant during the experiment. Book implementations are excluded.

The model is Bernoulli(logistic(β₀ + β₁ A)) with a caller-declared Gaussian
prior on both coefficients. The supplied binary exposure cohort is held
constant. The target is assignment-to-one versus observed treatment:

ψ = fraction(A = 0) × [logistic(β₀ + β₁) − logistic(β₀)].

Both exposure levels are present. This is neither an arm-versus-arm cohort
average nor longitudinal TMLE. Identification and the outcome model are
user-assumed. No outcome-derived prior scaling is used.

### Prespecified cells

Use n = 20 and 50. Fixed-parameter experiments include balanced exposure,
rare controls (one untreated row), separated-outcome stress (β = [-3, 6])
and a strong prior (covariance 0.25 I). Other fixed truths are [-0.3, 0.9],
with covariance 2.25 I except the strong-prior cell. All prior means are zero.
Rare controls test weak information and design imbalance; this is not a
general correlated-covariate or confounding validation. Prior-scale cells have
independent datasets and do not provide paired prior-choice comparisons.

Prior-predictive experiments use balanced, rare-control and strong-prior cells,
with β sampled from the declared prior. The separated fixed truth is not
relabelled as a separate prior-predictive experiment. Record realised complete,
quasi and single-class separation. There are 14 validation cells. Use 500
independent datasets per cell; a balanced n = 20 screen uses 50 per experiment.

### Numerical and statistical contracts

- Coverage of an equal-tailed 95% interval can be evaluated without quantile
  inversion: its CDF at the known truth must lie in [0.025, 0.975]. For the
  positive cohort multiplier above, this is the CDF of the unscaled contrast
  at its corresponding unscaled truth. A degenerate cohort is rejected.
- Evaluate posterior and Gaussian-law CDFs under independently tightened
  adaptive integration rules. Check normalising mass, CDF/tail agreement and
  integration estimates separately, retaining integrated inner as well as outer
  quadrature error estimates. These are numerical diagnostics, not formal error
  bounds. Numerically ambiguous boundary decisions
  remain unavailable, rather than being silently rounded into coverage.
- Compress identical binary-design/outcome rows using positive integer
  frequency counts. The weighted likelihood must equal the unfolded
  Bernoulli likelihood; no survey-weight or effective-sample-size assumption
  is introduced. Verify this before using the optimisation.
- Fit failures, raw covariance failures, corrected covariance failures and
  reference failures have separate denominators. A failed adaptive reference
  does not suppress an otherwise computable Gaussian-law coverage result,
  because the generating policy truth is known analytically.
- Report conditional coverage, all-attempt inclusion, their binomial MCSEs,
  availability and failure reasons. All-attempt inclusion counts unavailable
  intervals as noncoverage and is labelled accordingly.
- Compare raw VI and LRVB with paired coverage differences and paired MCSEs.
  Prior-predictive policy PIT histograms are distinct from fixed-parameter
  CDF diagnostics; only the former have an SBC interpretation.
- Gaussian pushforward Monte Carlo means and widths are descriptive summaries
  with draw counts/mean MCSE recorded. They are not used to decide coverage.
  Reference widths and errors are unavailable unless separately computed.
- Keep deterministic per-cell/per-replication RNG streams, raw replicate rows,
  source/configuration fingerprints, backend versions and timings. Resume
  only matching configurations and complete method triples. A malformed or
  incomplete checkpoint must be rejected explicitly.

### Implementation and verification

1. `test/logistic_tail_reference.jl`: add frequency-count support, retaining
   legacy constructors, and prove unfolded/weighted likelihood and CDF parity.
2. `test/stress/policy_calibration_reference.jl`: checked CDF-at-truth helpers;
   test analytic Gaussian signs, numerical failures and boundary ambiguity.
3. `test/stress/policy_calibration_support.jl`: cells, simulation, compression,
   summaries and paired differences, with small deterministic unit tests.
4. `test/stress/multivariate_policy_calibration.jl`: CLI, checkpointing, raw
   records and reports. Pilot all cells before the full 500-replication run.
5. Document observed coverage and Monte Carlo uncertainty without tuning the
   estimators to these datasets. Numerical-reference gates are computational
   checks; fixed-parameter Bayesian coverage need not be nominal 95%.

Run Julia with the isolated `test/stress` environment and `--threads=auto`.
Set BLAS to one thread for these small matrices. Record compilation separately
from warmed replicate timings; do not infer a speed ratio against MCMC.
