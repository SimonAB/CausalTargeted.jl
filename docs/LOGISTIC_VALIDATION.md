## Logistic approximation validation

The optional GraphPPL/RxInfer backend was checked with Julia 1.13.3 and RxInfer
6.1.3. The standalone script is `test/stress/logistic_calibration.jl`; run it
in an environment with CausalTargeted and its test extras. It sets BLAS threads
to one for small matrices and records compilation separately. The independent
reference is Gaussian-prior product Gauss–Hermite quadrature, refined until
posterior moments converge. No variational fit supplies reference nodes.

From the package directory, prepare the separate validation environment with
`julia --project=test/stress --threads=auto -e 'using Pkg; Pkg.instantiate()'`,
then run `julia --project=test/stress --threads=auto test/stress/logistic_calibration.jl`.
`LOGISTIC_REPLICATIONS` controls the batch size (default 500 per experiment).
Its optional dependencies do not enter the production package's hard dependencies.

### Compatibility

Isolated Julia 1.13.3 environments passed the engine, public API, quadrature
and scalar-tail tests (170 assertions each) with RxInfer/GraphPPL versions
4.7.3/4.6.6, 5.5.3/4.9.0 and 6.1.3/4.9.0. The release CI matrix additionally
covers these RxInfer major versions on Julia 1.12. These are backend
compatibility checks, separate from posterior-accuracy checks.

### Deterministic accuracy

Thirteen fixtures exercise the approximating Gaussian against prior-centred
Gauss–Hermite moments, including near-collinear columns, a strong prior
(`V = 0.25 I`), moderately scaled predictors (`W = 10 A` under a unit prior),
and gram-scale predictors (`W = 500 A`). Broad-prior and scaled cases can
require 400–960 nodes per dimension. For the prespecified moderate fixture,
maximum mean error was 0.0836 reference posterior SD. Relative covariance error
was 13.73% for raw VI and 5.46% for LRVB, passing the prespecified bounds of
0.20 SD and 25% respectively. Response derivatives also agree with fully
reconverged tilted fits at perturbations 1e-4 and 1e-5.

Near-collinear and strong-prior fixtures stay well inside the moderate moment
bounds (mean errors about 0.021 and 0.011 SD; LRVB covariance errors 0.53% and
1.22%). The moderately scaled fixture reaches moment convergence at 400 nodes,
but raw VI understates slope variance (mean error 0.236 SD; covariance error
60% raw / 14% LRVB).

Coefficient and policy interval contents from these Hermite grids remain
exploratory diagnostics. Adaptive results below apply to their recorded
interval endpoints; a matching design alone does not validate Hermite interval
masses or grid quantiles. Moment convergence does not establish convergence of
indicator-weighted interval masses or weighted-grid quantiles; the Gaussian
policy endpoints on the Hermite path also use an unrefined 80-node rule. For
example, holding those endpoints constant, increasing the moderately scaled
Hermite reference from 400 to 960 nodes changed LRVB policy-interval content
from 0.9570 to 0.9518. Near-collinear, strong-prior and moderately scaled
fixtures receive separate adaptive moment, content and endpoint gates in the
adaptive tail section.

Gram-scale columns under a unit prior are a documented reference failure:
prior-centred Hermite does not reach moment convergence by 960 nodes, and the
variational loop can hit `maxiter` before returning usable moments. Rescaling
predictor columns with the matching Gaussian prior transformation preserves
logits at corresponding prior-centred Hermite nodes and leaves their weights
unchanged. It therefore does not repair this rule's resolution. Use an
independently validated reference, such as adaptive integration with separate
moment, CDF and endpoint refinement, on the design being scored.

The all-one and separated fixtures retained mean errors of 0.231 and 0.237
posterior SD, exceeding the moderate-fixture mean criterion. LRVB covariance
errors there were 2.08% and 1.98%. A symmetric zero-mean fixture has exactly zero
correction and retains 7.09% covariance error. These outcomes show that improved
covariance cannot repair every variational mean or tail approximation, and that
predictor scale can defeat both the approximation and the Hermite reference.

### Calibration

Each experiment contains 500 independent n=20,p=2 datasets. All 1,000 fits
and quadrature moment checks were available; there were no numerical failures.
Intervals have nominal 95% content. Prior-predictive and fixed-parameter checks
answer different questions.

| Experiment | Method | Coefficient 1 / 2 coverage |
| --- | --- | --- |
| Prior predictive | Raw VI | 91.2% / 92.0% |
| Prior predictive | LRVB | 94.4% / 96.0% |
| Fixed parameters | Raw VI | 94.4% / 96.2% |
| Fixed parameters | LRVB | 97.0% / 97.4% |

Binomial Monte Carlo standard errors are approximately 0.7–1.3 percentage
points. Fixed-parameter CDF ranks are not simulation-based calibration ranks.
The table concerns normal coefficient intervals, whose endpoints require no
numerical integration. Quadrature moment convergence does not certify
weighted-grid quantiles: increasing the policy grid from 40 to 240 nodes per
dimension changed a prior-only upper endpoint from 0.3827 to 0.3638 despite
stable moments. The batch script retains grid-tail results explicitly as
exploratory diagnostics; they are excluded from this calibration table and
are not release evidence for policy intervals. The public Monte Carlo
pushforward is separately checked for shared-grid dependence, identity shifts,
reproducibility and mean MCSE.

### Adaptive tail validation

Run `julia --project=test/stress --threads=auto test/stress/logistic_tail_validation.jl`
for the independent QuadGK checks (`LOGISTIC_TAIL_REPLICATIONS` defaults to
500 per experiment). They integrate the original likelihood and scientific
Gaussian prior directly, rather than a variational approximation. Quantile
inversion, endpoint CDF residuals and integration-tolerance refinement are
checked separately from posterior moments.

All twelve scalar fixtures passed these numerical tail checks, including
prior sensitivity, separation and missing outcomes. In an exactly collinear
design, scalar reduction supplies an independent joint-moment reference: LRVB
covariance error was 1.16%, with the orthogonal prior-only direction preserved.
Prior-to-posterior variance ratios and approximation errors remain diagnostics;
successful computation does not establish data identification.

Two deterministic bivariate checks use the nonlinear policy functional
`logistic(β₀ + β₁) - logistic(β₀)`, replacing A=0 with A=1. Conditional on β₀,
an analytic boundary reduces its CDF to nested adaptive integration. Tightening
integration tolerance from 1e-7 to 1e-9 left both endpoints unchanged to the
reported precision; maximum CDF residual was 4.35e-9 and the tight-rule
integration error estimate was below 5.95e-11. These estimates are numerical
checks, not rigorous interval bounds.

For the moderate fixture, the reference 95% interval was
[-0.06527, 0.48623]. The raw Gaussian approximation's interval contained
93.11% of the reference posterior; the LRVB Gaussian approximation contained
94.31%, with lower and upper tail probabilities of 1.94% and 3.74%. This is
posterior content for one fixture, not repeated-sampling coverage. Public
80,000-draw intervals matched independently integrated Gaussian-law tails to
within 0.00068 probability in both this fixture and a correlated prior-only
fixture. Correct sampling from an approximation does not remove its error.

#### Hard bivariate interval checks

Near-collinear, strong-prior and moderately scaled fixtures are scored with the
same nested adaptive rule for coefficient marginals and the policy contrast
`logistic(β₀ + β₁) - logistic(β₀)`. Moments, interval content and endpoints each
have their own refinement gate. The reference's own 95% intervals are checked
first. The actual raw VI and LRVB coefficient limits and nonlinear Gaussian-law
policy limits are then held constant while the original posterior is integrated
at working (`rtol=1e-5`, `atol=1e-8`) and tight (`rtol=1e-7`, `atol=1e-10`)
tolerances. Content change, maximum tail-probability change and maximum reported
integration error must each be at most 5e-4. The driver records these quantities
separately for every interval.

The nonlinear policy endpoints also receive a separate Gaussian-law check:
quantiles are inverted once at `rtol=1e-7`, `atol=1e-10`, then the same endpoints
are checked at `rtol=1e-9`, `atol=1e-12`. Both rules' tail residuals from 0.025
and the maximum reported integration error must be at most 5e-6. This endpoint
check concerns the approximating Gaussian law; the original posterior's tails
at those endpoints need not be 0.025. Reference, content or Gaussian endpoint
failures withhold approximation comparisons; unavailable fits fail the gate.
Attempted, available, unavailable, withheld and completed comparisons are
reported explicitly.

All six raw VI/LRVB comparisons across the three fixtures passed, with no
unavailable fits, withheld comparisons or failed gates. Across their 18 actual
intervals, maximum content change was 1.10e-7, maximum tail-probability change
1.09e-7 and maximum reported integration error 1.43e-6. Gaussian-law policy
endpoint residuals were at most 1.49e-8.

| Fixture | Method | Coeff. content (1 / 2) | Policy content | Mean error / SD | Rel. cov. error |
| --- | --- | --- | --- | --- | --- |
| Near-collinear | Raw VI | 0.948 / 0.948 | 0.950 | 0.021 | 1.65% |
| Near-collinear | LRVB | 0.949 / 0.949 | 0.952 | 0.021 | 0.53% |
| Strong prior | Raw VI | 0.947 / 0.948 | 0.948 | 0.011 | 1.88% |
| Strong prior | LRVB | 0.948 / 0.949 | 0.949 | 0.011 | 1.22% |
| Moderately scaled | Raw VI | 0.851 / 0.382 | 0.704 | 0.236 | 60.2% |
| Moderately scaled | LRVB | 0.953 / 0.956 | 0.966 | 0.236 | 14.2% |

On the strong-prior and near-collinear designs, LRVB improves covariance more than
interval content: coefficient and policy contents already sit near 0.95 under
raw VI, so the correction adds little there. On the moderately scaled design,
raw VI badly understates slope variance and collapses interval content on the
scaled coefficient and policy contrast; LRVB restores contents near 0.95–0.97
even though the shared variational mean remains 0.236 reference SD off. These
are posterior contents for fixed fixtures, not repeated-sampling coverage.

A separate scalar experiment used 500 prior-predictive datasets and 500
fixed-parameter datasets, with no fit or reference failures:

| Experiment | Adaptive reference | Raw VI | LRVB |
| --- | --- | --- | --- |
| Prior predictive | 95.6% | 94.4% | 96.0% |
| Fixed parameters | 96.6% | 96.0% | 96.8% |

Monte Carlo standard errors are 0.8–1.0 percentage points. The same content
applies to the scalar monotone policy contrast `logistic(β) - 1/2`.
This validates the references and sampling layer on specified cases; it does
not certify general multivariate policy calibration.

### Performance and limits

The warmed complete LRVB fit took 15.2 ms and allocated 10,325,824 bytes on the
small benchmark. First-call wall time was 7.92 s including compilation. The
500-dataset batches took 31.48 s and 34.78 s, including reference work. These
local measurements do not establish a speed ratio against MCMC or other models.

This validates the numerical implementation and demonstrates useful accuracy on
specified small examples. It does not establish universal coverage, accuracy
under separation or misspecification, or a corrected posterior density. Inspect
raw and corrected moments, prior sensitivity and convergence diagnostics before
using approximate policy intervals for a scientific conclusion.
