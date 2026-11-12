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

### Deterministic accuracy

All nine reference fixtures reached moment convergence; broad-prior cases
required 640 nodes per dimension. For the prespecified moderate fixture,
maximum mean error was 0.0836 reference posterior SD. Relative covariance error
was 13.73% for raw VI and 5.46% for LRVB, passing the prespecified bounds of
0.20 SD and 25% respectively. Response derivatives also agree with fully
reconverged tilted fits at perturbations 1e-4 and 1e-5.

The all-one and separated fixtures retained mean errors of 0.231 and 0.237
posterior SD, exceeding the moderate-fixture mean criterion. LRVB covariance
errors there were 2.08% and 1.98%. A symmetric zero-mean fixture has exactly zero
correction and retains 7.09% covariance error. These outcomes show that improved
covariance cannot repair every variational mean or tail approximation.

### Calibration

Each experiment contains 500 independent n=20,p=2 datasets. All 1,000 fits
and quadrature moment checks were available; there were no numerical failures.
Intervals have nominal 95% content. Prior-predictive and fixed-parameter checks
answer different questions.

| Experiment | Method | Coefficient 1 / 2 coverage | Policy coverage |
| --- | --- | --- | --- |
| Prior predictive | Raw VI | 91.2% / 92.0% | 94.0% |
| Prior predictive | LRVB | 94.4% / 96.0% | 96.0% |
| Prior predictive | Quadrature grid | 94.2% / 95.0% | 96.4% |
| Fixed parameters | Raw VI | 94.4% / 96.2% | 94.2% |
| Fixed parameters | LRVB | 97.0% / 97.4% | 95.8% |
| Fixed parameters | Quadrature grid | 94.6% / 96.8% | 96.6% |

Binomial Monte Carlo standard errors are approximately 0.7–1.3 percentage
points. Fixed-parameter CDF ranks are not simulation-based calibration ranks.
Quadrature moment convergence does not certify weighted-grid quantiles; those
coverage columns remain numerical-reference diagnostics. Policy summaries
integrate the Gaussian moment approximation using quadrature to remove draw
noise. The public Monte Carlo pushforward is separately checked for shared-grid
dependence, identity shifts, reproducibility and mean MCSE.

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
