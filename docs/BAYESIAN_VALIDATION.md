## Exact Gaussian validation

Local development run: Julia 1.13.3, 20 threads. Reproduce from an environment
containing the package and test dependencies with
`julia --project=. --threads=auto test/stress/bayesian_calibration.jl`.
The seed is 431; each experiment uses 500 independent datasets, n=40, p=3.

| Experiment | Nominal | Coverage | MCSE | Mean width | RMSE | Posterior CDF mean |
| --- | --- | --- | --- | --- | --- | --- |
| Prior predictive | 95% | 93.8% | 1.08 percentage points | 0.6011 | 0.1607 | 0.5117 |
| Fixed parameters | 95% | 91.8% | 1.23 percentage points | 0.6205 | 0.1725 | 0.5064 |

Posterior-CDF decile counts (ten bins for the 500 probability-integral-transform
ranks of the treatment coefficient) are

- prior predictive: 47, 45, 51, 50, 44, 56, 51, 53, 53, 50
- fixed parameters: 69, 35, 54, 48, 44, 36, 42, 52, 60, 60

The prior-predictive coverage sits 1.1 Monte Carlo standard errors below 95%,
with nearly uniform deciles. That matches the conjugate calibration check.
The fixed-parameter coverage sits 2.6 Monte Carlo standard errors below 95%,
with heavy outer deciles rather than a shifted centre. Exact posterior
computation does not guarantee nominal repeated-sampling coverage for every
fixed parameter and prior. The shortfall is retained as evidence, not tuned
away. Prior-predictive calibration integrates over the declared prior and
answers a different question.

The warmed full fit on one n=40,p=3 dataset averaged 35.87 microseconds over
1,000 fits and allocated 33,616 bytes per fit. This local timing excludes
compilation and is not a claim about arbitrary datasets or hardware.

Analytic unit fixtures independently check coefficient posterior location,
Student-t scale versus covariance, degrees of freedom, pointwise quantiles,
paired policy-grid covariance, proper-prior collinear designs and observed-data
MAR likelihoods. The stress script reports posterior-CDF decile counts as well
as coverage. Inspect those diagnostics alongside scientific prior sensitivity.
