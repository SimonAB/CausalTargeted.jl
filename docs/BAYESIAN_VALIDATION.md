## Exact Gaussian validation

Local development run: Julia 1.13.3, 20 threads. Reproduce from an environment
containing the package and test dependencies with
`julia --project=. --threads=auto test/stress/bayesian_calibration.jl`.
The seed is 431; each experiment uses 500 independent datasets, n=40, p=3.

| Experiment | Nominal | Coverage | MCSE | Mean width | RMSE |
| --- | --- | --- | --- | --- | --- |
| Prior predictive | 95% | 93.8% | 1.08 percentage points | 0.6011 | 0.1607 |
| Fixed parameters | 95% | 91.8% | 1.23 percentage points | 0.6205 | 0.1725 |

The lower fixed-parameter coverage is retained as evidence, not tuned away.
Exact posterior computation does not guarantee nominal repeated-sampling
coverage for every fixed parameter and prior. Prior-predictive calibration
integrates over the declared prior and answers a different question.

The warmed full fit on one n=40,p=3 dataset averaged 35.87 microseconds over
1,000 fits and allocated 33,616 bytes per fit. This local timing excludes
compilation and is not a claim about arbitrary datasets or hardware.

Analytic unit fixtures independently check coefficient posterior location,
Student-t scale versus covariance, degrees of freedom, pointwise quantiles,
paired policy-grid covariance, proper-prior collinear designs and observed-data
MAR likelihoods. The stress script reports posterior-CDF decile counts as well
as coverage. Inspect those diagnostics alongside scientific prior sensitivity.
