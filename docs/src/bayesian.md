## Exact Bayesian Gaussian estimation

`fit_bayesian_gcomp` fits a Gaussian outcome regression with a declared
Normal–inverse-Gamma prior. Coefficients have a joint Student-t posterior after
integrating residual variance. The prior covariance of coefficients conditional
on residual variance is `sigma2 * covariance`; the constructor's `shape > 1`
restriction ensures finite marginal prior covariance. Scientific prior scales
are supplied explicitly and are never estimated from outcomes.
The conjugate parameterisation follows
[Murphy's Gaussian analysis](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).

```julia
using CausalTargeted, DataFrames, LinearAlgebra
prior = ConjugateGaussianPrior(mean = zeros(3),
    covariance = 4.0 * Matrix{Float64}(I, 3, 3), shape = 3.0, scale = 2.0)
fit = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
policy = smooth_bounded_logit_shift_policy(0.0, 1.0)
effect = bayesian_policy_contrast(fit, data, policy; deltas = [0.0, 0.2])
```

The default numeric design is intercept, treatment, then adjustment columns.
Categorical columns use the fitted schema. A custom fixed basis can include
interactions. The result describes `:policy_vs_observed`: the mean fitted
outcome under a policy minus that under the observed exposures on the supplied
target rows. Target rows, basis and policy constants are conditioned on.

The entire grid shares one coefficient posterior. Its covariance therefore
retains dependence between shifts, including legitimate rank deficiency. For
unknown variance use Student-t quantiles with `2 * shape` posterior degrees of
freedom. The Student-t scale matrix and covariance are different quantities.
These are model-based credible intervals, with no double-robustness claim.
Existing EIF inference and its eligibility checks remain separate.

### Missingness and adjustment

Missing outcomes require explicit Gaussian MAR handling. Fit the observed-data
likelihood; predictive draws of missing outcomes are separate from the posterior
of a mean effect. Missing treatment or adjustment values are unsupported.
Select `handle_missing=:mar` in the fit and request a draw separately with
`predictive_missing_outcomes(fit, data; rng)`. A custom fixed basis has signature
`(predictor_data, treatment_values) -> matrix`, retaining its column meanings.
With `observation_variance=...`, inference conditions on known residual variance
and the coefficient posterior is Gaussian rather than Student-t.

`graph_identification_status` records whether a matching certificate establishes
the query and actual adjustment set; without one it is `:user_assumed`.
`outcome_model_status` is always `:user_assumed`, for both default and custom
bases. A graph certificate does not establish regression adequacy. Interactions
do not themselves change the adjustment set. The conservative summary
`identification_status` stays `:user_assumed` for a supplied basis, while the
validated graph status and certificate remain available. Prior regularisation
can make rank-deficient models computable without demonstrating identification.

### Cross-fitted nuisance prediction

`BayesianGaussianLearner(; prior)` is an optional candidate in the existing
Super Learner. It fits each training fold separately and returns posterior mean
predictions. Its fitted state is distinct from a full-sample g-computation fit.
Coefficient covariance does not enter the existing EIF standard error.

### Validation

Independent analytic fixtures test posterior location, scale, covariance and
quantiles. The package `test/stress/bayesian_calibration.jl` script
separately reports prior-predictive calibration and fixed-parameter coverage,
with Monte Carlo uncertainty. Neither favourable simulation results nor a proper
prior guarantee validity under misspecification or poor overlap.
