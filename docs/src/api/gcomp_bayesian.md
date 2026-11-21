# Parametric & Bayesian g-comp

```@meta
CurrentModule = CausalTargeted
```

Cross-fitted g-computation, formula-based parametric GLM and MixedModels
g-computation (including dedicated subject-integrated NB2 random-intercept
models), and conjugate or variational Bayesian g-computation.

`fit_parametric_gcomp` defaults to the GLM backend. Set `backend=:mixed` and
provide `id` after loading `MixedModels`, `FastGaussQuadrature`, `NLopt`, and
`SpecialFunctions` for supported mixed fits. See [Methods and estimators](../methods.md)
for supported model, prediction, and inference combinations. Target data can omit
the outcome; the repeated-outcome compatibility wrappers retain their existing
static-treatment and observed-visit contract.

## Cross-fitted & parametric GLM g-computation

```@docs
run_gcomp
ParametricGComputationFit
fit_parametric_gcomp
run_parametric_gcomp
gcomp_mean
gcomp_contrast
gcomp_interaction
bootstrap_gcomp_interaction
```

## Mixed-effects & NB2 random-intercept g-computation

`ProfiledNB2MixedModel` is exported as a compatibility alias for
`NB2RandomInterceptModel`.

```@docs
MixedModelGComputationFit
MixedGComputationResult
StratifiedMixedGComputationResult
mixed_g_computation
NB2RandomInterceptModel
fit_profiled_nb2
quadrature_diagnostics
validate_fixed_theta_nb2_likelihood
theta
converged
fitdiagnostics
random_intercept_variance
```

## Bayesian Gaussian g-computation

Fits a conjugate Gaussian outcome model or a one-level random-intercept Gaussian
outcome model via structured variational inference ``q(\beta, b) q(\sigma_\varepsilon^{-2}) q(\sigma_b^{-2})``
with exact Schur-complement block algebra and exact affine prior
reparameterisation, integrating over the cluster random-effect distribution when
evaluating marginal policy contrasts. Loading `PSIS` (`using PSIS`) activates
`CausalTargetedPSISExt` for `correction = :psis`, which withholds intervals and
covariance if operational gates (`khat < 0.7`, `ess >= 400`, `mcse / sd <= 0.05`)
fail rather than falling back to raw VI. See
[Bayesian Gaussian & cluster](../bayesian.md) for the non-clustered conjugate
contract.

```@docs
ConjugateGaussianPrior
BayesianGCompFit
fit_bayesian_gcomp
BayesianGaussianLearner
predictive_missing_outcomes
ClusterGaussianPrior
BayesianClusterGCompFit
fit_bayesian_cluster_gcomp
```

## Bayesian logistic g-computation

Non-clustered logistic regression (`fit_bayesian_logistic`) uses
Jaakkola–Jordan variational inference with linear-response variance correction
(LRVB); see [Bayesian logistic (RxInfer)](../bayesian_logistic.md). Clustered
logistic regression (`fit_bayesian_cluster_logistic`) fits a one-level
random-intercept Bernoulli–logit outcome model via Jaakkola–Jordan weighted
Schur-complement structured VI ``q(\beta, b) q(\sigma_b^{-2})`` (zero dense
``N \times N`` or ``K \times K`` allocations). Marginal policy contrasts
integrate over ``b_k \sim \mathrm{Normal}(0, \sigma_b^2)`` via 25-node
Gauss–Hermite quadrature (distinct from setting `b_k = 0`). Optional
`correction = :psis` uses a heavy-tailed Student-``t`` proposal on
``(\beta, \log \sigma_b^2)`` with 21-node adaptive Laplace-centred 1D
Gauss–Hermite marginalised cluster log-likelihoods and withholds intervals when
operational gates fail.

```@docs
GaussianCoefficientPrior
BayesianLogisticFit
fit_bayesian_logistic
ClusterLogisticPrior
BayesianClusterLogisticFit
fit_bayesian_cluster_logistic
```

## Hierarchical Bayesian count & hurdle g-computation

Fits one-level random-intercept Poisson, negative binomial (NB2), and two-part
hurdle models via weighted Schur-complement structured Laplace–Gaussian VI (zero
dense ``N \times N`` or ``K \times K`` allocations). Marginal policy contrasts
integrate over ``b_k \sim \mathrm{Normal}(0, \sigma_b^2)`` via exact log-normal
expectation (`exp(η + 0.5 σ_b²)`) for standard counts and 25-node Gauss–Hermite
quadrature for presence and zero-truncated intensity.

```@docs
ClusterCountPrior
BayesianClusterCountFit
fit_bayesian_cluster_count
ClusterHurdlePrior
BayesianClusterHurdleFit
fit_bayesian_cluster_hurdle
bayesian_policy_contrast
```
