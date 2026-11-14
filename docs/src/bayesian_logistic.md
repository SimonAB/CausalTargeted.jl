## Logistic inference with RxInfer

The optional logistic backend approximates a Bernoulli-logit regression with
an explicit Gaussian coefficient prior. GraphPPL specifies Gaussian
subproblems, which RxInfer solves inside Pólya–Gamma/Jaakkola updates. The
coefficient block retains its full covariance; factorisation separates it from
the auxiliary variables. This is approximate logistic inference.

```julia
using CausalTargeted, RxInfer, GraphPPL, DataFrames, LinearAlgebra, Random
data = DataFrame(A = repeat([-0.7, -0.3, 0.3, 0.7], 8),
    Y = repeat([0, 1, 0, 1], 8))
prior = GaussianCoefficientPrior(mean = zeros(2),
    covariance = Matrix{Float64}(I, 2, 2))
fit = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior)
effect = bayesian_policy_contrast(fit, data,
    smooth_bounded_logit_shift_policy(-1.0, 1.0);
    deltas = [0.0, 0.5], n_draws = 2000, rng = Xoshiro(42))
```

### Covariance and convergence

The default `covariance=:linear_response` differentiates the converged
variational fixed point. Both `variational_covariance` and
`corrected_covariance` remain available. The latter is a posterior covariance
approximation, not a corrected joint posterior density. In particular, the
correction vanishes at a zero-mean symmetric solution; it cannot repair every
error in variational uncertainty.

Convergence checks coefficient means, covariance and second moments. Diagnostics
also check response conditioning, stability, solve residuals and positive
covariance. Failed checks withhold uncertainty: `posterior_covariance` and
policy limits are NaN, while a labelled plug-in effect remains available.
`covariance=:variational` explicitly selects raw moments. It is never silently
substituted after a correction fails. Inspect `fit.status`, `fit.diagnostics`
and `effect.metadata.uncertainty`.
`:converged` denotes numerical convergence, not certified posterior accuracy.
The default `maxiter=1000` allows more Jaakkola updates before withholding
uncertainty. If status is `:maxiter`, increasing the cap gives the existing
convergence checks more time; it does not establish posterior accuracy.

Standardising continuous predictors while retaining the same coefficient prior
changes the Bayesian model. For coefficient order `[intercept, treatment, W]`,
let `Wz = (W - c) / s`, with a declared centre `c` and scale `s > 0`. The equivalent
coefficients satisfy `βz = T * β`, where `T = [1 0 c; 0 1 0; 0 0 s]`. Preserve the
specified Gaussian prior by transforming its mean and covariance as
`μz = T * μ` and `Vz = T * V * T'`. Centring can therefore introduce prior
dependence between the intercept and slope even when their original priors were
independent. Use the same declared transformation in fitting and target designs.
With this prior transformation, the full Gaussian Jaakkola updates preserve
logit means and second moments; standardisation does not guarantee faster
convergence. Choosing a different prior on the standardised scale is a separate
modelling decision.

Policy effects are nonlinear risk contrasts. A shared set of joint normal
coefficient draws is pushed through the fixed target cohort and policy designs.
With corrected covariance, this normal moment approximation is an additional
approximation. Results include paired grid covariance, pointwise quantiles and
Monte Carlo standard errors of the means. Increase `n_draws` to assess simulation
stability; the exact Gaussian release's Student-t formulas do not apply here.

The mean's `mcse` does not describe the variability of interval endpoints.
`effect.endpoint_mc_bounds.lower_endpoint` and `.upper_endpoint` each contain
`lower` and `upper` vectors: pointwise Monte Carlo confidence bounds for the
corresponding ideal Gaussian-law quantile. Set `mc_level=0.95` to request 95%
bounds (the default). They use binomial order statistics of the same coefficient
draws, without additional outcome noise or an extra fit.

```julia
lower_endpoint_mc = effect.endpoint_mc_bounds.lower_endpoint
upper_endpoint_mc = effect.endpoint_mc_bounds.upper_endpoint
# Compare the bounds after increasing n_draws with a fresh declared RNG.
```

These bounds describe simulation precision conditional on the fitted Gaussian
moments. They do not measure prior sensitivity, correct posterior approximation
error, or provide a second causal confidence interval. They apply separately
to each endpoint and policy, rather than simultaneously across the grid. At
very small draw counts they can span the full risk-difference support [-1,1].
Unavailable posterior uncertainty also makes these diagnostics unavailable.
The distribution-free order-statistic construction is described in
[NIST Technical Note 2119](https://doi.org/10.6028/NIST.TN.2119).

### Prior sensitivity

Convergence and a proper prior do not guarantee useful fixed-truth coverage.
Compare scientifically defensible priors on the same data, basis, target and
policy before interpreting a sensitive effect. Declare coefficient scales from
the scientific units; do not estimate them from the outcome or select a prior
because it performs well on a known simulation truth.

```julia
# Illustrative scales only; replace them with justified scientific choices.
prior_scales = [0.5, 1.5, 3.0]
prior_comparison = map(prior_scales) do scale
    declared_prior = GaussianCoefficientPrior(mean = zeros(2),
        covariance = scale^2 * Matrix{Float64}(I, 2, 2))
    fitted = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior = declared_prior)
    contrast = bayesian_policy_contrast(fitted, data,
        smooth_bounded_logit_shift_policy(-1.0, 1.0);
        deltas = [0.5], n_draws = 2000, rng = Xoshiro(42))
    (; scale, status = fitted.status, contrast)
end
```

Retain failed fits and unavailable intervals in that comparison. The common RNG
helps separate changes due to prior choice from simulation variability; it does
not establish which prior is scientifically correct. Greater numerical precision
cannot repair an inappropriate model, prior or identification assumption.

### Assumptions and validation

Graph identification, model adequacy, basis provenance and MAR assumptions retain
separate metadata. Priors never use outcome-derived scales. Missing predictors
and hard-clamped policies remain unsupported. A proper prior keeps separated
binary regressions computable without guaranteeing accurate posterior tails.
Discrete policies require factual and assigned treatment levels to occur in
the fitting data, including numeric levels. This check does not establish
conditional positivity within covariate strata.

The independently reviewed derivation and numerical gates are in
[the logistic contract](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/LOGISTIC_INFERENCE.md). The package tests compare
Gaussian subproblems with analytic algebra, variational updates with an
independent implementation, and response columns with fully reconverged tilted
fits. Optional reference scripts compare posterior moments with converged
low-dimensional quadrature and separate prior-predictive calibration from
fixed-parameter coverage. Independent adaptive integration checks scalar tails
and two deterministic nonlinear bivariate policy cases; unchecked grid-tail
results remain exploratory. No universal calibration or improvement is claimed.
See the [validation report](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/LOGISTIC_VALIDATION.md),
which retains mean errors under separation and the zero-mean correction limit.

Sources: [Durante and Rigon](https://arxiv.org/abs/1711.06999),
[Giordano, Broderick and Jordan](https://www.jmlr.org/papers/v19/17-670.html).

```@docs
GaussianCoefficientPrior
BayesianLogisticFit
fit_bayesian_logistic
```
