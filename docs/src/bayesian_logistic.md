## Logistic inference with RxInfer

The development logistic backend approximates a Bernoulli-logit regression with
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

Policy effects are nonlinear risk contrasts. A shared set of joint normal
coefficient draws is pushed through the fixed target cohort and policy designs.
With corrected covariance, this normal moment approximation is an additional
approximation. Results include paired grid covariance, pointwise quantiles and
Monte Carlo standard errors of the means. Increase `n_draws` to assess simulation
stability; the exact Gaussian release's Student-t formulas do not apply here.

### Assumptions and validation

Graph identification, model adequacy, basis provenance and MAR assumptions retain
separate metadata. Priors never use outcome-derived scales. Missing predictors
and hard-clamped policies remain unsupported. A proper prior keeps separated
binary regressions computable without guaranteeing accurate posterior tails.

The independently reviewed derivation and numerical gates are in
[the logistic contract](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/LOGISTIC_INFERENCE.md). The package tests compare
Gaussian subproblems with analytic algebra, variational updates with an
independent implementation, and response columns with fully reconverged tilted
fits. Optional reference scripts compare posterior moments with converged
low-dimensional quadrature and separate prior-predictive calibration from
fixed-parameter coverage. No universal calibration or improvement is claimed.
See the [validation report](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/LOGISTIC_VALIDATION.md),
which retains mean errors under separation and the zero-mean correction limit.

Sources: [Durante and Rigon](https://arxiv.org/abs/1711.06999),
[Giordano, Broderick and Jordan](https://www.jmlr.org/papers/v19/17-670.html).

```@docs
GaussianCoefficientPrior
BayesianLogisticFit
fit_bayesian_logistic
```
