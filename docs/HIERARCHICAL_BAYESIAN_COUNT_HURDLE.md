# Hierarchical Bayesian Count and Hurdle Inference Contract

This document records the mathematical and software contract for one-level
random-intercept hierarchical Bayesian count (`:poisson`, `:nb2`) and two-part
hurdle (`fit_bayesian_cluster_hurdle`) models in `CausalTargeted.jl`.

## 1. Hierarchical count models (`ClusterCountPrior`, `fit_bayesian_cluster_count`)

For clusters $k = 1, \dots, K$, member $i \in k$, and non-negative integer
counts $Y_{ki} \in \{0, 1, 2, \dots\}$ with $\eta_{ki} = x_{ki}^\top \beta + b_k$,
$\mu_{ki} = \exp(\eta_{ki})$:

- **Priors**:
  $$\beta \sim \mathcal{N}(\mu_0, V_0), \qquad b_k \mid \sigma_b^2 \sim \mathcal{N}(0, \sigma_b^2), \qquad \sigma_b^2 \sim \operatorname{InvGamma}(a_b, b_b)$$
  and for `family = :nb2`, an explicit normal prior on log-dispersion:
  $$\log\theta \sim \mathcal{N}(\mu_{\log\theta}, \sigma_{\log\theta}^2), \qquad \operatorname{Var}(Y_{ki} \mid \mu_{ki}, \theta) = \mu_{ki} + \mu_{ki}^2 / \theta.$$
- **Structured Laplace–Gaussian VI (`q(β, b) q(σ_b⁻²) [q(log θ)]`)**:
  Updates the joint Gaussian block $q(\beta, b)$ using weighted Schur-complement
  elimination (zero dense $N \times N$ or $K \times K$ matrices) with exact
  log-likelihood weights $W_{ki} = -\partial^2 \ell_{ki} / \partial \eta_{ki}^2$,
  exact $q(\sigma_b^{-2})$ updates, and (for `:nb2`) a 1D Laplace update for
  $q(\log\theta)$. Exact affine design standardisation transforms $(\mu_0, V_0)$
  without changing the model.
- **Exact marginal mean over $b_k \sim \mathcal{N}(0, \sigma_b^2)$**:
  $$\mathbb{E}[Y_{ki} \mid x_{ki}, \beta, \sigma_b^2] = \exp\!\bigl(x_{ki}^\top \beta + \tfrac{1}{2}\sigma_b^2\bigr).$$
  Unlike setting $b_k = 0$, marginalisation preserves the exact log-normal
  factor $\exp(\sigma_b^2 / 2)$.
- **PSIS refinement (`correction = :psis`)**:
  Evaluates each cluster's exact marginalised log-likelihood
  $\log p(y_k \mid \beta, \sigma_b^2, [\theta])$ by 21-node adaptive
  Laplace-centred 1D Gauss–Hermite quadrature and applies Student-$t$ proposal
  PSIS gates ($\hat k < 0.7$, $\mathrm{ESS} \ge 400$, $\mathrm{MCSE}/\mathrm{SD} \le 5\%$).

## 2. Hierarchical two-part hurdle models (`ClusterHurdlePrior`, `fit_bayesian_cluster_hurdle`)

For zero-inflated / hurdle counts $Y_{ki} \in \{0, 1, 2, \dots\}$:

- **Part 1 (Presence)**: $I_{ki} = \mathbb{I}(Y_{ki} > 0) \sim \operatorname{Bernoulli}(\sigma(x_{ki}^\top \beta_1 + b_{1,k}))$, fitted via `fit_bayesian_cluster_logistic`.
- **Part 2 (Positive intensity)**: on $Y_{ki} > 0$, zero-truncated Poisson
  (`:poisson`) or zero-truncated NB2 (`:nb2`) with cluster random intercept
  $b_{2,k} \sim \mathcal{N}(0, \sigma_{b,2}^2)$ (independent of $b_{1,k}$ in
  this release).
- **Exact marginal hurdle expectation** (25-node Gauss–Hermite on both parts):
  $$\mathbb{E}[Y_{ki} \mid x_{ki}, \theta] = \mathbb{E}_{b_1}[\sigma(x_{ki}^\top \beta_1 + b_1)] \times \mathbb{E}_{b_2}\!\left[\frac{\exp(x_{ki}^\top \beta_2 + b_2)}{1 - p_0(x_{ki}^\top \beta_2 + b_2, \theta)}\right].$$
- `bayesian_policy_contrast(::BayesianClusterHurdleFit, ...)` propagates
  uncertainty jointly across shared draws, returning `overall` (combined count
  contrast), `presence` (binary prevalence contrast), and `intensity`
  (positive-part contrast), withholding intervals and covariance if either part
  fails to converge or fails PSIS gates.
