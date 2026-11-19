# Hierarchical Bayesian Logistic Inference Contract

This document records the mathematical and software contract for one-level
random-intercept hierarchical Bayesian logistic regression in
`CausalTargeted.jl` (Stage 3 Option A of `plans/cluster-effects-bayesian-inference.md`).

## 1. Model and prior (`ClusterLogisticPrior`)

For clusters $k = 1, \dots, K$ with sizes $N_k$ ($N = \sum_{k=1}^K N_k$) and
binary outcomes $Y_{ki} \in \{0, 1\}$:

$$
Y_{ki} \mid \beta, b_k \sim \operatorname{Bernoulli}\bigl(\sigma(x_{ki}^\top \beta + b_k)\bigr),
\qquad
b_k \mid \sigma_b^2 \sim \mathcal{N}(0, \sigma_b^2),
\qquad
\beta \sim \mathcal{N}(\mu_0, V_0),
\qquad
\sigma_b^2 \sim \operatorname{InvGamma}(a_b, b_b),
$$

where $\sigma(\eta) = (1 + e^{-\eta})^{-1}$, $V_0$ is symmetric positive
definite, $a_b > 1$, and $b_b > 0$. Prior locations and scales must be supplied
explicitly; no outcome-dependent rescaling is performed.

Exact affine design standardisation ($X_{\text{std}} = X A$, $\beta = A\tilde\beta$)
transforms the coefficient prior via $\tilde\mu_0 = A^{-1}\mu_0$,
$\tilde V_0 = A^{-1} V_0 A^{-\top}$ and maps posterior moments back exactly.

## 2. Weighted Schur-complement structured VI

Using the Jaakkola–Jordan (2000) quadratic lower bound with per-observation
variational parameters $\xi_{ki} \ge 0$, $\lambda(\xi) = \tanh(\xi/2) / (4\xi)$
($\lambda(0) = 1/8$), and diagonal weights $W_{ki} = 2\lambda(\xi_{ki})$:

$$
q(\beta, b)\,q(\sigma_b^{-2})
$$

retains joint $(\beta, b)$ dependence via exact block-arrow Schur-complement
elimination (zero dense $N \times N$ or $K \times K$ allocations):

$$
Q_{\beta\beta} = V_0^{-1} + \sum_{k,i} W_{ki} x_{ki} x_{ki}^\top,
\qquad
s_{Wx,k} = \sum_{i \in k} W_{ki} x_{ki},
\qquad
d_k = \bar\tau_b + \sum_{i \in k} W_{ki},
$$

$$
\Lambda_\beta = Q_{\beta\beta} - \sum_{k=1}^K \frac{1}{d_k} s_{Wx,k} s_{Wx,k}^\top,
\qquad
h_\beta = V_0^{-1}\mu_0 + \sum_{k,i}(y_{ki} - \tfrac{1}{2})x_{ki} - \sum_{k=1}^K \frac{h_{b,k}}{d_k} s_{Wx,k},
$$

with $h_{b,k} = \sum_{i \in k}(y_{ki} - \tfrac{1}{2})$. Updates iterate:
1. $q(\beta, b)$ via the weighted Schur complement,
2. $\xi_{ki} = \sqrt{(x_{ki}^\top m_\beta + m_{b_k})^2 + x_{ki}^\top \Sigma_\beta x_{ki} + 2 x_{ki}^\top \operatorname{Cov}(\beta, b_k) + \operatorname{Var}(b_k)}$,
3. $q(\tau_b) = \operatorname{Gamma}\bigl(a_b + K/2,\; \text{rate} = b_b + \frac{1}{2}\sum_{k=1}^K (m_{b_k}^2 + \operatorname{Var}(b_k))\bigr)$,

tracking the exact Jaakkola–Jordan ELBO until relative tolerance `tol` (default `1e-8`) or `maxiter` (default `1000`). Reaching `maxiter` sets `status = :maxiter_reached` and withholds downstream intervals.

## 3. True marginal policy contrasts (nonlinear link)

For a logistic link, integrating over the cluster random-intercept distribution
differs from setting $b_k = 0$ (Jensen attenuation):

$$
\mathbb{E}[Y_{ki} \mid x_{ki}(\delta), \beta, \sigma_b^2]
= \int_{-\infty}^{\infty} \sigma\!\bigl(x_{ki}(\delta)^\top \beta + \sigma_b u\bigr)\,\phi(u)\,du.
$$

`bayesian_policy_contrast(::BayesianClusterLogisticFit, ...)` evaluates this
1D integral for every shared draw $(\beta^{(s)}, \sigma_b^{2(s)})$ via
probabilist's Gauss–Hermite quadrature (25 nodes, Golub–Welsch), supporting
both `:individual` and `:cluster` target weighting.

## 4. Exact marginalised log-posterior and PSIS refinement

For `correction = :psis` (via `CausalTargetedPSISExt`), each cluster's
marginal log-likelihood $\log p(y_k \mid \beta, \sigma_b^2) = \log \int p(y_k \mid \beta, b_k)\,\mathcal{N}(b_k; 0, \sigma_b^2)\,db_k$
is evaluated by adaptive Laplace-centred 1D Gauss–Hermite quadrature (21 nodes).
Global log importance ratios on $(\beta, \log\sigma_b^2)$ pass through the
same operational gates ($\hat k < 0.7$, $\mathrm{ESS} \ge 400$,
$\mathrm{MCSE}/\mathrm{SD} \le 5\%$); failed gates withhold intervals and
covariance without falling back to raw VI.
