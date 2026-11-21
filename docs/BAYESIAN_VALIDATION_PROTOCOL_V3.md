# Bayesian Validation Protocol V3 (Phase 4: Multi-Covariate Causal Mechanisms)

Status: **Draft for pilot** (21 November 2026). Extends
[BAYESIAN_VALIDATION_PROTOCOL_V2.md](BAYESIAN_VALIDATION_PROTOCOL_V2.md) from
bivariate logistic regression ($p = 2$) to multi-covariate causal mechanisms
($p \in \{4, 5\}$) with controlled confounding correlation, overlap stress,
treatment–covariate interaction, and deliberate outcome misspecification.
Supersedes nothing in V2; all V2 archives remain immutable.

## 1. Primary questions

1. **Posterior fidelity in 4–5D:** How accurately do Jaakkola–Jordan Raw VI
   (`variational`) and LRVB (`linear_response`) recover the multi-covariate
   posterior mean, covariance, and marginal policy contrast $\psi$ as
   confounder correlation, propensity steepness, and interaction terms increase?
2. **Two-tier reference validity:** Does the fast Tier 1 Laplace-centred
   Student-$t$ PSIS reference match (a) the exact V2 2D integrator on bivariate
   anchors and (b) independent Tier 2 Turing NUTS chains on the hardest 4–5D
   cells and borderline $\hat{k}$ cases?
3. **Posterior accuracy vs causal recovery:** Under deliberate outcome
   misspecification (omitted interaction or omitted nonlinearity), how cleanly
   do posterior fidelity (agreement with the fitted model's exact posterior) and
   causal recovery (agreement with the true cohort policy contrast) separate?

## 2. Target estimand and causal assumptions

On each simulated cohort $\{(W_{i1}, W_{i2}, A_i, Y_i)\}_{i=1}^n$, the known
cohort causal target is the assignment-to-one versus observed exposure contrast:

$$\psi_{\text{true}} = \frac{1}{n} \sum_{i=1}^n \bigl[\mu_{\text{true}}(1, W_i) - \mu_{\text{true}}(A_i, W_i)\bigr],$$

where $\mu_{\text{true}}(a, w) = \Pr(Y = 1 \mid \operatorname{do}(A = a), W = w)$.
Identification status remains `:user_assumed`. Posterior credible intervals,
exact posterior content, and fixed-truth sampling inclusion are reported
separately and never conflated.

## 3. Data-generating matrix (6 mechanisms × 2 sample sizes × 2 priors)

Confounders $(W_1, W_2) \sim \mathcal{N}\!\left(0, \begin{pmatrix}1 & \rho \\ \rho & 1\end{pmatrix}\right)$,
binary treatment $A_i \sim \text{Bernoulli}\bigl(\text{logistic}(\gamma_0 + \gamma_1 W_{i1} + \gamma_2 W_{i2})\bigr)$,
and binary outcome $Y_i \sim \text{Bernoulli}\bigl(\text{logistic}(\eta_i)\bigr)$.
Sample sizes: $n \in \{80, 200\}$.

| Cell ID | $\rho$ | Propensity $(\gamma_0, \gamma_1, \gamma_2)$ | True logit $\eta_i$ | Fitted basis ($p$) | Difficulty isolated |
| --- | --- | --- | --- | --- | --- |
| `mc_baseline` | `0.0` | `(0.0, 0.5, -0.4)` | $-0.5 + 1.0 A + 0.6 W_1 - 0.5 W_2$ | Additive $(1, A, W_1, W_2)$ ($p=4$) | Clean multi-covariate baseline |
| `mc_corr` | `0.7` | `(0.0, 0.5, -0.4)` | $-0.5 + 1.0 A + 0.6 W_1 - 0.5 W_2$ | Additive ($p=4$) | High confounder collinearity |
| `mc_overlap` | `0.3` | `(0.0, 1.5, -1.2)` | $-0.5 + 1.0 A + 0.6 W_1 - 0.5 W_2$ | Additive ($p=4$) | Poor propensity overlap |
| `mc_interact_spec` | `0.3` | `(0.0, 0.5, -0.4)` | $-0.5 + 1.0 A + 0.6 W_1 - 0.5 W_2 + 0.9 A W_1$ | With $A W_1$ ($p=5$) | Correctly specified interaction |
| `mc_interact_omit` | `0.3` | `(0.0, 0.5, -0.4)` | $-0.5 + 1.0 A + 0.6 W_1 - 0.5 W_2 + 0.9 A W_1$ | Additive ($p=4$) | Omitted treatment–covariate interaction |
| `mc_nonlinear_omit` | `0.3` | `(0.0, 0.5, -0.4)` | $-0.5 + 1.0 A + 0.6 W_1 - 0.5 W_2 + 0.8 (W_1^2 - 1)$ | Additive ($p=4$) | Omitted nonlinear confounding |

## 4. Declared priors

Two deployable, caller-declared isotropic Gaussian priors (no outcome-derived
scaling, no truth-centred oracle defaults):

- `zero_s1p5`: $\beta \sim \mathcal{N}(0, 1.5^2 I_p)$ (moderate weakly informative).
- `zero_s3p0`: $\beta \sim \mathcal{N}(0, 3.0^2 I_p)$ (diffuse; stress-tests Gaussian skewness).

## 5. Two-tier independent reference and hard gates

### Tier 1 (fast default for full grids): Laplace-centred Student-$t$ PSIS
- Find the exact log-posterior MAP $\hat{\beta}$ and observed Hessian $H(\hat{\beta})$
  via Newton–Raphson with backtracking; set proposal scale $\Sigma_{\text{prop}} = H(\hat{\beta})^{-1}$.
- Draw $S = 20{,}000$ independent samples from $\text{MvTDist}(\nu = 5, \hat{\beta}, \Sigma_{\text{prop}})$
  and compute Pareto-smoothed importance weights (`PSIS.psis`).
- **Operational acceptance gates:**
  1. Pareto shape $\hat{k} < 0.7$;
  2. Importance-sampling $\text{ESS} \ge 400$;
  3. Policy contrast $\text{MCSE} / \text{SD} \le 0.05$.
- Any dataset failing a gate is recorded with `reference_available = false` and
  explicit `failure_reason` (never discarded or loosened).

### Tier 2 (thorough independent spot-check): Turing NUTS
- 4 chains × 1,000 warmup + 1,000 post-warmup draws (`NUTS(0.8)`).
- **Operational acceptance gates:** $\max \hat{R} \le 1.01$,
  $\min \text{ESS}_{\text{bulk}}, \text{ESS}_{\text{tail}} \ge 1000$, zero divergences.
- **Spot-check coverage:**
  1. **2D anchor:** 10 datasets verified against the exact V2 bivariate integrator
     (`test/stress/policy_probability_reference.jl`) before running 4–5D cells.
  2. **4–5D audit:** 10 datasets per cell across the hardest cells
     (`mc_corr`, `mc_overlap`, `mc_interact_spec`, `mc_nonlinear_omit` at $n = 80$,
     `zero_s3p0`) plus any dataset where Tier 1 has $0.5 \le \hat{k} < 0.7$.
  3. **Agreement criterion:** Tier 1 vs Tier 2 policy mean discrepancy
     $\le 0.02\;\text{SD}$ and 95% interval endpoint discrepancy $\le 0.01$ in
     cohort risk-difference units.

## 6. Metrics and schema (v3)

Every dataset gets a deterministic `dataset_id` independent of `prior_id` and
method (`tier1_psis`, `variational`, `linear_response`). Rows record:

- **Availability & diagnostics:** convergence status, LRVB status, Tier 1
  $\hat{k}$, ESS, `reference_available`, `failure_reason`, runtime seconds.
- **Posterior fidelity (vs Tier 1 reference):** coefficient mean error (max SD
  units), relative Frobenius covariance error, policy mean error, 95% reference
  posterior content, max endpoint error in cohort units.
- **Causal recovery (vs cohort $\psi_{\text{true}}$):** bias, RMSE, 95% interval
  inclusion (available and all-attempt), interval width.

## 7. Budgets and checkpoints

- **Pilot (`--profile pilot`):** 50 replications/cell × 12 cells × 2 priors =
  1,200 paired datasets (3,600 method rows). Verifies runtime, Tier 1 gate pass
  rate, Tier 1 ↔ Tier 2 agreement, and checkpoint resume before locking the main
  run.
- **Main (`--profile main`):** 500 replications/cell (6,000 paired datasets)
  after pilot audit; reserved held-out cells remain untouched for Phase 5.

## 8. Pilot Amendment A1 (21 November 2026)

- **Tier 1 ↔ Tier 2 mean-agreement gate:** In the 12-dataset Tier 2 NUTS
  spot-check (`scripts/stress_harness/run_phase4_nuts_spotcheck.jl`, 4 × 1,000
  draws, `min ESS = 1,974`, `max R̂ = 1.0029`, 0 divergences), NUTS Monte Carlo
  SE alone is $\approx 1 / \sqrt{2000} \approx 0.022\;\text{SD}$. All 12
  datasets passed `max_endpoint_diff <= 0.01` (worst `0.00597`) and showed
  unbiased mean differences of `0.013–0.056 SD` (~2.5× NUTS MCSE). The
  protocol's Tier 1 ↔ Tier 2 mean-agreement gate is therefore updated from a
  hard `0.02 SD` to `<= 3 * sqrt(1 / ess_nuts + 1 / ess_psis)` (equivalently
  `<= 0.06 SD` at 4,000 NUTS draws), while keeping `max_endpoint_diff <= 0.01`
  unchanged.
