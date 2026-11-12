## Bayesian inference contract

The first release conditions on supplied target rows, declared policy constants
and a fixed regression basis. It supplies exact Gaussian inference; it does not
replace existing EIF uncertainty or establish double robustness.

### Regression

Use beta | sigma² ~ Normal(mu0, sigma² V0), sigma² ~ InvGamma(a0,b0),
with density proportional to s^(-a0-1) exp(-b0/s), V0 positive definite,
a0 > 1 and b0 > 0. The shape restriction ensures finite prior covariance.

For observed outcomes only, Vn = (V0⁻¹ + X'X)⁻¹,
mun = Vn(V0⁻¹ mu0 + X'y), an = a0 + n_observed/2, and
bn = b0 + ((y-X mun)'(y-X mun) + (mun-mu0)'V0⁻¹(mun-mu0))/2.
Use factorisation solves. The marginal coefficient posterior has df = 2an,
scale (bn/an)Vn, and covariance (bn/(an-1))Vn. Scale is not covariance.

Policy contrasts are named `:policy_vs_observed`: each row of C averages
x(d_delta(A),W)-x(A,W) over target rows. The joint posterior is the linear
Student-t projection C beta. Identity policy gives an exact zero contrast;
dependent grid contrasts may have singular covariance. Only discrete and
fixed-bound smooth policies are supported initially. No simultaneous bands.

Missing Y is marginalised under declared MAR assumptions; predictive draws of
missing outcomes contain additional observation noise and are a separate output.
Missing treatment/covariates are rejected. Prior scales are supplied by callers,
never inferred from outcomes. A proper prior can regularise rank-deficient
designs without demonstrating causal identification.

Cross-fitted Bayesian nuisance candidates return posterior means only. Their
fits are distinct from full-sample g-computation fits. Never add posterior
covariance to EIF covariance. Certificates must match treatment, outcome and
the actual adjustment set. `graph_identification_status` records this claim;
`outcome_model_status` is always `:user_assumed`, including for the default basis.
The conservative `identification_status` summary stays `:user_assumed` for a
custom basis without discarding the validated graph status or certificate.
Interactions do not themselves invalidate adjustment; no certificate establishes
the adequacy of a default or user-supplied regression design.

### Dynamics

Known linear transitions and observations with known positive-definite noise
use exact Gaussian sum-product inference. Compare with independent Kalman/RTS
calculations. Retain filtered and predicted moments and smoothing gains Jt.
Cross-covariance Cov(x_t,x_(t+1)|Y) = Jt P_(t+1|T); recursively recover longer
lags. Missing measurement components use matching observation rows and noise
submatrices. An online update starts from the previous final filtered state
and consumes only new observations.

### Later approximations

Logistic inference has a separate specification in `LOGISTIC_INFERENCE.md`.
Count/hierarchical and uncertain-parameter dynamical models require separate
specifications and calibration before support is claimed. LRVB requires
supervisor-approved sufficient-statistic coordinates and derivatives of a
converged update map. It supplies corrected moments, not a corrected posterior
density. Never relax exact Gaussian fixtures to accommodate an approximation.

### Validation

Analytic Gaussian and dense joint-state fixtures are independent correctness
oracles. Separate prior-predictive calibration from fixed-parameter coverage;
report Monte Carlo uncertainty. Compare warmed runtime, compilation, allocations,
accuracy and failures. MCMC is an optional reference, never a production dependency.
