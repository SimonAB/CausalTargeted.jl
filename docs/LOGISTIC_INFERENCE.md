## Logistic variational inference contract

The likelihood is Bernoulli(logistic(X beta)), with beta ~ Normal(mu0,V0).
The prior covariance is positive definite and scientifically specified. There
is no residual-variance parameter. Missing outcomes contribute no likelihood
under the explicit ignorable-MAR option; missing predictors remain unsupported.

### Factorisation and computation

Use q(beta) product_i q(omega_i), retaining a full Gaussian coefficient block.
Pólya–Gamma expectations are w(r)=tanh(sqrt(r)/2)/(2sqrt(r)), with w(0)=1/4.
For kappa=y-1/2, Gaussian updates are S=(V0^-1+X'diag(w)X)^-1 and
m=S(V0^-1 mu0+X'kappa+t). Update r=diag(X S X')+(Xm).^2.
The vector t is a linear posterior tilt, used for response validation.

Each production Gaussian subproblem is specified in GraphPPL and solved by
RxInfer using pseudo-observations kappa/w and precisions w. This does not turn
the Gaussian graph's free energy into the logistic ELBO. No exact logistic
posterior claim is made. The default native BinomialPolya mean-only substitution
is not used as a substitute for these second-moment updates.

Convergence requires scaled changes in m, S and r below tol (default 1e-8),
with at most 200 iterations. Failure retains point summaries and withholds
requested uncertainty. The prior-only case is an exact Gaussian special case.

### Linear response

Let K=X S X', eta=Xm, g=dw/dr, H=K.^2+2diag(eta)Kdiag(eta).
The r-update Jacobian is J=-H diag(g), and B=2diag(eta)X S.
Solve (I-J)R=B and calculate SigmaLR=S-S X' diag(eta .* g)R.
These derivatives were independently checked against the augmented
mean-field fixed point. Near zero, g=-1/48+r/240-17r²/26880+O(r³).

Check the spectral radius using the symmetric similar matrix
sqrt(diag(-g)) H sqrt(diag(-g)). Reject radius >=1-1e-8, response condition
number >1e10, relative solve residual >1e-8, material covariance asymmetry
(relative error >1e-6), non-finite values or non-positive covariance. Round-off
symmetrisation is permitted; eigenvalue clipping is not. Keep the raw
variational covariance separately. Explicit covariance=:variational is an
alternative, not an automatic fallback after failed correction.

The current response calculation uses dense observation-space matrices:
memory grows quadratically and dense factorisation cost cubically with the
number of observed outcomes. It is intended for small datasets; the n=20
benchmark does not establish suitability for large cohorts.

When eta=0 the correction to mean response vanishes. This is a limitation of
this approximation, not evidence that the raw covariance is exact. Validate
response columns against fully reconverged tilted fits at steps 1e-4 and 1e-5.

### Policy uncertainty

Policy risk contrasts are nonlinear in beta. Push one shared set of joint
Gaussian coefficient draws through all fixed intervened designs. With LRVB
moments, this is an additional normal moment approximation; LRVB covariance
does not define a corrected joint posterior density. Return pointwise quantiles,
paired grid covariance, mean Monte Carlo error, plug-in contrast and explicit
approximation metadata. No outcome noise or population sampling is added.

### Validation gates

Independent one- and two-dimensional Gaussian-prior quadrature checks posterior
moments. Verify quadrature convergence across orders before using it as a
reference. Moderate deterministic fixtures require covariance relative error
<=0.25 and mean error <=0.2 reference posterior SD. These limits were chosen
before viewing benchmark results. Adversarial failures are retained, never
resolved by tuning the prior or weakening the limits.

Separate prior-predictive calibration and fixed-parameter coverage, each with
500 datasets. Include all-zero outcomes, separation, correlated predictors,
prior sensitivity and nonconvergence. Report failures and approximation error
alongside runtime. Quadrature grid quantiles need their own accuracy checks;
converged moments alone do not certify them.

Sources: [Durante and Rigon](https://arxiv.org/abs/1711.06999),
[Giordano, Broderick and Jordan](https://www.jmlr.org/papers/v19/17-670.html).
