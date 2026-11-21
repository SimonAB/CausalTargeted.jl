# API overview

```@meta
CurrentModule = CausalTargeted
```

Literature mapping lives in [Methods and estimators](methods.md); DOIs in [References](references.md).
Docstrings on exported symbols also carry `# References` sections where applicable.

## API sections

- [LMTP, sequential & survival](api/lmtp.md) — continuous LMTP grids and contrasts, shift policies, fold caching, targeting diagnostics, cluster-robust/simultaneous inference, and sequential/survival LMTP.
- [Discrete, hurdle, MSM & DiD](api/discrete_msm.md) — discrete LMTP, two-part hurdle LMTP, discrete bootstrap, repeated-outcome and parametric MSMs, MMRM, and difference-in-differences.
- [Parametric & Bayesian g-comp](api/gcomp_bayesian.md) — cross-fitted and parametric GLM/MixedModels g-computation (including NB2 random-intercept fits), non-clustered Bayesian Gaussian and logistic (LRVB) g-computation, and hierarchical Bayesian Gaussian, logistic, count, and hurdle models.
- [Super Learner & schemas](api/super_learner.md) — Super Learner fitting and libraries, small-*n* recommendations, cross-fitted nuisance interfaces, covariate schemas, design matrices, and estimand types.
- [Missingness & diagnostics](api/diagnostics_missingness.md) — missing-data handling, IPCW, posterior imputation and pooling, domain transportability, positivity reports, and sensitivity calibration.
- [Planning, certificates & plotting](api/planning_plotting.md) — temporal estimation plans, MTP execution planning, policy choice, identification certificates, run metadata, MTP effect curves, and synthetic DGPs.

## Module summary

```@docs
CausalTargeted
```

## Mediation

Mediation estimation lives in **CausalMediation.jl**. After `using CausalMediation`,
prefer `CausalMediation.run_mediation`, `CausalMediation.run_mediation_grid`, and
`CausalMediation.MediationSpec`. `CausalTargeted.jl` retains LMTP, g-computation,
MSM, DiD, certificates, and Super Learner workflows.
