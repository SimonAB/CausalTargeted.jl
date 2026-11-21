# Choosing an approach

```@meta
CurrentModule = CausalTargeted
```

This page maps common biological study designs, exposure types, and sample sizes
to the right estimator in `CausalTargeted`. It assumes you already have a causal
graph or adjustment set; if you are still deciding *which* covariates to adjust
for (or exclude), see [CausalDynamics.jl](https://simonab.github.io/CausalDynamics.jl/dev/orientation/).

## Where this package fits

| Task | Typical question | Package |
|------|------------------|---------|
| **Draw assumptions** | How do variables in my system influence one another? | [DAGMakie.jl](https://simonab.github.io/DAGMakie.jl/dev/) |
| **Identify & simulate** | Which covariates must I adjust for (or exclude)? | [CausalDynamics.jl](https://simonab.github.io/CausalDynamics.jl/dev/) |
| **Estimate effects** | Given data and an adjustment set, what is the effect of shifting an exposure or treatment? | **CausalTargeted.jl** (this package) |
| **Decompose pathways** | How much of an effect acts directly versus through an intermediate mechanism? | [CausalMediation.jl](https://simonab.github.io/CausalMediation.jl/dev/) |

## Why modified treatment policies and targeted learning?

Biological exposures are frequently **continuous** (parasite load, hormone
concentration, nutrient dose, temperature) or **repeated over time**. Asking
"what if every individual had zero parasites?" often violates *positivity*: in
some ages, seasons, or habitats, zero exposure never occurs, so any estimator is
forced to extrapolate beyond the data (Petersen et al., 2012).

A **modified treatment policy (MTP)** asks a realistic, data-supported question
instead: *"how would the population mean outcome change if each individual's
observed exposure were shifted by ``\delta`` (or scaled, or raised to a minimum
threshold)?"* (Díaz & van der Laan, 2012; Díaz et al., 2023).

Standard regression models assume a rigid functional form (such as linearity
with no interactions). **Targeted learning (TMLE / LMTP)** avoids that rigidity
by combining flexible cross-fitted machine learning (**Super Learner**; van der
Laan, Polley & Hubbard, 2007; Zheng & van der Laan, 2011; Chernozhukov et al.,
2018) with a targeting step that removes plug-in bias for the specific causal
estimand of interest (van der Laan & Rubin, 2006; van der Laan & Rose, 2011).

## Match your study design and data to an estimator

| Study design, exposure, or outcome | Causal estimand | Entry point in `CausalTargeted` | Basis / details |
|------------------------------------|-----------------|---------------------------------|-----------------|
| **Single time point, continuous exposure** (e.g. parasite load, biomarker, dose) | Shift curve ``\psi(\delta)`` under additive, multiplicative, or threshold shift | `run_lmtp_grid`, `additive_shift_policy`, `multiplicative_shift_policy`, `threshold_shift_policy` | Díaz & van der Laan (2012); Díaz et al. (2023). [Getting started](getting-started.md) |
| **Single time point, binary or categorical treatment** (e.g. control / low / high dose) | Interventional mean or contrast across treatment levels | `DiscreteTreatmentPolicy`, `run_discrete_lmtp`, `run_discrete_lmtp_contrast` | Díaz et al. (2023). For static binary ATE only, see also `TMLE.jl`. |
| **Zero-inflated / two-part biological outcome** (e.g. infection presence vs intensity; biomass with zeros) | Overall, presence, and intensity causal contrasts | `run_two_part_discrete_lmtp`, `TwoPartInterventionalMean` (or `fit_bayesian_cluster_hurdle`) | [API overview](api.md) |
| **Repeated exposures over time with time-varying confounders** (e.g. seasonal infection, body condition, treatment) | Longitudinal policy effect across waves | `SequentialPolicy`, `run_sequential_lmtp` | Robins (1986); Díaz et al. (2023). [Getting started](getting-started.md) |
| **Discrete-time survival / time-to-event** (e.g. overwinter survival across trapping sessions) | Event-free survival probability under an MTP | `SurvivalPolicy`, `run_survival_lmtp` | Díaz, Hoffman & Hejazi (2024). [Methods](methods.md) |
| **Baseline treatment with repeated outcomes across visits** (e.g. vaccine trial with longitudinal antibody titres) | Marginal treatment–time profile (MSM) | `RepeatedOutcomeMSM`, `run_repeated_outcome_msm`, `msm_contrast` (or `fit_mmrm` with `MixedModels`) | Rosenblum & van der Laan (2010). [Getting started](getting-started.md) |
| **Parametric standardisation (g-computation)** (transparent parametric baseline, mixed model, or comparison against LMTP) | Standardised interventional mean or contrast | `fit_parametric_gcomp`, `gcomp_contrast`, `run_parametric_gcomp` | Robins (1986); Hernán & Robins (2020). [Getting started](getting-started.md) |
| **Very small samples or clustered units** (cages, pens, litters, clinics; Gaussian, binary, count, or hurdle outcomes) | Posterior marginal policy contrast | `fit_bayesian_cluster_gcomp`, `fit_bayesian_cluster_logistic`, `fit_bayesian_cluster_count`, `fit_bayesian_cluster_hurdle` | [Bayesian Gaussian](bayesian.md), [Bayesian logistic](bayesian_logistic.md) |
| **Missing outcomes or attrition** (e.g. animals not recaptured, dropout) | Complete-population causal effect | Observable missingness policies | [Missingness](missingness.md) |
| **Direct vs mediated biological pathway** | Direct (NDE / IDE / CDE) and indirect (NIE / IIE) effects | Use [CausalMediation.jl](https://simonab.github.io/CausalMediation.jl/dev/) | Vansteelandt & Daniel (2017); Díaz & Hejazi (2020); Liu et al. (2024) |

## Practical checks before interpreting an estimate

Three checks separate a defensible observational estimate from an artefact:

1. **Match complexity to sample size (`recommend_run_options`).** Biological
   datasets often have tens to low hundreds of units rather than tens of
   thousands. `recommend_run_options(n; engine = :lmtp)` selects fold counts and
   lean Super Learner libraries (`SMALL_N_SL_LEARNERS`) appropriate for ``n``
   (Phillips et al., 2023; see [Small-*n* checklist](small_n.md)).
2. **Check positivity (`positivity_report`).** Even a shift policy can push
   individuals into exposure regions with few comparable observations. Inspect
   effective sample size and `weak_support` / `unsupported_shift` flags before
   interpreting large shifts (Petersen et al., 2012; Díaz et al., 2023).
3. **Quantify sensitivity to unmeasured confounding (`sensitivity_report`).**
   No adjustment set protects against an unmeasured confounder. Partial-``R^2``
   sensitivity reports how strongly an omitted variable would need to explain
   both exposure and outcome to overturn the result (Cinelli & Hazlett, 2020).

## Minimal workflow: small-*n* shift curve with diagnostics

```@example orientation-targeted
using CausalTargeted, StableRNGs

# Synthetic cohort: baseline confounder W, continuous exposure A, outcome Y
df, _ = simulate_linear_mtp(150; rng = StableRNG(101))
opts = recommend_run_options(size(df, 1); engine = :lmtp)

grid = run_lmtp_grid(
    df, :A, :Y;
    baseline = [:W],
    deltas = [-0.5, 0.0, 0.5],
    folds = opts.folds,
    learners_outcome = opts.learners_outcome,
    learners_trt = opts.learners_trt,
    parallel = false,
    positivity = opts.positivity,
    simultaneous = false,
    rng = StableRNG(102),
)

grid[:, [:delta, :est, :se]]
```

Inspect positivity support and omitted-confounder sensitivity across the grid:

```@example orientation-targeted
rep = positivity_report(df, :A; deltas = grid.delta)
sens = sensitivity_report(grid.est[end], grid.se[end]; n = size(df, 1))
(support = string.(rep.support_status), sensitivity_rows = size(sens, 1))
```

## References

Key works cited on this page (complete list with DOIs and BibTeX keys in
[References](references.md)):

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, *21*(1), C1–C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *JRSS-B*, *82*(1), 39–67. [doi:10.1111/rssb.12348](https://doi.org/10.1111/rssb.12348)
- Díaz Muñoz, I., & van der Laan, M. J. (2012). Population intervention causal effects based on stochastic interventions. *Biometrics*, *68*(2), 541–549. [doi:10.1111/j.1541-0420.2011.01685.x](https://doi.org/10.1111/j.1541-0420.2011.01685.x)
- Díaz, I., Williams, N., Hoffman, K. L., & Schenck, E. J. (2023). Nonparametric causal effects based on longitudinal modified treatment policies. *JASA*, *118*(542), 846–857. [doi:10.1080/01621459.2021.1955691](https://doi.org/10.1080/01621459.2021.1955691)
- Díaz, I., Hoffman, K. L., & Hejazi, N. S. (2024). Causal survival analysis under competing risks using longitudinal modified treatment policies. *Lifetime Data Analysis*, *30*, 213–236. [doi:10.1007/s10985-023-09606-7](https://doi.org/10.1007/s10985-023-09606-7)
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Petersen, M. L., et al. (2012). Diagnosing and responding to violations in the positivity assumption. *Statistical Methods in Medical Research*, *21*(1), 31–54. [doi:10.1177/0962280210386207](https://doi.org/10.1177/0962280210386207)
- Phillips, R. V., et al. (2023). Practical considerations for specifying a super learner. *International Journal of Epidemiology*, *52*(4), 1276–1285. [doi:10.1093/ije/dyad023](https://doi.org/10.1093/ije/dyad023)
- Robins, J. (1986). A new approach to causal inference in mortality studies with a sustained exposure period. *Mathematical Modelling*, *7*, 1393–1512.
- Rosenblum, M., & van der Laan, M. J. (2010). Targeted maximum likelihood estimation of the parameter of a marginal structural model. *International Journal of Biostatistics*, *6*(2). [doi:10.2202/1557-4679.1238](https://doi.org/10.2202/1557-4679.1238)
- van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. *SAGMB*, *6*(1).
- van der Laan, M. J., & Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer.
- van der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning. *International Journal of Biostatistics*, *2*(1).
- Zheng, W., & van der Laan, M. J. (2011). Cross-validated targeted minimum-loss-based estimation. In van der Laan & Rose (2011).

## Next steps

- Executable walk-throughs: [Getting started](getting-started.md).
- Finite-sample defaults and diagnostics: [Small-*n* checklist](small_n.md).
- Full method-to-API reference: [Methods and literature](methods.md).
