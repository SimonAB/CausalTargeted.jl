# Methods and literature

This page maps **CausalTargeted** APIs to the papers and books that define the
estimands, identification conditions, and estimators. The implementations are
Julia-native analogues of ideas developed in the LMTP / mediation literature
(including the R `lmtp` and `crumble` packages); they are not line-for-line ports.
Full bibliographic entries (with DOIs) are in [References](references.md). Keys
such as `diaz2023lmtp` match `references.bib` in the CDCS book. Engine naming
(`:lmtp` / `:mediation`, not “crumble”) is summarised in
[NAMING.md](https://github.com/SimonAB/CausalTargeted.jl/blob/main/NAMING.md).

## Modified treatment policies and LMTP

**Scientific problem.** Deterministic interventions that set a continuous exposure to a fixed
value are often scientifically uninteresting and exacerbate positivity violations. *Modified
treatment policies* (MTPs) shift or otherwise transform the *natural* value of treatment
(e.g. raise exposure by one standard deviation, subject to clamps).

| Topic | Primary sources | CausalTargeted surface |
|-------|-----------------|------------------------|
| Stochastic / population interventions | Díaz & van der Laan (2012), *Biometrics* | `ShiftPolicy`, additive / multiplicative / threshold policies |
| Longitudinal MTPs (LMTP): ID, EIF, TMLE & sequential DR | Díaz, Williams, Hoffman & Schenck (2023), *JASA* | `run_lmtp_grid`, `lmtp_tmle_contrast`, `LongitudinalPolicy` |
| Point treatment, repeated outcomes (joint ``Σ``) | Rosenblum & van der Laan (2010), *IJB* | `run_repeated_outcome_msm`, `msm_contrast` |
| Parametric treatment×time MSM | Rosenblum & van der Laan (2010), *IJB* | `run_parametric_repeated_msm`, `ParametricRepeatedOutcomeMSM` |
| Software reference (R) | Williams & Díaz (2023), *Observational Studies* | Conceptual parity, not API identity |
| Discrete-time survival / event-time LMTP | Díaz, Hoffman & Hejazi (2024), *Lifetime Data Analysis* | `SurvivalPolicy`, `run_survival_lmtp` (competing risks deferred) |

**Point-treatment continuous MTP.** `run_lmtp_grid` estimates a δ-indexed curve under a
user-chosen `ShiftPolicy`, with cross-fitted outcome and treatment nuisances and optional
TMLE fluctuation. Density-ratio options (`gaussian`, classification, hybrid) implement
practical continuous-exposure clever covariates in the spirit of the LMTP literature.

The identifying DAG for the linear synthetic DGP (`simulate_linear_mtp`) is baseline
confounding of a continuous exposure:

```@example methods-lmtp
using CausalDynamics, Graphs, DAGMakie, CairoMakie

g = DiGraph(3)
add_edge!(g, 1, 2)  # W → A
add_edge!(g, 1, 3)  # W → Y
add_edge!(g, 2, 3)  # A → Y
fig = plot_with_adjustment_set(g, 2, 3, [1]; node_labels = ["W", "A", "Y"])
fig
```

**Sequential / multi-time LMTP.** `SequentialPolicy` / `run_sequential_lmtp` implement a
practical recursive outcome regression with a TMLE-style correction at ``t = 1``, following the
sequential identification strategy of Díaz et al. (2023). Numeric `A_t` share an additive
`ShiftPolicy`. Categorical `A_t` take per-time `DiscreteTreatmentPolicy` values in `policies`
(dummy-coded Q; Díaz–Williams classification ratio at ``t = 1``). Mixed continuous/discrete
treatments are rejected. Pair with CausalDynamics
`TemporalEffectQuery` + `unroll_temporal_dag` → `identify`, then
`sequential_spec_from_identification` / `plan_sequential` (or
`sequential_identification_certificate`) so estimation carries an explicit ID certificate.
`estimand_from_query` on a `TemporalEffectQuery` still builds a numeric
`LongitudinalPolicy` by default; pass nonempty `policies` and wide `treatments`
(`:A1`, `:A2`, …) for a factor `SequentialPolicy`.
Observational panels from CausalDynamics `simulate_panel` use the same wide layout
(`baseline` / timed `:a1`,`:a2` / terminal `:y`) that `SequentialPolicy` expects;
`execute_estimand` merges the certificate before running sequential LMTP.
Latent/filter outputs enter via CausalDynamics `ObservationBridge` /
`panel_from_latent_series` before the same sequential path.

**Discrete-time survival / event-time LMTP.** `SurvivalPolicy` / `run_survival_lmtp`
estimate event-free probability at a horizon under a common MTP shift on
time-ordered treatments, with at-risk sequential regression and optional thin
IPCW for censoring. Pair with CausalDynamics `TemporalEffectQuery` (outcome =
event-free indicator at the horizon) and `survival_identification_certificate`.
Competing risks remain out of scope. Synthetic gate:
`simulate_discrete_survival_mtp`.

**Repeated outcomes under a static treatment.** When treatment is fixed once and
the same response is measured at several times (wide `Y1…YT`),
`run_repeated_outcome_msm` estimates the unstructured profile
``τ(t)=E[Y_t\mid do(A=1)]-E[Y_t\mid do(A=0)]`` with a **joint** influence-function
covariance. Shared cross-fit propensity and per-time outcome regressions yield
``\widehat\Sigma`` so contrasts such as ``τ(t_3)-τ(t_2)`` use
[`msm_contrast`](@ref). Default `estimator=:tmle` fluctuates ``Q_t`` on the
shared clever covariate ``A/g-(1-A)/(1-g)``; `:eif` is the untargeted one-step.
Long `(id, time, Y)` tables pivot with [`unstack_repeated_outcomes`](@ref).
Missingness is a complete-profile policy (`time_indexed=true`); ``Y_t`` are
never imputed. This is the point-treatment / multi-outcome setting
(Rosenblum & van der Laan 2010; R `tmle::tmleMSM` as a conceptual reference),
not sequential LMTP (time-varying ``A_t``). Pair with
`identify_repeated_outcomes` (CausalTargeted helper over CausalDynamics
`TotalEffectQuery`) when each ``Y_t`` shares a backdoor set.
Parametric MSMs project that profile (or stacked means ``μ(t,a)``) onto a
design with GLS: `run_parametric_repeated_msm` / `ParametricRepeatedOutcomeMSM`
(`:constant`, `:linear_time`, `:factor_time`, custom matrices, or
`:mean_treatment_time` with `target=:mean`). Pass `cluster=:cluster` (or a
length-`n` id vector) for a **cluster-robust** sandwich on ``\\widehat{\\Sigma}``;
point estimates are unchanged (sampling hierarchy, not BLUP). Synthetic gates:
`simulate_repeated_outcome_ate`, `simulate_mean_treatment_time_msm`. Generative
nested ``U`` DGPs live in CausalDynamics (`RandomEffectSpec`).

**Gaussian MMRM (optional MixedModels extension).** For a **static** treatment
and repeated Gaussian outcomes in long `(id, time, Y)` form, `fit_mmrm` /
`run_mmrm` fit `outcome ~ treatment * time + baseline + (1 | id)` (default) or
an `:unstructured` random-effects approximation `(1 + visit | id)` with an
internal categorical visit factor. [`mixed_g_computation`](@ref) supplies
visit-specific marginal contrasts (default `random_effects=:zero`). This is a
**parametric trial-style reference** beside LMTP/MSM, not a substitute for
discrete longitudinal LMTP. Random **slopes** for `:marginal` g-computation are
not supported in this release. Requires `using MixedModels`. Stress:
[`docs/stress/mmrm_stress.qmd`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/stress/mmrm_stress.qmd).

**Transport weights.** `domain_transport_weights` / `transport_weighted_mean` provide
marginal IPTW-style domain reweighting after a CausalDynamics `TransportQuery`
certificate. **Decision.** `choose_policy` evaluates labelled `Estimand`s with
`execute_estimand` and returns a `PolicyChoice` (max/min scalar TE).

## Targeted learning, Super Learner, and cross-fitting

Adjustment covariates are encoded before they enter Super Learner. An internal
fitted schema uses StatsModels default `DummyCoding` for string and categorical
columns and preserves its levels, dummy-column meanings, and column order across
all folds. Numeric values (including ordinary integer columns) remain one
continuous column, while `Bool` becomes numeric 0/1. Missing-data handling
(`:drop`, `:impute`, `:ipcw`, or `:ipcw_impute`) occurs before schema fitting;
the schema itself does not impute or standardise.

Finite-support treatments use `DiscreteTreatmentPolicy`: T=1 via
`run_discrete_lmtp`, multi-time via `SequentialPolicy` `policies`. Integer
columns remain numeric on the continuous MTP path unless factorised. Random
Forest, EvoTrees, and XGBoost receive numeric dummy columns rather than native
categorical features; Random Forest therefore computes `mtry` from the encoded
feature count.

**Missing data.** Full policy catalogue: [Missingness](missingness.md).
In brief: incompleteness is stratum × PCH rung (Structural $R$ in
CausalDynamics; Dynamical sequential / survival gaps; Observable policies
here). `handle_missing_data` supports `:drop`, `:impute`, `:ipcw`, and
`:ipcw_impute` before schema fitting; IPCW weights enter IF summaries so
`:drop` and `:ipcw` need not coincide. Survival censoring IPCW is not MAR
missing $S_T$. Opt-in `impute_posterior` + `run_lmtp_grid(...; imputation=)`
pools under a MAR certificate. Stress: [missingness grid](stress_missingness.md).

**G-computation SEs.** `run_gcomp` percentile intervals use a **refitting**
bootstrap (redraw rows, recompute cross-fitted $Q$). Set `n_boot = 0` for a
fast influence-function SE and normal Wald CI. Do not interpret older
ψ-only bootstrap SEs as sampling uncertainty for the plug-in functional.

This fitted-schema path covers CausalTargeted's g-computation, LMTP Gaussian and
classification/hybrid density-ratio nuisances, fold caching, sequential LMTP,
survival LMTP, and missing-data nuisance models. CausalMediation reuses
`fit_covariate_schema` / `design_matrix` for fold-stable string and categorical
covariates when running `run_mediation_grid`.

**Parametric regression standardisation.** `fit_parametric_gcomp` is a
formula-based complete-case GLM engine separate from cross-fitted `run_gcomp`
and LMTP. It supports Gaussian/identity, binomial/logit, Gamma/log, and
`:negbin`/log outcome models. `gcomp_mean` averages counterfactual response-scale
predictions over the empirical target rows; `gcomp_contrast` forms differences,
response-mean ratios, or log ratios; and `gcomp_interaction` forms joint
difference-of-differences or ratio-of-ratios contrasts. A target subgroup is
restricted before the intervention is applied. Formula predictors must be
complete-case (no silent row dropping). NB2 HC3/delta inference treats `theta`
as fixed.

The fitted StatsModels formula supplies categorical levels, contrasts, column
order, and interaction encoding for counterfactual prediction. Changing a
categorical intervention therefore rebuilds every affected formula term, and an
unseen level is rejected. Parametric uncertainty uses coefficient HC3 covariance
and analytic delta-method gradients; ratio intervals are constructed on the log
scale. `bootstrap_gcomp_interaction` provides an optional refitting bootstrap,
including resampling within user-specified strata. These inferential procedures
do not replace or modify the cross-fitted inference used by `run_gcomp`.

| Topic | Primary sources | CausalTargeted surface |
|-------|-----------------|------------------------|
| TMLE | van der Laan & Rubin (2006); van der Laan & Rose (2011, 2018) | `estimator=:tmle`, fluctuation helpers |
| Super Learner | van der Laan, Polley & Hubbard (2007); Phillips et al. (2023) | `fit_super_learner`; metalearners `:nnls`, `:nnloglik`, `:cv_selector` (dSL), `:invmse`; `family=:multinomial` (sl3-style simplex) |
| Optional MLJ linear nuisances | MLJ / MLJLinearModels (weakdep) | `:mlj_ridge`, `:mlj_lasso`, `:mlj_elasticnet`, `:mlj_logistic` after `using MLJ, MLJLinearModels` (features standardised; never in small-*n* presets) |
| Optional Random Forest nuisance | MLJ / MLJDecisionTreeInterface (DecisionTree.jl weakdep) | `:randomforest` after `using MLJ, MLJDecisionTreeInterface` (unscaled features; regression or probabilistic classification) |
| Optional XGBoost nuisance | MLJ / MLJXGBoostInterface (XGBoost.jl weakdep) | `:xgboost` after `using MLJ, MLJXGBoostInterface` (unscaled features; regression or probabilistic classification) |
| Optional neural nuisances | MLJFlux (Flux) | `:mlj_mlp`, `:mlj_nn_binary` after `using MLJFlux` — never in small-*n* presets |
| Cross-fitting / sample splitting | Zheng & van der Laan (2011); Chernozhukov et al. (2018) | `crossfit_indices`, fold caches |
| Applied TMLE overview | Schuler & Rose (2017) | Pedagogical pointer |

At **small *n***, rich libraries overfit. `recommend_run_options` / `adaptive_learners`
prefer lean GLM/mean stacks when `n < 80`, consistent with the Super Learner principle that
the library must be *estimable* at the sample size at hand. For binary propensity or outcome
nuisances where probability calibration matters, `fit_super_learner(...; family=:binomial)`
defaults to `:nnloglik` (R `method.NNloglik`). The discrete Super Learner is
`metalearner=:cv_selector` (Phillips dSL / sl3 `Lrnr_cv_selector`). Pass a
`nested_sl_candidate` in the library to nest an ensemble Super Learner inside
that selector (eSL-inside-dSL; opt-in, not a new default). LMTP density-ratio
classifiers stay on `:invmse`. Categorical
outcomes use `family=:multinomial` (convex combination of class-probability
matrices; sl3 `loss_loglik_multinomial`). Categorical *treatments* in LMTP use
`run_discrete_lmtp` with Díaz–Williams classification density ratios, not a
multinomial propensity. Arm contrasts (e.g. SS vs R) can use
`run_discrete_lmtp_contrast`, which fits two static policies and differences
the estimates (independent SE approximation). For binary presence outcomes
(`Y ∈ {0,1}`), pass `family_outcome = :binomial` so outcome nuisances use
logistic Super Learner fits rather than the default Gaussian location model:

```julia
run_discrete_lmtp_contrast(
    df, :arm, :infected;
    arm_hi = "SS", arm_ref = "R", levels = ["R", "SS", "SC"],
    baseline = [:weight],
    family_outcome = :binomial,
    learners_outcome = (:glm, :mean),
)
```

For nonnegative count outcomes (parasite totals, colony counts), use
`family_outcome = :poisson` or `:negbin`. Nuisance models use log-link GLMs;
profiled NB shape per cross-fit fold; TMLE targeting uses a log-link fluctuation
on the fitted mean. The reported estimand is ``E[Y \mid \mathrm{do}(A)]`` on
the count scale ([#36](https://github.com/SimonAB/CausalTargeted.jl/issues/36)):

```julia
run_discrete_lmtp_contrast(
    df, :arm, :count;
    arm_hi = 1, arm_ref = 0, levels = [0, 1],
    baseline = [:weight],
    family_outcome = :negbin,
    learners_outcome = (:glm_nb, :mean),
)
```

When excess zeros are scientifically co-primary, prefer the two-part hurdle
path below rather than a single count model.

The same `family_outcome` keyword is forwarded by `run_lmtp_grid` and
`run_discrete_lmtp`. Multi-time factor recodes use the same policies on
`run_sequential_lmtp`. Optional MLJ / MLP candidates are
**opt-in**: they can improve recovery on some DGPs in a single synthetic draw while diluting
others (overfitting vs generalisation). Prefer repeated Monte Carlo and library ablations before
changing defaults.

## Panel bridge (CausalDynamics → CausalTargeted)

Temporal identification in **CausalDynamics** yields adjustment certificates and
[`plan_targeted_estimation`](https://simonab.github.io/CausalDynamics.jl/dev/api/identification/#CausalDynamics.plan_targeted_estimation)
selects a runner symbol and wide column names. **CausalTargeted** executes the
plan with [`run_estimation_plan`](@ref).

| `EstimationPlan.engine` | Runner | Notes |
|-------------------------|--------|-------|
| `:discrete_lmtp` | `run_estimation_plan(...; arm_hi, arm_ref, levels)` | Same-occasion arm contrast; default for static grid designs |
| `:two_part_discrete_lmtp` | `run_estimation_plan(...; arm_hi, arm_ref, levels)` | Uses `plan.presence_col` / `plan.intensity_col` from `OutcomeKind.hurdle` |
| `:sequential_lmtp` | `sequential_spec_from_identification` + `run_sequential_lmtp` | Not wrapped by `run_estimation_plan` |
| `:lmtp_grid` | `run_lmtp_grid` | Continuous shift LMTP; pass `lower_q` / `upper_q`, not `shift` |

```julia
using CausalDynamics, CausalTargeted, DataFrames

spec = TemporalDAGSpec([:grid_type, :fec], [LaggedEdge(:grid_type, :fec, 0)])
u = unroll_temporal_dag(spec, 4)
query = TemporalEffectQuery(:grid_type, :fec, 2, 2)
plan = plan_targeted_estimation(
    u, query, names(wide);
    unit_level = [:grid_type],
    data = wide,              # optional: sets plan.estimability
    outcome_specs = Dict(     # optional: hurdle → :two_part_discrete_lmtp
        :fec => NodeOutcomeSpec(OutcomeKind.hurdle, :fec_bin, :fec_intensity),
    ),
)
res = run_estimation_plan(
    wide, plan;
    arm_hi = "SS", arm_ref = "R", levels = ["R", "SS", "SC"],
    learners_outcome = recommend_run_options(nrow(wide)).learners_outcome,
)
```

For dynamical queries (`t_treat < t_outcome` on the same variable), the planner
selects `:sequential_lmtp` or `:lmtp_grid`; see [`run_lmtp_grid`](@ref) kwargs
(`lower_q`, `upper_q`, `shift` is not a grid keyword). Stress notebook:
[`apodemus_panel_stress.qmd`](stress/apodemus_panel_stress.qmd).

### Super Learner candidate roles

- `:glm`, `:glm_interact`, and `:glm_quad` provide ordinary, interaction-expanded,
  and quadratic-expanded GLMs (logistic under `family=:binomial`).
- `:glmnet` and its lasso/ridge aliases provide regularised linear candidates
  through MLJLinearModels.
- `:randomforest` provides a bagged Random Forest through MLJ and DecisionTree.jl.
  Its conservative defaults support stable nuisance estimation at modest sample sizes.
- `:evotree` and `:evotree_deep` provide Julia gradient boosting through EvoTrees.jl.
- `:xgboost` provides gradient boosting through MLJ and XGBoost.jl.
- `:mlj_mlp` and `:mlj_nn_binary` provide optional MLJFlux neural candidates.
- `:mean` is the intercept-only reference and fallback learner. It is **not** a
  standalone treatment-contrast engine: `run_gcomp`, `run_lmtp_grid`, and related
  entry points reject `learners=(:mean,)` with an `ArgumentError`. Always pair
  `:mean` with at least one treatment-dependent candidate (typically `:glm`).

The base `DEFAULT_SL_LEARNERS` remains `(:glm, :mean)`. `RICH_SL_LEARNERS`
adds `:randomforest` for bagging/model-class diversity, while `:xgboost` remains
explicit opt-in because the rich library already contains EvoTrees boosting.
`SMALL_N_SL_LEARNERS` and `adaptive_learners` remain conservative and do not add
either flexible tree learner automatically.

```julia
using CausalTargeted, MLJ, MLJLinearModels
using MLJDecisionTreeInterface, MLJXGBoostInterface

fit = fit_super_learner(
    X,
    y;
    learners = (:glm, :glmnet, :randomforest, :xgboost, :mean),
)
```

Current Random Forest configuration:

| MLJ parameter or behaviour | Value |
|---|---:|
| `n_trees` | 500 |
| `n_subfeatures` | computed at fit time as `max(1, floor(sqrt(number of predictors)))` |
| `min_samples_leaf` | 5 |
| bootstrap sampling | enabled |
| `sampling_fraction` | 1.0 |
| `rng` | `MersenneTwister(42)` |

Current XGBoost defaults:

| MLJ parameter | Default |
|---|---:|
| `num_round` | 100 |
| `max_depth` | 2 |
| `eta` | 0.05 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `lambda` | 1 |
| `alpha` | 0 |
| `gamma` | 0 |
| `nthread` | 1 |
| `seed` | 42 |

## Interventional mediation grids

Natural direct/indirect effects (Pearl, 2001; Robins & Greenland, 1992; VanderWeele, 2015)
require cross-world assumptions that fail under intermediate confounding. *Interventional*
(randomised interventional) effects (Vansteelandt & Daniel, 2017) and *stochastic*
intervention mediation (Díaz & Hejazi, 2020; Hejazi et al., 2023) weaken those assumptions.
Liu, Williams, Rudolph & Díaz (2024) unify modern mediation estimands with MTPs; the R package
`crumble` (Liu et al., 2025 tutorial) is a software companion—Julia APIs use **mediation**
names (`run_mediation_grid`, engine `:mediation`), with soft-deprecated `run_crumble_*` / `:crumble` aliases.

| Topic | Primary sources | CausalTargeted surface |
|-------|-----------------|------------------------|
| Stochastic mediation (in)direct effects | Díaz & Hejazi (2020), *JRSS-B* | Conceptual basis for continuous-A mediation |
| Stochastic interventional effects with intermediate confounding | Hejazi et al. (2023), *Biostatistics* | Design target for robust mediation contrasts |
| Unified targeted mediation + MTP | Liu et al. (2024), arXiv:2408.14620 | `run_mediation_grid`, `MediationContrast` |
| Tutorial / R package companion | Liu et al. (2025), arXiv:2604.09902 | Estimand catalogue; cite, do not brand Julia after “crumble” |
| Classical mediation textbook | VanderWeele (2015) | Interpreting NDE/NIE vs interventional contrasts |

**Implementation note.** `run_mediation_grid` estimates TE / NDE / NIE under continuous MTP
shifts via nested Monte Carlo and cross-fitted nuisances. Nested-MC variability is first-class:
`mediation_n_mc_sweep` and `mediation_stability_summary` quantify SE and sign stability across
`n_mc` (essential at small *n*).

A minimal mediation DAG (`A → M → Y`, `A → Y`) for interpreting those contrasts:

```@example methods-mediation
using DAGMakie, CairoMakie

fig, _ax, _p = dagplot_mediation(["A", "M", "Y"])
fig
```

**Fold/δ cache.** `build_mediation_fold_cache` (and the LMTP analogue) reuse outcome / mediator /
exposure fits across δ within folds—same statistical estimand, lower wall time.

## Positivity and support

Positivity (overlap) is necessary for identification of interventional means
(Hernán & Robins, 2020; Petersen et al., 2012). MTPs are often *designed* so that shifted
exposures remain in the support of the observed treatment law (Díaz et al., 2023).

| Topic | Primary sources | CausalTargeted surface |
|-------|-----------------|------------------------|
| Diagnosing positivity violations | Petersen et al. (2012) | `positivity_report`, `positivity_markdown` |
| Clamp / support diagnostics under additive shifts | LMTP practice (Díaz et al., 2023) | support / clamp helpers in `mtp_common.jl`; grid `positivity=true` |

## Sensitivity to unmeasured confounding

Even with correct adjustment sets, estimates can tip under omitted confounding. CausalTargeted
exposes **diagnostic** tipping-point and partial-*R*² calibrations inspired by Cinelli &
Hazlett (2020); complementary classical tools include VanderWeele & Ding (2017) E-values and
Rosenbaum (2002) sensitivity models.

| Topic | Primary sources | CausalTargeted surface |
|-------|-----------------|------------------------|
| Partial *R*² / robustness-value style OVB | Cinelli & Hazlett (2020), *JRSS-B* | `partial_r2_calibration`, `sensitivity_report` |
| E-value | VanderWeele & Ding (2017) | Cite for reporting; not duplicated here |
| Discovery as *sensitivity*, not oracle | Pearl (2009); Spirtes et al. (2000) | `discovery_adjustment_sensitivity`, `merge_discovery_sensitivity!` |

**Never** silently replace a user DAG with a discovery graph in production defaults.

## Outcome paths and model choice

Biological and trial endpoints differ in support, time structure, and whether
zero inflation is scientifically primary. The table below maps common patterns
to **CausalTargeted** engines. All targeted paths share cross-fitted nuisances,
optional `handle_missing`, and `cluster=` sandwich SEs where noted. None of these
replace identification: pair with CausalDynamics certificates and the query
ladder (association → intervention → counterfactual).

| Outcome pattern | Primary estimand | Engine / API | When to prefer |
|-----------------|------------------|--------------|----------------|
| Continuous scalar under static arm | ``E[Y \mid \mathrm{do}(A)]`` | `run_discrete_lmtp` / `run_lmtp_grid` | Single visit or summary outcome |
| Binary presence ``I(Y>0)`` | Risk difference / probability | `run_discrete_lmtp` + `family_outcome=:binomial` ([#34](https://github.com/SimonAB/CausalTargeted.jl/issues/34)) | Infection, detection, any ``\{0,1\}`` endpoint |
| Zero-inflated semi-continuous | ``P(Y>0)`` **and** ``E[Y \mid Y>0]`` | `run_two_part_discrete_lmtp_contrast` ([#35](https://github.com/SimonAB/CausalTargeted.jl/issues/35)) | Parasite EPG, shedding scores: presence co-primary |
| Repeated Gaussian, static treatment | Visit-specific ``τ(t)`` (marginal) | `run_repeated_outcome_msm` (TMLE/IF) | Nonparametric profile + joint ``\\widehat{\\Sigma}`` |
| Repeated Gaussian, static treatment | LS-mean-style contrast (parametric) | `fit_mmrm` / `run_mmrm` ([#25](https://github.com/SimonAB/CausalTargeted.jl/issues/25); `using MixedModels`) | Trial-style reference beside MSM/LMTP |
| Repeated count / overdispersion, static treatment | Marginal g-comp on count scale | `mixed_g_computation` (NB2 ext) | Hierarchical reference; not LMTP |
| Time-varying treatment | Sequential LMTP | `run_sequential_lmtp` | ``A_t`` shifts or factor policies |
| Count / overdispersed integer | ``E[Y \mid \mathrm{do}(A)]`` on count scale | `run_discrete_lmtp` + `family_outcome=:poisson` or `:negbin` ([#36](https://github.com/SimonAB/CausalTargeted.jl/issues/36)) | Parasite totals, colony counts; prefer two-part when zeros are co-primary |

**Two-part hurdle LMTP.** Build `presence` (``I(Y>0)``) and `intensity` (e.g.
`log(y)` on positives, with `missing` when ``Y=0``). Presence uses binomial
outcome nuisances; intensity fits on the positive subsample only, so the
reported contrast is **conditional on observed infection**, not the marginal
``E[Y \mid \mathrm{do}(A)]`` unless you define a composite estimand separately.

```julia
res = run_two_part_discrete_lmtp_contrast(
    df, :grid_type;
    presence = :fec_bin,
    intensity = :fec_logpos,
    arm_hi = "SS", arm_ref = "R", levels = ["R", "SS", "SC"],
    baseline = [:weight],
    family_presence = :binomial,
    cluster = :mouse_id,
)
res.presence.estimate    # co-primary: infection risk difference
res.intensity.estimate   # secondary: log-burden among positives
```

Use [`suggest_family_outcome`](@ref) or `recommend_run_options(n; outcome=col)`
to set `family_outcome` for single-outcome runs. For DAG faithfulness on hurdle
nodes, see the next section; for repeated Gaussian visits, compare MSM (IF) with
MMRM (parametric) on the same panel before interpreting biology.

## Hurdle CI testing

Zero-inflated biological outcomes are often operationalised as a **two-part
hurdle**: presence ``I(Y > 0)`` and log intensity among positives. Gaussian
partial correlation on ``\log(1 + Y)`` is misspecified for faithfulness checks
on those nodes.

Use partial correlation for continuous block variables. Use
[`test_implied_hurdle_independences`](@ref) when a statement involves a
hurdle-split node (binomial presence GLM; Gaussian intensity GLM among
positives). String / categorical predictors (e.g. `grid_type`) use StatsModels
`DummyCoding`, aligned with discrete LMTP ([#34](https://github.com/SimonAB/CausalTargeted.jl/issues/34))
and two-part LMTP ([#35](https://github.com/SimonAB/CausalTargeted.jl/issues/35)).

```julia
results = test_implied_hurdle_independences(
    statements, df,
    Dict(:fec => (:fec_bin, :fec_intensity), :eimeria => (:eimeria_bin, :eimeria_intensity)),
)
```

## Identification certificates (CausalDynamics bridge)

Estimation attaches provenance via `identification_certificate` / `attach_run_metadata!`.
Upstream ID uses Pearl’s do-calculus toolkit (Pearl, 2009; Shpitser & Pearl, 2006) and
g-methods (Robins, 1986; Robins, 2000). Time-indexed queries use
`TemporalDAGSpec` / `unroll_temporal_dag` / `TemporalEffectQuery` in CausalDynamics
(see that package’s [References](https://simonab.github.io/CausalDynamics.jl/dev/references/)).

## Small-*n* profile

Conservation biology, ecology, and early trials often have tens to low hundreds of units.
`recommend_folds`, `SMALL_N_SL_LEARNERS`, and `recommend_run_options` encode memory-safe,
positivity-aware defaults (`parallel=false`, higher mediation `n_mc` when `n` is small).
See the [Small-*n* checklist](small_n.md).

## What we deliberately do *not* claim

- Full parity with every option in R `lmtp` / `crumble` (GPU Riesz nets, all mediation
  estimand flavours, competing-risks survival LMTP).
- That tipping-point / partial-*R*² helpers replace design-based identification.
- That Super Learner at *n* ≈ 30 recovers oracle rates—diagnostics exist precisely because
  they often do not.
