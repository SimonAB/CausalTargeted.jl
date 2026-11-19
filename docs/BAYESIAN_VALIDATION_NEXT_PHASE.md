## Bayesian package improvement guided by independent validation

### Resume here

Saved after the first package-improvement milestone on 14 November 2026.
Package commits `10a16b7` and `19ec260` (now pushed on `origin/main`, tip
`e3db1f2`); CDCS pointer commit `1c08c6f`. Source version remains 0.4.2 with
unreleased changes; no release or registration was performed.

Completed: public endpoint Monte Carlo bounds and interpretation; independent
probability-space reference; paired prior/public-API validation infrastructure;
dimensionless moment-gate amendment A1 and independent-metric summary amendment
A2. Verification passed 3,498 package assertions, 498 isolated validation
assertions, executable documentation and the local Documenter build.

The public-API study completed 800 datasets and 288,000 stream evaluations.
Read the [milestone report](validation/bayesian-improvement-20261115/report.md),
[decision record](BAYESIAN_IMPROVEMENT_DECISIONS.md), and
[protocol with amendments](BAYESIAN_VALIDATION_PROTOCOL_V2.md) before resuming.
Original failures, source snapshots and pilot results are preserved. The large
raw public table is retained locally, ignored by Git; its location and checksums
are documented in the report. Smaller evidence archives are committed.

Next work, in order:

1. Investigate/independently replicate the unresolved default-draw lower-endpoint
   MC-bound inclusion discrepancy. Rank calculations, seed uniqueness, selected
   reconstructed public draws and independent endpoint references passed; no
   defect was identified. Preserve the original results and avoid tuning seeds
   or defaults to erase the discrepancy.
2. Review pilot costs and lock a separate main-run budget/protocol for the
   1,000-replication paired factorial and prior-predictive studies. Those large
   studies have not run; the completed 50-replication pilots are not substitutes.
3. Use diagnosed weaknesses to justify further package remedies through phase
   3A, then evaluate continuous/correlated confounding and misspecification in
   phase 4 with a validated higher-dimensional reference.
4. Lock candidate behaviour before untouched-scenario confirmation and final
   release decisions in phases 5–6. Do not claim improved general posterior or
   causal coverage from the endpoint diagnostic alone.

Before restarting, inspect all repository statuses and concurrent edits. Preserve
unrelated CausalMediation/book/stress-harness work. Check live CI status on
`origin/main` before making further changes.

### Purpose and completion

Improve the packages' Bayesian inference, uncertainty diagnostics, public API
and efficiency using independent validation to identify weaknesses and verify
repairs. The deliverable is a set of justified, tested package improvements and
documented operating limits, supported by reproducible evidence. A validation
report alone does not complete the work.

Determine whether each weakness comes from approximation, numerical computation,
prior choice, model specification or causal identification. Implement remedies
where the package can address the cause. When it cannot, make the limitation
clear through appropriate diagnostics, API behaviour and documentation. Do not
manufacture an algorithmic change when existing behaviour is already adequate.
Nominal coverage is not an acceptance gate for every fixed truth.

This is an implementation plan, not authorisation to run every later phase at
once. Implement phases 0–3 and the package-improvement loop in phase 3A first;
then verify the candidate changes in phases 4–5. Phase 6 prepares the tested
changes and evidence for delivery. No book implementations or external releases
are included.

### Existing evidence and boundaries

Start from the [November study](validation/multivariate-policy-20261114/report.md)
and [original protocol](MULTIVARIATE_POLICY_CALIBRATION.md). Preserve their
archives, source fingerprints and original checkpoint schema. Extend existing
reference and driver utilities where suitable; do not create another estimator.

CausalTargeted owns regression, policy contrasts and their diagnostics. Keep
state-space changes in CausalDynamics and mediation changes in CausalMediation;
this phase supplies no evidence for changing either package's algorithms.
All runtime changes remain pure Julia and preserve optional RxInfer/GraphPPL
loading. Do not add research references or simulation dependencies to the core
package merely to support validation.

The current bivariate target remains

ψ = q₀ [logistic(β₀ + β₁) − logistic(β₀)], q₀ = n₀/n.

It is assignment-to-one versus observed exposure on the supplied cohort.
Identification remains `:user_assumed`. Keep posterior credible intervals,
fixed-truth sampling coverage and EIF confidence intervals distinct. Never add
posterior variance to EIF variance. Neither LRVB nor a good likelihood fit
repairs an incorrect causal identification assumption.

Use caller-declared priors, with no outcome-derived scaling. Truth-centred
priors below are labelled oracle diagnostics and must never become public
defaults. They do not constitute simulation-based calibration (SBC).

### Phase 0: lock questions, metrics and execution contracts

The supervisor owns the mathematical note, acceptance gates and final claims.
Before implementation, record the following primary questions:

1. How much of an interval discrepancy comes from the posterior approximation,
   rather than the chosen prior and likelihood?
2. Does LRVB improve posterior fidelity, and at what computational cost?
3. How does prior choice affect repeated-sampling performance on shared data?
4. Does the finite-draw public API behave like the ideal law being validated?
5. Which package changes follow from the evidence, and what user-visible
   improvement and regression protection will each change deliver?

Create `docs/BAYESIAN_IMPROVEMENT_DECISIONS.md` as a working decision record.
For each finding record the reproducer, cause, affected public operation,
proposed remedy, existing diagnostics, acceptance criteria, trade-offs and final
disposition. Distinguish implementation defects from statistical limitations.
Record the current public API, defaults, accuracy and performance as the
before-change baseline; preserve a commit and environment for paired reruns.

Create `docs/BAYESIAN_VALIDATION_PROTOCOL_V2.md` containing the exact cells,
RNG scheme, comparison contrasts, metrics, numerical gates and dataset budgets.
Treat the pilot as implementation and runtime validation. After the pilot,
record a protocol/source hash before inspecting confirmatory results. Log any
amendment and retain the superseded protocol. This is a repository protocol,
not a claim of independent external preregistration.

Reuse deterministic streams, source/configuration hashes and complete-checkpoint
validation. Introduce an explicit v2 schema; reject v1 checkpoints in a v2 run.
Every base dataset gets a `dataset_id` independent of the fitted prior. Rows
carry `prior_id`, method, experiment, availability, failure reason and reference
quality. Never discard a dataset because one prior or approximation failed.

### Phase 1: independently check the bivariate posterior

Create `test/stress/policy_probability_reference.jl` and
`test/test_policy_probability_reference.jl`. The supervisor derives and reviews
the change of variables; an implementation agent may translate the agreed note.

Use a separate probability-space reference with p₀ = logistic(β₀) and
p₁ = logistic(β₀ + β₁). Evaluate the binomial likelihood directly from arm counts.
The inverse transformation and Jacobian are

β = [logit(p₀), logit(p₁) − logit(p₀)],

|∂β/∂(p₀,p₁)| = 1/[p₀(1−p₀)p₁(1−p₁)].

Derive the transformed Gaussian-prior density explicitly. Integrate over the
unit square and over p₁ ≤ p₀ + u for the contrast CDF, with correct clipping and
domain splits. Use stable logarithmic evaluation and explicit endpoint limits;
do not impose undocumented finite integration cut-offs. Do not call the existing
posterior kernel or tail integration helpers from this implementation. Sharing
QuadGK is acceptable, but describe this as independent parameterisation and
implementation, not a completely independent numerical library.

Write fixtures before implementation: prior only; balanced counts; one-control
data; complete/quasi separation; strong prior; correlated proper coefficient
prior. Include positive and negative contrast thresholds and upper/lower tails.

Proposed gates, locked before the main run:

- Each reference passes its own tolerance-refinement, CDF/survival closure and
  normalisation checks; integrated error estimates must be ≤ 1e-5.
- Absolute CDF disagreement ≤ 5e-5 at declared thresholds and refined endpoints.
- Absolute arm-contrast quantile disagreement ≤ 1e-4, with posterior probability
  checked at each endpoint. Apply cohort scaling explicitly afterwards.
- Decisions near coverage boundaries remain unavailable when numerical
  uncertainty prevents classification. Never relax a gate to rescue a case.

These are practical numerical gates, not rigorous probability enclosures. If
probability-space quadrature cannot resolve a fixture, retain the failure and
use a third method on that fixture before making a posterior-accuracy claim.
Diagnosed MCMC is a possible reference fallback, never a hidden production
dependency or automatic gold standard.

### Phase 2: paired factorial prior sensitivity and posterior fidelity

Create `test/stress/policy_prior_factorial_support.jl`,
`test/stress/policy_prior_factorial.jl` and matching `test/test_*.jl` contracts.
Reuse the existing simulator and summaries where their contracts match.

Use eight fixed-truth base cells: n ∈ {20,50}, β₀ = −0.3,
β₁ ∈ {0.9,3.0}, and control fraction q₀ ∈ {0.5,0.1}. Counts are exact integers.
Generate each dataset once, then fit all twelve priors:

- Coefficient covariance s²I, s ∈ {0.5,1.5,3.0}.
- Means zero, β_true, β_true + 2s e₂ and β_true − 2s e₂.
- e₂ changes the treatment coefficient only; label all truth-dependent means
  oracle/offset diagnostics. The zero-centred family is the practical baseline.

This separates prior scale from deliberate centring error within each dataset.
Effect size, exposure allocation and n vary across base cells. Do not interpret
the oracle comparison as attainable scientific performance. Preserve the older
β = [−3,6] examples as named stress fixtures rather than pooling them into the
new primary matrix or attributing their behaviour to separation alone.

Use 50 datasets per base cell for the pilot and 1,000 for the main fixed-truth
study: 8,000 independent datasets, 96,000 prior fits, 288,000 method rows for
adaptive reference, raw VI and LRVB. The rows are not independent replications.
At 95% coverage, 1,000 replications give MCSE about 0.69 percentage points.
Do not make sub-percentage-point superiority claims from this design.

Primary outputs, with explicit available and attempted counts:

- Fixed-truth interval inclusion, posterior-mean bias/RMSE and mean interval
  width; paired differences on shared datasets with paired MCSEs.
- Approximate intervals' probability content under the reference posterior.
- Mean error in reference posterior SD units, covariance discrepancy, and
  lower/upper policy quantile error in arm and cohort units.
- Raw/LRVB comparison against the same fitted model and prior; prior comparisons
  against each prior's own reference. Keep those two comparisons separate.
- Reference and fit availability, numerical ambiguity, realised separation,
  convergence and prior/posterior variance diagnostics. Do not describe variance
  ratios alone as a complete measure of prior influence.

Compute cheap CDF-at-truth coverage for every replication. Preselect the first
100 deterministic replication IDs per base cell for expensive reference moments,
quantiles and full interval-fidelity metrics across every prior. The subset is
chosen before observing outcomes or convergence. Report its own denominator and
uncertainty; do not present it as 1,000-replication endpoint evidence.

For both kinds of paired comparison, report common-available differences and
availability differences. All-attempt inclusion counts unavailable intervals as
noncoverage and must retain that name. Width/fidelity contrasts cannot substitute
missing intervals with zero. Account for pairing across priors in any pooled
analysis; use dataset-level resampling within base cells if pooling is needed.

Run a separate prior-predictive experiment for zero-centred priors only:
two n values × two allocations × three prior scales, 1,000 datasets per cell.
Draw β from the exact fitted prior before simulating outcomes. This adds 12,000
independent datasets and 12,000 fits. Report PIT distributions and central
50/80/95% inclusion with Monte Carlo uncertainty. A 95% inclusion number alone
does not establish full posterior calibration. Oracle fixed-truth cells must
never be pooled with this experiment or labelled SBC.

Limit primary contrasts to LRVB versus raw VI and declared scale/centring
contrasts. Report effect sizes and MCSEs; label additional contrasts exploratory.
If formal testing is used, specify multiplicity handling before execution.

### Phase 3: the actual public procedure and practical cost

Create `test/stress/policy_public_api_validation.jl` and focused contract tests.
Use the existing `fit_bayesian_logistic` and `bayesian_policy_contrast` APIs;
do not duplicate their estimator equations or bypass eligibility gates.

Select at least 100 shared datasets per base cell before results are inspected.
For each deployable zero-centred prior, compare public raw VI and LRVB intervals
against their ideal Gaussian-law endpoints and the reference posterior. Use the
actual documented default draw count plus declared smaller/larger counts. Repeat
with 20 independent draw streams per dataset. Fit once per dataset/prior where
the public contract allows it, then vary only the interval draws.

Separate between-dataset sampling uncertainty from within-dataset endpoint Monte
Carlo variability. Resample datasets, not individual draw-stream records, for
uncertainty on aggregate performance. Check endpoint variability, reference
content, mean/width, failures, documented defaults and draw-count sensitivity.
Keep posterior simulation error separate from uncertainty about the estimand.

Benchmark identical model/design/prior configurations, with the same convergence
tolerances. Record first-call compilation, warmed fitting, covariance correction,
public interval draws and reference integration as distinct components. Include
allocations, draw counts, threads, hardware and backend versions. Fit-level
performance and reference-compression speedups must not be conflated.

### Phase 3A: turn diagnosed weaknesses into package improvements

This phase is required. Implement a small candidate change, test it, then
evaluate it against the preserved baseline. Avoid accumulating unrelated
changes before learning whether the first one helps. The supervisor owns the
statistical decision; implementation agents receive an agreed contract and
non-overlapping file ownership.

Inspect `src/bayesian_logistic.jl`, `ext/CausalTargetedRxInferExt.jl`,
`src/bayesian_gcomp.jl` and the existing Bayesian tests before choosing a remedy.
Reuse the current `status`, `diagnostics`, `metadata`, covariance selection and
unavailable-interval machinery. Names below describe responsibilities rather
than requiring new result types or duplicate APIs.

| Diagnosed problem | Candidate package action | Required evidence |
| --- | --- | --- |
| Numerical or convergence defect | Repair the responsible computation or stopping rule in the existing fit path | Minimal failing regression; independent reference agreement; convergence/failure behaviour retained |
| Inaccurate Gaussian interval approximation | Improve the justified approximation, expose a supported accuracy diagnostic, or withhold an unsupported interval | Same model/prior comparison against the reference; coverage, width, endpoint error and availability trade-offs |
| Important diagnostics exist internally but are inaccessible | Expose and document the existing diagnostic through fit/result metadata before adding new calculations | Public API tests for meaningful values, statuses and unavailable cases; backward compatibility |
| Scientifically consequential prior sensitivity | Provide or improve an explicit refit/comparison workflow on shared data, with caller-declared priors and transparent diagnostic summaries | Prior IDs/designs preserved; failures retained; no truth-dependent or outcome-scaled default; informative examples |
| Excessive finite-draw endpoint variation | Improve or document the sampling procedure, precision reporting or draw controls | Repeated draw-stream checks; endpoint variability and cost measured separately from posterior uncertainty |
| Avoidable fitting/correction/allocation cost | Optimise the existing implementation without changing the statistical target or declared tolerances | Equivalent numerical results and statuses; reproducible fit-level timing and allocation improvement |
| Misspecified likelihood or unsupported identification | Make assumptions and limitations actionable; maintain the existing identification/eligibility boundary | Public examples and tests prevent an accuracy or identification claim that the method cannot support |

For any withholding or warning rule, distinguish convergence, numerical validity
and statistical approximation quality. A positive-definite covariance is not a
certificate of interval accuracy. A candidate runtime rule may use only quantities
available from the supplied data, declared prior and fitted object. It must not
inspect simulated truth or require the validation reference. Develop and assess
the rule on development data, then lock it before held-out evaluation. Measure
false reassurance and unnecessary withholding as well as conditional coverage.
Do not claim a universal diagnostic from success on the bivariate examples.

Prior conflict is not generally identifiable from a small dataset alone. Do not
automatically declare a scientific prior wrong, widen intervals until they cover
the simulated truth, or replace the caller's prior. Report sensitivity as a
comparison conditional on declared choices. An oracle prior is never deployable.
Retain explicit unavailable results when the requested covariance correction
fails; never silently substitute another covariance or posterior law.

Each accepted change follows this sequence:

1. Register a concrete public behaviour and before/after acceptance criterion
   in the decision record. Define any accuracy loss or cost budget explicitly.
2. Write a failing regression or independently checked numerical fixture before
   changing runtime code. Preserve the original failure as a small CI test.
3. Make the smallest coherent Julia change. Add docstrings and public API tests;
   preserve return contracts or document/version a necessary API change.
4. Rerun matched baseline/candidate comparisons on the same development datasets
   and RNG streams. For speed changes, require accuracy within declared
   tolerances; for uncertainty changes, report width/error/availability costs.
5. Reject or revise changes without a defensible benefit. Retain negative
   findings in the record rather than tuning benchmarks or deleting hard cases.
6. Lock the candidate code, priors, thresholds and defaults before phases 4–5.

The first development milestone must include remedies for substantiated public
API or inference gaps, their regression tests, and updated user guidance. Where
no remedy is justified, record why the existing behaviour is adequate or which
limitation remains. A larger simulation archive alone is not that milestone.

### Phase 4: realistic causal mechanisms, separately specified

Only start after phases 1–3A are reviewed. Write an extension protocol before
running the new scenarios; use `test/stress/causal_policy_validation.jl` plus
small deterministic SCM/target fixtures.

Generate two observed baseline confounders with controlled correlation, exposure
from a declared propensity model, and outcomes from a declared structural
mechanism with independent exogenous errors. Vary one difficulty at a time:
covariate correlation, overlap, treatment interaction and nonlinear response.
Compare correctly specified bases with deliberate omissions. Keep missingness,
measurement error and longitudinal policies as later, separately identified
problems; do not add them all to this matrix.

The known target is n⁻¹Σᵢ[μ_true(1,Wᵢ)−μ_true(Aᵢ,Wᵢ)] on each supplied cohort.
Specify the adjustment set, policy and causal assumptions explicitly. In
misspecified fits, distinguish the causal truth from the model's posterior
functional and any pseudo-true projection. Their disagreement is not automatically
an inference-engine defect. Retain `:user_assumed` unless the existing certificate
and exactly certified design justify another status.

Additional coefficients require a validated higher-dimensional reference.
Do not reuse the bivariate integrator outside its domain or imply that these
examples inherit its accuracy. Specify and validate that reference first.

Compare the locked baseline and candidate package on shared scenarios. If these
cases expose further weaknesses, return them to the phase 3A development loop;
they then count as development evidence, not untouched confirmation. Require
evidence at the dimension and model class for which a public claim is intended.

### Phase 5: appropriate comparators and held-out confirmation

Before interpreting comparative results, match estimands, cohort, intervention,
model basis and interval meaning. Include an appropriate regularised logistic
baseline; fit Bayesian comparators to the same prior where posterior fidelity is
the question. Separate frequentist sampling comparisons from posterior-accuracy
comparisons. Do not rank methods by coverage alone or mix different targets.
MCMC comparisons require diagnostics and effective precision, not equal raw draw
counts. Report reference failures rather than declaring a winner without a check.

The primary practical comparison is the existing package versus the candidate
package. Appropriate external methods supply context; beating them is not a
prerequisite for fixing an implementation defect or improving diagnostics.

Lock any chosen public configuration before held-out evaluation. Reserve new
scenarios, not merely new seeds: for example n=80 with 24 controls and
β=[−0.6,1.6]; n=150 with 30 controls and β=[−2,4]; and n=30 with one control and
β=[−0.3,0.9]. Specify their deployable priors and metrics before generating them.
Use at least 1,000 datasets per held-out cell if the desired coverage MCSE is
about 0.7 percentage points near 95%. If configurations change after inspection,
those cells become development evidence; create a genuinely new confirmation
set. Do not silently reuse them as held out. Declare each change's acceptance
criteria before opening these results: reference fidelity and failure rates,
interval width/error trade-offs, diagnostic operating characteristics, or fit
time/allocations, as applicable. Do not substitute higher coverage for the
criterion originally chosen. A failed confirmation can justify reverting a
change, narrowing its supported scope or retaining experimental status.

### Phase 6: package delivery, claims and release readiness

Completion requires a reviewable package change set with:

- A decision record mapping every material finding to a shipped remedy,
  supported existing behaviour, or explicit remaining limitation.
- Implemented remedies in the owned runtime/API code where justified, with
  regressions, public API checks and proportionate integration tests.
- Before/after evidence on shared development data and untouched confirmation
  scenarios, including adverse effects and unsupported cases.
- User guidance for prior specification, diagnostics, unavailable results and
  the validated operating conditions, with executable public API examples.
- A cleanly reviewed series of local commits and appropriate version/changelog
  changes, ready for the user's push/release decision.

Create a dated validation archive with raw/compressed records, environment and
source hashes, exact commands, protocol amendments, timings and figures. Update
`docs/LOGISTIC_VALIDATION.md`, `docs/LOGISTIC_INFERENCE.md` and the changelog to
match the evidence. Render Documenter locally if its pages change. Do not
overwrite the November archive or imply that old results used new code.

Scientific success means identifying a useful, bounded operating regime or
documenting why no such regime has been established. To claim an improvement,
show fidelity/width/error trade-offs with uncertainty across prespecified
scenarios, not a single coverage increase. Do not claim general calibration,
robustness to misspecification, or speed over MCMC without the corresponding
experiment. Retain experimental labelling when the evidence warrants it.

Assess success at two levels: implementation accuracy conditional on the declared
model/prior, and usefulness for the intended scientific task. A successful
posterior calculation does not itself establish useful causal uncertainty.
Conversely, a truthful unavailable result or an actionable sensitivity workflow
can improve the package without increasing a validation coverage score.

No version bump is needed for validation-only additions. Behavioural fixes need
their own failing regression tests, focused change, appropriate versioning and
separate validation. Commit package changes before the CDCS submodule pointer;
push, release and external messages require the user's instruction.

### Agent allocation and execution gates

Keep the supervisor responsible for estimands, probability-space derivation,
numerical/statistical gates, reference failures and interpretation. Use at most
three bounded implementation agents concurrently, with explicit non-overlapping
file ownership. All agents must preserve concurrent work and stop when an
unresolved statistical decision would otherwise be invented.

| Packet | Owner and files | Delivery and verification |
| --- | --- | --- |
| A: protocol and theory | Supervisor; protocol v2 and mathematical fixtures | Written equations, fixed matrix/metrics/seeds, reviewed gates before agents code |
| B: independent reference | Supervised agent; probability reference and its tests only | Analytic fixtures and dual-reference agreement; supervisor signs off |
| C: paired data and summaries | Cheaper agent; factorial support and matching tests only | Same data across priors, valid counts/RNG, paired MCSE and failure tests |
| D: driver and provenance | Cheaper agent; factorial driver and checkpoint tests only | Complete dataset/prior/method records, interrupted resume, malformed/different-config rejection |
| E: public API and timing | Cheaper agent after B–D; public API validation and tests only | Actual defaults/gates, separate draw streams, dataset-clustered summaries |
| F: package remedies | Supervisor defines statistical contract; cheaper worker owns the specified runtime module and matching tests for one remedy | Reproducer first; concise implementation; public behaviour verified; baseline/candidate evidence |
| G: user guidance and delivery | Cheaper agent owns agreed docs pages; supervisor owns interpretation, versioning and integration | Executable public examples, decision record, confirmation evidence and reviewable commits |

Do not assign two workers `src/bayesian_logistic.jl` or the same extension at
once. Serialise overlapping remedies. The supervisor retains ownership of LRVB
derivations, new inference or gating rules, and decisions about supported claims;
cheaper agents implement only reviewed statistical specifications.

Split each packet into small edits: failing fixture; minimum implementation;
focused checks; interface handoff; then integration. Agree record/function
signatures in packet A before C and D work in parallel. Reuse module APIs from
the existing calibration helpers instead of copying them into each script.

From `packages/CausalTargeted.jl`, use the existing isolated stress environment:

```sh
julia --project=test/stress --threads=auto -e 'using Pkg; Pkg.instantiate()'
julia --project=test/stress --threads=auto test/test_policy_probability_reference.jl
julia --project=test/stress --threads=auto test/test_policy_prior_factorial_support.jl
julia --project=test/stress --threads=auto test/test_policy_prior_factorial_driver.jl
julia --project=test/stress --threads=auto test/stress/policy_prior_factorial.jl --profile pilot --replications 50 --seed 20261115 --out /tmp/policy-v2-pilot
```

These filenames/CLI options are implementation requirements, not existing APIs.
The pilot must cover every prior/method combination, exercise resume and measure
expensive endpoint work before a full-run runtime estimate is accepted. Estimate
the full coverage and endpoint budgets separately, cache deterministic reference
work only by complete validated inputs, and record any budget-driven protocol
change before confirmatory execution. Never thin failures or loosen tolerances
to meet a time budget. Keep large simulations out of ordinary CI; add only small
contract tests. Use BLAS=1 for small matrices and preserve protected package pins.

After focused checks, run the full package suite with visible progress, inspect
`git diff --check`, verify archive checksums and independently review the report.
Passing tests and numerical gates are prerequisites for reporting, not evidence
that a scientific hypothesis is true.

Runtime changes additionally require the matching existing Bayesian tests and
the RxInfer compatibility checks affected by the change. Documenter builds and
examples must use the candidate package. Keep unrelated dirty package/book work
out of these commits. The detailed protocol must state exact commands for each
remedy once its files and acceptance criteria have been agreed.

### Methodological sources

- [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086): simulation
  design, explicit estimands/performance measures and Monte Carlo uncertainty.
- [Talts et al.](https://arxiv.org/abs/1804.06788): simulation-based calibration;
  distinguish prior-predictive checks from fixed-parameter coverage.
- [Giordano, Broderick and Jordan (2018)](https://www.jmlr.org/papers/v19/17-670.html):
  covariance correction and local sensitivity do not establish general accuracy
  of the nonlinear posterior distribution.
