## Bayesian package improvement: first milestone

Implemented and locally verified on 14 November 2026, with master seed 20261115.
This delivers a public simulation-precision diagnostic, stronger independent
references and paired validation infrastructure. It does not improve or certify
the Gaussian posterior approximation itself. All studies use synthetic data.

### What changed for package users

`bayesian_policy_contrast(::BayesianLogisticFit, ...)` now returns
`endpoint_mc_bounds`, with separate lower/upper confidence bounds for each
ideal Gaussian-law interval endpoint. `mc_level=0.95` is the default.
Existing `mcse` still describes means. The new bounds reuse the coefficient
draws and use binomial order statistics, with risk-difference support bounds
when the requested order statistic falls outside the sample.

These are pointwise Monte Carlo confidence bounds conditional on the selected
Gaussian moments. They are not posterior credible intervals, simultaneous grid
bounds, or estimates of prior/model/approximation error. A proper covariance and
convergence still do not certify posterior accuracy. Failed fitted uncertainty
continues to withhold all uncertainty diagnostics; no covariance fallback or
default prior has been introduced.

At matching RNG states, the preserved baseline policy function and candidate
returned identical means, covariance, credible limits, mean MCSE and plug-in
estimates. The [benchmark](endpoint-benchmark.txt) found 48,016/141,280/467,744
additional allocated bytes for 1/3/10 policies at 2,000 draws, about 1–2% of
the original call allocations. Median times were noisy under concurrent work:
for 10 policies, 7.70 versus 7.85 ms. This is modest diagnostic overhead,
not an inference speedup claim. [Reproduction script](benchmark.jl).

The user manual adds endpoint interpretation and an executable same-data prior
comparison workflow. No outcome-derived prior scaling or automatic prior
selection is permitted. Existing LRVB equations, stopping criteria, posterior
selection and estimands are unchanged. All runtime additions remain Julia,
using existing dependencies and optional extension loading.

### Evidence for the endpoint diagnostic

The exact rank calculation was checked on 378 additional configurations by an
independent agent. The public-API study used eight declared bivariate scenarios,
100 independent datasets per scenario, three deployable zero-centred priors,
two public covariance modes, 20 streams and 500/2,000/8,000 draws. That is
800 datasets, 4,800 actual public fits and 288,000 stream evaluations; all public
results and ideal-law references were available. Driver-recorded runtime was
933.90 seconds, excluding imports and including compilation within main.
The two covariance modes were fitted separately; their correction time is
bundled in public fit time, not independently benchmarked.

Rates below average priors/streams within each dataset, then weight the eight
scenarios equally. Parentheses contain stratified dataset-level MCSEs in
percentage points. Streams and prior fits are not treated as independent data.

| Mode | Draws | Lower endpoint MC-bound inclusion | Upper endpoint MC-bound inclusion | Exact binomial expectation |
| --- | ---: | ---: | ---: | ---: |
| Raw VI | 500 | 96.89% (0.097) | 96.87% (0.108) | 97.00% |
| Raw VI | 2,000 | 95.09% (0.136) | 95.40% (0.126) | 95.54% |
| Raw VI | 8,000 | 95.35% (0.124) | 95.54% (0.130) | 95.51% |
| LRVB | 500 | 96.95% (0.104) | 96.84% (0.107) | 97.00% |
| LRVB | 2,000 | 95.04% (0.145) | 95.51% (0.123) | 95.54% |
| LRVB | 8,000 | 95.25% (0.133) | 95.55% (0.131) | 95.51% |

At the default 2,000 draws, LRVB endpoint RMSE relative to the ideal selected
Gaussian law was 0.00339 (lower) and 0.00237 (upper), in risk-difference units.
At 8,000 draws it was 0.00168 and 0.00119. More draws reduce endpoint simulation
error; this does not repair prior conflict or posterior approximation error.
These averages depend on the chosen scenarios and cohort multipliers.

**The default-draw lower-endpoint discrepancy remains unresolved.** The raw/LRVB
rates are respectively 3.33/3.46 estimated MCSEs below the exact discreteness
expectation, even though both are near nominal 95%. Their deviations are strongly
paired (dataset-level correlation 0.799), not independent confirmations. They
must not be described as automatic agreement or dismissed merely as multiplicity.

The audit found unique seeds, correctly reconstructed archived flags, and
same-seed Normal controls near their binomial expectations. Independently
reconstructing 16 fitted Gaussian laws and 320 streams reproduced order-statistic
fields within 5.55e-17; probability-space endpoint checks had maximum residual
9.76e-9. No implementation, rank or reference defect was identified. Chance is
plausible, but these checks neither establish all pseudo-stream independence
nor exclude a data-specific issue outside the reconstruction. Further independent
replication is warranted before a broad empirical calibration claim. The main
data, seeds and defaults were not changed in response to the finding.

See [analysis](public-analysis.txt), [seed audit](seed-audit.txt),
[stratification audit](strata-audit.txt) and
[independent reconstruction](independent-reconstruction.txt).

### Reference and prior-sensitivity pilots

The separate probability-space implementation evaluates the transformed
Gaussian prior, binomial arm likelihood and Jacobian without calling the
coefficient-space posterior kernel or tail helpers. It passed 323 fixture
assertions covering prior-only, balanced, one-control, separated, strong-prior
and correlated-prior cases, including direct tails and endpoint agreement.
This establishes selected numerical checks, not universal reference accuracy.

| Pilot | Independent datasets | Method rows | Available truth-inclusion evaluations |
| --- | ---: | ---: | ---: |
| Paired fixed-truth priors | 400 | 14,400 | 14,399 |
| Prior predictive | 600 | 1,800 | 1,800 |
| Original fidelity | 16 | 576 | 576 |
| Amended fidelity A1 | 16 | 576 | 576 |

The single fixed-truth unavailability is a Gaussian CDF refinement disagreement
for an oracle-offset prior; it is retained, not silently dropped or interpreted
as a failed fit. The fixed/prior-predictive pilots use 50 replications per cell
and do not replace the planned 1,000-replication studies. Oracle priors are
diagnostics, not deployable choices or SBC experiments. Summaries include
numerical-ambiguity-aware central 50/80/95% inclusion.

The fidelity pilot preselected replication 1 in each base cell, with replication
2 retaining coverage only. Of 288 selected method rows, all endpoints and
reference interval contents passed, but 75 moment comparisons were unavailable
under the original mixed raw-moment error gate. A reproduced case had excellent
refinement (standardised mean change 9.87e-8; covariance change 3.82e-7), while
the aggregate coarse error/mass was 7.89e-5 because the raw vector was dominated
by a second moment near 10.14. This was a diagnostic-contract mismatch, not
demonstrated posterior-moment inaccuracy.

Amendment A1 specifies centred, marginal-SD-scaled moment integration and
propagated dimensionless mean/covariance estimates. It was tested on analytic
Gaussian moments including very large centres, then rerun on exactly the same
pilot datasets in a new archive. All 288 selected moment/endpoint/content
comparisons passed. Probability and endpoint gates did not change. The original
75 unavailable comparisons remain in [the original archive](fidelity-original/report.md).
These quadrature estimates remain heuristic, not rigorous bounds.

Amendment A2 prevents ambiguous truth inclusion from discarding independently
validated fidelity metrics. Six regression assertions cover individual and
paired summaries. The original tables are retained; the amended pilot uses
the corrected summariser. No raw records were overwritten or retroactively
relabeled as successful.

### Verification, reproduction and remaining work

- Full package suite: 3,498 assertions passed, including plotting.
- Final isolated validation CI command: 498 assertions passed (v1/v2 driver,
  public validation contracts and independent reference). A1 contributes 28
  new driver assertions; A2 adds six separately passing support regressions.
- The three documented example blocks executed successfully after `Pkg.resolve`
  refreshed a stale ignored docs manifest. Local Documenter rendered successfully
  with deployment disabled; one pre-existing getting-started HTML-size warning
  remains. Hosted CI/deployment has not run on these local changes.
- Whitespace/YAML, checkpoint digests, raw checksums and generation-source hashes
  were checked. Original and amended generation source snapshots are separate.

Run the new scripts from the package root with the `test/stress` environment,
`--threads=auto`, and BLAS=1. The
[protocol and amendments](../../BAYESIAN_VALIDATION_PROTOCOL_V2.md) contain cells,
metrics, seeds and numerical gates. Driver `--help` is not implemented: use the
documented explicit options in the source and protocol.

`generation-source.tar.gz` reproduces the original pilots and public study;
`a1-source.tar.gz` records the amended fidelity generation. Package project and
isolated stress manifest snapshots are included for provenance. The source
version was 0.4.2 plus the unreleased candidate, identified by content hashes.
No version bump, release or external publication is implied by these studies.

Small raw pilots are committed as gzip files. The 193.7 MB main raw table is
retained locally as `public-main/raw.tsv.gz` (about 42 MiB) and deliberately
excluded from Git/package downloads. Its raw SHA256 is in metadata; the gzip
hash is in `RAW_ARTIFACTS.sha256`. Tracked reports, metadata, summaries, code and
small archives are covered by `SHA256SUMS`. To reproduce the main audit, unpack
the original source snapshot and decompress the raw table into a disposable
directory with metadata, then run `audit_public.jl STUDY_DIR SOURCE_DIR` using
the stress environment. Regeneration uses:

```sh
julia --project=test/stress --threads=auto test/stress/policy_public_api_validation.jl --profile main --replications 100 --streams 20 --draws 500,2000,8000 --seed 20261115 --out /tmp/public-api-main
```

The first package-improvement milestone is delivered. The large paired factorial
study, higher-dimensional/confounded mechanisms, misspecification and untouched
scenario confirmation remain unfinished. No improvement in general posterior
accuracy, fixed-truth coverage or causal identification is claimed here.
