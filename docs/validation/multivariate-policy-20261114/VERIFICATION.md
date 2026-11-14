## Local verification

14 November 2026, Julia 1.13.3 on macOS arm64.

- Focused reference and support checks: 327 assertions passed, including the
  existing logistic-tail regressions, frequency likelihood equivalence,
  analytic Gaussian policy probabilities, nested error accounting and
  simulation/summary contracts.
- Isolated driver/checkpoint checks: 71 assertions passed.
- Full study: 7,000 datasets, 21,000 available method evaluations; all passed
  declared numerical gates. The separate archive audit checked the real CSV
  round trip, source hashes, chained digest, counts and target identities.
- Frequency-compression benchmark script rerun successfully: 32.95 versus
  4.17 ms (7.89×); PIT difference 4.44e-16. Original measurements are retained
  in benchmark.toml, rather than overwritten by the rerun.
- Coverage SVG/PNG rendered and visually inspected. CI YAML parsed and
  `git diff --check` passed.
- Full `Pkg.test("CausalTargeted")` attempted in an isolated temporary
  environment. Dependency precompilation completed, but the test process
  was sampled in package-cache restoration and recursive method-table
  invalidation. The final log subsequently showed 2,364 passing assertions,
  zero failed assertions and one error caused by the requested interrupt.
  The outer testset had taken 14m18s, including 7m29s in the MLJFlux tests.
  This interrupted attempt did not establish a dependency-loading failure.
- A complete rerun in a persistent isolated environment, constructed from the
  package dependencies, test extras and compatibility bounds, passed **2,709
  assertions with zero failures or errors**. A temporary include wrapper
  exposed per-file timings without changing test order or scope. Registered
  CausalDynamics 0.5.1 and CausalMediation 0.2.0 were used. The main testset took
  14m23s; plotting tests also passed. Multi-minute numerical and MLJFlux checks,
  together with delayed summary output, had made the previous run appear
  stalled. The sampled cache invalidation was not evidence of a permanent hang.
- Permanent per-file progress logging was then checked separately: 13
  assertions passed for caller scope, relative paths, return values, immediate
  flushing and exception propagation, including integration inside a testset.
  The full-suite rerun used the temporary progress wrapper; the permanent
  logger was validated by these focused tests.
- CI has not run on this local validation pass. No version bump, release or
  push was performed.

Production estimator source is unchanged. These checks establish the focused
  validation contracts and study evidence; they do not establish universal
  statistical calibration. CI remains unverified until these commits are pushed.
