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
  **The full suite is unverified**, not passed or established as failing.
  The Memoization redefinition warning does not
  establish which dependency caused the loading problem.
- CI has not run on this local validation pass. No version bump, release or
  push was performed.

Production estimator source is unchanged. These checks establish the focused
validation contracts and study evidence; they do not establish universal
statistical calibration or substitute for the outstanding full-suite run.
