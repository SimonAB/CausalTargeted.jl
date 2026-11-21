# Phase 4 Multi-Covariate Causal Policy Pilot Archive (21 November 2026)

Compact summary tables (`metadata.toml`, `report.md`, `summary.tsv`,
`paired_methods.tsv`) from the 50-replication Phase 4 pilot under
[BAYESIAN_VALIDATION_PROTOCOL_V3.md](../../BAYESIAN_VALIDATION_PROTOCOL_V3.md)
(6 mechanisms × 2 sample sizes × 2 priors = 1,200 paired datasets, 3,600 method
rows, 100% availability). The raw table (`raw.tsv`) is retained locally outside
Git.

## Provenance & SHA-256 Checksums

- `summary.tsv`: `efe46a60efe15794b4b71b34484ac5ff8927934e36133f3c3ab24c6c612e9d43`
- `paired_methods.tsv`: `37fd3258a8953209bbc358a74ac3ef99066d2c30568f03c1a0014bb8bafb29fa`
- `metadata.toml`: `5824f9ede7ffda4a0807453880ed62e953d3a98ba88e038d3913250e102a38bc`
- `report.md`: `c84562c857a18514d1a5e063f069a62e222d685209e6780386ca0dcb0b8c3b8c`
- `raw.tsv` (local `/tmp/cdcs-phase4-pilot/raw.tsv`): `fa620a44e7d7fad9dee17144f1d4548639fbe496ca93bc2a2a45d5143820a75b`

## Headline Findings

1. **LRVB dominates Raw VI across all 24 cell×prior pairs:** relative Frobenius
   covariance error drops by **9.0–30.5 percentage points** in 24/24 pairs
   (largest gain in `mc_interact_spec_n80 / zero_s3p0`: `36.9% → 6.4%`), exact
   posterior content stays on nominal `94.7%–95.2%` (vs Raw VI `93.3%–94.1%`),
   and max 95% endpoint error improves in 24/24 pairs (`0.0026–0.0063` vs
   `0.0036–0.0085`).
2. **Two-tier reference verified:** Tier 1 Laplace-Student-$t$ PSIS achieved
   1,200/1,200 (100%) availability; Tier 2 Turing NUTS spot-check across the 12
   hardest `n = 80, s = 3.0` datasets (`scripts/stress_harness/run_phase4_nuts_spotcheck.jl`)
   confirmed `max_endpoint_diff <= 0.006`, `max_rhat <= 1.003`, 0 divergences,
   and `mean_diff_sd ∈ [0.013, 0.056]` (within ~2.5× NUTS MCSE at `ESS ≈ 2,000`).
