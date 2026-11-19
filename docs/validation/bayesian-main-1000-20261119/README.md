# Stage 2 Main-Run Validation Archive (1,000 Replications, 19 November 2026)

Compact summary tables (`metadata.toml`, `report.md`, `summary.tsv`, `paired_methods.tsv`) from the locked 1,000-replication studies in [BAYESIAN_VALIDATION_PROTOCOL_V2.md](../../BAYESIAN_VALIDATION_PROTOCOL_V2.md). Large raw tables (`raw.tsv`, `paired_priors.tsv`) are kept out of Git.

## Provenance & SHA-256 Checksums

- **Prior-predictive calibration** (`prior-predictive/`, 12 cells × 1,000 datasets = 12,000 datasets, 36,000 rows, 100.000% available, 0 failed):
  - `raw.tsv` (27 MB, local): `c20e9b8a0d257ae4e05ed66b07bd0d31f6705b81a0fdc6c3c964d9ca06ea977a`
  - `summary.tsv`: `3cf092890d8cf7406bdab6a62aae4626894a148504d22dcc6558759b8d9ed486`
  - `paired_methods.tsv`: `abbf9b4ed0bb5b8ac1c5dd27fd5bcea4487bb6b7bd75f6054b0eab5562313268`
  - `metadata.toml`: `c7b6c22b68715e89eb5ed4b2a317a2ed527f032715735eded517480bc515e2d3`
  - `report.md`: `f83debb2a7a05415706321d1ee436cc1abccb93adf8993ad15d19be4790449da`

- **Paired fixed-parameter factorial** (`fixed-parameter/`, 8 cells × 1,000 datasets × 12 priors = 288,000 rows, 99.995% available, 15 `cdf_disagreement` rows):
  - `raw.tsv` (207 MB, local): `c244b908529d8c5c4526b2958d59735816a998e77c63bbfdd9d19654d048de7c`
  - `paired_priors.tsv` (708 KB, local): `456ce2f8fb50d009535a328ebae20927a5179261e5a279cf5ffd75329712746d`
  - `summary.tsv`: `05f71281821522f1df6ba672a00e8f92b49513e82ad2c60bb579c4e564fc9a14`
  - `paired_methods.tsv`: `4da3ddea1ba2a15c2b85372c857eb3e2722832bf5948d300b27ef97c1601680b`
  - `metadata.toml`: `58cae4f9962ee6afc39e05ece51bd147bea7df6abba19d1929359748d4021959`
  - `report.md`: `6da34cc76512e8c7b179c0a5c30650763153ec1af8de3acaaa9401054da686c4`

## Headline Findings

1. **Prior-predictive calibration**: `adaptive_reference` achieves 94.2%–96.2% 95% coverage across all 12 cells. `linear_response` (LRVB) cuts relative Frobenius covariance error by 1.0–52.4 percentage points over `variational` (Raw VI) across every cell (`60.0% → 7.6%` in `n50_c25_s3p0`), avoiding Raw VI's undercoverage at `s = 3.0` (`86.1%–92.2%`) while trending mildly conservative (`96.8%–98.1%`).
2. **Paired fixed-parameter factorial**: confirms Decision D2 at 1,000-replication precision — under tight misspecified `zero_s0p5`, exact `adaptive_reference` drops to 36.5% mean coverage (min 0.0%). Across all 12 priors, LRVB reduces covariance error by 3.4–27.8 percentage points over Raw VI and holds exact posterior content at 94.9%–95.2% (vs Raw VI's 92.7%–94.9%).
