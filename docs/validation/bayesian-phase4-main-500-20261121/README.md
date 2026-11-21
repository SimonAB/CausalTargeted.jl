# Phase 4 Multi-Covariate Causal Policy Main Archive (500 Replications, 21 November 2026)

Compact summary tables (`metadata.toml`, `report.md`, `summary.tsv`,
`paired_methods.tsv`) from the 500-replication Phase 4 main study under
[BAYESIAN_VALIDATION_PROTOCOL_V3.md](../../BAYESIAN_VALIDATION_PROTOCOL_V3.md)
(6 mechanisms × 2 sample sizes × 2 priors = 12,000 paired datasets, 36,000
method rows, `35,986 / 36,000 = 99.961%` availability, fresh seed `2026112101`).
The raw table (`raw.tsv`) is retained locally outside Git.

## Provenance & SHA-256 Checksums

- `summary.tsv`: `659d9d9404f6a852dcf0f32a8324d197515a21074588dcddfabc0e180c0bd944`
- `paired_methods.tsv`: `b5f0fdb6727a51c37ada7e2a86ab3886609964e9f1fa1806671f3e71a3ddd0d1`
- `metadata.toml`: `d51ccfc22138d4261d951c14b1cfd0164278f1d2cd8ddd22f97deca196797a7e`
- `report.md`: `728fe836ad15e9c75c2583f5b780c2a8e33c12731e556184f44e172a2b662612`
- `raw.tsv` (local `/tmp/cdcs-phase4-main/raw.tsv`): `dd0a1272e5f02c4f1a926627ffd6ed4b98556cb733555742261329e470ab1a2a`

## Headline Findings

1. **`linear_response` (LRVB) strictly dominates `variational` (Raw VI) across
   all 24 cell×prior pairs:**
   - Relative Frobenius covariance error drops by **8.4–29.8 percentage points**
     in 24/24 pairs (largest gain in `mc_interact_spec_n80 / zero_s3p0`:
     `36.1% → 6.4%`; `mc_interact_spec_n200 / zero_s3p0`: `30.7% → 3.0%`).
   - Exact posterior content holds at nominal **`94.7%–95.3%`** across all 12
     cells (vs Raw VI `93.2%–94.1%`).
   - Max 95% endpoint error improves in **24/24** pairs (`0.0026–0.0060` vs
     `0.0038–0.0083`).
2. **Causal recovery vs posterior fidelity:** Under `mc_interact_spec_n80 / zero_s3p0`,
   Raw VI's causal coverage drops to `88.8%` while LRVB holds `92.2%–97.4%`
   across all 12 cells. Under deliberate misspecification (`mc_interact_omit`,
   `mc_nonlinear_omit`), LRVB maintains exact posterior fidelity (`94.8%–95.0%`
   content, `1.9%–4.3%` covariance error) to the fitted model's posterior;
   causal bias reflects model misspecification rather than variational error.
