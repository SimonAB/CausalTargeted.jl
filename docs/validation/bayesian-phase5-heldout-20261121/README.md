# Phase 5 Untouched Held-Out Confirmation Archive (1,000 Replications, 21 November 2026)

Compact summary tables (`metadata.toml`, `report.md`, `summary.tsv`,
`paired_methods.tsv`) from the 1,000-replication untouched held-out confirmation
study (`phase5_heldout_confirmation.jl`, 3 reserved scenarios × 2 deployable
priors × 4 methods = 24,000 method rows, seed `2026112105`). Large `raw.tsv` is
retained locally outside Git.

## Provenance & SHA-256 Checksums

- `summary.tsv`: `237afd4824a29e4ad48a52d4f7f1b9c09a2d96dece08e486732c4dcdbbf74f10`
- `paired_methods.tsv`: `4331409b6c27551923b94e5108351291c676d6c56afd60bcecd133a2fa439608`
- `metadata.toml`: `6c6426557d38a13dddc774aaa44d353122f62661c227286c73cc78ec7a8eee81`
- `report.md`: `873713d5460781559d73fa2591bdb4333934995d81a7b0e85d3d0a1943fc0b83`
- `raw.tsv` (local `/tmp/cdcs-phase5-heldout/raw.tsv`): `2ca0ce2bf91b307a6e92b341d5dd743da638f4e3e2875d40ef3ec93fcaf76911`

## Headline Findings

1. **`linear_response` (LRVB) outperforms both `variational` (Raw VI) and
   matched `laplace_map` across all 6 untouched held-out cell×prior pairs:**
   - `heldout_n150_c30_b4p0 / zero_s3p0` (near-separation large effect):
     covariance error `46.0%` (Raw VI) / `8.7%` (Laplace MAP) → **`6.3%`** (LRVB);
     exact posterior content `86.5%` / `94.2%` → **`94.7%`**;
     fixed-truth coverage `88.4%` / `96.0%` → **`96.4%`** (exact PSIS `96.2%`).
   - `heldout_n30_c1_b0p9 / zero_s3p0` (single control unit):
     covariance error `12.8%` / `11.8%` → **`5.3%`**; exact posterior content
     `94.2%` / `94.6%` → **`95.0%`**.
   - `heldout_n80_c24_b1p6 / zero_s3p0`:
     covariance error `11.8%` / `4.4%` → **`3.1%`**; exact posterior content
     `93.5%` / `94.5%` → **`94.7%`**; endpoint error `0.0050` / `0.0048` →
     **`0.0040`**.
2. **Prior-conflict confirmation (Decision D2):** under `heldout_n150_c30_b4p0 / zero_s1p5`
   ($\beta_1 = 4.0$ with prior SD $1.5$), even the exact PSIS reference drops to
   `82.8%` fixed-truth coverage while LRVB maintains `94.8%` exact posterior
   content (`covErr = 2.6%`), confirming that prior-truth disagreement is a
   declared-prior property rather than an inference-engine defect.
