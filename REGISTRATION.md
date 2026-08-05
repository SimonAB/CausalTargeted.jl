## Registration status

CausalTargeted.jl is on the Julia **General** registry.

Install: `Pkg.add("CausalTargeted")`. Requires Julia **1.12+**.
Hard dependency: [CausalDynamics.jl](https://github.com/SimonAB/CausalDynamics.jl) (`0.3` compat).

| Version | Status |
|---------|--------|
| **0.3.2** | On General ([#163199](https://github.com/JuliaRegistries/General/pull/163199), merged 2026-08-05) — first registration; tip matches `main` |

Local `main` is at **0.3.2**. Project.toml has **no `[sources]`** (path wiring belongs in the CDCS Manifest only).

## First registration checklist (completed)

1. CausalDynamics on General (`0.3` compat) — done
2. Clean-env `Pkg.test` against registry CausalDynamics — done
3. TagBot + CI workflows — present
4. [JuliaTeam Registrator](https://github.com/apps/juliateam-registrator/installations/new) installed on `SimonAB/CausalTargeted.jl`
5. `@JuliaRegistrator register` on [issue #1](https://github.com/SimonAB/CausalTargeted.jl/issues/1) / the `v0.3.2` commit — done; tag `v0.3.2` present

## Subsequent versions

Register **incrementally** (no version skips) so AutoMerge stays happy:

1. Bump `version` in `Project.toml`, commit, push
2. Comment `@JuliaRegistrator register` on an issue or the release commit
3. Wait for the General PR to AutoMerge; TagBot tags if needed

## Zenodo DOI

Deposit metadata is in `.zenodo.json` and `CITATION.cff`. See [packages/ZENODO.md](../ZENODO.md) in the CDCS repo.
