## Registration status

CausalTargeted is **not** yet on General. Project.toml has **no `[sources]`**
(path wiring belongs in the CDCS Manifest only).

First registration target: **v0.3.2** (drops path `[sources]`; adds CI / TagBot).

### Checklist

1. CausalDynamics on General (`0.3` compat) — done
2. Clean-env `Pkg.test` against registry CausalDynamics — done for this tree
3. TagBot + CI workflows — present
4. `@JuliaRegistrator register` on the `v0.3.2` commit / release issue
