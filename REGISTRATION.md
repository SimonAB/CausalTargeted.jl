## Registration status

CausalTargeted is **not** yet on General. Project.toml has **no `[sources]`**
(path wiring belongs in the CDCS Manifest only).

First registration target: **v0.3.2** (drops path `[sources]`; adds CI / TagBot).

### Checklist

1. CausalDynamics on General (`0.3` compat) — done
2. Clean-env `Pkg.test` against registry CausalDynamics — done for this tree
3. TagBot + CI workflows — present
4. Install [JuliaRegistrator](https://github.com/apps/juliareistrator) on `SimonAB/CausalTargeted.jl` if it is not already (required for `@JuliaRegistrator` to reply)
5. `@JuliaRegistrator register` on [issue #1](https://github.com/SimonAB/CausalTargeted.jl/issues/1) / the `v0.3.2` commit
