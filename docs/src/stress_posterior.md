# Missingness posterior stress

Opt-in Gaussian MAR nested-MC imputation (`impute_posterior`) pooled into
`run_lmtp_grid(...; imputation=draws)` under Rubin's rule. Cheap `:drop` /
`:ipcw` remain defaults; Turing / RxInfer backends are deferred.

**Quarto:**
[`docs/stress/missingness_posterior_stress.qmd`](https://github.com/SimonAB/CausalTargeted.jl/blob/main/docs/stress/missingness_posterior_stress.qmd)

```bash
cd docs/stress && quarto render missingness_posterior_stress.qmd
```

Structural claims come from CausalDynamics `MissingnessSpec` /
`certify_missingness` (see `identify(...; missingness=)`).
