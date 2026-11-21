# Phase 5 Untouched Held-Out Bivariate Confirmation Report

- Replications per cell: 1000
- Master seed: 2026112105
- Posterior contrast draws: 2000
- Threads: 20
- Elapsed seconds: 574.31

## Posterior Fidelity & Causal Recovery Summary

| Cell | Prior | Method | Avail | RefAvail | Cov(avail) | CausalBias | Width | CovErr | MeanErrSD | RefContent | EndpointErr | max(k̂) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout_n150_c30_b4p0 | zero_s1p5 | laplace_map | 988/1000 | 988/1000 | 0.7551 | -0.0179 | 0.0552 | 0.039 | 0.1438 | 0.9448 | 0.0038 | -0.517 |
| heldout_n150_c30_b4p0 | zero_s1p5 | linear_response | 1000/1000 | 988/1000 | 0.773 | -0.0173 | 0.0551 | 0.0263 | 0.0981 | 0.9476 | 0.0032 | -0.517 |
| heldout_n150_c30_b4p0 | zero_s1p5 | tier1_psis | 988/1000 | 988/1000 | 0.8279 | -0.016 | 0.0535 | 0.0 | 0.0 | 0.9501 | 0.0 | -0.517 |
| heldout_n150_c30_b4p0 | zero_s1p5 | variational | 1000/1000 | 988/1000 | 0.691 | -0.0166 | 0.0467 | 0.2978 | 0.0981 | 0.9013 | 0.0042 | -0.517 |
| heldout_n150_c30_b4p0 | zero_s3p0 | laplace_map | 984/1000 | 984/1000 | 0.9604 | -0.0062 | 0.054 | 0.0872 | 0.2142 | 0.9418 | 0.0063 | -0.495 |
| heldout_n150_c30_b4p0 | zero_s3p0 | linear_response | 1000/1000 | 984/1000 | 0.964 | -0.0056 | 0.0539 | 0.0626 | 0.1506 | 0.947 | 0.0055 | -0.495 |
| heldout_n150_c30_b4p0 | zero_s3p0 | tier1_psis | 984/1000 | 984/1000 | 0.9624 | -0.0035 | 0.0501 | 0.0 | 0.0 | 0.9501 | 0.0 | -0.495 |
| heldout_n150_c30_b4p0 | zero_s3p0 | variational | 1000/1000 | 984/1000 | 0.884 | -0.0044 | 0.0405 | 0.4596 | 0.1506 | 0.8648 | 0.0057 | -0.495 |
| heldout_n30_c1_b0p9 | zero_s1p5 | laplace_map | 1000/1000 | 1000/1000 | 0.999 | -0.0037 | 0.025 | 0.0325 | 0.0279 | 0.9472 | 0.0004 | -0.48 |
| heldout_n30_c1_b0p9 | zero_s1p5 | linear_response | 1000/1000 | 1000/1000 | 0.999 | -0.0037 | 0.0251 | 0.0192 | 0.0182 | 0.9483 | 0.0004 | -0.48 |
| heldout_n30_c1_b0p9 | zero_s1p5 | tier1_psis | 1000/1000 | 1000/1000 | 0.999 | -0.0037 | 0.0252 | 0.0 | 0.0 | 0.95 | 0.0 | -0.48 |
| heldout_n30_c1_b0p9 | zero_s1p5 | variational | 1000/1000 | 1000/1000 | 0.999 | -0.0037 | 0.0249 | 0.0271 | 0.0182 | 0.9464 | 0.0004 | -0.48 |
| heldout_n30_c1_b0p9 | zero_s3p0 | laplace_map | 1000/1000 | 1000/1000 | 1.0 | -0.0022 | 0.0314 | 0.1176 | 0.1254 | 0.9456 | 0.0007 | -0.595 |
| heldout_n30_c1_b0p9 | zero_s3p0 | linear_response | 1000/1000 | 1000/1000 | 1.0 | -0.0021 | 0.0317 | 0.0525 | 0.0677 | 0.9502 | 0.0007 | -0.595 |
| heldout_n30_c1_b0p9 | zero_s3p0 | tier1_psis | 1000/1000 | 1000/1000 | 1.0 | -0.002 | 0.0313 | 0.0 | 0.0 | 0.95 | 0.0 | -0.595 |
| heldout_n30_c1_b0p9 | zero_s3p0 | variational | 1000/1000 | 1000/1000 | 1.0 | -0.0021 | 0.0309 | 0.1276 | 0.0677 | 0.9424 | 0.0005 | -0.595 |
| heldout_n80_c24_b1p6 | zero_s1p5 | laplace_map | 1000/1000 | 1000/1000 | 0.928 | -0.0174 | 0.121 | 0.0318 | 0.0668 | 0.9463 | 0.0037 | -0.486 |
| heldout_n80_c24_b1p6 | zero_s1p5 | linear_response | 1000/1000 | 1000/1000 | 0.929 | -0.0168 | 0.1214 | 0.0216 | 0.0452 | 0.9476 | 0.0031 | -0.486 |
| heldout_n80_c24_b1p6 | zero_s1p5 | tier1_psis | 1000/1000 | 1000/1000 | 0.929 | -0.0156 | 0.1217 | 0.0 | 0.0 | 0.95 | 0.0 | -0.486 |
| heldout_n80_c24_b1p6 | zero_s1p5 | variational | 1000/1000 | 1000/1000 | 0.923 | -0.0166 | 0.1174 | 0.0833 | 0.0452 | 0.9392 | 0.0039 | -0.486 |
| heldout_n80_c24_b1p6 | zero_s3p0 | laplace_map | 999/1000 | 999/1000 | 0.9419 | -0.008 | 0.1265 | 0.0443 | 0.0871 | 0.945 | 0.0048 | -0.552 |
| heldout_n80_c24_b1p6 | zero_s3p0 | linear_response | 1000/1000 | 999/1000 | 0.943 | -0.0073 | 0.1269 | 0.0305 | 0.0593 | 0.947 | 0.004 | -0.552 |
| heldout_n80_c24_b1p6 | zero_s3p0 | tier1_psis | 999/1000 | 999/1000 | 0.9389 | -0.0056 | 0.127 | 0.0 | 0.0 | 0.95 | 0.0 | -0.552 |
| heldout_n80_c24_b1p6 | zero_s3p0 | variational | 1000/1000 | 999/1000 | 0.933 | -0.0069 | 0.1211 | 0.1178 | 0.0593 | 0.9347 | 0.005 | -0.552 |

## Paired Method Contrasts (`linear_response` minus comparator)

| Cell | Prior | Contrast | Paired | ΔCov | ΔWidth | ΔCovErr | ΔRefContent | ΔEndpointErr |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout_n150_c30_b4p0 | zero_s1p5 | linear_response_minus_variational | 1000/1000 | 0.082 | 0.0084 | -0.2715 | 0.0463 | -0.001 |
| heldout_n150_c30_b4p0 | zero_s1p5 | linear_response_minus_laplace_map | 988/1000 | 0.0192 | -0.0001 | -0.0127 | 0.0028 | -0.0006 |
| heldout_n150_c30_b4p0 | zero_s3p0 | linear_response_minus_variational | 1000/1000 | 0.08 | 0.0134 | -0.397 | 0.0822 | -0.0003 |
| heldout_n150_c30_b4p0 | zero_s3p0 | linear_response_minus_laplace_map | 984/1000 | 0.0051 | -0.0003 | -0.0246 | 0.0052 | -0.0009 |
| heldout_n30_c1_b0p9 | zero_s1p5 | linear_response_minus_variational | 1000/1000 | 0.0 | 0.0001 | -0.0079 | 0.0019 | -0.0 |
| heldout_n30_c1_b0p9 | zero_s1p5 | linear_response_minus_laplace_map | 1000/1000 | 0.0 | 0.0001 | -0.0134 | 0.001 | -0.0 |
| heldout_n30_c1_b0p9 | zero_s3p0 | linear_response_minus_variational | 1000/1000 | 0.0 | 0.0007 | -0.0751 | 0.0078 | 0.0001 |
| heldout_n30_c1_b0p9 | zero_s3p0 | linear_response_minus_laplace_map | 1000/1000 | 0.0 | 0.0002 | -0.0652 | 0.0046 | -0.0001 |
| heldout_n80_c24_b1p6 | zero_s1p5 | linear_response_minus_variational | 1000/1000 | 0.006 | 0.004 | -0.0617 | 0.0085 | -0.0007 |
| heldout_n80_c24_b1p6 | zero_s1p5 | linear_response_minus_laplace_map | 1000/1000 | 0.001 | 0.0004 | -0.0102 | 0.0014 | -0.0005 |
| heldout_n80_c24_b1p6 | zero_s3p0 | linear_response_minus_variational | 1000/1000 | 0.01 | 0.0058 | -0.0872 | 0.0123 | -0.001 |
| heldout_n80_c24_b1p6 | zero_s3p0 | linear_response_minus_laplace_map | 999/1000 | 0.002 | 0.0004 | -0.0138 | 0.002 | -0.0008 |
