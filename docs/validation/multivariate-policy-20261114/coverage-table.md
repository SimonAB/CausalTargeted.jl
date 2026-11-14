## Multivariate policy calibration

Completed 7000 datasets. Functional: assignment-to-one versus observed treatment in the supplied cohort.

Coverage uses an independently tightened adaptive CDF at the known truth. Gaussian Monte Carlo means, mean MCSEs and equal-tailed widths are descriptive only. Reference widths and errors are unavailable.

| Cell | Method | Available / attempts | Conditional coverage (MCSE) | All-attempt inclusion (MCSE) |
|---|---|---:|---:|---:|
| balanced_n20_fixed_parameter | adaptive_reference | 500 / 500 | 0.96 (0.0088) | 0.96 (0.0088) |
| balanced_n20_fixed_parameter | linear_response | 500 / 500 | 0.972 (0.0074) | 0.972 (0.0074) |
| balanced_n20_fixed_parameter | variational | 500 / 500 | 0.958 (0.009) | 0.958 (0.009) |
| balanced_n20_prior_predictive | adaptive_reference | 500 / 500 | 0.97 (0.0076) | 0.97 (0.0076) |
| balanced_n20_prior_predictive | linear_response | 500 / 500 | 0.982 (0.0059) | 0.982 (0.0059) |
| balanced_n20_prior_predictive | variational | 500 / 500 | 0.964 (0.0083) | 0.964 (0.0083) |
| balanced_n50_fixed_parameter | adaptive_reference | 500 / 500 | 0.954 (0.0094) | 0.954 (0.0094) |
| balanced_n50_fixed_parameter | linear_response | 500 / 500 | 0.954 (0.0094) | 0.954 (0.0094) |
| balanced_n50_fixed_parameter | variational | 500 / 500 | 0.952 (0.0096) | 0.952 (0.0096) |
| balanced_n50_prior_predictive | adaptive_reference | 500 / 500 | 0.958 (0.009) | 0.958 (0.009) |
| balanced_n50_prior_predictive | linear_response | 500 / 500 | 0.964 (0.0083) | 0.964 (0.0083) |
| balanced_n50_prior_predictive | variational | 500 / 500 | 0.932 (0.0113) | 0.932 (0.0113) |
| rare_control_n20_fixed_parameter | adaptive_reference | 500 / 500 | 0.998 (0.002) | 0.998 (0.002) |
| rare_control_n20_fixed_parameter | linear_response | 500 / 500 | 0.998 (0.002) | 0.998 (0.002) |
| rare_control_n20_fixed_parameter | variational | 500 / 500 | 0.998 (0.002) | 0.998 (0.002) |
| rare_control_n20_prior_predictive | adaptive_reference | 500 / 500 | 0.95 (0.0097) | 0.95 (0.0097) |
| rare_control_n20_prior_predictive | linear_response | 500 / 500 | 0.95 (0.0097) | 0.95 (0.0097) |
| rare_control_n20_prior_predictive | variational | 500 / 500 | 0.946 (0.0101) | 0.946 (0.0101) |
| rare_control_n50_fixed_parameter | adaptive_reference | 500 / 500 | 1.0 (0.0) | 1.0 (0.0) |
| rare_control_n50_fixed_parameter | linear_response | 500 / 500 | 1.0 (0.0) | 1.0 (0.0) |
| rare_control_n50_fixed_parameter | variational | 500 / 500 | 1.0 (0.0) | 1.0 (0.0) |
| rare_control_n50_prior_predictive | adaptive_reference | 500 / 500 | 0.948 (0.0099) | 0.948 (0.0099) |
| rare_control_n50_prior_predictive | linear_response | 500 / 500 | 0.952 (0.0096) | 0.952 (0.0096) |
| rare_control_n50_prior_predictive | variational | 500 / 500 | 0.944 (0.0103) | 0.944 (0.0103) |
| separated_n20_fixed_parameter | adaptive_reference | 500 / 500 | 0.0 (0.0) | 0.0 (0.0) |
| separated_n20_fixed_parameter | linear_response | 500 / 500 | 0.0 (0.0) | 0.0 (0.0) |
| separated_n20_fixed_parameter | variational | 500 / 500 | 0.0 (0.0) | 0.0 (0.0) |
| separated_n50_fixed_parameter | adaptive_reference | 500 / 500 | 0.084 (0.0124) | 0.084 (0.0124) |
| separated_n50_fixed_parameter | linear_response | 500 / 500 | 0.0 (0.0) | 0.0 (0.0) |
| separated_n50_fixed_parameter | variational | 500 / 500 | 0.0 (0.0) | 0.0 (0.0) |
| strong_prior_n20_fixed_parameter | adaptive_reference | 500 / 500 | 0.726 (0.0199) | 0.726 (0.0199) |
| strong_prior_n20_fixed_parameter | linear_response | 500 / 500 | 0.726 (0.0199) | 0.726 (0.0199) |
| strong_prior_n20_fixed_parameter | variational | 500 / 500 | 0.726 (0.0199) | 0.726 (0.0199) |
| strong_prior_n20_prior_predictive | adaptive_reference | 500 / 500 | 0.93 (0.0114) | 0.93 (0.0114) |
| strong_prior_n20_prior_predictive | linear_response | 500 / 500 | 0.932 (0.0113) | 0.932 (0.0113) |
| strong_prior_n20_prior_predictive | variational | 500 / 500 | 0.93 (0.0114) | 0.93 (0.0114) |
| strong_prior_n50_fixed_parameter | adaptive_reference | 500 / 500 | 0.764 (0.019) | 0.764 (0.019) |
| strong_prior_n50_fixed_parameter | linear_response | 500 / 500 | 0.752 (0.0193) | 0.752 (0.0193) |
| strong_prior_n50_fixed_parameter | variational | 500 / 500 | 0.752 (0.0193) | 0.752 (0.0193) |
| strong_prior_n50_prior_predictive | adaptive_reference | 500 / 500 | 0.938 (0.0108) | 0.938 (0.0108) |
| strong_prior_n50_prior_predictive | linear_response | 500 / 500 | 0.936 (0.0109) | 0.936 (0.0109) |
| strong_prior_n50_prior_predictive | variational | 500 / 500 | 0.934 (0.0111) | 0.934 (0.0111) |

All-attempt inclusion counts unavailable intervals as noncoverage. Failure reasons and separate denominators appear in summary.tsv. Paired LRVB minus raw VI coverage differences and paired MCSEs appear in paired.tsv.

Prior-predictive PIT deciles have an SBC interpretation; fixed-parameter CDF diagnostics do not. Fixed-parameter Bayesian coverage need not equal 95%.

Posterior/prior covariance trace ratios use tr(posterior_covariance)/tr(prior_covariance). Source hashes, backend versions, source head/dirty state and first-call compilation timing appear in metadata.toml.
