## Paired policy prior factorial validation

Completed 16 independent datasets. Schema v2; rows across priors and methods are paired.

Known-truth coverage uses tightened CDF integration for every attempt. A failed posterior reference does not suppress Gaussian-law coverage. All-attempt inclusion counts unavailable intervals as noncoverage.

Fidelity uses the prespecified replication subset and ideal-law endpoints, with separate availability and finite metric counts. Unselected fidelity metrics are NaN. Coefficient mean error is the maximum absolute mean discrepancy divided by its reference marginal SD; covariance error is relative Frobenius norm. Endpoint errors are maximum absolute errors in cohort units, with arm and signed endpoint errors retained.

Monte Carlo descriptive means use common standard draws across all priors and methods. Widths outside the fidelity subset are finite-draw descriptions; selected successful fidelity rows use ideal-law widths. Fixed-truth inclusion is not SBC. Prior-predictive calibration applies only to matching zero-centred priors.

See summary.tsv for separate availability and metric denominators, paired_methods.tsv for LRVB minus raw VI, and paired_priors.tsv for declared paired prior contrasts. Reference, fitting, Gaussian-law assessment and fidelity timings are separate; repeated method rows share the same fit time and must not be summed as separate fits.
