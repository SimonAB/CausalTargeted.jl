"""Categorical-treatment LMTP via Díaz–Williams 2n density-ratio classification."""

include("discrete_lmtp/policy.jl")
include("discrete_lmtp/density_ratio.jl")
include("discrete_lmtp/estimator.jl")

export DiscreteTreatmentPolicy, DiscreteInterventionalMean
export discrete_recode_policy, discrete_static_policy, discrete_shift_policy
export apply_discrete_policy, run_discrete_lmtp, run_discrete_lmtp_contrast, discrete_positivity
