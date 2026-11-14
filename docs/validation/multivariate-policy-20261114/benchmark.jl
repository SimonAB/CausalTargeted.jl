# Run from the package root with --project=test/stress --threads=auto.
using LinearAlgebra, Statistics
BLAS.set_num_threads(1)
include(joinpath(pwd(), "test/stress/policy_calibration_support.jl"))
include(joinpath(pwd(), "test/stress/policy_calibration_reference.jl"))
using .PolicyCalibrationSupport
cell = only(filter(c -> c.id == "balanced_n50_fixed_parameter", calibration_cells(:validation)))
data = simulate_policy_replication(cell, 1; seed=20261114)
compressed = compress_binary_likelihood(data.A, data.y)
unfolded() = checked_policy_cdf(data.X, data.y, cell.prior_mean,
    cell.prior_covariance, data.arm_truth)
grouped() = checked_policy_cdf(compressed.X, compressed.y, cell.prior_mean,
    cell.prior_covariance, data.arm_truth; weights=compressed.weights)
raw, weighted = unfolded(), grouped()
@assert raw.available && weighted.available
@assert isapprox(raw.pit, weighted.pit; atol=1e-10)
raw_seconds = median([@elapsed unfolded() for _ in 1:7])
compressed_seconds = median([@elapsed grouped() for _ in 1:7])
println((; raw_seconds, compressed_seconds, speed_ratio=raw_seconds/compressed_seconds,
    pit_difference=abs(raw.pit-weighted.pit)))
