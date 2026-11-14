using CSV, TOML, Statistics
include(joinpath(pwd(), "test/stress/multivariate_policy_calibration.jl"))
const D = MultivariatePolicyCalibration
length(ARGS) == 1 || error("provide a study directory containing decompressed raw.tsv")
directory = abspath(only(ARGS))
metadata = TOML.parsefile(joinpath(directory, "metadata.toml"))
config = (; (Symbol(k) => v for (k,v) in metadata["configuration"])...)
checkpoint = D.open_checkpoint(directory, config; resume = true)
@assert D.source_hashes() == config.source_hashes
rows = collect(CSV.File(joinpath(directory, "raw.tsv"); delim='\t'))
@assert length(rows) == 21000
@assert length(checkpoint.completed) == 7000
@assert all(r -> r.available && r.status == "ok", rows)
@assert all(r -> r.covered == (0.025 <= r.pit <= 0.975), rows)
@assert all(r -> r.control_n + r.treated_n == r.n && 0 < r.control_n < r.n, rows)
@assert all(r -> isapprox(r.policy_truth, r.policy_scale*r.arm_truth; atol=1e-14), rows)
for cell in config.cells, method in ("adaptive_reference", "variational", "linear_response")
    selected = filter(r -> r.cell_id == cell && r.method == method, rows)
    @assert sort(getproperty.(selected, :replication)) == collect(1:500)
end
println("AUDIT PASSED: digest, current source hashes, 21000 records, 7000 complete triples, cell counts, coverage decisions, target/count invariants; all available.")
for method in ("adaptive_reference", "variational", "linear_response")
    selected = filter(r -> r.method == method, rows)
    println((;method, median_seconds=median(getproperty.(selected,:seconds)), max_cdf_change=maximum(getproperty.(selected,:cdf_change)), max_tail_error=maximum(getproperty.(selected,:tail_error)), max_mass_change=maximum(getproperty.(selected,:mass_change))))
end
