using Test, CausalTargeted, StableRNGs, Statistics
import CausalMediation, Distributions

@testset "Supported uncapped inference coverage stress" begin
    repetitions = 200
    # Exact-binomial lower-tail alarm: under nominal 95% coverage, its false
    # alarm probability is at most 0.1% per scenario. Held-out calibration is
    # reported separately; this gate catches substantial regressions.
    cutoff = Int(Distributions.quantile(Distributions.Binomial(repetitions, 0.95), 0.001))
    covered = Dict(label => Bool[] for label in (:continuous, :discrete, :mediation))
    for replication in 1:repetitions
        seed = 900_000 + replication
        data, truth = simulate_linear_mtp(500; rng = StableRNG(seed))
        continuous = lmtp_tmle_contrast(data, :A, :Y, [:W], data.A .+ 0.5, data.A,
            3, StableRNG(seed + 1_000_000); weight_truncation = false)
        @test continuous.inference_status === :available
        @test isfinite(continuous.se) && continuous.se > 0
        push!(covered[:continuous], continuous.lower <= truth.shift_effect(0.5) <= continuous.upper)

        categorical, _ = simulate_categorical_treatment_mtp(500; rng = StableRNG(seed))
        discrete = run_discrete_lmtp_contrast(categorical, :A, :Y;
            arm_hi = "1", arm_ref = "0", levels = ["0", "1", "2"], baseline = [:W],
            weight_truncation = false, rng = StableRNG(seed + 1_000_000))
        @test discrete.inference_status === :available
        @test isfinite(discrete.se) && discrete.se > 0
        push!(covered[:discrete], discrete.lower <= 1.0 <= discrete.upper)

        mediation_data, mediation_truth = CausalMediation.simulate_mediation(500; rng = StableRNG(seed))
        mediation = CausalMediation.run_mediation_scalar(mediation_data, :A, :Y;
            covar = [:W], mediators = [:M], estimator = :onestep,
            weight_truncation = false, rng = StableRNG(seed + 1_000_000))
        te = only(eachrow(mediation[mediation.effect .== "TE", :]))
        @test te.inference_status == "available"
        @test isfinite(te.se) && te.se > 0
        push!(covered[:mediation], te.lower <= mediation_truth.te <= te.upper)
    end
    for (label, coverage) in covered
        @info "Supported coverage stress" label repetitions coverage = mean(coverage) cutoff
        @test count(coverage) >= cutoff
    end
end
