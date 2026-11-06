using CausalTargeted
using DataFrames
using Distributions
using Random
using StableRNGs
using Statistics
using Test

@testset "sequential one-step inference" begin
    rng = StableRNG(901)
    n = 420
    W = randn(rng, n)
    A1 = ifelse.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.4 .* W)), "1", "0")
    L1 = 0.5 .* (A1 .== "1") .+ 0.4 .* W .+ 0.3 .* randn(rng, n)
    p2 = 1 ./ (1 .+ exp.(-(-0.2 .+ 0.8 .* (A1 .== "1") .+ 0.35 .* L1)))
    A2 = ifelse.(rand(rng, n) .< p2, "2", "1")
    ε = 0.35 .* randn(rng, n)
    Y = 0.4 .* (A1 .== "1") .+ 1.1 .* (A2 .== "2") .+ 0.2 .* L1 .+ 0.3 .* W .+ ε
    df = DataFrame(; W, A1, L1, A2, Y)

    identity_policy = discrete_recode_policy(Dict{String, String}())
    identity_fit = run_sequential_lmtp(
        df, [:A1, :A2], :Y;
        baseline = [:W], time_vary = [Symbol[], [:L1]],
        policies = [identity_policy, identity_policy], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(902),
    )
    @test identity_fit.estimate ≈ mean(Y) atol = 1e-8
    @test identity_fit.se ≈ std(Y) / sqrt(n) atol = 1e-8
    @test identity_fit.score ≈ Y atol = 1e-8
    @test identity_fit.influence ≈ Y .- mean(Y) atol = 1e-8
    @test identity_fit.inference_status == :ok
    @test identity_fit.ci_lower < identity_fit.estimate < identity_fit.ci_upper

    t1 = run_sequential_lmtp(
        DataFrame(; W, A1, Y), [:A1], :Y;
        baseline = [:W], time_vary = [Symbol[]],
        policies = [identity_policy], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(913),
    )
    @test t1.estimate ≈ mean(Y) atol = 1e-8
    @test t1.se ≈ std(Y) / sqrt(n) atol = 1e-8
    @test t1.inference_status == :ok

    policy_t2 = discrete_recode_policy(Dict("2" => "1"))
    recoded = ifelse.(A2 .== "2", "1", A2)
    oracle = mean(
        0.4 .* (A1 .== "1") .+ 1.1 .* (recoded .== "2") .+
        0.2 .* L1 .+ 0.3 .* W,
    )
    fit_t2_only = run_sequential_lmtp(
        df, [:A1, :A2], :Y;
        baseline = [:W], time_vary = [Symbol[], [:L1]],
        policies = [identity_policy, policy_t2], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(903),
    )
    @test fit_t2_only.inference_status == :ok
    @test abs(fit_t2_only.estimate - oracle) < 0.24
    @test fit_t2_only.ratio_diagnostics[1].mean ≈ 1.0 atol = 0.2
    @test fit_t2_only.ratio_diagnostics[2].mean ≈ 1.0 atol = 0.2
    @test fit_t2_only.ratio_diagnostics[2].history == [:W, :L1, :A1]
    @test :A2 ∉ fit_t2_only.ratio_diagnostics[2].history

    unsupported_a2 = fill("2", n)
    structural_zero_data = DataFrame(; W, A1, L1, A2 = unsupported_a2, Y)
    structural_zero_fit = run_sequential_lmtp(
        structural_zero_data, [:A1, :A2], :Y;
        baseline = [:W], time_vary = [Symbol[], [:L1]],
        policies = [identity_policy, policy_t2], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(921),
    )
    @test all(iszero, structural_zero_fit.ratios[2])
    @test structural_zero_fit.inference_status == :positivity_violation
    @test isfinite(structural_zero_fit.estimate)
    @test all(isnan, (structural_zero_fit.se, structural_zero_fit.ci_lower,
                      structural_zero_fit.ci_upper))

    baseline_context_policy = discrete_shift_policy(
        (a, w) -> begin
            length(w) == 1 || error("sequential policy must receive baseline W only")
            w[1] > 0 && string(a) == "2" ? "1" : string(a)
        end;
        levels = ["0", "1", "2"],
    )
    baseline_context_fit = run_sequential_lmtp(
        structural_zero_data, [:A1, :A2], :Y;
        baseline = [:W], time_vary = [Symbol[], [:L1]],
        policies = [identity_policy, baseline_context_policy], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(923),
    )
    @test all(iszero, baseline_context_fit.ratios[2][W .> 0])
    @test any(x -> x > 0, baseline_context_fit.ratios[2][W .<= 0])

    fixed_folds = CausalTargeted.crossfit_indices(n, 3, StableRNG(920))
    heldout = fixed_folds[1]
    perturbed = copy(df)
    perturbed.Y[heldout] .+= 25.0
    nuisance_original = run_sequential_lmtp(
        df, [:A1, :A2], :Y;
        baseline = [:W], time_vary = [Symbol[], [:L1]],
        policies = [identity_policy, policy_t2], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(920),
    )
    nuisance_perturbed = run_sequential_lmtp(
        perturbed, [:A1, :A2], :Y;
        baseline = [:W], time_vary = [Symbol[], [:L1]],
        policies = [identity_policy, policy_t2], folds = 3,
        learners = (:glm, :mean), rng = StableRNG(920),
    )
    @test nuisance_perturbed.q_policy[1][heldout] ≈
        nuisance_original.q_policy[1][heldout] atol = 1e-10
    @test nuisance_perturbed.q_observed[1][heldout] ≈
        nuisance_original.q_observed[1][heldout] atol = 1e-10

    numeric = run_sequential_lmtp(
        DataFrame(W = W, A1 = randn(StableRNG(904), n),
            A2 = randn(StableRNG(905), n), Y = Y),
        [:A1, :A2], :Y;
        baseline = [:W], delta = 0.25, folds = 2,
        learners = (:glm, :mean), rng = StableRNG(906),
    )
    @test isfinite(numeric.estimate)
    @test numeric.inference_status == :unsupported_continuous_policy
    @test isnan(numeric.se) && isnan(numeric.ci_lower) && isnan(numeric.ci_upper)
end

@testset "survival clamped-shift inference is explicitly unsupported" begin
    df, truth = CausalTargeted.simulate_discrete_survival_mtp(
        100; T = 2, rng = StableRNG(911),
    )
    result = run_survival_lmtp(
        df, truth.treatments, truth.surv;
        baseline = [:W], delta = 0.25, folds = 2,
        learners = (:glm, :mean), rng = StableRNG(912),
    )
    @test isfinite(result.estimate)
    @test result.inference_status == :unsupported_continuous_policy
    @test isnan(result.se) && isnan(result.ci_lower) && isnan(result.ci_upper)
end
