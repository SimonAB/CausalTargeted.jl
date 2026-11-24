using Distributions: Normal, pdf

@testset "Feasible additive shifts" begin
    policy = feasible_additive_shift_policy(-2.0, 2.0; sd_scale = 0.5)
    a = [-3.0, -2.0, -1.5, 0.0, 1.5, 2.0, 3.0]
    @test apply_policy_values(a, 2, policy) == [-3.0, -1.0, -0.5, 1.0, 1.5, 2.0, 3.0]
    @test apply_policy_values(a, -2, policy) == [-3.0, -2.0, -1.5, -1.0, 0.5, 1.0, 3.0]
    @test apply_policy_values(a, 0, policy) == a
    @test apply_policy_values(a, 12, policy) == a
    @test apply_policy_values([1.0, 1.0], 2, policy; stratum_mask = [true, false]) == [2.0, 1.0]
    @test_throws ArgumentError feasible_additive_shift_policy(0, 1; sd_scale = 0)

    ratio(a, step, mask = nothing) =
        CausalTargeted._feasible_gaussian_ratio(a, zeros(length(a)), 1.0, policy, step, mask)
    law = Normal()
    @test ratio([-1.5, 0.0, 1.5, 3.0], 1.0) ≈ [0.0, pdf(law, -1) / pdf(law, 0), 1 + exp(1), 1.0]
    @test ratio([1.5, 0.0, -1.5, -3.0], -1.0) ≈ [0.0, pdf(law, 1) / pdf(law, 0), 1 + exp(1), 1.0]
    @test ratio(a, 0.0) == ones(length(a))
    @test ratio([1.5, 1.5], 1.0, [true, false]) ≈ [1 + exp(1), 1.0]
    grid = collect(-7.9995:0.001:7.9995)
    for step in [-1.0, 1.0]
        @test sum(pdf.(law, grid) .* ratio(grid, step)) * 0.001 ≈ 1 atol = 1e-7
    end

    # With supplied nuisance predictions, compare the point estimate and EIF
    # to the scalar weighted-intercept update and the observed-mean contrast.
    y = [2.0, 4.0, 7.0, 9.0]
    q = [2.0, 3.0, 5.0, 8.0]
    q_shifted = [3.0, 5.0, 5.0, 8.0]
    weights = ratio([-1.5, 0.0, 1.5, 3.0], 1.0)
    components = (; y, Q_obs = q, Q1 = q_shifted, Q0 = q,
        H1 = weights, H0 = ones(4), n = 4, trunc = Inf, family_outcome = :negbin,
        inference_status = :available)
    fit = lmtp_tmle_from_components(components; estimator = :tmle, epochs = 1)
    multiplier = sum(weights .* y) / sum(weights .* q)
    estimate = mean(multiplier .* q_shifted) - mean(y)
    contribution = weights .* (y .- multiplier .* q) .+ multiplier .* q_shifted .- y
    @test fit.estimate ≈ estimate
    @test fit.ic ≈ contribution .- estimate
    @test fit.se ≈ std(contribution) / sqrt(length(y))

    rng = StableRNG(47)
    exposure = randn(rng, 120)
    data = DataFrame(A = exposure, Y = exposure .+ randn(rng, length(exposure)))
    for cache in (true, false)
        grid = run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
            deltas = [-2.0, 0.0, 2.0], learners_outcome = (:glm,), learners_trt = (:glm,),
            stratify_by = nothing, cache_nuisances = cache, weight_truncation = false,
            simultaneous = false, parallel = false)
        @test grid.inference_status == [:available, :trivial, :available]
        @test grid.effective_shift ≈ [mean(apply_policy_values(exposure, d, policy) .- exposure)
            for d in grid.delta]
        @test all(isfinite, grid.se)
    end
end
