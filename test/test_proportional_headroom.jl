using Distributions: Normal, pdf

@testset "Proportional headroom shifts" begin
    policy = proportional_headroom_policy(-2.0, 2.0)
    @test policy isa ProportionalHeadroomPolicy
    @test policy.lower == -2.0 && policy.upper == 2.0

    # Bounds validation
    @test_throws ArgumentError proportional_headroom_policy(1.0, 1.0)
    @test_throws ArgumentError proportional_headroom_policy(2.0, 1.0)
    @test_throws ArgumentError proportional_headroom_policy(-Inf, 1.0)
    @test_throws ArgumentError proportional_headroom_policy(0.0, Inf)
    @test_throws ArgumentError proportional_headroom_policy(NaN, 1.0)
    @test_throws ArgumentError proportional_headroom_policy(-1e308, 1e308)

    # Positive, negative, and zero shifts; outside-bounds retention; exact endpoints
    a = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    @test apply_policy_values(a, 0.5, policy) ≈ [-3.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    @test apply_policy_values(a, -0.5, policy) ≈ [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 3.0]
    @test apply_policy_values(a, 0.0, policy) == a

    # Monotonicity and no boundary clamping on [L, U]
    interior = collect(-2.0:0.25:2.0)
    up = apply_policy_values(interior, 0.4, policy)
    down = apply_policy_values(interior, -0.4, policy)
    @test all(diff(up) .> 0) && all(-2.0 .<= up .<= 2.0)
    @test all(diff(down) .> 0) && all(-2.0 .<= down .<= 2.0)
    @test first(up) ≈ -2.0 + 0.4 * 4.0 && last(up) ≈ 2.0
    @test first(down) ≈ -2.0 && last(down) ≈ 2.0 - 0.4 * 4.0

    # Stratum mask
    @test apply_policy_values([0.0, 0.0, 1.0], 0.5, policy; stratum_mask = [true, false, true]) ≈
        [1.0, 0.0, 1.5]
    @test apply_policy_values([0.0, 0.0], -0.5, policy; stratum_mask = [false, true]) ≈
        [0.0, -1.0]

    # Delta, exposure, and stratum_mask validation
    for bad_delta in (-1.5, -1.0, 1.0, 1.5, -Inf, Inf, NaN)
        @test_throws ArgumentError apply_policy_values(a, bad_delta, policy)
    end
    @test_throws ArgumentError apply_policy_values([0.0, NaN], 0.2, policy)
    @test_throws ArgumentError apply_policy_values([0.0, Inf], 0.2, policy)
    @test_throws DimensionMismatch apply_policy_values([0.0, 1.0], 0.2, policy; stratum_mask = [true])

    # Exact single-branch Gaussian density ratios
    ratio(x, d, mask = nothing; μ = zeros(length(x)), σ = 1.0, weight_truncation = false) =
        CausalTargeted._headroom_gaussian_ratio(
            x, μ, σ, policy, d, mask;
            weight_truncation = weight_truncation,
        )

    law = Normal(0.0, 1.0)
    # Positive delta = 0.5 on [-2, 2]: threshold = 0.0, Jacobian = 2.0
    # a = -3.0 (outside) -> 1.0
    # a = -1.0 (in [-2, 0)) -> 0.0
    # a =  0.0 (threshold, u = -2.0) -> 2 * pdf(-2) / pdf(0)
    # a =  1.0 (u = 0.0) -> 2 * pdf(0) / pdf(1)
    # a =  2.0 (endpoint, u = 2.0) -> 2.0
    # a =  3.0 (outside) -> 1.0
    pts_pos = [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    expected_pos = [
        1.0,
        0.0,
        2.0 * pdf(law, -2.0) / pdf(law, 0.0),
        2.0 * pdf(law, 0.0) / pdf(law, 1.0),
        2.0,
        1.0,
    ]
    @test ratio(pts_pos, 0.5) ≈ expected_pos

    # Negative delta = -0.5 on [-2, 2]: threshold = 0.0, Jacobian = 2.0
    # a = -3.0 (outside) -> 1.0
    # a = -2.0 (endpoint, u = -2.0) -> 2.0
    # a = -1.0 (u = 0.0) -> 2 * pdf(0) / pdf(-1)
    # a =  0.0 (threshold, u = 2.0) -> 2 * pdf(2) / pdf(0)
    # a =  1.0 (in (0, 2]) -> 0.0
    # a =  3.0 (outside) -> 1.0
    pts_neg = [-3.0, -2.0, -1.0, 0.0, 1.0, 3.0]
    expected_neg = [
        1.0,
        2.0,
        2.0 * pdf(law, 0.0) / pdf(law, -1.0),
        2.0 * pdf(law, 2.0) / pdf(law, 0.0),
        0.0,
        1.0,
    ]
    @test ratio(pts_neg, -0.5) ≈ expected_neg

    # Non-standard mean and SD
    law_shift = Normal(0.6, 1.4)
    u_test = (1.2 - 0.3 * 2.0) / 0.7
    @test ratio([1.2], 0.3; μ = [0.6], σ = 1.4) ≈ [(1.0 / 0.7) * pdf(law_shift, u_test) / pdf(law_shift, 1.2)]

    # Zero delta and stratum mask
    @test ratio(a, 0.0) == ones(length(a))
    @test ratio([1.0, 1.0], 0.5, [true, false]) ≈ [2.0 * pdf(law, 0.0) / pdf(law, 1.0), 1.0]
    @test ratio([-1.0, -1.0], 0.5, [true, false]) == [0.0, 1.0]

    # Ratio validation guards
    @test_throws ArgumentError ratio([0.0], 1.0)
    @test_throws ArgumentError ratio([0.0], -1.0)
    @test_throws ArgumentError ratio([0.0, 1.0], 0.5; μ = [0.0])
    @test_throws DimensionMismatch ratio([0.0, 1.0], 0.5, [true])
    @test_throws DomainError ratio([0.0], 0.5; σ = NaN)

    # Extreme-tail behaviour: uncapped throws DomainError on underflow/overflow,
    # while truncated path stays finite via 1e-12 floor.
    @test_throws DomainError ratio([0.5], 0.5; μ = [100.0], σ = 0.1, weight_truncation = false)
    @test_throws DomainError ratio([0.5], 0.5; μ = [-100.0], σ = 0.1, weight_truncation = false)
    @test all(isfinite, ratio([0.5], 0.5; μ = [100.0], σ = 0.1, weight_truncation = true))
    @test all(isfinite, ratio([0.5], 0.5; μ = [-100.0], σ = 0.1, weight_truncation = true))

    # Numerical density normalisation: ∫ φ(a) r(a) da ≈ 1 to 1e-7
    h = 0.0005
    grid = collect((-8.0 + h / 2):h:(8.0 - h / 2))
    for d in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)
        @test sum(pdf.(law, grid) .* ratio(grid, d)) * h ≈ 1.0 atol = 1e-7
    end
    asym_policy = proportional_headroom_policy(-1.5, 2.5)
    asym_law = Normal(0.4, 1.2)
    asym_grid = collect((-10.0 + h / 2):h:(10.0 - h / 2))
    for d in (-0.5, 0.25, 0.5)
        r_asym = CausalTargeted._headroom_gaussian_ratio(
            asym_grid, fill(0.4, length(asym_grid)), 1.2, asym_policy, d, nothing;
            weight_truncation = false,
        )
        @test sum(pdf.(asym_law, asym_grid) .* r_asym) * h ≈ 1.0 atol = 1e-7
    end

    # Exact algebraic TMLE / EIF check (count family and Gaussian family)
    y = [2.0, 4.0, 7.0, 9.0]
    q = [2.0, 3.0, 5.0, 8.0]
    q_shifted = [3.0, 5.0, 6.0, 8.0]
    weights = ratio([-1.0, 0.0, 1.0, 3.0], 0.5)
    @test weights[1] == 0.0  # structural zero on [-2, 0)
    components_nb = (; y, Q_obs = q, Q1 = q_shifted, Q0 = q,
        H1 = weights, H0 = ones(4), n = 4, trunc = Inf, family_outcome = :negbin,
        inference_status = :available)
    fit_nb = lmtp_tmle_from_components(components_nb; estimator = :tmle, epochs = 1)
    multiplier = sum(weights .* y) / sum(weights .* q)
    estimate_nb = mean(multiplier .* q_shifted) - mean(y)
    contribution_nb = weights .* (y .- multiplier .* q) .+ multiplier .* q_shifted .- y
    @test fit_nb.estimate ≈ estimate_nb
    @test fit_nb.ic ≈ contribution_nb .- estimate_nb
    @test fit_nb.se ≈ std(contribution_nb) / sqrt(length(y))

    components_g = (; y, Q_obs = q, Q1 = q_shifted, Q0 = q,
        H1 = weights, H0 = ones(4), n = 4, trunc = Inf, family_outcome = :gaussian,
        inference_status = :available)
    fit_g = lmtp_tmle_from_components(components_g; estimator = :tmle, epochs = 1)
    eps_g = sum(weights .* (y .- q)) / sum(weights)
    estimate_g = mean(q_shifted .+ eps_g) - mean(y)
    contribution_g = weights .* (y .- (q .+ eps_g)) .+ (q_shifted .+ eps_g) .- y
    @test fit_g.estimate ≈ estimate_g
    @test fit_g.ic ≈ contribution_g .- estimate_g
    @test fit_g.se ≈ std(contribution_g) / sqrt(length(y))

    # End-to-end run_lmtp_grid: cached vs uncached, weight_truncation=false
    rng = StableRNG(73)
    exposure = randn(rng, 120)
    data = DataFrame(A = exposure, Y = 1.2 .* exposure .+ randn(rng, length(exposure)))
    for cache in (true, false)
        grid_res = run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
            deltas = [-0.4, 0.0, 0.4], learners_outcome = (:glm,), learners_trt = (:glm,),
            stratify_by = nothing, cache_nuisances = cache, weight_truncation = false,
            simultaneous = false, parallel = false, rng = StableRNG(74))
        @test grid_res.inference_status == [:available, :trivial, :available]
        @test all(grid_res.support_status .== "proportional_headroom_policy")
        @test grid_res.effective_shift ≈ [
            mean(apply_policy_values(exposure, d, policy) .- exposure)
            for d in grid_res.delta
        ]
        @test all(isfinite, grid_res.se)
        @test grid_res.se[2] == 0.0
        @test grid_res.se[1] > 0.0 && grid_res.se[3] > 0.0
    end

    # Cached component builder and lmtp_tmle_contrast match identically with shared seed
    a_shifted = apply_policy_values(exposure, 0.4, policy)
    direct = lmtp_tmle_contrast(
        data, :A, :Y, Symbol[], a_shifted, exposure, 4, StableRNG(75);
        learners_outcome = (:glm,), learners_trt = (:glm,),
        weight_truncation = false, headroom_policy = policy, headroom_delta = 0.4,
    )
    fold_cache = build_lmtp_fold_cache(
        data, :A, :Y, Symbol[], 4, StableRNG(75);
        learners_outcome = (:glm,), learners_trt = (:glm,),
        weight_truncation = false,
    )
    cached_comp = lmtp_components_from_cache(
        fold_cache, a_shifted, exposure;
        headroom_policy = policy, headroom_delta = 0.4, weight_truncation = false,
    )
    cached_fit = lmtp_tmle_from_components(cached_comp; estimator = :tmle, epochs = 3)
    @test direct.estimate ≈ cached_fit.estimate
    @test direct.se ≈ cached_fit.se
    @test direct.ic ≈ cached_fit.ic

    # weight_truncation = true withholds CIs (:truncated_weights) and skips stabilisation
    trunc_grid = run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [-0.4, 0.0, 0.4], learners_outcome = (:glm,), learners_trt = (:glm,),
        stratify_by = nothing, weight_truncation = true,
        simultaneous = false, parallel = false, rng = StableRNG(74))
    @test trunc_grid.inference_status == [:truncated_weights, :trivial, :truncated_weights]
    @test all(isnan, trunc_grid.se[[1, 3]])
    @test trunc_grid.se[2] == 0.0

    # Structural zeros must not be inflated by stabilize=true when weight_truncation=true
    a_zero_heavy = apply_policy_values(exposure, 0.85, policy)
    uncapped_zero_heavy = lmtp_components_from_cache(
        fold_cache, a_zero_heavy, exposure;
        headroom_policy = policy, headroom_delta = 0.85,
        weight_truncation = false,
    )
    trunc_comp = lmtp_components_from_cache(
        fold_cache, a_zero_heavy, exposure;
        headroom_policy = policy, headroom_delta = 0.85,
        weight_truncation = true, trunc = 5.0,
    )
    @test any(iszero, uncapped_zero_heavy.H1)
    @test trunc_comp.H1 ≈ prepare_clever_covariate(uncapped_zero_heavy.H1; trunc = 5.0, stabilize = false)
    @test !(trunc_comp.H1 ≈ prepare_clever_covariate(uncapped_zero_heavy.H1; trunc = 5.0, stabilize = true))
    @test maximum(trunc_comp.H1) <= 5.0

    # Stratified grid
    data_strat = copy(data)
    data_strat.group = repeat(["g1", "g2"], inner = length(exposure) ÷ 2)
    strat_grid = run_lmtp_grid(data_strat, :A, :Y; baseline = Symbol[], policy,
        deltas = [0.0, 0.4], learners_outcome = (:glm,), learners_trt = (:glm,),
        stratify_by = :group, weight_truncation = false,
        simultaneous = false, parallel = false, rng = StableRNG(76))
    @test nrow(strat_grid) == 4
    nonzero_strat = strat_grid[strat_grid.delta .== 0.4, :]
    @test all(nonzero_strat.inference_status .== :available)
    @test all(isfinite, nonzero_strat.se)

    # Configuration and component error guards
    @test_throws ArgumentError run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [0.4], density_ratio = :classification,
        stratify_by = nothing, simultaneous = false, parallel = false)
    @test_throws ArgumentError run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [0.4], density_ratio = :hybrid,
        stratify_by = nothing, simultaneous = false, parallel = false)
    @test_throws ArgumentError run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [0.4], positivity = true,
        stratify_by = nothing, simultaneous = false, parallel = false)
    @test_throws ArgumentError run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [0.4], weight_truncation = false, cv_trunc = true,
        stratify_by = nothing, simultaneous = false, parallel = false)
    @test_throws ArgumentError run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [1.0], weight_truncation = false,
        stratify_by = nothing, simultaneous = false, parallel = false)
    @test_throws ArgumentError run_lmtp_grid(data, :A, :Y; baseline = Symbol[], policy,
        deltas = [-1.0], weight_truncation = false,
        stratify_by = nothing, simultaneous = false, parallel = false)

    # Component-level guards
    @test_throws ArgumentError lmtp_components_from_cache(
        fold_cache, a_shifted, exposure;
        headroom_policy = policy, headroom_delta = nothing, weight_truncation = false,
    )
    @test_throws ArgumentError lmtp_components_from_cache(
        fold_cache, exposure, exposure;
        headroom_policy = policy, headroom_delta = 0.4, weight_truncation = false,
    )
    @test_throws ArgumentError lmtp_components_from_cache(
        fold_cache, a_shifted, a_shifted;
        headroom_policy = policy, headroom_delta = 0.4, weight_truncation = false,
    )
    @test_throws ArgumentError lmtp_components_from_cache(
        fold_cache, a_shifted, exposure;
        headroom_policy = policy, headroom_delta = 0.4,
        feasible_policy = feasible_additive_shift_policy(-2.0, 2.0; sd_scale = 1.0),
        feasible_delta = 0.4, weight_truncation = false,
    )
    @test_throws ArgumentError lmtp_components_from_cache(
        fold_cache, a_shifted, exposure;
        headroom_policy = policy, headroom_delta = 0.4,
        L = -2.0, U = 2.0, shift_amount = 0.4, weight_truncation = true,
    )
end
