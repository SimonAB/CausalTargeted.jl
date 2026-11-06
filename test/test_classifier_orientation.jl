using Test, StableRNGs

@testset "continuous classifier estimates policy over observed density" begin
    rng = StableRNG(5601)
    natural = randn(rng, 2000)
    policy = natural .+ 1.0
    fit = CausalTargeted._fit_density_ratio_classifier(natural, policy, zeros(2000, 0); rng = rng)
    points = [-0.5, 0.5, 1.5]
    ratio = CausalTargeted._ratio_from_classifier(fit, points, zeros(3, 0); trunc = 20)
    # N(1,1) / N(0,1) increases with a and equals exp(a - 1/2).
    @test ratio[1] < ratio[2] < ratio[3]
    @test all(isapprox.(ratio, exp.(points .- 0.5); rtol = 0.35))
end

@testset "discrete policy support preserves structural zeros" begin
    levels = ["0", "1", "2"]
    pure_recode = CausalTargeted.discrete_recode_policy(
        Dict("2" => "1"); levels = levels,
    )
    observed = ["2", "1", "0"]
    support = CausalTargeted._discrete_policy_support(
        observed, zeros(3, 0), pure_recode,
    )
    @test support == [false, true, true]
    @test CausalTargeted._discrete_policy_ratio(
        [0.5, 0.5, 0.5]; trunc = 10, support = support,
    ) == [0.0, 1.0, 1.0]

    cycle = CausalTargeted.discrete_recode_policy(
        Dict("0" => "1", "1" => "2", "2" => "0"); levels = levels,
    )
    @test CausalTargeted._discrete_policy_support(
        ["0"], zeros(1, 0), cycle; natural_levels = levels,
    ) == [true]

    conditional = CausalTargeted.discrete_shift_policy(
        (a, w) -> (w[1] > 0 && string(a) == "2") ? "1" : a;
        levels = levels,
    )
    conditional_support = CausalTargeted._discrete_policy_support(
        ["2", "2", "0"], reshape([1.0, -1.0, 1.0], :, 1), conditional,
    )
    @test conditional_support == [false, true, true]
end
