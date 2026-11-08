using DataFrames
using Distributions
using StableRNGs
using Statistics
using Test

@testset "count outcome Super Learner + LMTP (#36)" begin
    @testset "Poisson / NB learners" begin
        rng = StableRNG(1)
        n = 120
        X = hcat(ones(n), randn(rng, n))
        μ = exp.(0.2 .+ 0.3 .* X[:, 2])
        y = [rand(rng, NegativeBinomial(2.0, 2 / (2 + μᵢ))) for μᵢ in μ]

        sl_nb = fit_super_learner(
            X, Float64.(y);
            learners = (:glm_nb, :mean),
            family = :negbin,
            rng = StableRNG(2),
        )
        pred = predict_super_learner(sl_nb, X)
        @test all(>=(0), pred)
        @test mean(pred) > 0

        sl_pois = fit_super_learner(
            X, Float64.(y);
            learners = (:glm_poisson, :mean),
            family = :poisson,
            rng = StableRNG(3),
        )
        @test all(>=(0), predict_super_learner(sl_pois, X))
    end

    @testset "discrete LMTP NB recovery (issue #36 example)" begin
        rng = StableRNG(3)
        df = DataFrame(arm = rand(rng, [0, 1], 400), w = randn(rng, 400))
        μ = @. exp(0.5 + 0.4 * df.arm + 0.1 * df.w)
        df.count = [rand(rng, NegativeBinomial(2.0, 2 / (2 + μᵢ))) for μᵢ in μ]

        res = run_discrete_lmtp_contrast(
            df, :arm, :count;
            arm_hi = 1,
            arm_ref = 0,
            levels = [0, 1],
            baseline = [:w],
            family_outcome = :negbin,
            learners_outcome = (:glm_nb, :mean),
            weight_truncation = false,
            rng = StableRNG(4),
        )
        @test res.estimate > 0
        @test isfinite(res.se)
        @test res.lower < res.upper
    end

    @testset "Poisson family_outcome on discrete LMTP" begin
        rng = StableRNG(11)
        df = DataFrame(
            arm = rand(rng, [0, 1], 300),
            w = randn(rng, 300),
        )
        μ = @. exp(0.3 + 0.35 * df.arm + 0.05 * df.w)
        df.count = rand.(rng, Poisson.(μ))

        res = run_discrete_lmtp_contrast(
            df, :arm, :count;
            arm_hi = 1,
            arm_ref = 0,
            levels = [0, 1],
            baseline = [:w],
            family_outcome = :poisson,
            learners_outcome = (:glm_poisson, :mean),
            rng = StableRNG(12),
        )
        @test res.estimate > 0
    end
end

@testset "continuous shift count LMTP (#41)" begin
    rng = StableRNG(5)
    n = 200
    df = DataFrame(
        a = rand(rng, n) .+ 0.5,
        w = randn(rng, n),
    )
    μ = @. exp(0.2 + 0.15 * df.a + 0.05 * df.w)
    df.count = [rand(rng, NegativeBinomial(2.0, 2 / (2 + μᵢ))) for μᵢ in μ]

    grid = run_lmtp_grid(
        df, :a, :count;
        baseline = [:w],
        deltas = [0.1, 0.2],
        family_outcome = :negbin,
        learners_outcome = (:glm_nb, :mean),
        folds = 2,
        simultaneous = false,
        parallel = false,
        rng = StableRNG(6),
    )
    @test nrow(grid) >= 1
    pos = filter(r -> r.delta > 0, grid)
    @test !isempty(pos)
    @test all(isfinite, pos.est)

    grid_pois = run_lmtp_grid(
        df, :a, :count;
        baseline = [:w],
        deltas = [0.1],
        family_outcome = :poisson,
        learners_outcome = (:glm_poisson, :mean),
        folds = 2,
        simultaneous = false,
        parallel = false,
        rng = StableRNG(7),
    )
    @test nrow(grid_pois) >= 1
end

@testset "ZINB learner (#40)" begin
    rng = StableRNG(21)
    n = 350
    df = DataFrame(arm = rand(rng, [0, 1], n), w = randn(rng, n))
    π₀ = @. 0.35 - 0.1 * df.arm
    μ = @. exp(0.4 + 0.5 * df.arm + 0.1 * df.w)
    df.count = zeros(Int, n)
    for i in 1:n
        if rand(rng) < π₀[i]
            df.count[i] = 0
        else
            df.count[i] = rand(rng, NegativeBinomial(2.0, 2 / (2 + μ[i])))
        end
    end

    sl = fit_super_learner(
        hcat(ones(n), df.arm, df.w), Float64.(df.count);
        learners = (:zeroinflated_nb, :glm_nb, :mean),
        family = :zeroinflated_nb,
        rng = StableRNG(22),
    )
    pred = predict_super_learner(sl, hcat(ones(n), df.arm, df.w))
    @test all(>=(0), pred)

    res = run_discrete_lmtp_contrast(
        df, :arm, :count;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :zeroinflated_nb,
        learners_outcome = (:zeroinflated_nb, :glm_nb, :mean),
        folds = 2,
        rng = StableRNG(23),
    )
    @test res.estimate > 0
end
