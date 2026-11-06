using DataFrames
using Distributions
using StableRNGs
using Test

@testset "bootstrap discrete LMTP (#42)" begin
    resampled_rows, copy_ids = CausalTargeted._expand_cluster_sample(
        [:a, :a, :b, :b], [:a, :a],
    )
    @test resampled_rows == [1, 2, 1, 2]
    @test copy_ids == [1, 1, 2, 2]

    rng = StableRNG(1)
    n = 180
    df = DataFrame(arm = rand(rng, [0, 1], n), w = randn(rng, n))
    logit_p = @. 0.2 + 0.6 * df.arm + 0.1 * df.w
    df.y = rand.(rng, Bernoulli.(1 ./ (1 .+ exp.(-logit_p))))

    point = run_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :binomial,
        learners_outcome = (:glm, :mean),
        folds = 2,
        rng = StableRNG(2),
    )

    boot = bootstrap_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :binomial,
        learners_outcome = (:glm, :mean),
        B = 40,
        folds = 2,
        rng = StableRNG(3),
    )
    @test boot.estimate == boot.point.estimate
    @test boot.n_attempted == 40
    @test boot.n_failed == 40 - boot.n_success
    @test boot.status == :ok
    @test isfinite(point.estimate)
    @test boot.n_success >= 10
    @test isfinite(boot.se)
    @test boot.lower <= boot.estimate <= boot.upper || boot.n_success < 40

    insufficient = bootstrap_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1, arm_ref = 0, levels = [0, 1], baseline = [:w],
        family_outcome = :binomial, B = 1, folds = 2,
        rng = StableRNG(31),
    )
    @test insufficient.status == :insufficient_success
    @test insufficient.n_attempted == 1
    @test insufficient.n_success <= 1
    @test isnan(insufficient.se) && isnan(insufficient.lower) && isnan(insufficient.upper)

    rng_c = StableRNG(4)
    df.cluster = repeat(1:30, inner = 6)
    boot_c = bootstrap_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1,
        arm_ref = 0,
        levels = [0, 1],
        baseline = [:w],
        family_outcome = :binomial,
        B = 20,
        folds = 2,
        cluster = :cluster,
        rng = rng_c,
    )
    @test boot_c.n_success >= 5
    @test boot_c.n_attempted == 20
    @test boot_c.status == :ok
    boot_vector = bootstrap_discrete_lmtp_contrast(
        df, :arm, :y;
        arm_hi = 1, arm_ref = 0, levels = [0, 1], baseline = [:w],
        family_outcome = :binomial, B = 5, folds = 2,
        cluster = df.cluster, rng = StableRNG(5), min_success_fraction = 0.4,
    )
    @test boot_vector.n_attempted == 5
    @test boot_vector.status == :ok

    unsupported = copy(df)
    unsupported.arm .= 0
    unsupported_boot = bootstrap_discrete_lmtp_contrast(
        unsupported, :arm, :y;
        arm_hi = 1, arm_ref = 0, levels = [0, 1], baseline = [:w],
        family_outcome = :binomial, B = 2, folds = 2,
        min_success_fraction = 0.5, rng = StableRNG(51),
    )
    @test unsupported_boot.n_success == 0
    @test unsupported_boot.n_positivity_failures == 2
    @test unsupported_boot.status == :insufficient_success
    @test isempty(unsupported_boot.boot_samples)
end

@testset "bootstrap two-part LMTP (#42)" begin
    rng = StableRNG(5)
    n = 200
    df = DataFrame(
        arm = rand(rng, ["R", "SS"], n),
        w = randn(rng, n),
    )
    logit_p = @. -0.5 + 0.7 * (df.arm == "SS") + 0.1 * df.w
    df.pres = Float64.(rand.(rng, Bernoulli.(1 ./ (1 .+ exp.(-logit_p)))))
    df.int = [p > 0 ? 1.0 + 0.2 * w + randn(rng) * 0.3 : missing for (p, w) in zip(df.pres, df.w)]

    boot = bootstrap_two_part_discrete_lmtp_contrast(
        df, :arm;
        presence = :pres,
        intensity = :int,
        arm_hi = "SS",
        arm_ref = "R",
        levels = ["R", "SS"],
        baseline = [:w],
        B = 25,
        folds = 2,
        rng = StableRNG(6),
    )
    @test boot.presence.n_success >= 5
    @test boot.intensity.n_success >= 3
    @test boot.presence.n_success == boot.intensity.n_success
    @test boot.presence.n_attempted == boot.intensity.n_attempted == 25
    @test boot.presence.status == boot.intensity.status == :ok

    insufficient = bootstrap_two_part_discrete_lmtp_contrast(
        df, :arm; presence = :pres, intensity = :int,
        arm_hi = "SS", arm_ref = "R", levels = ["R", "SS"],
        baseline = [:w], B = 1, folds = 2, rng = StableRNG(61),
    )
    @test insufficient.presence.status == :insufficient_success
    @test insufficient.presence.n_success == insufficient.intensity.n_success
    @test isnan(insufficient.presence.lower) && isnan(insufficient.intensity.upper)
    @test all(isfinite, (boot.presence.estimate, boot.intensity.estimate))

    unsupported = copy(df)
    unsupported.arm .= "R"
    unsupported_boot = bootstrap_two_part_discrete_lmtp_contrast(
        unsupported, :arm;
        presence = :pres, intensity = :int,
        arm_hi = "SS", arm_ref = "R", levels = ["R", "SS"],
        baseline = [:w], B = 2, folds = 2,
        min_success_fraction = 0.5, rng = StableRNG(62),
    )
    @test unsupported_boot.presence.n_success == 0
    @test unsupported_boot.presence.n_positivity_failures == 2
    @test unsupported_boot.intensity.n_success == 0
    @test unsupported_boot.presence.status == :insufficient_success
end
