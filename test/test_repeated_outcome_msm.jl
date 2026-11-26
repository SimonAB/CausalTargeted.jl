using LinearAlgebra: I, Symmetric, diag, eigvals, issymmetric

@testset "repeated-outcome MSM (point treatment)" begin
    df, truth = simulate_repeated_outcome_ate(800; T = 4, rng = StableRNG(101))
    @test length(truth.tau) == 4
    @test Symbol.(names(df)) == [:W, :A, :Y1, :Y2, :Y3, :Y4]

    res = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W],
        folds = 3,
        learners = (:glm, :mean),
        rng = StableRNG(102),
    )
    @test length(res.estimates) == 4
    @test size(res.covariance) == (4, 4)
    @test all(isfinite, res.estimates)
    @test all(>(0), res.se)
    @test issymmetric(res.covariance) ||
        maximum(abs.(res.covariance .- res.covariance')) < 1e-10
    @test maximum(abs.(res.estimates .- truth.tau)) < 0.15
    # Marginal SE should match sqrt(diag(Σ))
    @test res.se ≈ sqrt.(diag(res.covariance)) atol = 1e-10
    explicit_default = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W],
        folds = 3, learners = (:glm, :mean), rng = StableRNG(102),
        strata = nothing, propensity = nothing,
    )
    for field in (:estimates, :se, :covariance, :ic, :mu1, :mu0,
                  :covariance_mu1, :covariance_mu0, :ic_mu1, :ic_mu0)
        @test getproperty(explicit_default, field) == getproperty(res, field)
    end
    @test explicit_default.outcomes == res.outcomes
    @test explicit_default.n == res.n
    @test explicit_default.positivity == res.positivity

    # Joint contrast τ₃ − τ₂
    c = msm_contrast(res, 3, 2)
    @test isfinite(c.estimate)
    @test c.se > 0
    @test abs(c.estimate - (truth.tau[3] - truth.tau[2])) < 0.20
    # Linear form with explicit weights
    c2 = msm_contrast(res, [0.0, -1.0, 1.0, 0.0])
    @test c2.estimate ≈ c.estimate atol = 1e-12
    @test c2.se ≈ c.se atol = 1e-12

    estimand = RepeatedOutcomeMSM(:A, [:Y1, :Y2, :Y3, :Y4], [:W])
    @test estimand_engine(estimand) == :repeated_msm
    grid = execute_estimand(estimand, df; folds = 2, rng = StableRNG(103), metadata = false)
    @test nrow(grid) == 4
    @test all(isfinite, grid.est)

    plan = plan_mtp(estimand, df; folds = 2)
    @test plan.estimated_fits >= 2 * (1 + 4)  # shared g + per-outcome Q

    # Binary treatment required for MVP
    df_bad = copy(df)
    df_bad.A = df_bad.A .+ 0.1
    @test_throws ArgumentError run_repeated_outcome_msm(
        df_bad, :A, [:Y1, :Y2];
        baseline = [:W], folds = 2, rng = StableRNG(104),
    )

    # ID handoff: same backdoor set for each outcome
    g = DiGraph(6)
    add_edge!(g, 1, 2)  # W → A
    for y in 3:6
        add_edge!(g, 1, y)  # W → Y_t
        add_edge!(g, 2, y)  # A → Y_t
    end
    node_names = [:W, :A, :Y1, :Y2, :Y3, :Y4]
    ids = identify_repeated_outcomes(
        g, :A, [:Y1, :Y2, :Y3, :Y4]; node_names = node_names,
    )
    @test length(ids) == 4
    @test all(r -> r.identifiable, ids)
    @test all(r -> Set(r.adjustment) == Set([:W]), ids)

    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, Symbol[]; baseline = [:W], folds = 2, rng = StableRNG(105),
    )
    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, [:Y1, :Y1]; baseline = [:W], folds = 2, rng = StableRNG(105),
    )
    @test_throws BoundsError msm_contrast(res, 9, 1)
    @test_throws ArgumentError msm_contrast(res, [1.0, 0.0])
    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2];
        baseline = [:W], folds = 2, rng = StableRNG(105), estimator = :bogus,
    )

    eif = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean),
        rng = StableRNG(102), estimator = :eif,
    )
    @test maximum(abs.(eif.estimates .- truth.tau)) < 0.18

    grid_meta = execute_estimand(estimand, df; folds = 2, rng = StableRNG(106))
    @test only(unique(grid_meta.meta_engine)) == "repeated_msm"
    @test only(unique(grid_meta.meta_density_ratio)) == "propensity"
    Σ_grid = metadata(grid_meta, "causal_targeted_msm_covariance")
    @test size(Σ_grid) == (4, 4)

    n = nrow(df)
    long = DataFrame(id = Int[], time = Int[], A = Float64[], W = Float64[], Y = Float64[])
    for i in 1:n
        for t in 1:4
            push!(long, (id = i, time = t, A = df.A[i], W = df.W[i], Y = df[i, Symbol("Y", t)]))
        end
    end
    wide = unstack_repeated_outcomes(
        long; id = :id, time = :time, outcome = :Y, treatment = :A, covariates = [:W],
    )
    @test Symbol.(names(wide)) == [:A, :W, :Y1, :Y2, :Y3, :Y4]
    @test wide.Y1 ≈ df.Y1
    res_long = run_repeated_outcome_msm(
        wide, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 2, learners = (:glm, :mean), rng = StableRNG(107),
    )
    @test all(isfinite, res_long.estimates)

    # Missingness: shared complete profile; do not impute Y
    df_miss = copy(df)
    df_miss.Y2 = Vector{Union{Float64, Missing}}(df.Y2)
    df_miss.Y2[1:40] .= missing
    df_miss.W = Vector{Union{Float64, Missing}}(df.W)
    df_miss.W[41:50] .= missing
    for strat in (:drop, :ipcw, :impute, :ipcw_impute)
        rmiss = run_repeated_outcome_msm(
            df_miss, :A, [:Y1, :Y2, :Y3, :Y4];
            baseline = [:W], folds = 2, learners = (:glm, :mean),
            rng = StableRNG(108), handle_missing = strat,
        )
        @test all(isfinite, rmiss.estimates)
        @test rmiss.missingness.strategy === strat
        @test rmiss.missingness.time_indexed === true
        @test rmiss.n < n
        @test !any(ismissing, rmiss.ic)
    end

    # Cluster-robust Σ: same point estimates; sampling hierarchy only
    df_c = copy(df)
    df_c.cluster = Float64.(mod1.(1:n, 20))
    rng_u = StableRNG(210)
    u_by = Dict{Float64, Float64}(c => randn(rng_u) for c in unique(df_c.cluster))
    uvec = [u_by[c] for c in df_c.cluster]
    for y in (:Y1, :Y2, :Y3, :Y4)
        df_c[!, y] = df_c[!, y] .+ uvec
    end
    res_unit = run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(211),
    )
    res_cl = run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(211),
        cluster = :cluster,
    )
    @test res_cl.covariance_kind === :cluster
    @test res_unit.covariance_kind === :unit
    @test res_cl.estimates ≈ res_unit.estimates atol = 1e-12
    @test res_cl.covariance ≉ res_unit.covariance
    # Cluster sandwich re-aggregates ICs; finite-sample SE_cluster ≱ SE_unit always
    @test all(isfinite, res_cl.se)
    @test all(≥(0), eigvals(Symmetric(res_cl.covariance)))
    @test length(unique(res_cl.cluster)) == 20
    # Vector form of cluster= matches Symbol form
    res_vec = run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 3, learners = (:glm, :mean), rng = StableRNG(211),
        cluster = df_c.cluster,
    )
    @test res_vec.covariance ≈ res_cl.covariance atol = 1e-12
    @test_throws ArgumentError run_repeated_outcome_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], folds = 2, learners = (:glm, :mean), rng = StableRNG(212),
        cluster = :missing_col,
    )
    pres = run_parametric_repeated_msm(
        df_c, :A, [:Y1, :Y2, :Y3, :Y4];
        baseline = [:W], design = :constant, folds = 3,
        learners = (:glm, :mean), rng = StableRNG(211), cluster = :cluster,
    )
    @test pres.covariance_kind === :cluster

    rec = CausalTargeted.run_julia_synthetic_once(
        :repeated_outcome_ate; n = 600, folds = 3,
        rng = StableRNG(109), learners = (:glm, :mean),
    )
    @test maximum(rec.abs_error) < 0.18
end

@testset "joint effect-modifier stratification" begin
    rng = StableRNG(310)
    cells = [(age, sex) for age in ("younger", "older") for sex in ("F", "M")]
    age_raw = String[]
    sex = String[]
    A = Float64[]
    for (age, sx) in cells
        append!(age_raw, fill(age, 24))
        append!(sex, fill(sx, 24))
        append!(A, repeat([0.0, 1.0], 12))
    end
    age = categorical(age_raw; levels = ["older", "younger"], ordered = true)
    n = length(A)
    W = randn(rng, n)
    g = ifelse.(age_raw .== "older", 0.65, 0.35)
    shared = 0.35 .* randn(rng, n)
    df = DataFrame(W = W, Age = age, Sex = sex, A = A, g_known = g)
    truth_tau = zeros(4, 4)
    for t in 1:4
        tau = @. 0.15 * t + 0.25 * t * (age_raw == "older") + 0.08 * (sex == "M")
        noise = 0.25 .* randn(rng, n)
        df[!, Symbol("Y", t)] = @. 1.0 + 0.5 * W + 0.2 * (age_raw == "older") +
            tau * A + shared + noise
    end

    # Ordered categorical Age first, then stable first occurrence for Sex.
    res = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W],
        strata = [:Age, :Sex], propensity = :g_known,
        folds = 3, learners = (:glm, :mean), rng = StableRNG(311),
    )
    @test res.strata.columns == [:Age, :Sex]
    @test res.strata.levels == [
        (Age = "older", Sex = "F"), (Age = "older", Sex = "M"),
        (Age = "younger", Sex = "F"), (Age = "younger", Sex = "M"),
    ]
    @test res.strata.nuisance_adjustment == [:W, :Age, :Sex]
    @test res.outcomes == [:Y1, :Y2, :Y3, :Y4]
    @test length(res.estimates) == 16
    @test size(res.ic) == (n, 16)
    @test size(res.covariance) == (16, 16)
    @test maximum(abs.(res.covariance .- res.covariance')) < 1e-12
    @test minimum(eigvals(Symmetric(res.covariance))) > -1e-10
    @test [x.position for x in res.parameter_index] == collect(1:16)
    @test res.parameter_index[5].stratum == (Age = "older", Sex = "M")
    @test res.parameter_index[5].outcome === :Y1
    @test res.propensity_metadata == (estimated = false, source = :column, column = :g_known)
    @test maximum(abs.(res.targeting_scores)) < 1e-10
    @test maximum(abs.(res.targeting_scores_by_fold)) < 1e-10

    # Manual conditional Hajek IC denominator for the first stratum/time cell.
    mask = (age_raw .== "older") .& (sex .== "F")
    manual_ic = Float64.(mask) ./ mean(mask) .* (res.uncentered_tau[:, 1] .- res.estimates[1])
    @test res.ic[:, 1] ≈ manual_ic atol = 1e-12
    @test abs(mean(res.ic[:, 1])) < 1e-12
    @test res.se ≈ sqrt.(diag(res.covariance)) atol = 1e-12

    # Single-column profile and age-modification contrast Gamma(t).
    age_res = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W],
        strata = :Age, propensity = g,
        folds = 3, learners = (:glm, :mean), rng = StableRNG(312),
    )
    @test age_res.strata.levels == [(Age = "older",), (Age = "younger",)]
    gamma = msm_stratum_contrast(age_res, "older", "younger")
    @test gamma.estimates ≈ gamma.contrast_matrix * age_res.estimates atol = 1e-12
    @test gamma.covariance ≈ gamma.contrast_matrix * age_res.covariance *
        gamma.contrast_matrix' atol = 1e-12
    @test gamma.se ≈ sqrt.(diag(gamma.covariance)) atol = 1e-12
    @test gamma.outcomes == age_res.outcomes
    @test maximum(abs.(gamma.estimates .- 0.25 .* (1:4))) < 0.22
    arbitrary = msm_contrast(age_res, vcat([1.0, -1.0], zeros(6)))
    @test arbitrary.estimate ≈ age_res.estimates[1] - age_res.estimates[2]

    # Scalar, column, vector, and estimated propensity metadata.
    scalar_res = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
        propensity = 0.5, folds = 2, learners = (:glm, :mean), rng = StableRNG(313),
    )
    @test scalar_res.propensity == fill(0.5, n)
    @test scalar_res.propensity_metadata.source === :scalar
    estimated_res = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
        folds = 2, learners = (:glm, :mean), rng = StableRNG(314),
    )
    @test estimated_res.propensity_metadata.estimated
    @test estimated_res.propensity_metadata.source === :estimated

    # Input-row-aligned vector is subset identically after complete-profile drop.
    df_missing = copy(df)
    df_missing.Y2 = Vector{Union{Missing, Float64}}(df_missing.Y2)
    df_missing.Y2[7] = missing
    vec_res = run_repeated_outcome_msm(
        df_missing, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
        propensity = g, folds = 2, learners = (:glm, :mean), rng = StableRNG(315),
    )
    @test vec_res.n == n - 1
    @test vec_res.propensity == g[setdiff(1:n, 7)]
    @test_throws ArgumentError run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
        propensity = g[1:end-1], folds = 2, learners = (:glm, :mean),
    )
    for bad in (0.0, 1.0, NaN, Inf)
        @test_throws ArgumentError run_repeated_outcome_msm(
            df, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
            propensity = bad, folds = 2, learners = (:glm, :mean),
        )
    end

    # Cluster re-aggregation changes covariance, never the targeted profile.
    cluster = repeat(1:48, inner = 2)
    clustered = run_repeated_outcome_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W], strata = :Age,
        propensity = g, cluster = cluster, folds = 3,
        learners = (:glm, :mean), rng = StableRNG(312),
    )
    @test clustered.estimates ≈ age_res.estimates atol = 1e-12
    @test clustered.covariance_kind === :cluster
    @test clustered.covariance != age_res.covariance

    # Unsupported cells and explicit empty categorical levels fail early.
    one_arm = copy(df)
    one_arm.A[age_raw .== "older"] .= 1.0
    @test_throws ArgumentError run_repeated_outcome_msm(
        one_arm, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
        propensity = g, folds = 2, learners = (:glm, :mean),
    )
    empty_level = copy(df)
    empty_level.Age = categorical(age_raw; levels = ["older", "younger", "oldest"])
    @test_throws ArgumentError run_repeated_outcome_msm(
        empty_level, :A, [:Y1, :Y2]; baseline = [:W], strata = :Age,
        propensity = g, folds = 2, learners = (:glm, :mean),
    )

    # Stratified parametric projection requires an explicit K*T-row design.
    @test_throws ArgumentError run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W], strata = :Age,
        propensity = g, design = :linear_time, folds = 2,
    )
    D = hcat(ones(8), repeat(0.0:3.0, 2), repeat([1.0, 0.0], inner = 4))
    projected = run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W], strata = :Age,
        propensity = g, design = D, folds = 2,
        learners = (:glm, :mean), rng = StableRNG(316),
    )
    @test length(projected.fitted_tau) == 8
    @test projected.strata.columns == [:Age]
    mean_design = Matrix{Float64}(I, 16, 16)
    mean_projected = run_parametric_repeated_msm(
        df, :A, [:Y1, :Y2, :Y3, :Y4]; baseline = [:W], strata = :Age,
        propensity = g, target = :mean, design = mean_design, folds = 2,
        learners = (:glm, :mean), rng = StableRNG(317),
    )
    @test length(mean_projected.coefficients) == 16
    @test length(mean_projected.fitted_tau) == 8
end

@testset "IPCW-weighted TMLE fluctuation consistency (unstratified vs single-stratum)" begin
    rng = StableRNG(410)
    n = 300
    W = randn(rng, n)
    g_true = 1.0 ./ (1.0 .+ exp.(-0.5 .* W))
    A = Float64.(rand(rng, n) .< g_true)
    Y1 = Vector{Union{Missing, Float64}}(1.0 .+ 0.7 .* A .+ 0.8 .* W .+ 0.4 .* randn(rng, n))
    Y2 = Vector{Union{Missing, Float64}}(0.5 .+ 1.1 .* A .+ 0.6 .* W .+ 0.4 .* randn(rng, n))
    # MAR outcome missingness depending strongly on W so IPCW weights vary materially.
    p_miss = 1.0 ./ (1.0 .+ exp.(-(1.2 .* W .- 1.0)))
    miss_mask = rand(rng, n) .< p_miss
    Y2[miss_mask] .= missing

    df = DataFrame(W = W, A = A, Y1 = Y1, Y2 = Y2, all_one = fill("all", n))
    df_cc = dropmissing(df, [:W, :A])
    R = CausalTargeted._complete_profile_R(df_cc, [:Y1, :Y2])
    ipcw_w = CausalTargeted._ipcw_weights_from_R(df_cc, R, [:W]; rng = StableRNG(411))
    keep = R .== 1.0
    df_clean = df_cc[keep, :]
    w_keep = ipcw_w[keep]
    @test std(w_keep) > 0.1

    folds = CausalTargeted.crossfit_indices(nrow(df_clean), 3, StableRNG(412))
    g_keep = g_true[keep]

    psi_u, est_u, _, mu1_u, _, mu0_u = CausalTargeted._aipw_influence_matrix(
        df_clean, :A, [:Y1, :Y2], [:W], folds, g_keep, w_keep;
        learners = (:glm, :mean), rng = StableRNG(413), estimator = :tmle,
    )
    strat = CausalTargeted._stratified_aipw_influence_matrix(
        df_clean, :A, [:Y1, :Y2], [:W], folds, g_keep, w_keep,
        [:all_one], [(all_one = "all",)];
        learners = (:glm, :mean), rng = StableRNG(413), estimator = :tmle,
    )
    Σ_u, se_u, ic_u = CausalTargeted._msm_covariance(psi_u, est_u, w_keep)
    Σ_s, se_s = CausalTargeted._msm_covariance_from_ic(strat.ic)

    @test est_u ≈ strat.estimates atol = 1e-12
    @test mu1_u ≈ strat.mu1 atol = 1e-12
    @test mu0_u ≈ strat.mu0 atol = 1e-12
    @test ic_u ≈ strat.ic atol = 1e-12
    @test se_u ≈ se_s atol = 1e-12
    @test Σ_u ≈ Σ_s atol = 1e-12
    @test maximum(abs.(strat.targeting_scores)) < 1e-10
end
