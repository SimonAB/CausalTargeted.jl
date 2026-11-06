using Test, DataFrames, StableRNGs, Statistics

@testset "G-computation uncertainty contracts" begin
    rng = StableRNG(6701)
    n = 160
    A = repeat([0.0, 1.0], n ÷ 2)
    W = randn(rng, n)
    data = DataFrame(A = A, W = W, Y = 1 .+ 2 .* A .+ W .+ randn(rng, n))

    plugin = run_gcomp(data, :A, :Y; covariates = [:W], n_boot = 0,
                       folds = 2, rng = StableRNG(6702))
    @test isnan(plugin.se)
    @test isnan(plugin.ci_lower) && isnan(plugin.ci_upper)
    @test plugin.uncertainty === :unavailable
    @test isfinite(plugin.prediction_se)

    one = run_gcomp(data, :A, :Y; covariates = [:W], estimator = :onestep,
                    propensity = 0.5, folds = 2, rng = StableRNG(6702))
    @test one.estimator === :onestep
    @test one.uncertainty === :influence_function
    @test isapprox(mean(one.ic), 0; atol = 1e-12)
    @test isapprox(one.se, std(one.ic) / sqrt(n); rtol = 1e-12)
    @test one.ci_lower < 2 < one.ci_upper
    @test one.se > plugin.prediction_se
    @test one.propensity_metalearner === :known

    fitted = run_gcomp(data, :A, :Y; covariates = [:W], estimator = :onestep,
                       folds = 2, rng = StableRNG(6702))
    @test isfinite(fitted.se) && fitted.se > 0
    @test fitted.propensity_source === :estimated
    @test fitted.propensity_metalearner === :nnloglik
    @test_throws ArgumentError run_gcomp(data, :A, :Y; estimator = :onestep, propensity = 1.0)
    @test_throws ArgumentError run_gcomp(data, :A, :Y; estimator = :onestep, propensity = [0.5])
    @test_throws ArgumentError run_gcomp(data, :A, :Y; estimator = :unknown)
    @test_throws ArgumentError run_gcomp(data, :A, :Y; estimator = :onestep, delta = 0.5)
    @test_throws ArgumentError run_gcomp(data, :A, :Y; n_boot = 1)
    continuous = copy(data)
    continuous.A = W
    @test_throws ArgumentError run_gcomp(continuous, :A, :Y; estimator = :onestep)
    incomplete = copy(data)
    allowmissing!(incomplete, :Y)
    incomplete.Y[1] = missing
    @test_throws ArgumentError run_gcomp(incomplete, :A, :Y; estimator = :onestep)
end

@testset "g-computation bootstrap refits missingness processing" begin
    rng = StableRNG(6730)
    n = 120
    W = randn(rng, n)
    A = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.4 .* W)))
    Y = 1 .+ 2 .* A .+ W .+ randn(rng, n)
    observed = rand(rng, n) .< 1 ./ (1 .+ exp.(0.4 .* W))
    Y_missing = Union{Missing, Float64}[observed[i] ? Y[i] : missing for i in 1:n]
    data = DataFrame(A = A, W = W, Y = Y_missing)
    observed_idx = findall(observed)
    boot_idx = repeat(observed_idx; outer = cld(n, length(observed_idx)))[1:n]

    original = CausalTargeted.handle_missing_data(
        data, :Y, [:W, :A], :ipcw;
        folds = 2, rng = StableRNG(6731),
    )
    original_weights = zeros(n)
    original_weights[observed_idx] .= original.weights
    boot = CausalTargeted._gcomp_refit_missingness_bootstrap(
        data, :Y, [:W, :A], [:W], :ipcw, boot_idx;
        folds = 2, rng = StableRNG(6731),
    )

    @test boot.n_input == n
    @test nrow(boot.data) == n
    @test all(boot.weights .== 1.0)
    @test !isapprox(boot.weights, original_weights[boot_idx]; atol = 1e-8)

    W_missing = Union{Missing, Float64}[
        rand(rng) < 0.15 ? missing : W[i] for i in 1:n
    ]
    incomplete = DataFrame(A = A, W = W_missing, Y = Y_missing)
    for strategy in (:ipcw, :impute, :ipcw_impute)
        fit = run_gcomp(
            incomplete, :A, :Y; covariates = [:W], folds = 2,
            learners = (:glm, :mean), rng = StableRNG(6732),
            n_boot = 2, handle_missing = strategy,
        )
        @test isfinite(fit.estimate) && isfinite(fit.se)
        @test isfinite(fit.ci_lower) && isfinite(fit.ci_upper)
        @test fit.bootstrap_refits_missingness
    end

    complete = DataFrame(A = A, W = W, Y = Y)
    complete_fit = run_gcomp(
        complete, :A, :Y; covariates = [:W], folds = 2,
        learners = (:glm, :mean), rng = StableRNG(6733),
        n_boot = 2, handle_missing = :drop,
    )
    @test isfinite(complete_fit.estimate) && isfinite(complete_fit.se)
    @test !complete_fit.bootstrap_refits_missingness
end
