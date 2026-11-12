using Test
using CausalTargeted
using LinearAlgebra

function logistic_jj_oracle(X, y, μ₀, V₀; tilt = zeros(length(μ₀)), tol = 1e-12)
    p = length(μ₀)
    Q₀ = cholesky(Symmetric(V₀)) \ Matrix{Float64}(I, p, p)
    m, S = copy(μ₀), copy(V₀)
    r = vec(sum((X * S) .* X; dims = 2)) + (X * m).^2
    for _ in 1:10_000
        w = [v == 0 ? 0.25 : tanh(sqrt(v) / 2) / (2sqrt(v)) for v in r]
        Q = cholesky(Symmetric(Q₀ + X' * Diagonal(w) * X))
        Snext = Q \ Matrix{Float64}(I, p, p)
        mnext = Q \ (Q₀ * μ₀ + X' * (y .- 0.5) + tilt)
        rnext = vec(sum((X * Snext) .* X; dims = 2)) + (X * mnext).^2
        change = maximum((maximum(abs.(mnext - m)), maximum(abs.(Snext - S)), maximum(abs.(rnext - r))))
        m, S, r = mnext, Snext, rnext
        change < tol && return (mean = m, covariance = S, second_moment = r)
    end
    error("Independent JJ oracle did not converge")
end

function run_logistic_engine_tests(engine)
    @testset "Logistic engine" begin
        @testset "Gaussian pseudo-observation update" begin
            X = [1.0 -1.0; 1.0 0.5; 1.0 2.0]
            y = [0.0, 1.0, 1.0]
            μ₀ = [0.3, -0.2]
            V₀ = [1.5 0.4; 0.4 0.8]
            w = [0.2, 0.25, 0.1]
            tilt = [0.02, -0.03]
            Q₀ = cholesky(Symmetric(V₀)) \ Matrix{Float64}(I, 2, 2)
            Q = Q₀ + X' * Diagonal(w) * X
            expected_covariance = cholesky(Symmetric(Q)) \ Matrix{Float64}(I, 2, 2)
            expected_mean = Q \ (Q₀ * μ₀ + X' * (y .- 0.5) + tilt)
            posterior = engine._logistic_gaussian_update(X, y, μ₀, V₀, w; tilt)
            @test posterior.mean ≈ expected_mean atol = 1e-10 rtol = 1e-10
            @test posterior.covariance ≈ expected_covariance atol = 1e-10 rtol = 1e-10
            @test posterior.raw !== nothing
            algebra = engine._logistic_gaussian_algebra(X, y, μ₀, V₀, w; tilt)
            @test algebra.mean ≈ expected_mean
            @test algebra.covariance ≈ expected_covariance
        end

        @testset "Logistic weights and derivative" begin
            @test engine._logistic_weights([0.0])[1] == 0.25
            @test engine._logistic_weight_derivative(0.0) == -1 / 48
            for r in (1e-7, 0.2, 5.0, 100.0)
                h = max(1e-9, r * 1e-5)
                derivative = (engine._logistic_weights([r + h])[1] - engine._logistic_weights([r - h])[1]) / (2h)
                @test engine._logistic_weight_derivative(r) ≈ derivative atol = 1e-8 rtol = 1e-5
            end
        end

        @testset "Full Gaussian logistic fixed point" begin
            μ₀ = [0.2, -0.1]
            V₀ = [1.5 0.3; 0.3 0.9]
            designs = (
                ([1.0 -2; 1 -1; 1 0; 1 0.5; 1 1; 1 2], [0., 1, 0, 1, 0, 1]),
                ([1.0 -2; 1 -1; 1 -0.5; 1 0.5; 1 1; 1 2], [0., 0, 0, 1, 1, 1]),
                ([1.0 -2; 1 -1; 1 0; 1 0.5; 1 1; 1 2], ones(6)),
                ([1.0 1.001; 2 1.999; -1 -1.001; -2 -1.999; 0.5 0.501; -0.5 -0.499], [0., 1, 0, 1, 0, 1]),
            )
            for (X, y) in designs
                fit = engine._fit_logistic_engine(X, y, μ₀, V₀; covariance = :variational, tol = 1e-11)
                oracle = logistic_jj_oracle(X, y, μ₀, V₀)
                @test fit.converged
                @test fit.status == :converged
                @test fit.mean ≈ oracle.mean atol = 1e-9
                @test fit.variational_covariance ≈ oracle.covariance atol = 1e-9
                @test fit.corrected_covariance === nothing
                @test isposdef(Symmetric(fit.variational_covariance))
            end
            X, y = first(designs)
            fit = engine._fit_logistic_engine(X, y, μ₀, V₀; maxiter = 1)
            @test !fit.converged
            @test fit.status == :maxiter
            @test fit.n_iterations == 1
            @test fit.corrected_covariance === nothing
        end

        @testset "Prior and zero-design limits" begin
            μ₀ = [0.2, -0.1]
            V₀ = [1.5 0.3; 0.3 0.9]
            emptyfit = engine._fit_logistic_engine(zeros(0, 2), Float64[], μ₀, V₀)
            @test emptyfit.mean == μ₀
            @test emptyfit.variational_covariance == V₀
            @test emptyfit.corrected_covariance == V₀
            @test emptyfit.converged
            @test emptyfit.n_iterations == 0
            @test emptyfit.raw === nothing
            zerofit = engine._fit_logistic_engine(zeros(4, 2), [0., 0, 1, 1], μ₀, V₀)
            @test zerofit.mean ≈ μ₀
            @test zerofit.variational_covariance ≈ V₀
            @test zerofit.corrected_covariance ≈ V₀
            @test zerofit.converged
        end

        @testset "Linear response matches re-fitted tilt derivatives" begin
            X = [1.0 -2; 1 -1; 1 0; 1 0.5; 1 1; 1 2]
            y = [0., 0, 1, 0, 1, 1]
            μ₀ = [0.1, 0.2]
            V₀ = [1.7 0.4; 0.4 1.2]
            fit = engine._fit_logistic_engine(X, y, μ₀, V₀; tol = 1e-12)
            @test fit.converged
            @test fit.corrected_covariance !== nothing
            @test fit.diagnostics.linear_response.valid
            @test isposdef(Symmetric(fit.corrected_covariance))
            for ε in (1e-4, 1e-5)
                derivatives = zeros(2, 2)
                for j in 1:2
                    tilt = zeros(2)
                    tilt[j] = ε
                    plus = engine._fit_logistic_engine(X, y, μ₀, V₀; covariance = :variational, tilt, tol = 1e-12)
                    minus = engine._fit_logistic_engine(X, y, μ₀, V₀; covariance = :variational, tilt = -tilt, tol = 1e-12)
                    @test plus.converged && minus.converged
                    derivatives[:, j] = (plus.mean - minus.mean) / (2ε)
                end
                @test fit.corrected_covariance ≈ derivatives atol = 1e-7 rtol = 1e-6
            end
        end

        @testset "Invalid numerical inputs fail explicitly" begin
            X, y, μ₀, V₀ = ones(3, 2), [0., 1, 0], zeros(2), Matrix{Float64}(I, 2, 2)
            @test_throws DimensionMismatch engine._fit_logistic_engine(X, y[1:2], μ₀, V₀)
            @test_throws DimensionMismatch engine._fit_logistic_engine(X, y, zeros(3), V₀)
            @test_throws DimensionMismatch engine._fit_logistic_engine(X, y, μ₀, ones(3, 3))
            @test_throws DimensionMismatch engine._fit_logistic_engine(X, y, μ₀, V₀; tilt = zeros(3))
            @test_throws ArgumentError engine._fit_logistic_engine(X, [0., 2, 0], μ₀, V₀)
            @test_throws ArgumentError engine._fit_logistic_engine(fill(NaN, 3, 2), y, μ₀, V₀)
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, [Inf, 0], V₀)
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, μ₀, [1. 2; 0 1])
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, μ₀, [1. 2; 2 1])
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, μ₀, V₀; maxiter = 0)
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, μ₀, V₀; tol = NaN)
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, μ₀, V₀; covariance = :sandwich)
            @test_throws ArgumentError engine._fit_logistic_engine(X, y, μ₀, V₀; tilt = [NaN, 0])
        end

        @testset "Unstable response is withheld without repair" begin
            response = engine._logistic_linear_response(ones(1, 1), zeros(1), fill(100.0, 1, 1), zeros(1))
            @test response.covariance === nothing
            @test !response.diagnostics.valid
            @test response.diagnostics.reason == :unstable_fixed_point
            @test response.diagnostics.spectral_radius >= 1
        end
    end
end

if !isdefined(@__MODULE__, :LOGISTIC_ENGINE_DEFER_TESTS)
    run_logistic_engine_tests(Base.get_extension(CausalTargeted, :CausalTargetedRxInferExt))
end
