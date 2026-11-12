# Run independent prior-predictive and fixed-parameter Gaussian calibration.
using CausalTargeted, Distributions, DataFrames, StableRNGs
using LinearAlgebra, Statistics, Random, Printf

function calibrate_bayesian_gaussian(; replications = 500, n = 40, seed = 431)
    rng = StableRNG(seed)
    prior = ConjugateGaussianPrior(mean = zeros(3), covariance = 4.0 * Matrix{Float64}(I, 3, 3), shape = 3.0, scale = 2.0)
    rows = NamedTuple[]
    for experiment in (:prior_predictive, :fixed_parameter)
        covered = 0
        widths = Float64[]
        squared_errors = Float64[]
        ranks = Float64[]
        elapsed = @elapsed for _ in 1:replications
            a, w = randn(rng, n), randn(rng, n)
            X = hcat(ones(n), a, w)
            sigma2 = experiment === :prior_predictive ? rand(rng, InverseGamma(3.0, 2.0)) : 1.0
            beta = experiment === :prior_predictive ? sqrt(4sigma2) .* randn(rng, 3) : [0.0, 0.8, -0.4]
            data = DataFrame(A = a, W = w, Y = X * beta + sqrt(sigma2) .* randn(rng, n))
            fit = fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
            # Default basis order is asserted by the unit tests: intercept, A, W.
            location = fit.posterior_mean[2]
            scale = sqrt(fit.posterior_scale[2, 2])
            reference = TDist(fit.posterior_df)
            q = quantile(reference, 0.975)
            covered += abs(beta[2] - location) <= q * scale
            push!(widths, 2q * scale)
            push!(squared_errors, (beta[2] - location)^2)
            push!(ranks, cdf(reference, (beta[2] - location) / scale))
        end
        rate = covered / replications
        push!(rows, (; experiment, replications, n, coverage = rate,
            mcse = sqrt(rate * (1-rate) / replications),
            mean_width = mean(widths), rmse = sqrt(mean(squared_errors)),
            posterior_cdf_mean = mean(ranks),
            posterior_cdf_deciles = [count(r -> (j-1)/10 <= r < j/10, ranks) for j in 1:10],
            seconds = elapsed))
    end
    return DataFrame(rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Compile before timing the two calibration experiments.
    calibrate_bayesian_gaussian(replications = 2)
    result = calibrate_bayesian_gaussian()
    show(stdout, MIME("text/plain"), result)
    println()
    rng = StableRNG(15)
    data = DataFrame(A = randn(rng, 40), W = randn(rng, 40), Y = randn(rng, 40))
    prior = ConjugateGaussianPrior(mean = zeros(3), covariance = 4.0 * Matrix{Float64}(I, 3, 3), shape = 3.0, scale = 2.0)
    fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
    seconds = @elapsed for _ in 1:1000
        fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
    end
    bytes = @allocated fit_bayesian_gcomp(data, :Y, :A, [:W]; prior)
    @printf("Warmed Gaussian fit, n=40 p=3: %.3f microseconds/fit, %d bytes/fit\n", 1e6 * seconds / 1000, bytes)
end
