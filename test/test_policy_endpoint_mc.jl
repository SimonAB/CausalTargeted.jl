using Test, DataFrames, LinearAlgebra, Random, Statistics, Distributions
using CausalTargeted

@testset "Policy endpoint Monte Carlo bounds" begin
    # Exact rank coverage, independent of posterior approximation or data.
    for n in (2, 20, 2000), p in (0.025, 0.5, 0.975), level in (0.8, 0.95)
        ranks = CausalTargeted._policy_quantile_mc_ranks(n, p, level)
        binomial = Binomial(n, p)
        probability = cdf(binomial, ranks.upper - 1) - cdf(binomial, ranks.lower - 1)
        @test probability >= level - 1e-12
        @test 0 <= ranks.lower < ranks.upper <= n + 1
    end
    @test_throws ArgumentError CausalTargeted._policy_quantile_mc_ranks(0, 0.5, 0.95)
    @test_throws ArgumentError CausalTargeted._policy_quantile_mc_ranks(20, 0.0, 0.95)
    for level in (0.0, 1.0, NaN, Inf)
        @test_throws ArgumentError CausalTargeted._policy_quantile_mc_ranks(20, 0.5, level)
    end

    # A bounded uniform law has analytic quantiles. Replications test the
    # implementation, not calibration of a causal posterior.
    rng = Xoshiro(20261115)
    contained = count(1:2000) do _
        draws = sort!(2rand(rng, 200) .- 1)
        bounds = CausalTargeted._policy_quantile_mc_bounds(draws, 0.025, 0.95)
        bounds.lower <= -0.95 <= bounds.upper
    end
    @test 0.925 <= contained / 2000 <= 0.985

    data = DataFrame(A=repeat([0,1], 10), Y=repeat([0,1,1,0], 5))
    prior = GaussianCoefficientPrior(mean=zeros(2), covariance=Matrix{Float64}(I,2,2))
    fit = fit_bayesian_logistic(data, :Y, :A, Symbol[]; prior)
    policy = smooth_bounded_logit_shift_policy(-1, 2)
    result = bayesian_policy_contrast(fit, data, policy; deltas=[0,0.5,0.5],
        n_draws=2000, rng=Xoshiro(12))
    repeated = bayesian_policy_contrast(fit, data, policy; deltas=[0,0.5,0.5],
        n_draws=2000, mc_level=0.8, rng=Xoshiro(12))
    @test result.mean == repeated.mean
    @test result.covariance == repeated.covariance
    @test result.lower == repeated.lower && result.upper == repeated.upper
    @test result.mcse == repeated.mcse
    @test result.endpoint_mc_bounds.lower_endpoint.lower[1] == 0
    @test result.endpoint_mc_bounds.lower_endpoint.upper[1] == 0
    @test result.endpoint_mc_bounds.upper_endpoint.lower[1] == 0
    @test result.endpoint_mc_bounds.upper_endpoint.upper[1] == 0
    for endpoint in (:lower_endpoint, :upper_endpoint)
        bounds = getproperty(result.endpoint_mc_bounds, endpoint)
        narrower = getproperty(repeated.endpoint_mc_bounds, endpoint)
        @test bounds.lower[2] == bounds.lower[3]
        @test bounds.upper[2] == bounds.upper[3]
        @test bounds.lower[2] <= narrower.lower[2] <= narrower.upper[2] <= bounds.upper[2]
    end
    @test result.metadata.endpoint_mc_method == :binomial_order_statistics
    @test result.metadata.endpoint_mc_level == 0.95
    @test result.metadata.endpoint_mc_scope == :conditional_on_gaussian_moments
    @test result.metadata.endpoint_mc_simultaneous == false
    @test_throws ArgumentError bayesian_policy_contrast(fit,data,policy; deltas=[0],mc_level=1)
    tiny = bayesian_policy_contrast(fit,data,policy; deltas=[0.5],n_draws=2)
    @test tiny.endpoint_mc_bounds.lower_endpoint.lower == [-1.0]
    @test tiny.endpoint_mc_bounds.upper_endpoint.upper == [1.0]
    unavailable = fit_bayesian_logistic(data,:Y,:A,Symbol[];prior,maxiter=1)
    @test !unavailable.metadata.uncertainty_available
    failed = bayesian_policy_contrast(unavailable,data,policy;deltas=[0.5])
    @test all(isnan, failed.endpoint_mc_bounds.lower_endpoint.lower)
    @test all(isnan, failed.endpoint_mc_bounds.upper_endpoint.upper)
end
