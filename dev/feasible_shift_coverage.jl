# Run from an environment that includes this checkout:
# julia --project dev/feasible_shift_coverage.jl 400
using CausalTargeted, DataFrames, Distributions, LinearAlgebra, StableRNGs, Statistics

const CT = CausalTargeted
const POLICY = feasible_additive_shift_policy(-1.85, 1.70; sd_scale = 1)
const DELTAS = (-2.0, -1.0, 1.0, 2.0)
const COVARIATES = [:age, :vaccination, :history]
const SCENARIOS = (("moderate", 0.35, 0.85), ("weak", 0.85, 0.48))
const BETA = 0.25
const GAMMA = 0.35

function simulate(rng, n, slope, sigma)
    age, vaccination = Float64.(rand(rng, Bool, n)), Float64.(rand(rng, Bool, n))
    history = randn(rng, n)
    conditional_mean = 0.35 .* age .- 0.30 .* vaccination .+ slope .* history
    a = conditional_mean .+ sigma .* randn(rng, n)
    mean_y = 800 .* exp.(BETA .* a .+ GAMMA .* history .+ 0.2 .* age .- 0.3 .* vaccination)
    y = [rand(rng, NegativeBinomial(2, 2 / (2 + m))) for m in mean_y]
    return DataFrame(A = a, Y = y, age = age, vaccination = vaccination, history = history),
        conditional_mean
end

function true_effect(delta, slope, sigma)
    lo = max(POLICY.lower, POLICY.lower - delta)
    hi = min(POLICY.upper, POLICY.upper - delta)
    # Integrate the exponential outcome mean on the interval eligible to move.
    variance = (BETA * sigma)^2 + (BETA * slope + GAMMA)^2
    integral = 0.0
    for age in (0.0, 1.0), vaccination in (0.0, 1.0)
        μ = 0.35 * age - 0.30 * vaccination
        tilted = Normal(μ + BETA * sigma^2 + slope * (BETA * slope + GAMMA), hypot(slope, sigma))
        probability = cdf(tilted, hi) - cdf(tilted, lo)
        integral += 800 * exp(0.2 * age - 0.3 * vaccination + BETA * μ + variance / 2) * probability
    end
    return expm1(BETA * delta) * integral / 4
end

function assess(n, slope, sigma, nrep, seed)
    rng = StableRNG(seed)
    covered = zeros(Int, length(DELTAS), 2)
    available = zero(covered)
    biases = zeros(size(covered))
    truth = [true_effect(d, slope, sigma) for d in DELTAS]
    for replicate in 1:nrep
        data, conditional_mean = simulate(rng, n, slope, sigma)
        cache = CT.build_lmtp_fold_cache(data, :A, :Y, COVARIATES, 3, StableRNG(20261124 + replicate);
            learners_outcome = (:glm, :mean), learners_trt = (:glm,),
            family_outcome = :negbin, weight_truncation = false)
        for (index, delta) in enumerate(DELTAS)
            shifted = apply_policy_values(data.A, delta, POLICY)
            components = CT.lmtp_components_from_cache(cache, shifted, data.A;
                feasible_policy = POLICY, feasible_delta = delta)
            # Independent oracle evaluates the Gaussian densities at each preimage.
            weights = map(data.A, conditional_mean) do a, μ
                retained = !(POLICY.lower <= a <= POLICY.upper &&
                    POLICY.lower <= a + delta <= POLICY.upper)
                shifted_here = POLICY.lower <= a <= POLICY.upper &&
                    POLICY.lower <= a - delta <= POLICY.upper
                law = Normal(μ, sigma)
                Float64(retained) + (shifted_here ? pdf(law, a - delta) / pdf(law, a) : 0.0)
            end
            for (method, fit_components) in enumerate((components, merge(components, (; H1 = weights))))
                fit = lmtp_tmle_from_components(fit_components; estimator = :tmle, epochs = 1)
                if fit.inference_status === :available
                    available[index, method] += 1
                    covered[index, method] += fit.lower <= truth[index] <= fit.upper
                    biases[index, method] += fit.estimate - truth[index]
                end
            end
        end
    end
    return truth, available, covered, biases
end

BLAS.set_num_threads(1)
replicates = isempty(ARGS) ? 400 : parse(Int, only(ARGS))
println("n,overlap,delta,density,truth,attempted,available,covered,coverage,mcse,bias")
for n in (30, 200), (index, (scenario, slope, sigma)) in enumerate(SCENARIOS)
    truth, available, covered, biases = assess(n, slope, sigma, replicates, 20261124 + 10n + index)
    for (i, delta) in enumerate(DELTAS), (j, density) in enumerate(("estimated", "oracle"))
        proportion = covered[i, j] / available[i, j]
        mcse = sqrt(proportion * (1 - proportion) / available[i, j])
        println(join((n, scenario, delta, density, truth[i], replicates, available[i, j],
            covered[i, j], proportion, mcse, biases[i, j] / available[i, j]), ','))
    end
    flush(stdout)
end
