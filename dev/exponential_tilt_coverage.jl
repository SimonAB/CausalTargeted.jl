# Benchmark bounded and linear ExponentialTiltPolicy (calibration-balanced TMLE,
# Wald vs studentised Rademacher bootstrap) against FeasibleAdditiveShiftPolicy on the
# exact NegBin DGP from dev/feasible_shift_coverage.jl.
#
# Run from the workspace root:
#   julia --project=packages/CausalTargeted.jl --threads=auto \
#       packages/CausalTargeted.jl/dev/exponential_tilt_coverage.jl 300

using CausalTargeted
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using StableRNGs
using Statistics

const CT = CausalTargeted
const FEASIBLE_POLICY = feasible_additive_shift_policy(-1.85, 1.70; sd_scale = 1)
const LINEAR_TILT = exponential_tilt_policy(bound = nothing, center = 0.0)
const BOUNDED_TILT = exponential_tilt_policy(bound = 2.0, center = 0.0)
const DELTAS = (-2.0, -1.0, 1.0, 2.0)
const COVARIATES = [:age, :vaccination, :history]
const SCENARIOS = (("moderate", 0.35, 0.85), ("weak", 0.85, 0.48))
const BETA = 0.25
const GAMMA = 0.35

# Exact NegBin DGP from dev/feasible_shift_coverage.jl
function simulate_negbin_dgp(rng, n, slope, sigma)
    age = Float64.(rand(rng, Bool, n))
    vaccination = Float64.(rand(rng, Bool, n))
    history = randn(rng, n)
    conditional_mean = 0.35 .* age .- 0.30 .* vaccination .+ slope .* history
    a = conditional_mean .+ sigma .* randn(rng, n)
    mean_y = 800.0 .* exp.(BETA .* a .+ GAMMA .* history .+ 0.2 .* age .- 0.3 .* vaccination)
    y = [rand(rng, NegativeBinomial(2, 2.0 / (2.0 + m))) for m in mean_y]
    return DataFrame(A = a, Y = y, age = age, vaccination = vaccination, history = history)
end

# Golub-Welsch Gauss-Hermite nodes & probabilities for Z ~ Normal(0, 1)
function standard_normal_quadrature(m::Int = 60)
    offdiag = [sqrt(Float64(k)) for k in 1:(m - 1)]
    J = SymTridiagonal(zeros(Float64, m), offdiag)
    eig = eigen(J)
    nodes = eig.values
    weights = abs2.(eig.vectors[1, :])
    return nodes, weights
end

const GH_NODES, GH_WEIGHTS = standard_normal_quadrature(60)

# Exact E[Y] under the NegBin DGP
function true_baseline_mean_y(slope, sigma)
    v = (BETA * sigma)^2 + (BETA * slope + GAMMA)^2
    s = 0.0
    for age in (0.0, 1.0), vaccination in (0.0, 1.0)
        μ = 0.35 * age - 0.30 * vaccination
        s += 800.0 * exp(0.2 * age - 0.3 * vaccination + BETA * μ + 0.5 * v)
    end
    return s / 4.0
end

# Exact ground truth for FeasibleAdditiveShiftPolicy
function true_feasible_truth(delta, slope, sigma)
    lo = max(FEASIBLE_POLICY.lower, FEASIBLE_POLICY.lower - delta)
    hi = min(FEASIBLE_POLICY.upper, FEASIBLE_POLICY.upper - delta)
    v = (BETA * sigma)^2 + (BETA * slope + GAMMA)^2
    sd_a = hypot(slope, sigma)
    integral_y = 0.0
    prob_shift = 0.0
    for age in (0.0, 1.0), vaccination in (0.0, 1.0)
        μ = 0.35 * age - 0.30 * vaccination
        tilted = Normal(μ + BETA * sigma^2 + slope * (BETA * slope + GAMMA), sd_a)
        prob_y = cdf(tilted, hi) - cdf(tilted, lo)
        integral_y += 800.0 * exp(0.2 * age - 0.3 * vaccination + BETA * μ + 0.5 * v) * prob_y
        prob_shift += cdf(Normal(μ, sd_a), hi) - cdf(Normal(μ, sd_a), lo)
    end
    tau = expm1(BETA * delta) * integral_y / 4.0
    shift = delta * prob_shift / 4.0
    return (tau = tau, shift = shift)
end

# Exact closed-form ground truth for linear ExponentialTiltPolicy (bound = nothing)
function true_linear_tilt_truth(delta, slope, sigma)
    shift = delta * sigma^2
    tau = expm1(BETA * shift) * true_baseline_mean_y(slope, sigma)
    return (tau = tau, shift = shift)
end

# High-precision 2D Gauss-Hermite ground truth for arbitrary ExponentialTiltPolicy
function true_tilt_truth_quadrature(delta, slope, sigma, policy::ExponentialTiltPolicy)
    e_y0 = true_baseline_mean_y(slope, sigma)
    e_a0 = 0.0
    psi_y = 0.0
    psi_a = 0.0
    for age in (0.0, 1.0), vaccination in (0.0, 1.0)
        base_mu = 0.35 * age - 0.30 * vaccination
        base_y_factor = 800.0 * exp(0.2 * age - 0.3 * vaccination)
        e_a0 += base_mu / 4.0
        for (ih, h) in enumerate(GH_NODES)
            wh = GH_WEIGHTS[ih]
            mu_w = base_mu + slope * h
            y_w_factor = base_y_factor * exp(GAMMA * h)
            # Inner expectation over Z ~ N(0, 1), A = mu_w + sigma * Z
            # Shift log-weights by max for overflow safety
            max_lw = -Inf
            for z in GH_NODES
                lw = CT.tilt_log_weight(mu_w + sigma * z, delta, policy)
                lw > max_lw && (max_lw = lw)
            end
            m0_s = 0.0
            num_a_s = 0.0
            num_y_s = 0.0
            for (iz, z) in enumerate(GH_NODES)
                wz = GH_WEIGHTS[iz]
                a_val = mu_w + sigma * z
                wt = wz * exp(CT.tilt_log_weight(a_val, delta, policy) - max_lw)
                m0_s += wt
                num_a_s += wt * a_val
                num_y_s += wt * exp(BETA * a_val)
            end
            psi_a += 0.25 * wh * (num_a_s / m0_s)
            psi_y += 0.25 * wh * y_w_factor * (num_y_s / m0_s)
        end
    end
    return (tau = psi_y - e_y0, shift = psi_a - e_a0)
end

# Sanity-check Gauss-Hermite quadrature against closed-form linear tilt
function verify_quadrature_accuracy()
    for (_, slope, sigma) in SCENARIOS, d in DELTAS
        exact = true_linear_tilt_truth(d, slope, sigma)
        quad = true_tilt_truth_quadrature(d, slope, sigma, LINEAR_TILT)
        err_tau = abs(exact.tau - quad.tau) / max(1.0, abs(exact.tau))
        err_shift = abs(exact.shift - quad.shift) / max(1.0, abs(exact.shift))
        @assert err_tau < 1e-11 "Quadrature τ mismatch: $err_tau"
        @assert err_shift < 1e-11 "Quadrature Δ mismatch: $err_shift"
    end
end

const METHOD_NAMES = (
    "feasible_shift",
    "tilt_linear_plugin",
    "tilt_bounded_plugin",
    "tilt_bounded_direct",
    "tilt_bounded_lam01",
)

function run_scenario(n, slope, sigma, nrep, base_seed)
    n_deltas = length(DELTAS)
    n_methods = length(METHOD_NAMES)

    truths = Matrix{NamedTuple{(:tau, :shift), Tuple{Float64, Float64}}}(undef, n_deltas, n_methods)
    for (id, d) in enumerate(DELTAS)
        truths[id, 1] = true_feasible_truth(d, slope, sigma)
        truths[id, 2] = true_linear_tilt_truth(d, slope, sigma)
        bnd_t = true_tilt_truth_quadrature(d, slope, sigma, BOUNDED_TILT)
        truths[id, 3] = bnd_t
        truths[id, 4] = bnd_t
        truths[id, 5] = bnd_t
    end

    # Pre-simulate datasets deterministically so threading order does not affect results
    master_rng = StableRNG(base_seed)
    datasets = [simulate_negbin_dgp(master_rng, n, slope, sigma) for _ in 1:nrep]

    # Per-replicate metric storage: (nrep, n_deltas, n_methods).
    # Use byte-addressed Array{Bool} (not BitArray) to avoid bit-packing races under @threads.
    avail = fill(false, nrep, n_deltas, n_methods)
    est_tau = zeros(Float64, nrep, n_deltas, n_methods)
    est_shift = zeros(Float64, nrep, n_deltas, n_methods)
    ess_ratio = zeros(Float64, nrep, n_deltas, n_methods)
    wald_cov = fill(false, nrep, n_deltas, n_methods)
    boot_cov = fill(false, nrep, n_deltas, n_methods)
    shift_wald_cov = fill(false, nrep, n_deltas, n_methods)
    shift_boot_cov = fill(false, nrep, n_deltas, n_methods)

    Threads.@threads for rep in 1:nrep
        data = datasets[rep]
        # 1. FeasibleAdditiveShiftPolicy (exact setup from dev/feasible_shift_coverage.jl)
        cache = CT.build_lmtp_fold_cache(
            data, :A, :Y, COVARIATES, 3, StableRNG(20261124 + rep);
            learners_outcome = (:glm, :mean),
            learners_trt = (:glm,),
            family_outcome = :negbin,
            weight_truncation = false,
        )
        for (id, d) in enumerate(DELTAS)
            t_feas = truths[id, 1]
            shifted = apply_policy_values(data.A, d, FEASIBLE_POLICY)
            comps = CT.lmtp_components_from_cache(
                cache, shifted, data.A;
                feasible_policy = FEASIBLE_POLICY, feasible_delta = d,
            )
            fit = lmtp_tmle_from_components(comps; estimator = :tmle, epochs = 1)
            if fit.inference_status === :available
                avail[rep, id, 1] = true
                est_tau[rep, id, 1] = fit.estimate
                est_shift[rep, id, 1] = mean(shifted .- data.A)
                H = comps.H1
                sH2 = sum(abs2, H)
                ess_ratio[rep, id, 1] = sH2 > 0 ? (sum(H)^2 / sH2) / n : 0.0
                wald_cov[rep, id, 1] = fit.lower <= t_feas.tau <= fit.upper
                bci = studentised_multiplier_ci(
                    fit.estimate, fit.ic; B = 999, weights = :rademacher,
                    rng = StableRNG(900000 + 100 * rep + id),
                )
                boot_cov[rep, id, 1] = bci.lower <= t_feas.tau <= bci.upper
            end

            # 2-5. ExponentialTiltPolicy variants
            tilt_specs = (
                (2, LINEAR_TILT, :plugin, 0.0),
                (3, BOUNDED_TILT, :plugin, 0.0),
                (4, BOUNDED_TILT, :direct, 0.0),
                (5, BOUNDED_TILT, :plugin, 0.1),
            )
            for (im, pol, ometh, lam) in tilt_specs
                tr = truths[id, im]
                res = estimate_exponential_tilt(
                    data, :A, :Y, COVARIATES, d, pol;
                    family_outcome = :negbin,
                    folds = 3,
                    independent_folds = true,
                    lambda = lam,
                    outcome_method = ometh,
                    bootstrap_B = 999,
                    bootstrap_weights = :rademacher,
                    rng = StableRNG(20261124 + 1000 * im + 10 * rep + id),
                )
                if res.inference_status === :available
                    avail[rep, id, im] = true
                    est_tau[rep, id, im] = res.estimate
                    est_shift[rep, id, im] = res.shift_estimate
                    ess_ratio[rep, id, im] = res.ess_ratio
                    wald_cov[rep, id, im] = res.lower <= tr.tau <= res.upper
                    boot_cov[rep, id, im] = res.boot_lower <= tr.tau <= res.boot_upper
                    shift_wald_cov[rep, id, im] = res.shift_lower <= tr.shift <= res.shift_upper
                    shift_boot_cov[rep, id, im] = res.shift_boot_lower <= tr.shift <= res.shift_boot_upper
                end
            end
        end
    end

    rows = NamedTuple[]
    for (id, d) in enumerate(DELTAS), (im, mname) in enumerate(METHOD_NAMES)
        idx = findall(@view avail[:, id, im])
        navail = length(idx)
        tr = truths[id, im]
        if navail == 0
            push!(rows, (
                n = n, overlap = "", delta = d, method = mname,
                true_shift = tr.shift, mean_shift = NaN,
                true_tau = tr.tau, ess_ratio = NaN,
                avail = 0, nrep = nrep,
                bias = NaN, rmse = NaN,
                wald_cov = NaN, boot_cov = NaN,
                shift_wald_cov = NaN, shift_boot_cov = NaN,
            ))
            continue
        end
        errs = est_tau[idx, id, im] .- tr.tau
        push!(rows, (
            n = n, overlap = "", delta = d, method = mname,
            true_shift = tr.shift,
            mean_shift = mean(est_shift[idx, id, im]),
            true_tau = tr.tau,
            ess_ratio = mean(ess_ratio[idx, id, im]),
            avail = navail, nrep = nrep,
            bias = mean(errs),
            rmse = sqrt(mean(abs2, errs)),
            wald_cov = mean(wald_cov[idx, id, im]),
            boot_cov = mean(boot_cov[idx, id, im]),
            shift_wald_cov = im == 1 ? NaN : mean(shift_wald_cov[idx, id, im]),
            shift_boot_cov = im == 1 ? NaN : mean(shift_boot_cov[idx, id, im]),
        ))
    end
    return rows
end

function main()
    BLAS.set_num_threads(1)
    verify_quadrature_accuracy()
    replicates = isempty(ARGS) ? 300 : parse(Int, only(ARGS))
    println("# NegBin exponential-tilt vs feasible-shift benchmark (replicates = $replicates, threads = $(Threads.nthreads()))")
    println("n,overlap,delta,method,true_shift,mean_shift,true_tau,ess_ratio,avail,nrep,bias,rmse,wald_cov,boot_cov,shift_wald_cov,shift_boot_cov")
    t0 = time()
    for n in (30, 200), (sidx, (scenario, slope, sigma)) in enumerate(SCENARIOS)
        rows = run_scenario(n, slope, sigma, replicates, 20261125 + 100 * n + sidx)
        for r in rows
            @printf(
                "%d,%s,%.1f,%s,%.4f,%.4f,%.2f,%.4f,%d,%d,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f\n",
                r.n, scenario, r.delta, r.method, r.true_shift, r.mean_shift,
                r.true_tau, r.ess_ratio, r.avail, r.nrep, r.bias, r.rmse,
                r.wald_cov, r.boot_cov, r.shift_wald_cov, r.shift_boot_cov,
            )
        end
        flush(stdout)
    end
    @printf("# Elapsed: %.1fs\n", time() - t0)
end

main()
