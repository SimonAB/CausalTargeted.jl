"""Phase 5 untouched held-out bivariate confirmation driver.

Evaluates 3 reserved held-out bivariate scenarios × 2 deployable isotropic
Gaussian priors (`zero_s1p5`, `zero_s3p0`) across 4 methods:
  1. `tier1_psis`: Laplace-centred Student-t PSIS reference (`logistic_psis_reference.jl`)
  2. `variational`: Jaakkola-Jordan Raw VI (`covariance = :variational`)
  3. `linear_response`: Linear-Response Variational Bayes (`covariance = :linear_response`)
  4. `laplace_map`: Matched regularised logistic baseline (exact MAP + observed
     negative Hessian inverse on the identical Gaussian prior, 2,000 MvNormal draws)

Uses `LinearAlgebra.BLAS.set_num_threads(1)` and deterministic `Threads.@threads`
over `(cell, rep)` jobs with pre-allocated slot vectors and FNV-1a `StableRNG`
streams so all statistical outputs are bit-for-bit deterministic across thread counts.
"""
module Phase5HeldoutConfirmation

using CausalTargeted, RxInfer, GraphPPL
using DataFrames, CSV, SHA, TOML, LinearAlgebra, Statistics, Random, StableRNGs
using Distributions: Bernoulli, MvNormal, logpdf

export heldout_cells, prior_specs, replication_seed,
    simulate_heldout_dataset, fit_laplace_map,
    evaluate_dataset_prior, evaluate_heldout_dataset,
    summarise_confirmation_rows, paired_method_summaries,
    parse_options, run_confirmation, main

const METHODS = (:tier1_psis, :variational, :linear_response, :laplace_map)
const PRIOR_SCALES = (1.5, 3.0)

const HELDOUT_SCENARIOS = (
    (
        id = "heldout_n80_c24_b1p6",
        n = 80,
        n0 = 24,
        n1 = 56,
        beta_true = [-0.6, 1.6],
        description = "Moderate n=80, 30% controls, moderate positive effect",
    ),
    (
        id = "heldout_n150_c30_b4p0",
        n = 150,
        n0 = 30,
        n1 = 120,
        beta_true = [-2.0, 4.0],
        description = "Larger n=150, 20% controls, strong effect with frequent separation",
    ),
    (
        id = "heldout_n30_c1_b0p9",
        n = 30,
        n0 = 1,
        n1 = 29,
        beta_true = [-0.3, 0.9],
        description = "Small n=30, single control (n0=1), extreme exposure imbalance",
    ),
)

"""Return the 3 reserved Phase 5 held-out bivariate scenarios."""
function heldout_cells()
    return [
        (;
            id = s.id,
            n = s.n,
            n0 = s.n0,
            n1 = s.n1,
            beta_true = copy(s.beta_true),
            description = s.description,
        )
        for s in HELDOUT_SCENARIOS
    ]
end

scale_id(s::Real) = replace(string(Float64(s)), "." => "p")

"""Return the two deployable isotropic Gaussian priors (`zero_s1p5`, `zero_s3p0`) in 2D."""
function prior_specs(p::Integer = 2)
    p > 0 || throw(ArgumentError("prior dimension p must be positive"))
    return [
        (;
            id = "zero_s$(scale_id(s))",
            scale = Float64(s),
            mean = zeros(Float64, p),
            covariance = Float64(s)^2 * Matrix{Float64}(I, p, p),
        )
        for s in PRIOR_SCALES
    ]
end

# FNV-1a over an explicit encoding avoids Julia's process-dependent hash.
# Channel separates dataset simulation (0), Gaussian draws (1), and PSIS draws (2).
function replication_seed(cell_id::AbstractString, replication::Integer, seed::Integer, channel::Integer)
    value = UInt64(0xcbf29ce484222325)
    for byte in codeunits("$(seed)|$(ncodeunits(cell_id)):$(cell_id)|$(replication)|$(channel)")
        value = (value ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return value
end

@inline logistic(x::Real) = x >= 0 ? inv(1.0 + exp(-x)) : (ex = exp(x); ex / (1.0 + ex))
@inline _log1pexp(x::Real) = x <= 0 ? log1p(exp(x)) : x + log1p(exp(-x))

function classify_separation(A::AbstractVector, y::AbstractVector)
    all(==(first(y)), y) && return :single_class
    control = y[A .== 0]
    treated = y[A .== 1]
    pure_control = all(==(first(control)), control)
    pure_treated = all(==(first(treated)), treated)
    pure_control && pure_treated && return :complete
    (pure_control || pure_treated) && return :quasi
    return :none
end

"""Simulate one held-out bivariate dataset and exact cohort `psi_true`."""
function simulate_heldout_dataset(cell, replication::Integer; seed::Integer = 20261121)
    replication > 0 || throw(ArgumentError("replication must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    cell.n0 + cell.n1 == cell.n || throw(ArgumentError("n0 + n1 must equal n"))
    cell.n0 > 0 && cell.n1 > 0 || throw(ArgumentError("both exposure arms must be nonempty"))

    sim_seed = replication_seed(cell.id, replication, seed, 0)
    rng = StableRNG(sim_seed)
    A = vcat(zeros(Int, cell.n0), ones(Int, cell.n1))
    X_obs = hcat(ones(Float64, cell.n), Float64.(A))
    X_alt = hcat(ones(Float64, cell.n), ones(Float64, cell.n))
    η = X_obs * cell.beta_true
    y = [Int(rand(rng, Bernoulli(logistic(η[i])))) for i in 1:cell.n]

    β0, β1 = Float64(cell.beta_true[1]), Float64(cell.beta_true[2])
    psi_true = (cell.n0 / cell.n) * (logistic(β0 + β1) - logistic(β0))
    df = DataFrame(A = A, Y = y)
    separation = string(classify_separation(A, y))

    dataset_id = "$(cell.id)|seed=$(seed)|rep=$(replication)"
    draw_seed = replication_seed(cell.id, replication, seed, 1)
    psis_seed = replication_seed(cell.id, replication, seed, 2)

    return (;
        df,
        psi_true,
        dataset_id,
        sim_seed,
        draw_seed,
        psis_seed,
        X = X_obs,
        X_obs,
        X_alt,
        y = Vector{Float64}(y),
        A,
        separation,
    )
end

function _laplace_log_posterior(
    X::Matrix{Float64},
    y::Vector{Float64},
    prior::MvNormal,
    β::AbstractVector{<:Real},
)
    lp = Float64(logpdf(prior, β))
    isfinite(lp) || return -Inf
    n = size(X, 1)
    if n > 0
        η = X * β
        @inbounds for i in 1:n
            ηi = Float64(η[i])
            lp += y[i] == 1.0 ? -_log1pexp(-ηi) : -_log1pexp(ηi)
        end
    end
    return lp
end

"""Fit exact MAP and observed negative Hessian inverse under Gaussian prior `N(μ0, Σ0)`."""
function fit_laplace_map(
    X::AbstractMatrix,
    y::AbstractVector,
    μ0::AbstractVector,
    Σ0::AbstractMatrix;
    max_iter::Int = 200,
)
    X_m = Matrix{Float64}(X)
    y_v = Float64.(y)
    μ_v = Vector{Float64}(μ0)
    Σ_m = Matrix{Float64}(Σ0)
    n, p = size(X_m)
    length(y_v) == n || throw(DimensionMismatch("X and y row counts must match"))
    length(μ_v) == p || throw(DimensionMismatch("μ0 length must match columns of X"))
    size(Σ_m) == (p, p) || throw(DimensionMismatch("Σ0 dimensions must match columns of X"))

    all(isfinite, X_m) && all(isfinite, y_v) && all(isfinite, μ_v) && all(isfinite, Σ_m) ||
        return (; converged = false, status = "newton_failed",
            posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = 0)

    Σ0_sym = Symmetric(0.5 .* (Σ_m .+ transpose(Σ_m)))
    prior_chol = cholesky(Σ0_sym; check = false)
    issuccess(prior_chol) ||
        return (; converged = false, status = "hessian_not_pd",
            posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = 0)

    inv_Σ0 = Matrix{Float64}(Symmetric(inv(prior_chol)))
    prior = MvNormal(μ_v, Matrix(Σ0_sym))
    β = copy(μ_v)
    f_cur = _laplace_log_posterior(X_m, y_v, prior, β)
    isfinite(f_cur) ||
        return (; converged = false, status = "newton_failed",
            posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = 0)

    η = Vector{Float64}(undef, n)
    w_diag = Vector{Float64}(undef, n)
    residual = Vector{Float64}(undef, n)
    g = Vector{Float64}(undef, p)
    H = Matrix{Float64}(undef, p, p)

    iters = 0
    for iter in 1:max_iter
        iters = iter
        if n > 0
            mul!(η, X_m, β)
            @inbounds for i in 1:n
                pi = logistic(η[i])
                w_diag[i] = pi * (1.0 - pi)
                residual[i] = y_v[i] - pi
            end
            mul!(g, transpose(X_m), residual)
            g .-= inv_Σ0 * (β .- μ_v)
            mul!(H, transpose(X_m), X_m .* w_diag)
            H .+= inv_Σ0
        else
            g .= -inv_Σ0 * (β .- μ_v)
            H .= inv_Σ0
        end

        H_sym = Symmetric(0.5 .* (H .+ transpose(H)))
        all(isfinite, H_sym) ||
            return (; converged = false, status = "hessian_not_pd",
                posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)
        H_chol = cholesky(H_sym; check = false)
        issuccess(H_chol) ||
            return (; converged = false, status = "hessian_not_pd",
                posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)

        d = H_chol \ g
        all(isfinite, d) ||
            return (; converged = false, status = "newton_failed",
                posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)

        if norm(d, Inf) < 1e-10 && norm(g, Inf) < 1e-8
            Σ_lap = Matrix{Float64}(Symmetric(inv(H_chol)))
            return (; converged = true, status = "converged",
                posterior_mean = β, posterior_covariance = Σ_lap, n_iterations = iters)
        end

        slope = dot(g, d)
        if slope <= 0.0
            if norm(g, Inf) < 1e-6
                Σ_lap = Matrix{Float64}(Symmetric(inv(H_chol)))
                return (; converged = true, status = "converged",
                    posterior_mean = β, posterior_covariance = Σ_lap, n_iterations = iters)
            end
            return (; converged = false, status = "newton_failed",
                posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)
        end

        α = 1.0
        accepted = false
        while α >= 1e-12
            β_cand = β .+ α .* d
            f_cand = _laplace_log_posterior(X_m, y_v, prior, β_cand)
            if isfinite(f_cand) && f_cand >= f_cur + 1e-4 * α * slope
                β = β_cand
                f_cur = f_cand
                accepted = true
                break
            end
            α *= 0.5
        end

        if !accepted
            if norm(g, Inf) < 1e-6
                Σ_lap = Matrix{Float64}(Symmetric(inv(H_chol)))
                return (; converged = true, status = "converged",
                    posterior_mean = β, posterior_covariance = Σ_lap, n_iterations = iters)
            end
            return (; converged = false, status = "newton_failed",
                posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)
        end
    end

    if n > 0
        mul!(η, X_m, β)
        @inbounds for i in 1:n
            pi = logistic(η[i])
            w_diag[i] = pi * (1.0 - pi)
            residual[i] = y_v[i] - pi
        end
        mul!(g, transpose(X_m), residual)
        g .-= inv_Σ0 * (β .- μ_v)
        mul!(H, transpose(X_m), X_m .* w_diag)
        H .+= inv_Σ0
    else
        g .= -inv_Σ0 * (β .- μ_v)
        H .= inv_Σ0
    end
    H_sym = Symmetric(0.5 .* (H .+ transpose(H)))
    H_chol = cholesky(H_sym; check = false)
    issuccess(H_chol) ||
        return (; converged = false, status = "hessian_not_pd",
            posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)
    if norm(g, Inf) < 1e-6
        Σ_lap = Matrix{Float64}(Symmetric(inv(H_chol)))
        return (; converged = true, status = "converged",
            posterior_mean = β, posterior_covariance = Σ_lap, n_iterations = iters)
    end
    return (; converged = false, status = "newton_failed",
        posterior_mean = fill(NaN, p), posterior_covariance = fill(NaN, p, p), n_iterations = iters)
end

const _PSIS_INCLUDE_LOCK = ReentrantLock()

function ensure_psis_reference!()
    lock(_PSIS_INCLUDE_LOCK) do
        if !isdefined(@__MODULE__, :evaluate_logistic_psis_reference) &&
                !isdefined(@__MODULE__, :LogisticPSISReference)
            path = joinpath(@__DIR__, "logistic_psis_reference.jl")
            isfile(path) || throw(ArgumentError("PSIS reference script not found at $path"))
            Base.include(@__MODULE__, path)
        end
    end
    return nothing
end

function resolve_psis_functions()
    ensure_psis_reference!()
    if isdefined(@__MODULE__, :evaluate_logistic_psis_reference)
        ref_fn = Base.invokelatest(getfield, @__MODULE__, :evaluate_logistic_psis_reference)
        wic_fn = isdefined(@__MODULE__, :weighted_interval_content) ?
            Base.invokelatest(getfield, @__MODULE__, :weighted_interval_content) : nothing
        return (; ref_fn, wic_fn)
    elseif isdefined(@__MODULE__, :LogisticPSISReference)
        mod = Base.invokelatest(getfield, @__MODULE__, :LogisticPSISReference)
        ref_fn = Base.invokelatest(getfield, mod, :evaluate_logistic_psis_reference)
        wic_fn = isdefined(mod, :weighted_interval_content) ?
            Base.invokelatest(getfield, mod, :weighted_interval_content) : nothing
        return (; ref_fn, wic_fn)
    end
    error("logistic_psis_reference.jl did not define evaluate_logistic_psis_reference")
end

function invoke_psis_reference(ref_fn, X, y, mu0, Sigma0, X_alt, X_obs, psis_seed::UInt64)
    ms = Base.invokelatest(methods, ref_fn)
    kw = isempty(ms) ? Symbol[] : Base.invokelatest(Base.kwarg_decl, first(ms))
    if :rng in kw
        return Base.invokelatest(ref_fn, X, y, mu0, Sigma0, X_alt, X_obs; rng = StableRNG(psis_seed))
    elseif :seed in kw
        return Base.invokelatest(ref_fn, X, y, mu0, Sigma0, X_alt, X_obs; seed = psis_seed)
    else
        return Base.invokelatest(ref_fn, X, y, mu0, Sigma0, X_alt, X_obs)
    end
end

function compute_reference_content(wic_fn, ref, lo::Real, hi::Real)
    wic_fn === nothing && return NaN
    isfinite(lo) && isfinite(hi) && lo <= hi || return NaN
    lo_f, hi_f = Float64(lo), Float64(hi)
    if hasproperty(ref, :policy_draws) && hasproperty(ref, :weights) &&
            Base.invokelatest(hasmethod, wic_fn, Tuple{typeof(ref.policy_draws), typeof(ref.weights), Float64, Float64})
        res = Base.invokelatest(wic_fn, ref.policy_draws, ref.weights, lo_f, hi_f)
        return res isa Real ? Float64(res) : Float64(res.content)
    elseif hasproperty(ref, :draws) && hasproperty(ref, :weights) &&
            Base.invokelatest(hasmethod, wic_fn, Tuple{typeof(ref.draws), typeof(ref.weights), Float64, Float64})
        res = Base.invokelatest(wic_fn, ref.draws, ref.weights, lo_f, hi_f)
        return res isa Real ? Float64(res) : Float64(res.content)
    elseif Base.invokelatest(hasmethod, wic_fn, Tuple{typeof(ref), Float64, Float64})
        res = Base.invokelatest(wic_fn, ref, lo_f, hi_f)
        return res isa Real ? Float64(res) : Float64(res.content)
    end
    return NaN
end

function extract_prop(obj, names::Tuple, default)
    for name in names
        hasproperty(obj, name) && return getproperty(obj, name)
    end
    return default
end

isfinite_real(x) = x isa Real && isfinite(x)
safe_mean(xs) = isempty(xs) ? NaN : mean(xs)
safe_max(xs) = isempty(xs) ? NaN : maximum(xs)
safe_min(xs) = isempty(xs) ? NaN : minimum(xs)
binomial_mcse(p, n) = n == 0 ? NaN : sqrt(p * (1 - p) / n)

function valid_covariance(V, p::Int)
    return V isa AbstractMatrix && size(V) == (p, p) && all(isfinite, V)
end

function compute_moment_fidelity(fit_mean, fit_cov, ref_mean, ref_cov)
    p = length(ref_mean)
    length(fit_mean) == p && all(isfinite, fit_mean) && all(isfinite, ref_mean) || return (NaN, NaN)
    valid_covariance(fit_cov, p) && valid_covariance(ref_cov, p) || return (NaN, NaN)
    ref_vars = diag(ref_cov)
    all(v -> isfinite(v) && v > 0, ref_vars) || return (NaN, NaN)
    ref_norm = norm(ref_cov)
    isfinite(ref_norm) && ref_norm > 0 || return (NaN, NaN)
    mean_err_sd = maximum(abs.(fit_mean .- ref_mean) ./ sqrt.(ref_vars))
    cov_err = norm(fit_cov .- ref_cov) / ref_norm
    return (Float64(mean_err_sd), Float64(cov_err))
end

function base_row(cell, dataset, prior, replication::Integer, seed::Integer, method::Symbol)
    return (;
        dataset_id = dataset.dataset_id,
        cell_id = cell.id,
        n = Int(cell.n),
        n0 = Int(cell.n0),
        n1 = Int(cell.n1),
        separation = dataset.separation,
        replication = Int(replication),
        prior_id = prior.id,
        prior_scale = Float64(prior.scale),
        method = string(method),
        seed = Int(seed),
        draw_seed = UInt64(dataset.draw_seed),
        psi_true = Float64(dataset.psi_true),
        available = false,
        status = "not_run",
        lrvb_status = "not_applicable",
        fit_iters = 0,
        reference_available = false,
        fidelity_available = false,
        failure_reason = "",
        pareto_k = NaN,
        psis_ess = NaN,
        psis_mcse = NaN,
        policy_mean = NaN,
        policy_lower = NaN,
        policy_upper = NaN,
        mean_mcse = NaN,
        width = NaN,
        causal_bias = NaN,
        causal_covered = missing,
        mean_error_sd = NaN,
        covariance_error = NaN,
        policy_error_vs_ref = NaN,
        reference_content = NaN,
        endpoint_error = NaN,
        seconds = 0.0,
        exception = "",
    )
end

"""Evaluate all 4 methods (`tier1_psis`, `variational`, `linear_response`, `laplace_map`) for one dataset and prior."""
function evaluate_dataset_prior(cell, dataset, prior, replication::Integer;
        seed::Integer = 20261121,
        n_draws::Integer = 2000,
        reference_fn = nothing,
        interval_content_fn = nothing,
        fitter = fit_bayesian_logistic,
        contrast_fn = bayesian_policy_contrast)

    if reference_fn === nothing
        resolved = resolve_psis_functions()
        reference_fn = resolved.ref_fn
        interval_content_fn === nothing && (interval_content_fn = resolved.wic_fn)
    end

    # 1. Tier 1 PSIS reference
    ref = nothing
    ref_exception = ""
    ref_timing = @timed try
        ref = invoke_psis_reference(reference_fn, dataset.X, dataset.y,
            prior.mean, prior.covariance, dataset.X_alt, dataset.X_obs, dataset.psis_seed)
    catch ex
        ex isa InterruptException && rethrow()
        ref_exception = sprint(showerror, ex)
        ref = nothing
    end

    ref_avail = ref !== nothing && Bool(extract_prop(ref, (:available,), false))
    pareto_k = ref === nothing ? NaN : Float64(extract_prop(ref, (:khat, :pareto_k, :k_hat, :k), NaN))
    psis_ess = ref === nothing ? NaN : Float64(extract_prop(ref, (:ess, :psis_ess), NaN))
    psis_mcse = ref === nothing ? NaN : Float64(extract_prop(ref, (:mcse, :policy_mcse), NaN))
    ref_failure = if ref === nothing
        "reference_exception"
    elseif !ref_avail
        string(extract_prop(ref, (:failure_reason, :reason, :status), "psis_gate_failed"))
    else
        ""
    end

    ref_coef_mean = ref === nothing ? nothing : extract_prop(ref, (:beta_mean, :posterior_mean, :mean, :coef_mean), nothing)
    ref_coef_cov = ref === nothing ? nothing : extract_prop(ref, (:beta_cov, :posterior_covariance, :covariance, :cov, :coef_cov), nothing)
    ref_pmean = ref === nothing ? NaN : Float64(extract_prop(ref, (:policy_mean, :psi_mean, :contrast_mean), NaN))
    ref_lo = ref === nothing ? NaN : Float64(extract_prop(ref, (:policy_lower, :lower), NaN))
    ref_hi = ref === nothing ? NaN : Float64(extract_prop(ref, (:policy_upper, :upper), NaN))

    ref_valid_interval = ref_avail && isfinite(ref_pmean) && isfinite(ref_lo) && isfinite(ref_hi) && ref_lo <= ref_hi
    ref_avail_eff = ref_valid_interval

    ref_content_self = if ref_avail_eff
        compute_reference_content(interval_content_fn, ref, ref_lo, ref_hi)
    else
        NaN
    end

    ref_mean_err_sd, ref_cov_err = if ref_avail_eff && ref_coef_mean !== nothing && ref_coef_cov !== nothing
        compute_moment_fidelity(ref_coef_mean, ref_coef_cov, ref_coef_mean, ref_coef_cov)
    else
        (ref_avail_eff ? 0.0 : NaN, ref_avail_eff ? 0.0 : NaN)
    end

    row_psis = merge(base_row(cell, dataset, prior, replication, seed, :tier1_psis), (;
        available = ref_avail_eff,
        status = ref_avail_eff ? "ok" : (ref === nothing ? "reference_exception" : ref_failure),
        lrvb_status = "not_applicable",
        fit_iters = 0,
        reference_available = ref_avail_eff,
        fidelity_available = ref_avail_eff,
        failure_reason = ref_failure,
        pareto_k,
        psis_ess,
        psis_mcse,
        policy_mean = ref_avail_eff ? ref_pmean : NaN,
        policy_lower = ref_avail_eff ? ref_lo : NaN,
        policy_upper = ref_avail_eff ? ref_hi : NaN,
        mean_mcse = ref_avail_eff ? psis_mcse : NaN,
        width = ref_avail_eff ? (ref_hi - ref_lo) : NaN,
        causal_bias = ref_avail_eff ? (ref_pmean - dataset.psi_true) : NaN,
        causal_covered = ref_avail_eff ? (ref_lo <= dataset.psi_true <= ref_hi) : missing,
        mean_error_sd = ref_mean_err_sd,
        covariance_error = ref_cov_err,
        policy_error_vs_ref = ref_avail_eff ? 0.0 : NaN,
        reference_content = ref_content_self,
        endpoint_error = ref_avail_eff ? 0.0 : NaN,
        seconds = ref_timing.time,
        exception = ref_exception,
    ))

    method_rows = NamedTuple[row_psis]

    # 2 & 3. Variational (VI) and Linear Response (LRVB)
    gprior = GaussianCoefficientPrior(; mean = prior.mean, covariance = prior.covariance)
    policy = discrete_static_policy(1; levels = [0, 1])

    for cov_method in (:variational, :linear_response)
        fit = nothing
        contrast = nothing
        meth_exception = ""
        timing = @timed try
            fit = fitter(dataset.df, :Y, :A, Symbol[];
                prior = gprior, maxiter = 1000, tol = 1e-10,
                covariance = cov_method)
            contrast = contrast_fn(fit, dataset.df, policy;
                deltas = [1.0], n_draws = n_draws, rng = StableRNG(dataset.draw_seed))
        catch ex
            ex isa InterruptException && rethrow()
            meth_exception = sprint(showerror, ex)
        end

        lrvb_diag = if fit !== nothing && hasproperty(fit, :diagnostics) &&
                hasproperty(fit.diagnostics, :linear_response) &&
                hasproperty(fit.diagnostics.linear_response, :reason)
            string(fit.diagnostics.linear_response.reason)
        else
            fit === nothing ? "fit_exception" : "unknown"
        end
        fit_iters = fit !== nothing && hasproperty(fit, :metadata) && hasproperty(fit.metadata, :n_iterations) ?
            Int(fit.metadata.n_iterations) : 0

        pmean = contrast === nothing ? NaN : Float64(only(contrast.mean))
        lo = contrast === nothing ? NaN : Float64(only(contrast.lower))
        hi = contrast === nothing ? NaN : Float64(only(contrast.upper))
        mcse = contrast === nothing ? NaN : Float64(only(contrast.mcse))

        fit_ok = fit !== nothing && fit.status == :converged &&
            hasproperty(fit, :metadata) && fit.metadata.uncertainty_available
        contrast_ok = contrast !== nothing &&
            hasproperty(contrast, :metadata) && contrast.metadata.uncertainty != :unavailable &&
            isfinite(pmean) && isfinite(lo) && isfinite(hi) && lo <= hi
        avail = fit_ok && contrast_ok

        meth_status = if !isempty(meth_exception)
            "fit_exception"
        elseif fit === nothing
            "fit_exception"
        elseif !fit_ok
            string(fit.status)
        elseif !contrast_ok
            "contrast_unavailable"
        else
            "ok"
        end

        fidelity_ok = avail && ref_avail_eff
        mean_err_sd, cov_err = if fidelity_ok && ref_coef_mean !== nothing && ref_coef_cov !== nothing
            compute_moment_fidelity(fit.posterior_mean, fit.posterior_covariance, ref_coef_mean, ref_coef_cov)
        else
            (NaN, NaN)
        end
        pol_err_ref = fidelity_ok ? (pmean - ref_pmean) : NaN
        ref_content = fidelity_ok ? compute_reference_content(interval_content_fn, ref, lo, hi) : NaN
        ep_err = fidelity_ok ? max(abs(lo - ref_lo), abs(hi - ref_hi)) : NaN

        push!(method_rows, merge(base_row(cell, dataset, prior, replication, seed, cov_method), (;
            available = avail,
            status = meth_status,
            lrvb_status = lrvb_diag,
            fit_iters,
            reference_available = ref_avail_eff,
            fidelity_available = fidelity_ok,
            failure_reason = ref_failure,
            pareto_k,
            psis_ess,
            psis_mcse,
            policy_mean = isfinite(pmean) ? pmean : NaN,
            policy_lower = avail ? lo : NaN,
            policy_upper = avail ? hi : NaN,
            mean_mcse = avail && isfinite(mcse) ? mcse : NaN,
            width = avail ? (hi - lo) : NaN,
            causal_bias = isfinite(pmean) ? (pmean - dataset.psi_true) : NaN,
            causal_covered = avail ? (lo <= dataset.psi_true <= hi) : missing,
            mean_error_sd = mean_err_sd,
            covariance_error = cov_err,
            policy_error_vs_ref = pol_err_ref,
            reference_content = ref_content,
            endpoint_error = ep_err,
            seconds = timing.time,
            exception = meth_exception,
        )))
    end

    # 4. Laplace MAP baseline (matched Gaussian prior, exact MAP + observed H⁻¹)
    lap_fit = nothing
    lap_pmean = NaN
    lap_lo = NaN
    lap_hi = NaN
    lap_mcse = NaN
    lap_exception = ""
    lap_timing = @timed try
        lap_fit = fit_laplace_map(dataset.X, dataset.y, prior.mean, prior.covariance)
        if lap_fit.converged
            Σ_lap = Matrix{Float64}(Symmetric(0.5 .* (lap_fit.posterior_covariance .+ transpose(lap_fit.posterior_covariance))))
            lap_dist = MvNormal(lap_fit.posterior_mean, Σ_lap)
            β_draws = rand(StableRNG(dataset.draw_seed), lap_dist, Int(n_draws))
            η_alt = dataset.X_alt * β_draws
            η_obs = dataset.X_obs * β_draws
            pol_draws = vec(mean(logistic.(η_alt) .- logistic.(η_obs); dims = 1))
            lap_pmean = Float64(mean(pol_draws))
            lap_lo = Float64(quantile(pol_draws, 0.025))
            lap_hi = Float64(quantile(pol_draws, 0.975))
            lap_mcse = Float64(std(pol_draws) / sqrt(n_draws))
        end
    catch ex
        ex isa InterruptException && rethrow()
        lap_exception = sprint(showerror, ex)
    end

    lap_avail = isempty(lap_exception) && lap_fit !== nothing && lap_fit.converged &&
        isfinite(lap_pmean) && isfinite(lap_lo) && isfinite(lap_hi) && lap_lo <= lap_hi
    lap_status = if !isempty(lap_exception)
        "fit_exception"
    elseif lap_fit === nothing
        "fit_exception"
    elseif !lap_fit.converged
        lap_fit.status
    elseif !lap_avail
        "contrast_unavailable"
    else
        "ok"
    end
    lap_iters = lap_fit === nothing ? 0 : lap_fit.n_iterations

    lap_fidelity_ok = lap_avail && ref_avail_eff
    lap_mean_err_sd, lap_cov_err = if lap_fidelity_ok && ref_coef_mean !== nothing && ref_coef_cov !== nothing
        compute_moment_fidelity(lap_fit.posterior_mean, lap_fit.posterior_covariance, ref_coef_mean, ref_coef_cov)
    else
        (NaN, NaN)
    end
    lap_pol_err_ref = lap_fidelity_ok ? (lap_pmean - ref_pmean) : NaN
    lap_ref_content = lap_fidelity_ok ? compute_reference_content(interval_content_fn, ref, lap_lo, lap_hi) : NaN
    lap_ep_err = lap_fidelity_ok ? max(abs(lap_lo - ref_lo), abs(lap_hi - ref_hi)) : NaN

    push!(method_rows, merge(base_row(cell, dataset, prior, replication, seed, :laplace_map), (;
        available = lap_avail,
        status = lap_status,
        lrvb_status = "not_applicable",
        fit_iters = lap_iters,
        reference_available = ref_avail_eff,
        fidelity_available = lap_fidelity_ok,
        failure_reason = ref_failure,
        pareto_k,
        psis_ess,
        psis_mcse,
        policy_mean = isfinite(lap_pmean) ? lap_pmean : NaN,
        policy_lower = lap_avail ? lap_lo : NaN,
        policy_upper = lap_avail ? lap_hi : NaN,
        mean_mcse = lap_avail && isfinite(lap_mcse) ? lap_mcse : NaN,
        width = lap_avail ? (lap_hi - lap_lo) : NaN,
        causal_bias = isfinite(lap_pmean) ? (lap_pmean - dataset.psi_true) : NaN,
        causal_covered = lap_avail ? (lap_lo <= dataset.psi_true <= lap_hi) : missing,
        mean_error_sd = lap_mean_err_sd,
        covariance_error = lap_cov_err,
        policy_error_vs_ref = lap_pol_err_ref,
        reference_content = lap_ref_content,
        endpoint_error = lap_ep_err,
        seconds = lap_timing.time,
        exception = lap_exception,
    )))

    return method_rows
end

"""Simulate one held-out dataset and evaluate across both deployable priors (8 rows total)."""
function evaluate_heldout_dataset(cell, replication::Integer;
        seed::Integer = 20261121, n_draws::Integer = 2000, kwargs...)
    dataset = simulate_heldout_dataset(cell, replication; seed)
    priors = prior_specs(2)
    rows = NamedTuple[]
    for prior in priors
        append!(rows, evaluate_dataset_prior(cell, dataset, prior, replication;
            seed, n_draws, kwargs...))
    end
    return rows
end

"""Summarise finite paired differences (`b - a`) with sample-SD Monte Carlo error."""
function paired_metric_summary(values_a, values_b)
    length(values_a) == length(values_b) || throw(ArgumentError("paired vectors must have equal lengths"))
    diffs = Float64[Float64(b) - Float64(a) for (a, b) in zip(values_a, values_b)
        if isfinite_real(a) && isfinite_real(b)]
    count = length(diffs)
    return (; count, delta = safe_mean(diffs), mcse = count < 2 ? NaN : std(diffs) / sqrt(count))
end

function covered_binary(row)
    v = row.causal_covered
    ismissing(v) && return missing
    v isa Real && v in (0, 1) || throw(ArgumentError("causal_covered must be Bool/binary or missing"))
    return Float64(v)
end

function summarise_group(rows)
    first_r = first(rows)
    attempted = length(rows)
    avail_rows = filter(r -> r.available, rows)
    available = length(avail_rows)
    ref_avail_count = count(r -> r.reference_available, rows)
    fidelity_count = count(r -> r.fidelity_available, rows)

    inclusions = Float64[covered_binary(r) for r in avail_rows if !ismissing(covered_binary(r))]
    cov_count = length(inclusions)
    covered_sum = sum(inclusions; init = 0.0)
    cov_avail = cov_count == 0 ? NaN : covered_sum / cov_count
    cov_all = covered_sum / attempted

    causal_errs = Float64[r.causal_bias for r in avail_rows if isfinite_real(r.causal_bias)]
    widths = Float64[r.width for r in avail_rows if isfinite_real(r.width)]
    mean_errs = Float64[r.mean_error_sd for r in rows if r.fidelity_available && isfinite_real(r.mean_error_sd)]
    cov_errs = Float64[r.covariance_error for r in rows if r.fidelity_available && isfinite_real(r.covariance_error)]
    pol_ref_errs = Float64[r.policy_error_vs_ref for r in rows if r.fidelity_available && isfinite_real(r.policy_error_vs_ref)]
    contents = Float64[r.reference_content for r in rows if r.fidelity_available && isfinite_real(r.reference_content)]
    ep_errs = Float64[r.endpoint_error for r in rows if r.fidelity_available && isfinite_real(r.endpoint_error)]

    khats = Float64[r.pareto_k for r in rows if isfinite_real(r.pareto_k)]
    ess_vals = Float64[r.psis_ess for r in rows if isfinite_real(r.psis_ess)]
    secs = Float64[r.seconds for r in rows if isfinite_real(r.seconds)]

    failures = Dict{String, Int}()
    for r in rows
        r.available && continue
        st = string(r.status)
        failures[st] = get(failures, st, 0) + 1
    end
    failure_reasons = join(["$k=$(failures[k])" for k in sort!(collect(keys(failures)))], ";")

    return (;
        cell_id = String(first_r.cell_id),
        n = Int(first_r.n),
        n0 = Int(first_r.n0),
        n1 = Int(first_r.n1),
        prior_id = String(first_r.prior_id),
        prior_scale = Float64(first_r.prior_scale),
        method = String(first_r.method),
        attempted,
        available,
        failed = attempted - available,
        availability = available / attempted,
        reference_available_count = ref_avail_count,
        fidelity_available_count = fidelity_count,
        causal_coverage_available = cov_avail,
        causal_coverage_all_attempts = cov_all,
        causal_coverage_mcse_available = binomial_mcse(cov_avail, cov_count),
        causal_coverage_mcse_all_attempts = binomial_mcse(cov_all, attempted),
        causal_bias_mean = safe_mean(causal_errs),
        causal_rmse = isempty(causal_errs) ? NaN : sqrt(mean(abs2, causal_errs)),
        causal_bias_count = length(causal_errs),
        width_mean = safe_mean(widths),
        width_count = length(widths),
        mean_error_sd_mean = safe_mean(mean_errs),
        mean_error_sd_max = safe_max(mean_errs),
        mean_error_sd_count = length(mean_errs),
        covariance_error_mean = safe_mean(cov_errs),
        covariance_error_max = safe_max(cov_errs),
        covariance_error_count = length(cov_errs),
        policy_error_vs_ref_mean = safe_mean(pol_ref_errs),
        policy_error_vs_ref_rmse = isempty(pol_ref_errs) ? NaN : sqrt(mean(abs2, pol_ref_errs)),
        policy_error_vs_ref_count = length(pol_ref_errs),
        reference_content_mean = safe_mean(contents),
        reference_content_count = length(contents),
        endpoint_error_mean = safe_mean(ep_errs),
        endpoint_error_max = safe_max(ep_errs),
        endpoint_error_count = length(ep_errs),
        pareto_k_mean = safe_mean(khats),
        pareto_k_max = safe_max(khats),
        psis_ess_min = safe_min(ess_vals),
        psis_ess_mean = safe_mean(ess_vals),
        seconds_mean = safe_mean(secs),
        seconds_total = isempty(secs) ? 0.0 : sum(secs),
        failure_reasons,
    )
end

"""Aggregate rows by `(cell_id, prior_id, method)`."""
function summarise_confirmation_rows(rows)
    groups = Dict{Tuple{String, String, String}, Vector{Any}}()
    seen = Set{Tuple{String, String, String, String}}()
    for row in rows
        key = (String(row.cell_id), String(row.prior_id), String(row.method))
        ukey = (key..., String(row.dataset_id))
        ukey in seen && throw(ArgumentError("duplicate confirmation row for $ukey"))
        push!(seen, ukey)
        push!(get!(groups, key, Any[]), row)
    end
    keys_sorted = sort!(collect(keys(groups)))
    return [summarise_group(groups[k]) for k in keys_sorted]
end

"""Compute paired `linear_response` minus `variational` and `linear_response` minus `laplace_map` summaries."""
function paired_method_summaries(rows)
    groups = Dict{Tuple{String, String, String}, Vector{Any}}()
    seen = Set{Tuple{String, String, String, String}}()
    for row in rows
        key = (String(row.cell_id), String(row.prior_id), String(row.method))
        ukey = (key..., String(row.dataset_id))
        ukey in seen && throw(ArgumentError("duplicate confirmation row for $ukey"))
        push!(seen, ukey)
        push!(get!(groups, key, Any[]), row)
    end

    cell_priors = sort!(unique((String(r.cell_id), String(r.prior_id)) for r in rows))
    paired = NamedTuple[]
    for (cell_id, prior_id) in cell_priors
        haskey(groups, (cell_id, prior_id, "linear_response")) || continue
        lr_rows = groups[(cell_id, prior_id, "linear_response")]
        by_lr = Dict(String(r.dataset_id) => r for r in lr_rows)

        for method_a in ("variational", "laplace_map")
            haskey(groups, (cell_id, prior_id, method_a)) || continue
            a_rows = groups[(cell_id, prior_id, method_a)]
            by_a = Dict(String(r.dataset_id) => r for r in a_rows)
            ids = sort!(collect(intersect(keys(by_a), keys(by_lr))))
            isempty(ids) && continue

            a = [by_a[id] for id in ids]
            b = [by_lr[id] for id in ids]
            first_r = first(a)

            avail_delta = paired_metric_summary([Float64(r.available) for r in a], [Float64(r.available) for r in b])
            joint_avail = [(ra, rb) for (ra, rb) in zip(a, b) if ra.available && rb.available]
            cov_delta = paired_metric_summary(
                [covered_binary(ra) for (ra, _) in joint_avail],
                [covered_binary(rb) for (_, rb) in joint_avail])
            incl_val(r) = r.available && !ismissing(covered_binary(r)) ? covered_binary(r) : 0.0
            all_incl = paired_metric_summary(incl_val.(a), incl_val.(b))

            width_s = paired_metric_summary(
                [ra.width for (ra, rb) in joint_avail],
                [rb.width for (ra, rb) in joint_avail])
            abs_bias_s = paired_metric_summary(
                [abs(ra.causal_bias) for (ra, rb) in joint_avail],
                [abs(rb.causal_bias) for (ra, rb) in joint_avail])

            joint_fid = [(ra, rb) for (ra, rb) in zip(a, b) if ra.fidelity_available && rb.fidelity_available]
            mean_err_s = paired_metric_summary(
                [ra.mean_error_sd for (ra, rb) in joint_fid],
                [rb.mean_error_sd for (ra, rb) in joint_fid])
            cov_err_s = paired_metric_summary(
                [ra.covariance_error for (ra, rb) in joint_fid],
                [rb.covariance_error for (ra, rb) in joint_fid])
            pol_ref_abs_s = paired_metric_summary(
                [abs(ra.policy_error_vs_ref) for (ra, rb) in joint_fid],
                [abs(rb.policy_error_vs_ref) for (ra, rb) in joint_fid])
            content_s = paired_metric_summary(
                [ra.reference_content for (ra, rb) in joint_fid],
                [rb.reference_content for (ra, rb) in joint_fid])
            ep_err_s = paired_metric_summary(
                [ra.endpoint_error for (ra, rb) in joint_fid],
                [rb.endpoint_error for (ra, rb) in joint_fid])
            sec_s = paired_metric_summary([ra.seconds for ra in a], [rb.seconds for rb in b])

            push!(paired, (;
                cell_id,
                n = Int(first_r.n),
                n0 = Int(first_r.n0),
                n1 = Int(first_r.n1),
                prior_id,
                prior_scale = Float64(first_r.prior_scale),
                method_a,
                method_b = "linear_response",
                contrast = "linear_response_minus_$(method_a)",
                attempts = length(ids),
                paired_available = length(joint_avail),
                paired_fidelity = length(joint_fid),
                availability_delta = avail_delta.delta,
                availability_delta_mcse = avail_delta.mcse,
                causal_coverage_delta = cov_delta.delta,
                causal_coverage_delta_mcse = cov_delta.mcse,
                all_attempts_inclusion_delta = all_incl.delta,
                all_attempts_inclusion_delta_mcse = all_incl.mcse,
                abs_causal_bias_delta = abs_bias_s.delta,
                abs_causal_bias_delta_mcse = abs_bias_s.mcse,
                width_delta = width_s.delta,
                width_delta_mcse = width_s.mcse,
                width_count = width_s.count,
                mean_error_sd_delta = mean_err_s.delta,
                mean_error_sd_delta_mcse = mean_err_s.mcse,
                mean_error_sd_count = mean_err_s.count,
                covariance_error_delta = cov_err_s.delta,
                covariance_error_delta_mcse = cov_err_s.mcse,
                covariance_error_count = cov_err_s.count,
                abs_policy_error_vs_ref_delta = pol_ref_abs_s.delta,
                abs_policy_error_vs_ref_delta_mcse = pol_ref_abs_s.mcse,
                policy_error_vs_ref_count = pol_ref_abs_s.count,
                reference_content_delta = content_s.delta,
                reference_content_delta_mcse = content_s.mcse,
                reference_content_count = content_s.count,
                endpoint_error_delta = ep_err_s.delta,
                endpoint_error_delta_mcse = ep_err_s.mcse,
                endpoint_error_count = ep_err_s.count,
                seconds_delta = sec_s.delta,
                seconds_delta_mcse = sec_s.mcse,
            ))
        end
    end
    return paired
end

function parse_options(arguments)
    options = Dict{Symbol, Any}(
        :replications => 1000,
        :seed => 20261121,
        :draws => 2000,
        :out => joinpath(@__DIR__, "phase5_heldout_confirmation"),
        :cells => String[],
    )
    i = 1
    while i <= length(arguments)
        arg = arguments[i]
        key = startswith(arg, "--") ? Symbol(replace(arg[3:end], '-' => '_')) : :unknown
        haskey(options, key) || throw(ArgumentError("unknown option $arg"))
        i < length(arguments) || throw(ArgumentError("missing value for $arg"))
        val = arguments[i + 1]
        options[key] = key in (:replications, :seed, :draws) ? parse(Int, val) :
            key == :cells ? String.(split(val, ','; keepempty = false)) : val
        i += 2
    end
    options[:replications] > 0 || throw(ArgumentError("replications must be positive"))
    options[:draws] >= 2 || throw(ArgumentError("draws must be at least 2"))
    options[:seed] >= 0 || throw(ArgumentError("seed must be nonnegative"))
    return (; options...)
end

function source_hashes()
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = [
        @__FILE__,
        joinpath(@__DIR__, "logistic_psis_reference.jl"),
        joinpath(package_root, "src", "bayesian_logistic.jl"),
        joinpath(package_root, "ext", "logistic_engine.jl"),
        joinpath(package_root, "docs", "BAYESIAN_VALIDATION_NEXT_PHASE.md"),
        joinpath(package_root, "Project.toml"),
    ]
    return Dict(relpath(p, package_root) => bytes2hex(sha256(read(p))) for p in paths if isfile(p))
end

fmt(x; digits = 4) = x isa Real && isfinite(x) ? string(round(Float64(x); digits)) : "NA"

function write_report_md(path, summaries, paired, options, elapsed_seconds)
    open(path, "w") do io
        println(io, "# Phase 5 Untouched Held-Out Bivariate Confirmation Report\n")
        println(io, "- Replications per cell: ", options.replications)
        println(io, "- Master seed: ", options.seed)
        println(io, "- Posterior contrast draws: ", options.draws)
        println(io, "- Threads: ", Threads.nthreads())
        println(io, "- Elapsed seconds: ", fmt(elapsed_seconds; digits = 2), "\n")

        println(io, "## Posterior Fidelity & Causal Recovery Summary\n")
        println(io, "| Cell | Prior | Method | Avail | RefAvail | Cov(avail) | CausalBias | Width | CovErr | MeanErrSD | RefContent | EndpointErr | max(k̂) |")
        println(io, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in summaries
            println(io, "| ", s.cell_id, " | ", s.prior_id, " | ", s.method,
                " | ", s.available, "/", s.attempted,
                " | ", s.reference_available_count, "/", s.attempted,
                " | ", fmt(s.causal_coverage_available),
                " | ", fmt(s.causal_bias_mean),
                " | ", fmt(s.width_mean),
                " | ", fmt(s.covariance_error_mean),
                " | ", fmt(s.mean_error_sd_mean),
                " | ", fmt(s.reference_content_mean),
                " | ", fmt(s.endpoint_error_mean),
                " | ", fmt(s.pareto_k_max; digits = 3), " |")
        end

        println(io, "\n## Paired Method Contrasts (`linear_response` minus comparator)\n")
        println(io, "| Cell | Prior | Contrast | Paired | ΔCov | ΔWidth | ΔCovErr | ΔRefContent | ΔEndpointErr |")
        println(io, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for p in paired
            println(io, "| ", p.cell_id, " | ", p.prior_id, " | ", p.contrast,
                " | ", p.paired_available, "/", p.attempts,
                " | ", fmt(p.causal_coverage_delta),
                " | ", fmt(p.width_delta),
                " | ", fmt(p.covariance_error_delta),
                " | ", fmt(p.reference_content_delta),
                " | ", fmt(p.endpoint_error_delta), " |")
        end
    end
end

"""Run the Phase 5 held-out confirmation grid using deterministic `Threads.@threads`."""
function run_confirmation(options;
        reference_fn = nothing,
        interval_content_fn = nothing,
        fitter = fit_bayesian_logistic,
        contrast_fn = bayesian_policy_contrast,
        quiet::Bool = false)
    LinearAlgebra.BLAS.set_num_threads(1)

    if reference_fn === nothing
        resolved = resolve_psis_functions()
        reference_fn = resolved.ref_fn
        interval_content_fn === nothing && (interval_content_fn = resolved.wic_fn)
    end

    cells = heldout_cells()
    if !isempty(options.cells)
        valid_ids = Set(c.id for c in cells)
        all(id -> id in valid_ids, options.cells) || throw(ArgumentError("unknown selected cell"))
        cells = filter(c -> c.id in options.cells, cells)
    end

    isdir(options.out) && !isempty(readdir(options.out)) &&
        throw(ArgumentError("output directory must be empty: $(options.out)"))
    mkpath(options.out)

    started = time()
    metadata = Dict{String, Any}(
        "schema" => "phase5-heldout-confirmation-v1",
        "replications" => options.replications,
        "seed" => options.seed,
        "draws" => options.draws,
        "cells" => [c.id for c in cells],
        "priors" => ["zero_s1p5", "zero_s3p0"],
        "methods" => [string(m) for m in METHODS],
        "julia" => string(VERSION),
        "threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "CausalTargeted" => string(Base.pkgversion(CausalTargeted)),
        "RxInfer" => string(Base.pkgversion(RxInfer)),
        "GraphPPL" => string(Base.pkgversion(GraphPPL)),
        "source_hashes" => source_hashes(),
        "status" => "running",
        "started_unix" => started,
    )
    meta_path = joinpath(options.out, "metadata.toml")
    write_meta() = open(io -> TOML.print(io, metadata; sorted = true), meta_path, "w")
    write_meta()

    jobs = [(cell = c, replication = r) for c in cells for r in 1:options.replications]
    job_blocks = Vector{Vector{NamedTuple}}(undef, length(jobs))
    completed = Threads.Atomic{Int}(0)
    total_jobs = length(jobs)

    Threads.@threads for idx in eachindex(jobs)
        job = jobs[idx]
        block = evaluate_heldout_dataset(job.cell, job.replication;
            seed = options.seed, n_draws = options.draws,
            reference_fn, interval_content_fn, fitter, contrast_fn)
        job_blocks[idx] = block
        done = Threads.atomic_add!(completed, 1) + 1
        if !quiet && (done == total_jobs || done % 25 == 0)
            println("phase5_heldout_confirmation progress: ", done, "/", total_jobs,
                " (", job.cell.id, " rep ", job.replication, ")")
            flush(stdout)
        end
    end

    rows = NamedTuple[]
    for block in job_blocks
        append!(rows, block)
    end

    raw_path = joinpath(options.out, "raw.tsv")
    CSV.write(raw_path, rows; delim = '\t', missingstring = "NA")

    summaries = summarise_confirmation_rows(rows)
    paired = paired_method_summaries(rows)
    CSV.write(joinpath(options.out, "summary.tsv"), summaries; delim = '\t', missingstring = "NA")
    CSV.write(joinpath(options.out, "paired_methods.tsv"), paired; delim = '\t', missingstring = "NA")

    elapsed = time() - started
    write_report_md(joinpath(options.out, "report.md"), summaries, paired, options, elapsed)

    metadata["status"] = "complete"
    metadata["elapsed_seconds"] = elapsed
    metadata["datasets"] = length(jobs)
    metadata["rows"] = length(rows)
    metadata["raw_sha256"] = bytes2hex(sha256(read(raw_path)))
    write_meta()

    return (; rows, summaries, paired, metadata, directory = options.out)
end

function main(arguments = ARGS)
    LinearAlgebra.BLAS.set_num_threads(1)
    options = parse_options(arguments)
    return run_confirmation(options)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Phase5HeldoutConfirmation.main(ARGS)
end
