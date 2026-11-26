# Two-part hierarchical Bayesian hurdle model (Bernoulli-logit presence +
# zero-truncated Poisson / NB2 positive intensity).

"""
    ClusterHurdlePrior(; presence::ClusterLogisticPrior, intensity::ClusterCountPrior)

Explicit two-part hierarchical hurdle prior combining a presence-part
`ClusterLogisticPrior` and a positive-intensity `ClusterCountPrior`.
"""
struct ClusterHurdlePrior
    presence::ClusterLogisticPrior
    intensity::ClusterCountPrior
end

ClusterHurdlePrior(; presence::ClusterLogisticPrior, intensity::ClusterCountPrior) =
    ClusterHurdlePrior(presence, intensity)

"""
Typed two-part hierarchical hurdle fit combining a Bernoulli-logit presence fit
(`BayesianClusterLogisticFit`) and a zero-truncated count intensity fit
(`BayesianClusterCountFit`).
"""
struct BayesianClusterHurdleFit
    presence_fit::BayesianClusterLogisticFit
    intensity_fit::BayesianClusterCountFit
    family::Symbol
    status::Symbol
    converged::Bool
    prior::ClusterHurdlePrior
    outcome::Symbol
    treatment::Symbol
    covariates::Vector{Symbol}
    cluster::Symbol
    metadata::NamedTuple
end

"""
    fit_bayesian_cluster_hurdle(data, outcome, treatment, covariates; cluster,
        prior::ClusterHurdlePrior, family=:nb2, basis=nothing,
        identification=nothing, handle_missing=:error, maxiter=1000,
        tol=1e-8, standardize=true)

Fit a two-part hierarchical Bayesian hurdle model for zero-inflated counts:
Part 1 fits binary presence `I(Y > 0)` via `fit_bayesian_cluster_logistic`, and
Part 2 fits positive intensity `Y | Y > 0` via `fit_bayesian_cluster_count`
with `zero_truncated = true` (`family = :poisson` or `:nb2`).
"""
function fit_bayesian_cluster_hurdle(data::AbstractDataFrame, outcome::Symbol,
        treatment::Symbol, covariates::AbstractVector{Symbol};
        cluster::Symbol, prior::ClusterHurdlePrior, family::Symbol = :nb2,
        basis = nothing, identification = nothing, handle_missing::Symbol = :error,
        maxiter::Integer = 1000, tol::Real = 1e-8, standardize::Bool = true)
    family in (:poisson, :nb2) || throw(ArgumentError("family must be :poisson or :nb2"))
    _validate_requested_columns(data, [outcome, cluster])
    observed = findall(!ismissing, data[!, outcome])
    length(observed) == nrow(data) || handle_missing == :mar || throw(ArgumentError(
        "missing outcomes require explicit handle_missing=:mar"))

    # Build binary presence column cleanly without mutating caller DataFrame
    df_pres = copy(data)
    pres_col = Vector{Union{Missing, Int}}(undef, nrow(data))
    @inbounds for i in 1:nrow(data)
        v = data[i, outcome]
        if ismissing(v)
            pres_col[i] = missing
        else
            (v isa Real && isfinite(v) && isinteger(v) && v >= 0) ||
                throw(ArgumentError("hurdle outcome must be non-negative integer counts"))
            pres_col[i] = v > 0 ? 1 : 0
        end
    end
    df_pres[!, outcome] = pres_col

    pos_rows = [i for i in observed if data[i, outcome] > 0]
    !isempty(pos_rows) || throw(ArgumentError("hurdle fitting requires at least one positive outcome"))
    df_pos = data[pos_rows, :]

    pres_fit = fit_bayesian_cluster_logistic(df_pres, outcome, treatment, covariates;
        cluster, prior = prior.presence, basis, identification, handle_missing,
        maxiter, tol, standardize)
    int_fit = fit_bayesian_cluster_count(df_pos, outcome, treatment, covariates;
        cluster, prior = prior.intensity, family, zero_truncated = true,
        basis, identification, handle_missing = :error, maxiter, tol, standardize)

    converged = pres_fit.converged && int_fit.converged
    status = converged ? :converged : :maxiter_reached
    metadata = (; identification_status = pres_fit.metadata.identification_status,
        outcome_model_status = :user_assumed,
        likelihood = Symbol("hierarchical_hurdle_$(family)"),
        family, n_observed = length(observed), n_positive = length(pos_rows),
        zero_fraction = 1.0 - length(pos_rows) / length(observed),
        part_statuses = (; presence = pres_fit.status, intensity = int_fit.status),
        independent_part_random_effects = true)
    return BayesianClusterHurdleFit(pres_fit, int_fit, family, status, converged,
        prior, outcome, treatment, collect(covariates), cluster, metadata)
end

function _slice_hurdle_part(μ, sd, lo, hi, mc, cov_full, rng_idx)
    return (; mean = μ[rng_idx], sd = sd[rng_idx], lower = lo[rng_idx],
        upper = hi[rng_idx], mcse = mc[rng_idx],
        covariance = Matrix(Symmetric(cov_full[rng_idx, rng_idx])))
end

"""
    bayesian_policy_contrast(fit::BayesianClusterHurdleFit, target, policy;
        deltas, n_draws=2000, rng=Random.default_rng(), level=0.95,
        mc_level=0.95, weighting=:individual, cluster=fit.cluster,
        correction=:none, marginalize_random_effects=true)

Joint marginal `:policy_vs_observed` contrasts from a fitted two-part
hierarchical hurdle model, returning `overall` (combined expected count
contrast), `presence` (binary prevalence contrast), and `intensity`
(positive-part intensity contrast) evaluated on shared posterior draws.
"""
function bayesian_policy_contrast(fit::BayesianClusterHurdleFit, target::AbstractDataFrame, policy;
        deltas, n_draws::Integer = 2000, rng::AbstractRNG = Random.default_rng(),
        level::Real = 0.95, mc_level::Real = 0.95, weighting::Symbol = :individual,
        cluster::Union{Nothing, Symbol} = fit.cluster, correction::Symbol = :none,
        marginalize_random_effects::Bool = true)
    pf, inf = fit.presence_fit, fit.intensity_fit
    inp = _prepare_bayesian_contrast_inputs(pf, target, policy;
        deltas, n_draws, level, mc_level, weighting, cluster, correction, check_mc_level = false)
    int_designs = _build_bayesian_policy_designs(inf, inp.frame, inp.a, inp.discrete_values,
        inp.deltas, policy)
    m, n_target, w = inp.m, inp.n_target, inp.w
    p1 = inp.p
    X1_obs, X1_pols = inp.X_obs, inp.X_policies
    X2_obs, X2_pols = int_designs.X_obs, int_designs.X_policies

    target_kind = marginalize_random_effects ? :fixed_cohort_marginal : :fixed_cohort_conditional_b0
    if fit.status !== :converged
        nan_vec = fill(NaN, m)
        nan_mat = fill(NaN, m, m)
        metadata = (; fit.metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level = Float64(level), n_draws = Int(n_draws), correction,
            uncertainty = :unavailable, intervals = :unavailable, intervals_available = false)
        part_na = (; mean = nan_vec, sd = nan_vec, lower = nan_vec, upper = nan_vec,
            mcse = nan_vec, covariance = nan_mat)
        return (; deltas = inp.deltas, mean = nan_vec, sd = nan_vec, lower = nan_vec, upper = nan_vec,
            mcse = nan_vec, covariance = nan_mat,
            overall = part_na, presence = part_na, intensity = part_na, metadata)
    end

    S = Int(n_draws)
    use_t = correction === :psis
    if use_t
        d1 = _draw_student_t_logistic_proposal(pf.block_state, pf.cluster_shape, pf.cluster_scale, S, rng)
    else
        Z1 = randn(rng, p1, S)
        ldiv!(pf.block_state.beta_precision_chol.U, Z1)
        s2_1 = [rand(rng, InverseGamma(pf.cluster_shape, pf.cluster_scale)) for _ in 1:S]
        d1 = (; beta_draws = Z1 .+ pf.posterior_mean, σ2_b_draws = s2_1,
            log_σ2_b_draws = log.(s2_1), log_q = Float64[])
    end
    d2 = _sample_count_global_draws(inf, S, rng, use_t)

    η1_obs = X1_obs * d1.beta_draws
    η2_obs = X2_obs * d2.beta_draws
    η1_pols = [X1_pols[j] * d1.beta_draws for j in 1:m]
    η2_pols = [X2_pols[j] * d2.beta_draws for j in 1:m]

    # Stack [overall (1:m); presence (m+1:2m); intensity (2m+1:3m)]
    all_draws = Matrix{Float64}(undef, 3 * m, S)
    @inbounds for s in 1:S
        σ1 = marginalize_random_effects ? sqrt(d1.σ2_b_draws[s]) : 0.0
        σ2 = marginalize_random_effects ? sqrt(d2.σ2_b_draws[s]) : 0.0
        θs = inf.family === :nb2 ? exp(d2.log_disp_draws[s]) : 1.0
        obs_p = 0.0; obs_int = 0.0; obs_ov = 0.0
        for i in 1:n_target
            pi_ = _marginal_logistic_prob(η1_obs[i, s], σ1)
            mi_ = _marginal_count_mean(η2_obs[i, s], σ2, inf.family, θs, true)
            wi = w[i]
            obs_p += wi * pi_
            obs_int += wi * mi_
            obs_ov += wi * (pi_ * mi_)
        end
        for j in 1:m
            p_mat = η1_pols[j]; i_mat = η2_pols[j]
            pol_p = 0.0; pol_int = 0.0; pol_ov = 0.0
            for i in 1:n_target
                pi_ = _marginal_logistic_prob(p_mat[i, s], σ1)
                mi_ = _marginal_count_mean(i_mat[i, s], σ2, inf.family, θs, true)
                wi = w[i]
                pol_p += wi * pi_
                pol_int += wi * mi_
                pol_ov += wi * (pi_ * mi_)
            end
            all_draws[j, s] = pol_ov - obs_ov
            all_draws[m + j, s] = pol_p - obs_p
            all_draws[2 * m + j, s] = pol_int - obs_int
        end
    end

    if correction === :psis
        log_ratios = Vector{Float64}(undef, S)
        @inbounds for s in 1:S
            lp1 = _marginal_cluster_logistic_log_posterior(pf.layout_raw, pf.prior,
                view(d1.beta_draws, :, s), d1.log_σ2_b_draws[s])
            lp2 = _marginal_cluster_count_log_posterior(inf.layout_raw, inf.prior,
                inf.family, true, view(d2.beta_draws, :, s),
                d2.log_σ2_b_draws[s], d2.log_disp_draws[s])
            log_ratios[s] = (lp1 - d1.log_q[s]) + (lp2 - d2.log_q[s])
        end
        pr = _psis_weighted_summary(all_draws, log_ratios; level = Float64(level))
        psis_diagnostics = (; passed = pr.passed, failures = pr.failures,
            khat = pr.khat, ess = pr.ess, max_mcse_to_sd = pr.max_mcse_to_sd)
        ov = _slice_hurdle_part(pr.mean, pr.sd, pr.lower, pr.upper, pr.mcse, pr.covariance, 1:m)
        ps = _slice_hurdle_part(pr.mean, pr.sd, pr.lower, pr.upper, pr.mcse, pr.covariance, (m + 1):(2 * m))
        it = _slice_hurdle_part(pr.mean, pr.sd, pr.lower, pr.upper, pr.mcse, pr.covariance, (2 * m + 1):(3 * m))
        metadata = (; fit.metadata..., functional = :policy_vs_observed,
            factual_reference = :observed, target = target_kind, weighting, n_target,
            level = Float64(level), n_draws = S, correction,
            uncertainty = pr.passed ? :psis_refined_moments : :unavailable,
            intervals = pr.passed ? :pointwise : :unavailable,
            intervals_available = pr.passed, psis_diagnostics)
        return (; deltas = inp.deltas, mean = ov.mean, sd = ov.sd, lower = ov.lower, upper = ov.upper,
            mcse = ov.mcse, covariance = ov.covariance,
            overall = ov, presence = ps, intensity = it, metadata)
    end

    mc = _summarise_mc_draws(all_draws, Float64(level))
    ov = _slice_hurdle_part(mc.mean, mc.sd, mc.lower, mc.upper, mc.mcse, mc.covariance, 1:m)
    ps = _slice_hurdle_part(mc.mean, mc.sd, mc.lower, mc.upper, mc.mcse, mc.covariance, (m + 1):(2 * m))
    it = _slice_hurdle_part(mc.mean, mc.sd, mc.lower, mc.upper, mc.mcse, mc.covariance, (2 * m + 1):(3 * m))
    metadata = (; fit.metadata..., functional = :policy_vs_observed,
        factual_reference = :observed, target = target_kind, weighting, n_target,
        level = Float64(level), n_draws = S, correction,
        uncertainty = :structured_laplace_vi, intervals = :pointwise, intervals_available = true)
    return (; deltas = inp.deltas, mean = ov.mean, sd = ov.sd, lower = ov.lower, upper = ov.upper,
        mcse = ov.mcse, covariance = ov.covariance,
        overall = ov, presence = ps, intensity = it, metadata)
end
