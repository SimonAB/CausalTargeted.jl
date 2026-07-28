"""Interventional mediation MTP grid (NDE / NIE / TE) with SuperLearner and nested mediator MC.

Julia-native continuous-exposure mediation under modified treatment policies.
Inspired by the R `crumble` package (Liu et al.); Julia public APIs use
`run_mediation_grid` / `:mediation` (legacy aliases `run_crumble_grid` / `:crumble`
remain). Estimators are EIF / nested-MC analogues, not a port of torch Riesz nets.

# References

- Liu, Williams, Rudolph & Díaz (2024), arXiv:2408.14620 — unified mediation + MTP
- Liu et al. (2025), arXiv:2604.09902 — crumble tutorial companion
- Díaz & Hejazi (2020), *JRSS-B*; Hejazi et al. (2023), *Biostatistics*
- Vansteelandt & Daniel (2017), *Epidemiology* — interventional effects
"""

using DataFrames
using Statistics
using Random
using StableRNGs
using Distributions

function _fit_sl_outcome(df, cols, y; treatment = nothing, learners = DEFAULT_SL_LEARNERS, rng = StableRNG(1))
    X = design_matrix(df, cols; treatment = treatment)
    return fit_super_learner(X, y; learners = learners, rng = rng)
end

function _predict_sl(sl, df, cols; treatment = nothing, treatment_values = nothing)
    X = design_matrix(df, cols; treatment = treatment, treatment_values = treatment_values)
    return predict_super_learner(sl, X)
end

"""
    _mediator_residual_sd(train, med_model, m_col, covar, trt) -> Float64

Residual SD for Gaussian nested draws of a continuous mediator.
"""
function _mediator_residual_sd(train::DataFrame, med_model, m_col::Symbol, covar, trt)
    μ = _predict_sl(med_model, train, covar; treatment = trt)
    return robust_residual_sd(Float64.(train[!, m_col]) .- μ)
end

"""
    _nested_mediator_outcome_means(...) -> (y_a0_m0, y_a1_m0, y_a1_m1)

Average outcome predictions over nested Gaussian draws of mediators under
intervened treatment values (Monte Carlo approximation to ∫ Q(a,m,w) dG(m|a,w)).
"""
function _nested_mediator_outcome_means(
    ols_y,
    block::DataFrame,
    adjust::Vector{Symbol},
    mediators::Vector{Symbol},
    med_models,
    σ_m::Vector{Float64},
    covar::Vector{Symbol},
    trt::Symbol,
    a0::AbstractVector{<:Real},
    a1::AbstractVector{<:Real},
    n_mc::Int,
    rng::AbstractRNG,
)
    n_te = nrow(block)
    n_med = length(mediators)
    y_a0_m0 = zeros(n_te)
    y_a1_m0 = zeros(n_te)
    y_a1_m1 = zeros(n_te)

    μ0 = hcat([_predict_sl(mm, block, covar; treatment = trt, treatment_values = a0) for mm in med_models]...)
    μ1 = hcat([_predict_sl(mm, block, covar; treatment = trt, treatment_values = a1) for mm in med_models]...)

    if n_mc <= 1
        # Plug-in conditional means (legacy path)
        block_m0 = copy(block)
        block_m1 = copy(block)
        for j in 1:n_med
            block_m0[!, mediators[j]] = μ0[:, j]
            block_m1[!, mediators[j]] = μ1[:, j]
        end
        y_a0_m0 .= _predict_sl(ols_y, block_m0, adjust; treatment = trt, treatment_values = a0)
        y_a1_m0 .= _predict_sl(ols_y, block_m0, adjust; treatment = trt, treatment_values = a1)
        y_a1_m1 .= _predict_sl(ols_y, block_m1, adjust; treatment = trt, treatment_values = a1)
        return y_a0_m0, y_a1_m0, y_a1_m1
    end

    block_m0 = copy(block)
    block_m1 = copy(block)
    for _ in 1:n_mc
        for j in 1:n_med
            noise0 = σ_m[j] .* randn(rng, n_te)
            noise1 = σ_m[j] .* randn(rng, n_te)
            block_m0[!, mediators[j]] = μ0[:, j] .+ noise0
            block_m1[!, mediators[j]] = μ1[:, j] .+ noise1
        end
        y_a0_m0 .+= _predict_sl(ols_y, block_m0, adjust; treatment = trt, treatment_values = a0)
        y_a1_m0 .+= _predict_sl(ols_y, block_m0, adjust; treatment = trt, treatment_values = a1)
        y_a1_m1 .+= _predict_sl(ols_y, block_m1, adjust; treatment = trt, treatment_values = a1)
    end
    inv_mc = 1 / n_mc
    y_a0_m0 .*= inv_mc
    y_a1_m0 .*= inv_mc
    y_a1_m1 .*= inv_mc
    return y_a0_m0, y_a1_m0, y_a1_m1
end

"""
    _mediator_density_ratio(m_obs, μ0, μ1, σ; trunc) -> Vector

Gaussian conditional density ratio g(M|A=a1,W) / g(M|A=a0,W) at observed mediators.
For multiple mediators, multiply independent ratios.
"""
function _mediator_density_ratio(
    m_obs::AbstractMatrix{<:Real},
    μ0::AbstractMatrix{<:Real},
    μ1::AbstractMatrix{<:Real},
    σ::Vector{Float64};
    trunc::Real = 5.0,
)
    n, p = size(m_obs)
    r = ones(n)
    for j in 1:p
        σj = max(σ[j], 1e-6)
        num = [_gaussian_density(m_obs[i, j], μ1[i, j], σj) for i in 1:n]
        den = [max(_gaussian_density(m_obs[i, j], μ0[i, j], σj), 1e-12) for i in 1:n]
        r .*= clamp.(num ./ den, 1 / trunc, trunc)
    end
    return r
end

"""
    _eif_update!(psi, H, resid; weight=1) -> nothing

Efficient one-step update: `psi += weight * H * resid` (EIF fluctuation term).
"""
function _eif_update!(
    psi::AbstractVector{<:Real},
    H::AbstractVector{<:Real},
    resid::AbstractVector{<:Real};
    weight::Real = 1.0,
)
    w = Float64(weight)
    @inbounds for i in eachindex(psi)
        psi[i] += w * H[i] * resid[i]
    end
    return nothing
end

"""
    _iterated_projection!(psi, H, resid; epochs, trunc_ε) -> nothing

Legacy OLS projection onto H. Prefer `_eif_update!` for continuous outcomes.
Kept for binary-propensity paths that request multiple targeting epochs.
"""
function _iterated_projection!(
    psi::AbstractVector{<:Real},
    H::AbstractVector{<:Real},
    resid::AbstractVector{<:Real};
    epochs::Int = 1,
    trunc_ε::Real = 5.0,
)
    epochs = max(0, epochs)
    for _ in 1:epochs
        d = sum(abs2, H)
        d <= 1e-12 && break
        ε = clamp(sum(H .* resid) / d, -trunc_ε, trunc_ε)
        abs(ε) < 1e-8 && break
        psi .+= ε .* H
        resid .-= ε .* H
    end
    return nothing
end

"""
    _mediation_effects(...) -> (est, se)

Interventional mediation with shared-fold SuperLearner and nested EIF.

```
ψ(a_t, a_m) = E[∫ Q(a_t,m,W) g(m|a_m,W) dm]
NDE = ψ(a₁,a₀)−ψ(a₀,a₀),  NIE = ψ(a₁,a₁)−ψ(a₁,a₀),  TE = ψ(a₁,a₁)−ψ(a₀,a₀)
```

- **Binary A:** full nested EIF (outcome + mediator scores with propensity weights).
- **Continuous MTP:** nested-MC plugin Q̄ for NDE; plugin + mediator density-ratio
  outcome residual for NIE. The treatment-weighted mediator score is omitted for
  continuous MTP because `a_m = d(A)` makes `H^{a_m}(Q−Q̄)` cancel plugin NIE in
  finite samples.
"""
function _mediation_effects(
    df::DataFrame,
    outcome::Symbol,
    trt::Symbol,
    covar::Vector{Symbol},
    mediators::Vector{Symbol},
    a_nat::Vector{Float64},
    a_shift::Vector{Float64},
    folds::Int,
    epochs::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
    n_mc::Int = 32,
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift::Union{Nothing, Real} = nothing,
    nie_weight::Real = 1.0,
    fold_cache::Union{Nothing, MediationFoldCache} = nothing,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, trt])
    adjust = unique(vcat(covar, mediators))
    binary_a = all(x -> x == 0.0 || x == 1.0, a)
    psi_te = zeros(n)
    psi_nde = zeros(n)
    psi_nie = zeros(n)
    fold_sets = fold_cache === nothing ? crossfit_indices(n, folds, rng) : fold_cache.fold_sets
    n_polish = binary_a ? 0 : clamp(epochs - 1, 0, 2)

    for (fi, test_idx) in enumerate(fold_sets)
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        block = df[test_idx, :]
        y_tr = y[train_idx]
        a0 = a_nat[test_idx]
        a1 = a_shift[test_idx]
        A_te = a[test_idx]
        y_te = y[test_idx]

        if fold_cache === nothing
            ols_y = _fit_sl_outcome(train, adjust, y_tr; treatment = trt, learners = learners, rng = rng)
            med_models = [
                _fit_sl_outcome(train, covar, Float64.(train[!, m]); treatment = trt, learners = learners, rng = rng)
                for m in mediators
            ]
            σ_m = [
                _mediator_residual_sd(train, med_models[j], mediators[j], covar, trt)
                for j in eachindex(mediators)
            ]
        else
            ols_y = fold_cache.outcome_models[fi]
            med_models = fold_cache.mediator_models[fi]
            σ_m = fold_cache.sigma_m[fi]
        end

        Q̄00, Q̄10, Q̄11 = _nested_mediator_outcome_means(
            ols_y, block, adjust, mediators, med_models, σ_m, covar, trt,
            a0, a1, n_mc, rng,
        )
        Q_obs = _predict_sl(ols_y, block, adjust; treatment = trt)
        Q_a0_M = _predict_sl(ols_y, block, adjust; treatment = trt, treatment_values = a0)
        Q_a1_M = _predict_sl(ols_y, block, adjust; treatment = trt, treatment_values = a1)

        μ0 = hcat([_predict_sl(mm, block, covar; treatment = trt, treatment_values = a0) for mm in med_models]...)
        μ1 = hcat([_predict_sl(mm, block, covar; treatment = trt, treatment_values = a1) for mm in med_models]...)
        μ_obs = hcat([_predict_sl(mm, block, covar; treatment = trt) for mm in med_models]...)
        m_obs = hcat([Float64.(block[!, m]) for m in mediators]...)
        ρ0 = mediator_density_ratio_vs_obs(m_obs, μ0, μ_obs, σ_m; trunc = 5.0)
        ρ1 = mediator_density_ratio_vs_obs(m_obs, μ1, μ_obs, σ_m; trunc = 5.0)

        if binary_a
            Xw = design_matrix(train, covar)
            sl_e = fit_super_learner(
                Xw, a[train_idx];
                learners = (:logistic, :mean),
                family = :binomial,
                metalearner = :invmse,
                rng = rng,
            )
            e = clamp.(predict_super_learner(sl_e, design_matrix(block, covar)), 1e-3, 1 - 1e-3)
            H1 = truncate_weights(A_te ./ e; trunc = 10.0)
            H0 = truncate_weights((1 .- A_te) ./ (1 .- e); trunc = 10.0)
            ic10 = eif_psi_interventional(Q̄10, Q_a1_M, Q_obs, y_te, H1, H0, ρ0)
            ic00 = eif_psi_interventional(Q̄00, Q_a0_M, Q_obs, y_te, H0, H0, ρ0)
            ic11 = eif_psi_interventional(Q̄11, Q_a1_M, Q_obs, y_te, H1, H1, ρ1)
            parts = decompose_mediation_eif(ic10, ic00, ic11)
            nde, nie, te = parts.nde, parts.nie, parts.te
        else
            if fold_cache === nothing
                sl_a = fit_super_learner(
                    design_matrix(train, covar), a[train_idx];
                    learners = learners, rng = rng,
                )
            else
                sl_a = fold_cache.exposure_models[fi]
            end
            mu_tr = predict_super_learner(sl_a, design_matrix(train, covar))
            mu_te = predict_super_learner(sl_a, design_matrix(block, covar))
            σ_a = robust_residual_sd(a[train_idx] .- mu_tr)
            if L !== nothing && U !== nothing && shift !== nothing
                H1_raw = _mtp_clever_covariate_clamp_aware(A_te, mu_te, σ_a, shift, L, U)
                H0_raw = _mtp_clever_covariate_clamp_aware(A_te, mu_te, σ_a, 0.0, L, U)
            else
                H1_raw = _mtp_clever_covariate_gaussian(A_te, a1, mu_te, σ_a)
                H0_raw = _mtp_clever_covariate_gaussian(A_te, a0, mu_te, σ_a)
            end
            H1 = truncate_weights(H1_raw; trunc = 10.0)
            H0 = truncate_weights(H0_raw; trunc = 10.0)
            resid = y_te .- Q_obs
            Δρ = clamp.(ρ1 .- ρ0, -5.0, 5.0)
            λ = clamp(Float64(nie_weight), 0.0, 1.0)
            # Plugin Q̄ for NDE; NIE = plugin + weighted mediator-ratio residual
            nde = Q̄10 .- Q̄00
            nie = (Q̄11 .- Q̄10) .+ (λ .* H1 .* Δρ) .* resid
            if n_polish > 0
                _iterated_projection!(
                    nie, λ .* H1 .* Δρ, copy(resid);
                    epochs = n_polish, trunc_ε = 1.0,
                )
            end
            te = nde .+ nie
        end

        psi_nde[test_idx] = nde
        psi_nie[test_idx] = nie
        psi_te[test_idx] = te
    end
    est = (nde = mean(psi_nde), nie = mean(psi_nie), te = mean(psi_te))
    se = (
        nde = std(psi_nde .- est.nde) / sqrt(n),
        nie = std(psi_nie .- est.nie) / sqrt(n),
        te = std(psi_te .- est.te) / sqrt(n),
    )
    return est, se
end

"""
    run_mediation_grid(data, trt, outcome; covar, mediators, kwargs...) -> DataFrame
"""
function run_mediation_grid(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    covar::Vector{Symbol},
    mediators::Vector{Symbol},
    deltas = default_deltas(),
    lower_q = mtp_settings().lower_q,
    upper_q = mtp_settings().upper_q,
    folds = mtp_settings().folds,
    epochs::Int = 1,
    stratify_by = resolved_stratify_by(),
    shift_scale = mtp_settings().shift_scale,
    learners = DEFAULT_SL_LEARNERS,
    n_mc::Int = 32,
    rng::AbstractRNG = StableRNG(42),
    parallel::Bool = false,
    cache_nuisances::Bool = true,
    positivity::Bool = false,
)
    df = make_analysis_strata(data, stratify_by)
    pooled = stratify_by !== nothing
    covar = columns_present(df, unique(vcat(covar, pooled ? [stratify_by] : Symbol[])))
    covar = [c for c in covar if c != trt]
    mediators = columns_present(df, mediators)
    a = Float64.(df[!, trt])
    sd_a = std(a)
    L, U = exposure_bounds(a, lower_q, upper_q)
    a_nat = apply_shift_policy(a, 0.0, L, U)

    if sparse_exposure_diagnostic(a).sparse
        learners = (:evotree, :mean)
    end

    fold_cache = cache_nuisances ? build_mediation_fold_cache(
        df, outcome, trt, covar, mediators, folds, rng; learners = learners,
    ) : nothing

    rows = Dict{String, Any}[]
    strata = get_target_strata(df)
    n_jobs = count(d -> !isapprox(d, 0; atol = 1e-12), deltas) * length(strata)
    job_i = 0
    for stratum in strata
        stratum_mask = BitVector(string.(df.STRAT) .== stratum)
        scale_by = pooled ? mean(stratum_mask) : 1.0
        for d in deltas
            diag = support_diagnostics(
                df, trt, stratum, stratify_by, lower_q, upper_q, d, shift_scale;
                min_stratum_n = mtp_settings().min_stratum_n,
                max_stratum_clamp_prop = mtp_settings().max_stratum_clamp_prop,
                min_shift_retention = mtp_settings().min_shift_retention,
            )
            if isapprox(d, 0; atol = 1e-12)
                for lab in ("NDE", "NIE", "TE")
                    push!(rows, _mediation_row(d, lab, 0.0, 0.0, 0.0, 0.0, diag, lower_q, upper_q, sd_a; stratum = stratum))
                end
                continue
            end
            job_i += 1
            @info "mediation grid" trt outcome stratum delta = d progress = "$job_i/$n_jobs" n_mc
            req = diag.requested_shift
            if !isfinite(req)
                for lab in ("NDE", "NIE", "TE")
                    push!(rows, _mediation_row(d, lab, NaN, NaN, NaN, NaN, diag, lower_q, upper_q, sd_a; stratum = stratum))
                end
                continue
            end
            a_shift = apply_shift_policy(a, req, L, U; stratum_mask = pooled ? stratum_mask : nothing)
            add_diag = additive_clamp_diagnostics(
                pooled ? a[stratum_mask] : a, req, L, U,
            )
            nw = targeting_weight_from_clamp(add_diag.clamp)
            try
                est, se = _mediation_effects(
                    df, outcome, trt, covar, mediators, a_nat, a_shift, folds, epochs, rng;
                    learners = learners,
                    n_mc = n_mc,
                    L = L,
                    U = U,
                    shift = req,
                    nie_weight = nw,
                    fold_cache = fold_cache,
                )
                for (lab, e, s) in (("NDE", est.nde, se.nde), ("NIE", est.nie, se.nie), ("TE", est.te, se.te))
                    e_s = e / scale_by
                    s_s = s / scale_by
                    lwr, upr = wald_ci(e_s, s_s)
                    push!(rows, _mediation_row(
                        d, lab, e_s, s_s, lwr, upr, diag, lower_q, upper_q, sd_a;
                        severity = add_diag.severity, clamp_rate = add_diag.clamp, stratum = stratum,
                    ))
                end
            catch
                for lab in ("NDE", "NIE", "TE")
                    push!(rows, _mediation_row(d, lab, NaN, NaN, NaN, NaN, diag, lower_q, upper_q, sd_a; stratum = stratum))
                end
            end
        end
    end
    out = DataFrame(rows)
    if positivity
        rep = positivity_report(
            data, trt;
            deltas = deltas, stratify_by = stratify_by,
            lower_q = lower_q, upper_q = upper_q, shift_scale = shift_scale,
        )
        attach_positivity_summary!(out, rep)
    end
    return out
end

function _mediation_row(d, lab, est, se, lwr, upr, diag, lower_q, upper_q, sd_a; severity = 0.0, clamp_rate = nothing, stratum = "full_population")
    return Dict(
        "delta" => d,
        "estimand" => lab,
        "est" => est,
        "se" => se,
        "lwr" => lwr,
        "upr" => upr,
        "clamp" => clamp_rate === nothing ? coalesce(diag.stratum_clamp_prop, diag.global_clamp_prop, 0.0) : clamp_rate,
        "severity" => severity,
        "effective_shift" => diag.effective_shift_mean,
        "shift_retention" => diag.shift_retention,
        "lower_q" => lower_q,
        "upper_q" => upper_q,
        "sd_exposure" => sd_a,
        "support_status" => diag.support_status,
        "stratum" => string(stratum),
    )
end

export run_mediation_grid
const run_crumble_grid = run_mediation_grid  # legacy alias (R crumble naming)
const _crumble_mediation_effects = _mediation_effects  # legacy
export run_crumble_grid, _mediation_effects, _crumble_mediation_effects
