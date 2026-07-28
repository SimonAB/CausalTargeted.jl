"""Cross-fitted LMTP TMLE / EIF / SDR for modified treatment policies.

Uses a **single shared fold partition** for outcome and density-ratio fits.
Policy means use stabilised, adaptively truncated density-ratio clever covariates
`H_d = π^d / π`. Optional CV truncation selects the hard cap by clever-covariate
variance. Density ratios: `:gaussian`, `:classification`, or `:hybrid`.

Estimators (`estimator`):
- `:tmle` — score-solving separate-policy TMLE (default; stable at small n)
- `:eif` / `:aipw` / `:sdr` — EIF one-step (sequential DR at T = 1)
- `:itmle` — iterative TMLE until the targeting score is approximately solved
"""

using DataFrames
using Statistics
using Distributions
using Random
using StableRNGs

"""
    apply_shift_policy(a, requested_shift, L, U; stratum_mask=nothing) -> Vector{Float64}

Match R `.make_shift_LU`: `clamp(a + δ, L, U)` (shift raw exposure, then clamp).
Natural policy (`δ ≈ 0`) is `clamp(a, L, U)`.
"""
function apply_shift_policy(
    a::AbstractVector{<:Real},
    requested_shift::Real,
    L::Real,
    U::Real;
    stratum_mask::Union{Nothing, BitVector} = nothing,
)
    a = Float64.(a)
    if !isfinite(requested_shift) || isapprox(requested_shift, 0; atol = 1e-12)
        return clamp.(a, L, U)
    end
    if stratum_mask === nothing
        return clamp.(a .+ requested_shift, L, U)
    end
    out = clamp.(a, L, U)
    out[stratum_mask] .= clamp.(a[stratum_mask] .+ requested_shift, L, U)
    return out
end

function _gaussian_density(x::Real, μ::Real, σ::Real)
    σ = max(σ, 1e-6)
    return pdf(Normal(μ, σ), x)
end

"""
    _mtp_clever_covariate_gaussian(a_obs, a_policy, mu_a, sigma) -> Vector

Interior MTP clever covariate using `a_policy` to recover the shift
(`a − δ ≈ 2a − a_policy`). Prefer `_mtp_clever_covariate_clamp_aware` when
bounds `(L,U)` and the requested shift are known.
"""
function _mtp_clever_covariate_gaussian(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma::Real,
)
    a_minus_δ = 2 .* Float64.(a_obs) .- Float64.(a_policy)
    num = _gaussian_density.(a_minus_δ, mu_a, sigma)
    den = max.(_gaussian_density.(a_obs, mu_a, sigma), 1e-12)
    return num ./ den
end

"""
    _mtp_clever_covariate_clamp_aware(a_obs, mu_a, sigma, δ, L, U) -> Vector

Clamp-aware additive MTP density ratio for `d(a) = clamp(a+δ, L, U)`.

On the absolutely continuous interior
`{a : L < a < U and L < a−δ < U}`, returns `g(a−δ|w)/g(a|w)`.
Elsewhere returns `0` (boundary atoms / non-overlapping support do not enter
the continuous Radon–Nikodym fluctuation; g-computation still uses `Q(d(A),W)`).
"""
function _mtp_clever_covariate_clamp_aware(
    a_obs::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma::Real,
    δ::Real,
    L::Real,
    U::Real;
    eps::Real = 1e-8,
)
    n = length(a_obs)
    H = zeros(n)
    if !isfinite(δ) || isapprox(δ, 0; atol = 1e-12)
        fill!(H, 1.0)
        return H
    end
    σ = max(Float64(sigma), 1e-6)
    Lo = Float64(L) + eps
    Up = Float64(U) - eps
    @inbounds for i in 1:n
        a = Float64(a_obs[i])
        a_back = a - Float64(δ)
        if a > Lo && a < Up && a_back > Lo && a_back < Up
            den = max(_gaussian_density(a, mu_a[i], σ), 1e-12)
            H[i] = _gaussian_density(a_back, mu_a[i], σ) / den
        end
    end
    return H
end

"""
    _mtp_clever_covariate_gaussian_het(a_obs, a_policy, mu_a, sigma_a) -> Vector

Heteroscedastic Gaussian density ratio with per-row residual SD `sigma_a`.
"""
function _mtp_clever_covariate_gaussian_het(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma_a::AbstractVector{<:Real},
)
    n = length(a_obs)
    H = ones(n)
    for i in 1:n
        σ = max(sigma_a[i], 1e-6)
        a_m = 2 * Float64(a_obs[i]) - Float64(a_policy[i])
        num = _gaussian_density(a_m, mu_a[i], σ)
        den = max(_gaussian_density(a_obs[i], mu_a[i], σ), 1e-12)
        H[i] = num / den
    end
    return H
end

function _covariate_matrix(df::DataFrame, covariates::Vector{Symbol})
    n = nrow(df)
    isempty(covariates) && return zeros(n, 0)
    return Matrix{Float64}(df[:, covariates])
end

"""
    _ratio_from_classifier(sl, a, W; trunc=10) -> Vector{Float64}
"""
function _ratio_from_classifier(sl, a::AbstractVector{<:Real}, W::Matrix{Float64}; trunc::Real = 10.0)
    X = hcat(ones(length(a)), Float64.(a), W)
    p = clamp.(predict_super_learner(sl, X), 1e-4, 1 - 1e-4)
    r = p ./ (1 .- p)
    return clamp.(r, 1 / trunc, trunc)
end

"""
    _fit_density_ratio_classifier(a_obs, a_policy, W; learners, rng) -> SuperLearner
"""
function _fit_density_ratio_classifier(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    W::Matrix{Float64};
    learners = (:logistic, :mean),
    rng = StableRNG(1),
)
    n = length(a_obs)
    A_tr = vcat(Float64.(a_obs), Float64.(a_policy))
    W_tr = vcat(W, W)
    S_tr = vcat(ones(n), zeros(n))
    X_tr = hcat(ones(2n), A_tr, W_tr)
    return fit_super_learner(
        X_tr, S_tr;
        learners = learners,
        family = :binomial,
        metalearner = :invmse,
        rng = rng,
    )
end

"""
    _shared_fold_lmtp_components(...) -> NamedTuple

When `L`, `U`, and `shift_policy` / `shift_reference` are provided, Gaussian
density ratios use the clamp-aware interior formula (zeros on non-overlapping
support). Otherwise falls back to the legacy `2a − a_policy` ratio.
"""
function _shared_fold_lmtp_components(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector{<:Real},
    a_reference::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0),
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_policy::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = nothing,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, trt])
    a1 = Float64.(a_policy)
    a0 = Float64.(a_reference)
    W = _covariate_matrix(df, covariates)
    clamp_aware = L !== nothing && U !== nothing && shift_policy !== nothing

    Q_obs = zeros(n)
    Q1 = zeros(n)
    Q0 = zeros(n)
    H1 = ones(n)
    H0 = ones(n)

    fold_sets = crossfit_indices(n, folds, rng)
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]

        Xtr = design_matrix(train, covariates; treatment = trt)
        sl_y = fit_super_learner(Xtr, y[train_idx]; learners = learners_outcome, rng = rng)

        Q_obs[test_idx] = predict_super_learner(sl_y, design_matrix(test, covariates; treatment = trt))
        Q1[test_idx] = predict_super_learner(
            sl_y, design_matrix(test, covariates; treatment = trt, treatment_values = a1[test_idx]),
        )
        Q0[test_idx] = predict_super_learner(
            sl_y, design_matrix(test, covariates; treatment = trt, treatment_values = a0[test_idx]),
        )

        if density_ratio == :gaussian || density_ratio == :hybrid
            sl_a = fit_super_learner(
                design_matrix(train, covariates), a[train_idx];
                learners = learners_trt, rng = rng,
            )
            mu_tr = predict_super_learner(sl_a, design_matrix(train, covariates))
            mu_te = predict_super_learner(sl_a, design_matrix(test, covariates))
            σ_fold = robust_residual_sd(a[train_idx] .- mu_tr)
            if clamp_aware
                Hg1 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], mu_te, σ_fold, shift_policy, L, U,
                )
                δ0 = shift_reference === nothing ? 0.0 : Float64(shift_reference)
                Hg0 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], mu_te, σ_fold, δ0, L, U,
                )
            else
                Hg1 = _mtp_clever_covariate_gaussian(a[test_idx], a1[test_idx], mu_te, σ_fold)
                Hg0 = _mtp_clever_covariate_gaussian(a[test_idx], a0[test_idx], mu_te, σ_fold)
            end
            if density_ratio == :hybrid
                W_tr = W[train_idx, :]
                clf1 = _fit_density_ratio_classifier(a[train_idx], a1[train_idx], W_tr; rng = rng)
                clf0 = _fit_density_ratio_classifier(a[train_idx], a0[train_idx], W_tr; rng = rng)
                Hc1 = _ratio_from_classifier(clf1, a[test_idx], W[test_idx, :]; trunc = trunc)
                Hc0 = _ratio_from_classifier(clf0, a[test_idx], W[test_idx, :]; trunc = trunc)
                H1[test_idx] = [
                    g > 1e-12 ? sqrt(g * c) : 0.0 for (g, c) in zip(Hg1, Hc1)
                ]
                H0[test_idx] = [
                    g > 1e-12 ? sqrt(g * c) : 0.0 for (g, c) in zip(Hg0, Hc0)
                ]
            else
                H1[test_idx] = Hg1
                H0[test_idx] = Hg0
            end
        else
            W_tr = W[train_idx, :]
            clf1 = _fit_density_ratio_classifier(a[train_idx], a1[train_idx], W_tr; rng = rng)
            clf0 = _fit_density_ratio_classifier(a[train_idx], a0[train_idx], W_tr; rng = rng)
            H1[test_idx] = _ratio_from_classifier(clf1, a[test_idx], W[test_idx, :]; trunc = trunc)
            H0[test_idx] = _ratio_from_classifier(clf0, a[test_idx], W[test_idx, :]; trunc = trunc)
        end
    end

    trunc_used = Float64(trunc)
    # Clamp-aware H has structural zeros — truncate without forcing mean(H)=1 on
    # the full sample (that would inflate interior weights by the zero mass).
    stabilize = !clamp_aware
    if cv_trunc
        trunc_used, H1 = cv_select_truncation(H1; candidates = trunc_candidates, stabilize = stabilize)
        _, H0 = cv_select_truncation(H0; candidates = trunc_candidates, stabilize = stabilize)
    else
        H1 = prepare_clever_covariate(H1; trunc = trunc, stabilize = stabilize)
        H0 = prepare_clever_covariate(H0; trunc = trunc, stabilize = stabilize)
    end

    return (
        y = y, Q_obs = Q_obs, Q1 = Q1, Q0 = Q0, H1 = H1, H0 = H0, n = n,
        trunc = trunc_used, density_ratio = density_ratio, clamp_aware = clamp_aware,
    )
end

"""
    _solve_tmle_scores!(Q1, Q0, resid, H1, H0; λ, max_epochs, tol) -> Int

Iterate score-solving TMLE updates until scores are small or `max_epochs` reached.
Depletes the working residual by the **sum** of both policy projections
(not the contrast), which keeps multi-epoch updates stable.
Returns the number of epochs used.
"""
function _solve_tmle_scores!(
    Q1::AbstractVector{<:Real},
    Q0::AbstractVector{<:Real},
    resid::AbstractVector{<:Real},
    H1::AbstractVector{<:Real},
    H0::AbstractVector{<:Real};
    λ::Real = 1.0,
    max_epochs::Int = 1,
    tol::Real = 1e-10,
)
    n_used = 0
    for ep in 1:max_epochs
        d1 = sum(abs2, H1)
        d0 = sum(abs2, H0)
        ε1 = d1 > 1e-12 ? clamp(sum(H1 .* resid) / d1, -2.0, 2.0) : 0.0
        ε0 = d0 > 1e-12 ? clamp(sum(H0 .* resid) / d0, -2.0, 2.0) : 0.0
        abs(ε1) + abs(ε0) < tol && break
        step = ep == 1 ? 1.0 : 0.5^(ep - 1)
        Q1 .+= (λ * step * ε1) .* H1
        Q0 .+= (λ * step * ε0) .* H0
        resid .-= (λ * step) .* (ε1 .* H1 .+ ε0 .* H0)
        n_used = ep
    end
    return n_used
end

"""
    lmtp_tmle_contrast(...) -> NamedTuple

`estimator`:
- `:tmle` (default) — score-solving separate-policy TMLE (stable at small n)
- `:eif` / `:aipw` / `:sdr` — full EIF one-step `P_n[Q(d)+H(Y−Q)]` (SDR at T = 1)
- `:itmle` — iterative TMLE until targeting scores ≈ 0 (then residual EIF polish)

`density_ratio`: `:gaussian` (default), `:classification`, or `:hybrid`.

`cv_trunc=true` selects the hard truncation among `trunc_candidates` by
clever-covariate variance. `targeting_weight` scales fluctuations toward
g-computation when clamp is high. `epochs > 1` iterates TMLE (capped; iTMLE uses more).
"""
function lmtp_tmle_contrast(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector{<:Real},
    a_reference::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0),
    targeting_weight::Real = 1.0,
    epochs::Int = 1,
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_policy::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
)
    c = _shared_fold_lmtp_components(
        df, trt, outcome, covariates, a_policy, a_reference, folds, rng;
        learners_outcome = learners_outcome,
        density_ratio = density_ratio,
        learners_trt = learners_trt,
        trunc = trunc,
        cv_trunc = cv_trunc,
        trunc_candidates = trunc_candidates,
        L = L,
        U = U,
        shift_policy = shift_policy,
        shift_reference = shift_reference,
    )
    λ = clamp(Float64(targeting_weight), 0.0, 1.0)
    n_epochs = clamp(max(1, epochs), 1, 5)

    Q1 = copy(c.Q1)
    Q0 = copy(c.Q0)
    resid = c.y .- c.Q_obs
    epochs_used = 0

    if estimator in (:eif, :aipw, :sdr)
        # Full EIF / sequential DR one-step (prefer large n / synthetic)
        ic1 = Q1 .+ λ .* c.H1 .* resid
        ic0 = Q0 .+ λ .* c.H0 .* resid
        ψ1 = mean(ic1)
        ψ0 = mean(ic0)
        est = ψ1 - ψ0
        ic = (ic1 .- ic0) .- est
    elseif estimator == :itmle
        # Iterative TMLE: up to 5 damped score steps (linear fluctuation).
        epochs_used = _solve_tmle_scores!(
            Q1, Q0, resid, c.H1, c.H0;
            λ = λ, max_epochs = clamp(max(n_epochs, 5), 1, 5), tol = 1e-8,
        )
        ψ1 = mean(Q1)
        ψ0 = mean(Q0)
        est = ψ1 - ψ0
        ic_raw = (Q1 .- Q0) .+ λ .* ((c.H1 .- c.H0) .* resid)
        ic = ic_raw .- mean(ic_raw)
    else
        # Score-solving TMLE (default)
        epochs_used = _solve_tmle_scores!(
            Q1, Q0, resid, c.H1, c.H0;
            λ = λ, max_epochs = n_epochs, tol = 1e-10,
        )
        ψ1 = mean(Q1)
        ψ0 = mean(Q0)
        est = ψ1 - ψ0
        ic_raw = (Q1 .- Q0) .+ λ .* ((c.H1 .- c.H0) .* resid)
        ic = ic_raw .- mean(ic_raw)
    end

    se = std(ic) / sqrt(c.n)
    lwr, upr = wald_ci(est, se)
    return (
        estimate = est, se = se, lower = lwr, upper = upr, ic = ic,
        estimator = estimator, targeting_weight = λ, epochs = max(epochs_used, n_epochs),
        psi1 = ψ1, psi0 = ψ0, trunc = c.trunc,
    )
end

"""
    lmtp_tmle_from_components(components; estimator, targeting_weight, epochs) -> NamedTuple

Targeting step only (for cached fold nuisances or diagnostics).
"""
function lmtp_tmle_from_components(
    components::NamedTuple;
    estimator::Symbol = :tmle,
    targeting_weight::Real = 1.0,
    epochs::Int = 1,
)
    λ = clamp(Float64(targeting_weight), 0.0, 1.0)
    n_epochs = clamp(max(1, epochs), 1, 5)
    Q1 = copy(components.Q1)
    Q0 = copy(components.Q0)
    resid = components.y .- components.Q_obs
    epochs_used = 0

    if estimator in (:eif, :aipw, :sdr)
        ic1 = Q1 .+ λ .* components.H1 .* resid
        ic0 = Q0 .+ λ .* components.H0 .* resid
        ψ1 = mean(ic1)
        ψ0 = mean(ic0)
        est = ψ1 - ψ0
        ic = (ic1 .- ic0) .- est
    elseif estimator == :itmle
        epochs_used = _solve_tmle_scores!(
            Q1, Q0, resid, components.H1, components.H0;
            λ = λ, max_epochs = clamp(max(n_epochs, 5), 1, 5), tol = 1e-8,
        )
        ψ1 = mean(Q1)
        ψ0 = mean(Q0)
        est = ψ1 - ψ0
        ic_raw = (Q1 .- Q0) .+ λ .* ((components.H1 .- components.H0) .* resid)
        ic = ic_raw .- mean(ic_raw)
    else
        epochs_used = _solve_tmle_scores!(
            Q1, Q0, resid, components.H1, components.H0;
            λ = λ, max_epochs = n_epochs, tol = 1e-10,
        )
        ψ1 = mean(Q1)
        ψ0 = mean(Q0)
        est = ψ1 - ψ0
        ic_raw = (Q1 .- Q0) .+ λ .* ((components.H1 .- components.H0) .* resid)
        ic = ic_raw .- mean(ic_raw)
    end

    se = std(ic) / sqrt(components.n)
    lwr, upr = wald_ci(est, se)
    diag = tmle_score_diagnostics(components; λ = λ)
    return (
        estimate = est, se = se, lower = lwr, upper = upr, ic = ic,
        estimator = estimator, targeting_weight = λ, epochs = max(epochs_used, n_epochs),
        psi1 = ψ1, psi0 = ψ0, trunc = components.trunc,
        targeting_diagnostics = diag,
    )
end

"""
    lmtp_tmle_policy_mean(...) -> (psi, ic)

Single-policy LMTP mean using shared-fold density ratios.
"""
function lmtp_tmle_policy_mean(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
)
    a_ref = Float64.(a_policy)
    c = _shared_fold_lmtp_components(
        df, trt, outcome, covariates, a_policy, a_ref, folds, rng;
        learners_outcome = learners_outcome,
        density_ratio = density_ratio,
        learners_trt = learners_trt,
        trunc = trunc,
        cv_trunc = cv_trunc,
    )
    resid = c.y .- c.Q_obs
    ic_raw = c.Q1 .+ c.H1 .* resid
    psi = mean(ic_raw)
    ic = ic_raw .- psi
    return psi, ic
end

export apply_shift_policy, lmtp_tmle_policy_mean, lmtp_tmle_contrast, lmtp_tmle_from_components
