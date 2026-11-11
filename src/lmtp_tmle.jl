"""Cross-fitted LMTP TMLE / EIF / SDR for modified treatment policies.

Uses a **single shared fold partition** for outcome and density-ratio fits.
Policy means use stabilised, adaptively truncated density-ratio clever covariates
`H_d = π^d / π`. Optional CV truncation selects the hard cap by clever-covariate
variance. Density ratios: `:gaussian`, `:classification`, or `:hybrid`.

Estimators (`estimator`):
- `:tmle` — score-solving separate-policy TMLE (default)
- `:eif` / `:aipw` / `:sdr` — EIF one-step (sequential DR at T = 1)
- `:itmle` — compatibility label for the same canonical weighted-intercept targeting step

# References

- Díaz et al. (2023), *JASA* — LMTP EIF, TMLE, and sequential double robustness
- van der Laan & Rubin (2006); van der Laan & Rose (2011) — TMLE
- Zheng & van der Laan (2011); Chernozhukov et al. (2018) — cross-fitting
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

"""
    _gaussian_density(x, μ, σ) -> Float64

Univariate Gaussian density at `x` with mean `μ`; `σ` is lower-bounded for numerical stability in
clever-covariate ratios.
"""
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

"""Uncapped Gaussian MTP ratio computed on the log scale."""
function _mtp_clever_covariate_gaussian_uncapped(
    a_obs::AbstractVector{<:Real},
    a_policy::AbstractVector{<:Real},
    mu_a::AbstractVector{<:Real},
    sigma::Real,
)
    length(a_obs) == length(a_policy) == length(mu_a) ||
        throw(ArgumentError("treatment and density prediction lengths must match"))
    σ = max(Float64(sigma), 1e-6)
    out = Vector{Float64}(undef, length(a_obs))
    lo, hi = log(nextfloat(0.0)), log(floatmax(Float64))
    for i in eachindex(out)
        a = Float64(a_obs[i])
        ap = Float64(a_policy[i])
        μ = Float64(mu_a[i])
        all(isfinite, (a, ap, μ)) || throw(DomainError(
            (a, ap, μ), "uncapped Gaussian MTP ratios require finite treatments and means",
        ))
        δi = (a - ap) / σ
        δi == 0 && (out[i] = 1.0; continue)
        za = (a - μ) / σ
        zap = (2a - ap - μ) / σ
        log_ratio = -δi * (za / 2 + zap / 2)
        (isfinite(log_ratio) && lo <= log_ratio <= hi) || throw(DomainError(
            log_ratio, "uncapped Gaussian MTP ratio is not representable in Float64",
        ))
        out[i] = exp(log_ratio)
    end
    return out
end

function _constant_additive_shift(a::AbstractVector{<:Real}, policy::AbstractVector{<:Real})
    length(a) == length(policy) || return false
    isempty(a) && return false
    shifts = Float64.(policy) .- Float64.(a)
    all(isfinite, shifts) || return false
    δ = first(shifts)
    return all(x -> isapprox(x, δ; atol = 1e-10, rtol = 1e-10), shifts)
end

function _lmtp_weight_inference_status(
    a::AbstractVector{<:Real},
    a1::AbstractVector{<:Real},
    a0::AbstractVector{<:Real};
    weight_truncation::Bool,
    clamp_config::Bool,
    smooth_policy::Bool = false,
)
    all(isequal.(a1, a0)) && return :trivial
    clamp_config && return :unsupported_clamped_policy
    smooth_policy && return weight_truncation ? :truncated_weights : :available
    (_constant_additive_shift(a, a1) && _constant_additive_shift(a, a0)) ||
        return :unsupported_policy
    return weight_truncation ? :truncated_weights : :available
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
    _covariate_matrix(schema, df)

Matrix of encoded covariates from a fitted `CovariateSchema` (same contract as `transform_covariates`).
Used for stacking `W` beside treatment in density-ratio designs.
"""
function _covariate_matrix(schema::CovariateSchema, df::AbstractDataFrame)
    return transform_covariates(schema, df)
end

"""
    _covariate_matrix(df, covariates::Vector{Symbol})

Fit `fit_covariate_schema(df, covariates)` and return the design matrix via `_covariate_matrix(schema, df)`.
"""
function _covariate_matrix(df::AbstractDataFrame, covariates::Vector{Symbol})
    schema = fit_covariate_schema(df, covariates)
    return _covariate_matrix(schema, df)
end

"""
    _ratio_from_classifier(sl, a, W; trunc=10) -> Vector{Float64}

Predict Pr(label=`observed` | A, W) with fitted binomial Super Learner `sl` on `[1, A, W]`. Training uses
label 1 for natural `A` and 0 for policy `A`; Bayes odds give ``g_{\\text{policy}} / g_{\\text{obs}} = (1-p)/p``,
then clamp to `[1/trunc, trunc]`.
"""
function _ratio_from_classifier(sl, a::AbstractVector{<:Real}, W::Matrix{Float64}; trunc::Real = 10.0)
    X = hcat(ones(length(a)), Float64.(a), W)
    # The classifier labels observed samples 1 and policy samples 0, so the
    # shared inverse-odds transform applies unchanged.
    return _discrete_policy_ratio(predict_super_learner(sl, X); trunc = trunc)
end

"""
    _fit_density_ratio_classifier(a_obs, a_policy, W; learners, rng)

Continuous-treatment 2n stack: rows are `([a_obs; a_policy], [W; W])` with binary label `S=1` on the first
half (observed law) and `S=0` on the second (policy law). Returns the fitted Super Learner for use with
`_ratio_from_classifier`.
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

When `L`, `U`, and `shift_amount` / `shift_reference` are provided, Gaussian
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
    family_outcome::Symbol = :gaussian,
    density_ratio::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_amount::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = nothing,
    smooth_policy::Union{Nothing, SmoothBoundedLogitShiftPolicy} = nothing,
    smooth_shift::Union{Nothing, Real} = nothing,
    policy_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    weight_truncation::Bool = true,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, trt])
    a1 = Float64.(a_policy)
    a0 = Float64.(a_reference)
    length(a1) == n == length(a0) || throw(ArgumentError("policy vectors must match the data length"))
    covariate_schema = fit_covariate_schema(df, covariates)
    W = _covariate_matrix(covariate_schema, df)
    clamp_aware = L !== nothing && U !== nothing && shift_amount !== nothing
    clamp_config = L !== nothing || U !== nothing || shift_amount !== nothing
    smooth_active = smooth_policy !== nothing
    density_a = smooth_active ? _bounded_logit_values(a, smooth_policy) : a
    smooth_active && (smooth_shift === nothing || !isfinite(smooth_shift)) &&
        throw(ArgumentError("smooth policy requires a finite smooth_shift"))
    if smooth_active
        expected = apply_policy_values(a, smooth_shift, smooth_policy; stratum_mask = policy_mask)
        all(isapprox.(a1, expected; atol = 1e-10, rtol = 1e-10)) || throw(ArgumentError(
            "a_policy does not match the supplied smooth bounded policy and shift",
        ))
    end
    if !weight_truncation
        density_ratio === :gaussian || throw(ArgumentError(
            "weight_truncation=false supports only Gaussian LMTP density ratios",
        ))
        cv_trunc && throw(ArgumentError(
            "weight_truncation=false cannot be combined with cv_trunc=true",
        ))
        clamp_config && !smooth_active && throw(ArgumentError(
            "weight_truncation=false supports only unclamped LMTP policies",
        ))
        ((!smooth_active && _constant_additive_shift(a, a1) && _constant_additive_shift(a, a0)) ||
         (smooth_active && all(isapprox.(a0, a)))) ||
            throw(ArgumentError(
                "weight_truncation=false requires constant additive shifts or a smooth bounded policy with the natural reference",
            ))
    end
    inference_status = _lmtp_weight_inference_status(
        a, a1, a0; weight_truncation = weight_truncation, clamp_config = clamp_config,
        smooth_policy = smooth_active,
    )

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

        Xtr = design_matrix(covariate_schema, train; treatment = trt)
        sl_y = fit_super_learner(
            Xtr, y[train_idx];
            learners = learners_outcome, family = family_outcome, rng = rng,
        )

        Q_obs[test_idx] = predict_super_learner(
            sl_y,
            design_matrix(covariate_schema, test; treatment = trt),
        )
        Q1[test_idx] = predict_super_learner(
            sl_y,
            design_matrix(
                covariate_schema,
                test;
                treatment = trt,
                treatment_values = a1[test_idx],
            ),
        )
        Q0[test_idx] = predict_super_learner(
            sl_y,
            design_matrix(
                covariate_schema,
                test;
                treatment = trt,
                treatment_values = a0[test_idx],
            ),
        )

        if density_ratio == :gaussian || density_ratio == :hybrid
            sl_a = fit_super_learner(
                design_matrix(covariate_schema, train), density_a[train_idx];
                learners = learners_trt, rng = rng,
            )
            mu_tr = predict_super_learner(sl_a, design_matrix(covariate_schema, train))
            mu_te = predict_super_learner(sl_a, design_matrix(covariate_schema, test))
            σ_fold = robust_residual_sd(density_a[train_idx] .- mu_tr)
            if smooth_active
                mask_test = policy_mask === nothing ? ones(length(test_idx)) : Float64.(policy_mask[test_idx])
                z1 = density_a[test_idx] .+ Float64(smooth_shift) .* mask_test
                Hg1 = weight_truncation ? _mtp_clever_covariate_gaussian(density_a[test_idx], z1, mu_te, σ_fold) :
                    _mtp_clever_covariate_gaussian_uncapped(density_a[test_idx], z1, mu_te, σ_fold)
                Hg0 = ones(length(test_idx))
            elseif clamp_aware
                Hg1 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], mu_te, σ_fold, shift_amount, L, U,
                )
                δ0 = shift_reference === nothing ? 0.0 : Float64(shift_reference)
                Hg0 = _mtp_clever_covariate_clamp_aware(
                    a[test_idx], mu_te, σ_fold, δ0, L, U,
                )
            else
                if weight_truncation
                    Hg1 = _mtp_clever_covariate_gaussian(a[test_idx], a1[test_idx], mu_te, σ_fold)
                    Hg0 = _mtp_clever_covariate_gaussian(a[test_idx], a0[test_idx], mu_te, σ_fold)
                else
                    Hg1 = _mtp_clever_covariate_gaussian_uncapped(a[test_idx], a1[test_idx], mu_te, σ_fold)
                    Hg0 = _mtp_clever_covariate_gaussian_uncapped(a[test_idx], a0[test_idx], mu_te, σ_fold)
                end
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

    trunc_used = weight_truncation ? Float64(trunc) : Inf
    # Clamp-aware H has structural zeros — truncate without forcing mean(H)=1 on
    # the full sample (that would inflate interior weights by the zero mass).
    stabilize = !clamp_aware
    if !weight_truncation
        # Opt-out removes LMTP-level quantile/cap/floor and stabilisation.
    elseif cv_trunc
        trunc_used, H1 = cv_select_truncation(H1; candidates = trunc_candidates, stabilize = stabilize)
        _, H0 = cv_select_truncation(H0; candidates = trunc_candidates, stabilize = stabilize)
    else
        H1 = prepare_clever_covariate(H1; trunc = trunc, stabilize = stabilize)
        H0 = prepare_clever_covariate(H0; trunc = trunc, stabilize = stabilize)
    end

    return (
        y = y, Q_obs = Q_obs, Q1 = Q1, Q0 = Q0, H1 = H1, H0 = H0, n = n,
        trunc = trunc_used, density_ratio = density_ratio, clamp_aware = clamp_aware,
        family_outcome = family_outcome, weight_truncation = weight_truncation,
        inference_status = inference_status,
    )
end

"""
    _clamp_count_predictions!(Q)

Ensure fitted means stay on the nonnegative count scale.
"""
function _clamp_count_predictions!(Q::AbstractVector{<:Real})
    Q .= max.(Q, 0.0)
    return Q
end

@inline _expit_stable(x::Float64) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))

function _weighted_intercept_tilt(
    y::Vector{Float64},
    q_obs::Vector{Float64},
    q_policy::Vector{Float64},
    H::Vector{Float64},
    family::Symbol,
    λ::Float64,
)
    all(isfinite, y) && all(isfinite, q_obs) && all(isfinite, q_policy) && all(isfinite, H) ||
        throw(DomainError((y, q_obs, q_policy, H), "TMLE tilting requires finite inputs"))
    all(H .>= 0) || throw(DomainError(H, "TMLE clever covariates must be nonnegative"))
    sum_h = sum(H)
    isfinite(sum_h) || throw(DomainError(sum_h, "TMLE clever-covariate sum must be finite"))
    sum_h > 0 || return copy(q_obs), copy(q_policy), 0.0, false

    if family in COUNT_OUTCOME_FAMILIES
        all(y .>= 0) || throw(DomainError(y, "count TMLE outcomes must be nonnegative"))
        all(q_obs .>= 0) && all(q_policy .>= 0) || throw(DomainError(
            (q_obs, q_policy), "count TMLE outcome predictions must be nonnegative",
        ))
        denominator = sum(H .* q_obs)
        numerator = sum(H .* y)
        isfinite(denominator) && isfinite(numerator) || throw(DomainError(
            (denominator, numerator), "count TMLE weighted sums must be finite",
        ))
        if numerator == 0
            if λ == 1.0
                return zeros(length(q_obs)), zeros(length(q_policy)), -Inf, true
            end
            updated_scale = exp(λ * log(floatmin(Float64)))
            return updated_scale .* q_obs, updated_scale .* q_policy, log(updated_scale), false
        end
        denominator > 0 || throw(DomainError(denominator, "count TMLE requires a positive weighted outcome offset"))
        scale = numerator / denominator
        (isfinite(scale) && scale > 0) || throw(DomainError(
            scale, "count TMLE weighted-intercept scale must be finite and positive",
        ))
        updated_scale = exp(λ * log(scale))
        return updated_scale .* q_obs, updated_scale .* q_policy, log(updated_scale), false
    elseif family === :binomial
        all(x -> 0 <= x <= 1, y) || throw(DomainError(y, "binomial TMLE outcomes must lie in [0, 1]"))
        all(x -> 0 <= x <= 1, q_obs) && all(x -> 0 <= x <= 1, q_policy) ||
            throw(DomainError((q_obs, q_policy), "binomial outcome predictions must lie in [0, 1]"))
        if all(iszero, y) || all(==(1.0), y)
            target = all(iszero, y) ? 0.0 : 1.0
            if λ == 1.0
                return fill(target, length(q_obs)), fill(target, length(q_policy)),
                    target == 0 ? -Inf : Inf, true
            end
            ε = λ * (target == 0 ? -40.0 : 40.0)
            qfloor = eps(Float64)
            p_obs = clamp.(q_obs, qfloor, 1 - qfloor)
            p_policy = clamp.(q_policy, qfloor, 1 - qfloor)
            offset_obs = log.(p_obs ./ (1 .- p_obs))
            offset_policy = log.(p_policy ./ (1 .- p_policy))
            return _expit_stable.(offset_obs .+ ε),
                _expit_stable.(offset_policy .+ ε), ε, false
        end
        qfloor = eps(Float64)
        p_obs = clamp.(q_obs, qfloor, 1 - qfloor)
        p_policy = clamp.(q_policy, qfloor, 1 - qfloor)
        offset_obs = log.(p_obs ./ (1 .- p_obs))
        offset_policy = log.(p_policy ./ (1 .- p_policy))
        score(ε) = sum(H .* (y .- _expit_stable.(offset_obs .+ ε)))
        left, right = -40.0, 40.0
        step = 40.0
        while score(left) < 0 && abs(left) < 1e6
            left -= step
            step *= 2
        end
        step = 40.0
        while score(right) > 0 && abs(right) < 1e6
            right += step
            step *= 2
        end
        score(left) >= 0 && score(right) <= 0 || throw(DomainError(
            (score(left), score(right)), "binomial weighted-intercept score has no finite root",
        ))
        for _ in 1:200
            mid = (left + right) / 2
            score(mid) > 0 ? (left = mid) : (right = mid)
        end
        root = (left + right) / 2
        abs(score(root)) <= 1e-8 * max(1.0, sum_h) || throw(DomainError(
            score(root), "binomial weighted-intercept score root did not converge",
        ))
        ε = λ * root
        return _expit_stable.(offset_obs .+ ε), _expit_stable.(offset_policy .+ ε), ε, false
    elseif family === :gaussian
        ε = λ * sum(H .* (y .- q_obs)) / sum_h
        return q_obs .+ ε, q_policy .+ ε, ε, false
    end
    throw(ArgumentError("weighted-intercept LMTP TMLE does not support family_outcome=$family"))
end

"""
    lmtp_tmle_contrast(...) -> NamedTuple

`estimator`:
- `:tmle` (default) — score-solving separate-policy TMLE
- `:eif` / `:aipw` / `:sdr` — full EIF one-step `P_n[Q(d)+H(Y−Q)]` (SDR at T = 1)
- `:itmle` — compatibility label for the same canonical weighted-intercept targeting step

`density_ratio`: `:gaussian` (default), `:classification`, or `:hybrid`.

`cv_trunc=true` selects the hard truncation among `trunc_candidates` by
clever-covariate variance. `targeting_weight` scales fluctuations toward
g-computation. `epochs` remains available for compatibility; each arm's
canonical score is solved in one step.
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
    family_outcome::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    trunc_candidates = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
    weight_truncation::Bool = true,
    targeting_weight::Real = 1.0,
    epochs::Int = 3,
    L::Union{Nothing, Real} = nothing,
    U::Union{Nothing, Real} = nothing,
    shift_amount::Union{Nothing, Real} = nothing,
    shift_reference::Union{Nothing, Real} = 0.0,
    smooth_policy::Union{Nothing, SmoothBoundedLogitShiftPolicy} = nothing,
    smooth_shift::Union{Nothing, Real} = nothing,
    policy_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
)
    c = _shared_fold_lmtp_components(
        df, trt, outcome, covariates, a_policy, a_reference, folds, rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        density_ratio = density_ratio,
        learners_trt = learners_trt,
        trunc = trunc,
        cv_trunc = cv_trunc,
        trunc_candidates = trunc_candidates,
        L = L,
        U = U,
        shift_amount = shift_amount,
        shift_reference = shift_reference,
        smooth_policy = smooth_policy,
        smooth_shift = smooth_shift,
        policy_mask = policy_mask,
        weight_truncation = weight_truncation,
    )
    return _lmtp_targeting_step(
        c; estimator = estimator, targeting_weight = targeting_weight,
        epochs = epochs, family_outcome = family_outcome,
    )
end

"""
    _lmtp_targeting_step(components; estimator, targeting_weight, epochs, family_outcome) -> NamedTuple

Apply the targeting step to cross-fitted nuisances and return the Wald interval.

`estimator` selects the one-step EIF path (`:eif`, `:aipw`, `:sdr`) or either
TMLE label (`:tmle`, `:itmle`). The TMLE paths fit separate weighted canonical-link
intercept tilts for each policy arm, applying each fitted intercept to both the
factual offset and its policy prediction. `epochs` is accepted for compatibility;
the canonical weighted-intercept score is solved in one step per arm.
"""
function _lmtp_targeting_step(
    components::NamedTuple;
    estimator::Symbol,
    targeting_weight::Real,
    epochs::Int,
    family_outcome::Union{Nothing, Symbol},
)
    λ = clamp(Float64(targeting_weight), 0.0, 1.0)
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "unsupported LMTP estimator=$estimator; expected :tmle, :itmle, :eif, :aipw, or :sdr",
    ))
    fam = family_outcome === nothing ?
        (hasproperty(components, :family_outcome) ? components.family_outcome : :gaussian) :
        family_outcome
    Q1 = copy(components.Q1)
    Q0 = copy(components.Q0)
    y = Float64.(components.y)
    Q_obs = Float64.(components.Q_obs)
    H1 = Float64.(components.H1)
    H0 = Float64.(components.H0)
    base_status = hasproperty(components, :inference_status) ?
        Symbol(components.inference_status) : :available

    # The component builder marks an algebraically identical policy pair as
    # trivial. Do not infer that status from coincidentally equal fitted arrays:
    # those can arise under degenerate fits while a base inference gate applies.
    same_arms = base_status === :trivial
    if same_arms
        shortcut_status = base_status === :available ? :trivial : base_status
        point_score = zeros(Float64, components.n)
        shortcut_ic = shortcut_status === :trivial ? copy(point_score) : fill(NaN, components.n)
        shortcut_se = shortcut_status === :trivial ? 0.0 : NaN
        diagnostics = (
            score_policy1 = 0.0, score_policy0 = 0.0,
            score_norm_policy1 = 0.0, score_norm_policy0 = 0.0,
            score_combined_norm = 0.0, mean_abs_residual = 0.0,
            mean_h1 = mean(H1), mean_h0 = mean(H0), targeting_ok = true,
        )
        return (
            estimate = 0.0, se = shortcut_se,
            lower = shortcut_status === :trivial ? 0.0 : NaN,
            upper = shortcut_status === :trivial ? 0.0 : NaN,
            ic = shortcut_ic, point_score = point_score, estimator = estimator,
            targeting_weight = λ, epochs = 0, psi1 = 0.0, psi0 = 0.0,
            trunc = components.trunc, inference_status = shortcut_status,
            targeting_diagnostics = diagnostics,
        )
    end

    if base_status === :available && (all(iszero, H1) || all(iszero, H0))
        base_status = :positivity_violation
    end
    λ < 1.0 && base_status === :available && (base_status = :attenuated_targeting)
    inference_status = base_status
    Q_obs1 = copy(Q_obs)
    Q_obs0 = copy(Q_obs)
    boundary_outcome = false
    epochs_used = 0

    if estimator in (:eif, :aipw, :sdr)
        fam in COUNT_OUTCOME_FAMILIES && begin
            _clamp_count_predictions!(Q1)
            _clamp_count_predictions!(Q0)
        end
        raw1 = Q1 .+ λ .* H1 .* (y .- Q_obs)
        raw0 = Q0 .+ λ .* H0 .* (y .- Q_obs)
        ψ1 = mean(raw1)
        ψ0 = mean(raw0)
        est = ψ1 - ψ0
        point_score = raw1 .- raw0
        Q_obs1 .= Q_obs
        Q_obs0 .= Q_obs
    else
        Q_obs1, Q1, _, boundary1 = _weighted_intercept_tilt(
            y, Q_obs, Q1, H1, fam, λ,
        )
        Q_obs0, Q0, _, boundary0 = _weighted_intercept_tilt(
            y, Q_obs, Q0, H0, fam, λ,
        )
        boundary_outcome = boundary1 || boundary0
        ψ1 = mean(Q1)
        ψ0 = mean(Q0)
        est = ψ1 - ψ0
        resid1 = y .- Q_obs1
        resid0 = y .- Q_obs0
        raw1 = Q1 .+ H1 .* resid1
        raw0 = Q0 .+ H0 .* resid0
        point_score = (raw1 .- mean(raw1) .+ ψ1) .-
            (raw0 .- mean(raw0) .+ ψ0)
        epochs_used = 1
    end

    inference_status === :available && boundary_outcome &&
        (inference_status = :boundary_outcome)

    all(isfinite, point_score) && isfinite(est) || throw(DomainError(
        (point_score, est), "LMTP targeting produced non-finite point scores",
    ))
    abs(mean(point_score) - est) <= 1e-9 * (1 + abs(est)) || throw(DomainError(
        (mean(point_score), est), "LMTP point-score mean does not equal the point estimate",
    ))
    if inference_status === :available
        ic = point_score .- est
        all(isfinite, ic) || throw(DomainError(ic, "LMTP EIF contains non-finite values"))
        se = std(ic) / sqrt(components.n)
        isfinite(se) || throw(DomainError(se, "LMTP EIF standard error is not finite"))
        lwr, upr = wald_ci(est, se)
    else
        ic = fill(NaN, components.n)
        se = lwr = upr = NaN
    end
    diagnostics = tmle_score_diagnostics(
        components; λ = 1.0,
        residual_policy1 = y .- Q_obs1,
        residual_policy0 = y .- Q_obs0,
    )
    return (
        estimate = est, se = se, lower = lwr, upper = upr, ic = ic,
        point_score = Float64.(point_score), estimator = estimator,
        targeting_weight = λ, epochs = epochs_used,
        psi1 = ψ1, psi0 = ψ0, trunc = components.trunc,
        inference_status = inference_status, targeting_diagnostics = diagnostics,
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
    family_outcome::Union{Nothing, Symbol} = nothing,
)
    return _lmtp_targeting_step(
        components; estimator = estimator, targeting_weight = targeting_weight,
        epochs = epochs, family_outcome = family_outcome,
    )
end

"""
    lmtp_tmle_policy_mean(...) -> (psi, ic)

Single-policy interventional mean under the supplied treatment policy. `:tmle`
and `:itmle` use the canonical weighted-intercept update; `:eif`, `:aipw`, and
`:sdr` use the initial one-step score. The centred influence curve is all `NaN`
when fixed weight truncation or another unsupported route gates inference. An
identity policy returns the natural sample mean and its centred outcome values.
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
    family_outcome::Symbol = :gaussian,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    cv_trunc::Bool = false,
    weight_truncation::Bool = true,
)
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "unsupported LMTP estimator=$estimator; expected :tmle, :itmle, :eif, :aipw, or :sdr",
    ))
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, trt])
    policy = Float64.(a_policy)
    length(policy) == length(a) || throw(DimensionMismatch(
        "policy length $(length(policy)) does not match sample size $(length(a))",
    ))
    if all(isequal.(policy, a))
        ψ = mean(y)
        return ψ, y .- ψ
    end
    c = _shared_fold_lmtp_components(
        df, trt, outcome, covariates, policy, a, folds, rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        density_ratio = density_ratio,
        learners_trt = learners_trt,
        trunc = trunc,
        cv_trunc = cv_trunc,
        weight_truncation = weight_truncation,
    )
    if estimator in (:eif, :aipw, :sdr)
        ic_raw = c.Q1 .+ c.H1 .* (c.y .- c.Q_obs)
        ψ = mean(ic_raw)
        all(isfinite, ic_raw) && isfinite(ψ) || throw(DomainError(
            (ic_raw, ψ), "single-policy LMTP mean produced non-finite point scores",
        ))
        ic = ic_raw .- ψ
        all(isfinite, ic) || throw(DomainError(
            ic, "single-policy LMTP mean EIF contains non-finite values",
        ))
    else
        qobs, qpolicy, _, boundary = _weighted_intercept_tilt(
            c.y, c.Q_obs, c.Q1, c.H1, family_outcome, 1.0,
        )
        ψ = mean(qpolicy)
        ic_raw = qpolicy .+ c.H1 .* (c.y .- qobs)
        all(isfinite, ic_raw) && isfinite(ψ) || throw(DomainError(
            (ic_raw, ψ), "single-policy LMTP mean produced non-finite point scores",
        ))
        ic = ic_raw .- mean(ic_raw)
        all(isfinite, ic) || throw(DomainError(
            ic, "single-policy LMTP mean EIF contains non-finite values",
        ))
        status = c.inference_status === :available && boundary ? :boundary_outcome : c.inference_status
        return ψ, status === :available ? ic : fill(NaN, length(ic))
    end
    return ψ, c.inference_status === :available ? ic : fill(NaN, length(ic))
end

export apply_shift_policy, lmtp_tmle_policy_mean, lmtp_tmle_contrast, lmtp_tmle_from_components
