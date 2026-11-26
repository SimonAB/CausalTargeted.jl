"""Density-ratio truncation selection and simultaneous inference utilities."""

"""
    cv_select_truncation(H_raw; candidates, stabilize) -> (trunc, H)

Choose hard truncation among `candidates` by minimising the empirical variance of
the stabilised clever covariate (proxy for EIF variance under mild misspecification).
"""
function cv_select_truncation(
    H_raw::AbstractVector{<:Real};
    candidates = (5.0, 10.0, 20.0, 50.0),
    stabilize::Bool = true,
)
    best_t = Float64(first(candidates))
    best_v = Inf
    best_H = prepare_clever_covariate(H_raw; trunc = best_t, stabilize = stabilize)
    for t in candidates
        H = prepare_clever_covariate(H_raw; trunc = t, stabilize = stabilize)
        v = var(H)
        if v < best_v
            best_v = v
            best_t = Float64(t)
            best_H = H
        end
    end
    return best_t, best_H
end

"""
    multiplier_simultaneous_critical(ic_cols; n_boot, alpha, rng) -> Float64

Multiplier-bootstrap critical value for simultaneous Wald bands across columns
(δ-grid). Each column is an influence curve for one estimand; columns are
centred before resampling, as required for a multiplier approximation.

Returns `c` such that `est ± c * se` has approximate simultaneous coverage
`1 − alpha`, where `se = std(ic)/√n` for each column.
"""
function multiplier_simultaneous_critical(
    ic_cols::AbstractMatrix{<:Real};
    n_boot::Int = 999,
    alpha::Real = 0.05,
    rng = StableRNG(1),
)
    n, p = size(ic_cols)
    n >= 2 || throw(ArgumentError("ic_cols must contain at least two observations"))
    p >= 1 || throw(ArgumentError("ic_cols must contain at least one estimand"))
    n_boot >= 1 || throw(ArgumentError("n_boot must be ≥ 1"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between 0 and 1"))
    all(isfinite, ic_cols) || throw(ArgumentError("ic_cols must contain only finite values"))
    centred = Matrix{Float64}(ic_cols)
    scales = Vector{Float64}(undef, p)
    for j in 1:p
        column = view(centred, :, j)
        column .-= mean(column)
        scales[j] = std(column)
    end
    active = findall(>(0.0), scales)
    isempty(active) && return 0.0
    m_active = length(active)
    C_scaled = Matrix{Float64}(undef, n, m_active)
    inv_sqrt_n = 1.0 / sqrt(n)
    @inbounds for (k, j) in enumerate(active)
        inv_denom = inv_sqrt_n / scales[j]
        for i in 1:n
            C_scaled[i, k] = centred[i, j] * inv_denom
        end
    end
    Ct = transpose(C_scaled)
    ξ = Vector{Float64}(undef, n)
    scores = Vector{Float64}(undef, m_active)
    maxima = Vector{Float64}(undef, n_boot)
    @inbounds for b in 1:n_boot
        randn!(rng, ξ)
        mul!(scores, Ct, ξ)
        m = 0.0
        for k in 1:m_active
            z = abs(scores[k])
            z > m && (m = z)
        end
        maxima[b] = m
    end
    return quantile(maxima, 1 - alpha)
end

"""
    simultaneous_wald_bands(estimates, ses, critical) -> (lwr, upr)

Apply a common multiplier critical value to pointwise SEs.
"""
function simultaneous_wald_bands(
    estimates::AbstractVector{<:Real},
    ses::AbstractVector{<:Real},
    critical::Real,
)
    lwr = estimates .- critical .* ses
    upr = estimates .+ critical .* ses
    return lwr, upr
end

"""
    studentised_multiplier_ci(estimate, ic; alpha = 0.05, B = 999, weights = :rademacher, rng = StableRNG(2026))

Compute a studentised multiplier (wild) bootstrap confidence interval for an
asymptotically linear estimator with influence function vector `ic` (mean zero).

For each bootstrap draw ``b = 1, \\dots, B``:
1. Draw i.i.d. multipliers ``\\xi_i^{(b)}`` (`:rademacher` on ``\\{-1, +1\\}``, default, or
   `:mammen` two-point distribution matching ``\\mathbb{E}[\\xi]=0, \\mathbb{E}[\\xi^2]=1,
   \\mathbb{E}[\\xi^3]=1``).
2. Form ``z_i^{(b)} = \\xi_i^{(b)} \\hat{\\varphi}_i``, bootstrap shift
   ``\\bar{z}^{(b)} = \\frac{1}{n}\\sum_i z_i^{(b)}``, and bootstrap SE
   ``\\widehat{\\mathrm{SE}}^{(b)} = \\mathrm{sd}(z^{(b)}) / \\sqrt{n}``.
3. Studentise ``T^{(b)} = \\bar{z}^{(b)} / \\widehat{\\mathrm{SE}}^{(b)}``.
4. Invert quantiles ``[q_{\\alpha/2}, q_{1-\\alpha/2}]``:
   ``[\\hat{\\tau} - q_{1-\\alpha/2} \\widehat{\\mathrm{SE}},\\; \\hat{\\tau} - q_{\\alpha/2} \\widehat{\\mathrm{SE}}]``.

`:rademacher` is the default because ``\\xi_i^2 \\equiv 1`` yields strictly heavier-than-normal
``t``-like finite-sample quantiles, whereas `:mammen` satisfies ``\\xi_i^2 = \\xi_i + 1`` so
large positive draws inflate ``\\widehat{\\mathrm{SE}}^{(b)}`` and can compress the lower bound
in small samples.
"""
function studentised_multiplier_ci(
    estimate::Real,
    ic::AbstractVector{<:Real};
    alpha::Real = 0.05,
    B::Int = 999,
    weights::Symbol = :rademacher,
    rng::AbstractRNG = StableRNG(2026),
)
    est = Float64(estimate)
    phi = Float64.(ic)
    n = length(phi)
    n >= 2 || throw(ArgumentError("studentised_multiplier_ci requires n >= 2"))
    (0.0 < alpha < 1.0) || throw(ArgumentError("alpha must lie in (0, 1); got $alpha"))
    B >= 19 || throw(ArgumentError("bootstrap replicates B must be >= 19; got $B"))
    weights in (:mammen, :rademacher) || throw(ArgumentError(
        "bootstrap weights must be :mammen or :rademacher; got $weights",
    ))

    # Centre EIF defensively in case of tiny numerical drift
    phi_c = phi .- mean(phi)
    se = std(phi_c) / sqrt(n)
    if !isfinite(se) || se <= 1e-15
        return (
            lower = est, upper = est, se = se,
            q_lower = 0.0, q_upper = 0.0, weights = weights, B = B,
        )
    end

    t_stats = Vector{Float64}(undef, B)
    inv_n = inv(Float64(n))
    inv_nm1 = inv(Float64(n - 1))
    inv_sqrt_n = inv(sqrt(n))
    valid = 0

    if weights === :rademacher
        # For Rademacher multipliers ξ_i ∈ {-1, +1}, z_i^2 ≡ phi_c[i]^2 is invariant
        # across draws, so ∑(z_i - z̄)^2 = ∑phi_c^2 - n z̄^2 in a single pass.
        sum_phi2 = sum(abs2, phi_c)
        for _ in 1:B
            s1 = 0.0
            @inbounds for i in 1:n
                s1 += ifelse(rand(rng, Bool), phi_c[i], -phi_c[i])
            end
            zbar = s1 * inv_n
            ssq = max(sum_phi2 - n * zbar * zbar, 0.0)
            se_b = sqrt(ssq * inv_nm1) * inv_sqrt_n
            if isfinite(se_b) && se_b > 1e-15 * se
                valid += 1
                t_stats[valid] = zbar / se_b
            end
        end
    else
        # Mammen's two-point distribution: E[ξ]=0, E[ξ²]=1, E[ξ³]=1
        sq5 = sqrt(5.0)
        m_lo = -(sq5 - 1.0) / 2.0
        m_hi = (sq5 + 1.0) / 2.0
        p_lo = (sq5 + 1.0) / (2.0 * sq5)
        z = Vector{Float64}(undef, n)

        for _ in 1:B
            s1 = 0.0
            @inbounds for i in 1:n
                ξ = rand(rng) < p_lo ? m_lo : m_hi
                zi = ξ * phi_c[i]
                z[i] = zi
                s1 += zi
            end
            zbar = s1 * inv_n
            ssq = 0.0
            @inbounds for i in 1:n
                d = z[i] - zbar
                ssq += d * d
            end
            se_b = sqrt(ssq * inv_nm1) * inv_sqrt_n
            if isfinite(se_b) && se_b > 1e-15 * se
                valid += 1
                t_stats[valid] = zbar / se_b
            end
        end
    end

    if valid < 10
        lwr, upr = wald_ci(est, se; alpha = alpha)
        zcrit = quantile(Normal(0, 1), 1 - alpha / 2)
        return (
            lower = lwr, upper = upr, se = se,
            q_lower = -zcrit, q_upper = zcrit, weights = weights, B = valid,
        )
    end

    resize!(t_stats, valid)
    q_lo = quantile(t_stats, alpha / 2)
    q_hi = quantile(t_stats, 1.0 - alpha / 2)
    # Pivot inversion: P(q_lo <= (est - truth) / se <= q_hi) ≈ 1 - alpha
    lower = est - q_hi * se
    upper = est - q_lo * se
    return (
        lower = lower, upper = upper, se = se,
        q_lower = q_lo, q_upper = q_hi, weights = weights, B = valid,
    )
end
