"""Density-ratio truncation selection and simultaneous inference utilities."""

using Statistics
using Random
using StableRNGs
using Distributions

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
    maxima = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        ξ = randn(rng, n)
        m = 0.0
        for j in active
            z = abs(sum(ξ .* view(centred, :, j)) / (sqrt(n) * scales[j]))
            m = max(m, z)
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

export cv_select_truncation
export multiplier_simultaneous_critical, simultaneous_wald_bands
