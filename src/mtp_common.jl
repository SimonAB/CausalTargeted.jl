"""Shared MTP shift, clamp, and strata utilities."""

using DataFrames
using Statistics
using StatsBase
using Random
using StableRNGs

"""
    exposure_bounds(x, lower_q, upper_q) -> (L, U)
"""
function exposure_bounds(x::AbstractVector{<:Real}, lower_q::Real, upper_q::Real)
    xv = collect(skipmissing(Float64.(x)))
    L = quantile(xv, lower_q)
    U = quantile(xv, upper_q)
    return L, U
end

clamp_exposure(a, L, U) = clamp.(Float64.(a), L, U)

"""
    shifted_exposure(a, delta, sd_a, L, U) -> Vector{Float64}

SD-unit shift policy (crumble / z-scale).
"""
function shifted_exposure(a, delta::Real, sd_a::Real, L::Real, U::Real)
    a = Float64.(a)
    return clamp.(a .+ delta * sd_a, L, U)
end

"""
    clamp_diagnostics(a, delta, sd_a, L, U) -> NamedTuple

Returns clamp proportion, severity, effective shift mean (SD-scaled shift).
"""
function clamp_diagnostics(a, delta::Real, sd_a::Real, L::Real, U::Real)
    a = Float64.(a)
    unclamped = a .+ delta * sd_a
    clamped = clamp.(unclamped, L, U)
    hit = (unclamped .< L) .| (unclamped .> U)
    clamp_rate = mean(hit)
    severity = mean(abs.(clamped .- unclamped))
    effective_shift = mean(clamped .- a)
    retention = isapprox(delta, 0; atol = 1e-12) ? 1.0 : effective_shift / (delta * sd_a)
    return (clamp = clamp_rate, severity = severity, effective_shift = effective_shift, shift_retention = retention)
end

"""
    additive_clamp_diagnostics(a, requested_shift, L, U) -> NamedTuple

Clamp diagnostics for additive (z/raw) shift policies used by LMTP grids.
Matches R `.effective_shift`: shift raw `a`, then clamp (not clamp-then-shift).
"""
function additive_clamp_diagnostics(a, requested_shift::Real, L::Real, U::Real)
    a = Float64.(a)
    a_nat = clamp.(a, L, U)
    if !isfinite(requested_shift) || isapprox(requested_shift, 0; atol = 1e-12)
        return (clamp = 0.0, severity = 0.0, effective_shift = 0.0, shift_retention = 1.0)
    end
    unclamped = a .+ requested_shift
    clamped = clamp.(unclamped, L, U)
    hit = (unclamped .< L) .| (unclamped .> U)
    clamp_rate = mean(hit)
    severity = mean(abs.(clamped .- unclamped))
    effective_shift = mean(clamped .- a_nat)
    retention = effective_shift / requested_shift
    return (clamp = clamp_rate, severity = severity, effective_shift = effective_shift, shift_retention = retention)
end

"""
    targeting_weight_from_clamp(clamp_rate; soft=0.25, hard=0.75) -> Float64

Down-weight TMLE fluctuation when clamping is severe (hybrid g-comp / TMLE).
"""
function targeting_weight_from_clamp(clamp_rate::Real; soft::Real = 0.50, hard::Real = 0.95)
    c = Float64(clamp_rate)
    c <= soft && return 1.0
    c >= hard && return 0.0
    return (hard - c) / (hard - soft)
end

"""
    make_analysis_strata(df, stratify_by) -> DataFrame

Add `STRAT` column; `nothing` → single `full_population` stratum.
"""
function make_analysis_strata(df::DataFrame, stratify_by)
    out = copy(df)
    if stratify_by === nothing
        out.STRAT = fill("full_population", nrow(out))
        return out
    end
    col = string(stratify_by)
    col in names(out) || error("Stratify column $col not in data")
    out.STRAT = string.(out[!, col])
    return out
end

"""
    crossfit_indices(n, folds, rng) -> Vector{Vector{Int}}
"""
function crossfit_indices(n::Int, folds::Int, rng::AbstractRNG)
    folds = max(2, folds)
    perm = randperm(rng, n)
    fold_id = mod.(perm .- 1, folds) .+ 1
    return [findall(==(k), fold_id) for k in 1:folds]
end

"""
    wald_ci(est, se; alpha=0.05) -> (lwr, upr)
"""
function wald_ci(est::Real, se::Real; alpha = 0.05)
    z = quantile(Normal(0, 1), 1 - alpha / 2)
    return est - z * se, est + z * se
end

using Distributions: Normal

"""
    robust_residual_sd(resid) -> Float64

Scale estimate: max of sample SD and MAD-based σ (Gaussian consistency factor).
"""
function robust_residual_sd(resid::AbstractVector{<:Real})
    r = Float64.(resid)
    isempty(r) && return 1e-6
    σ_sd = std(r)
    med = median(r)
    σ_mad = 1.4826 * median(abs.(r .- med))
    return max(σ_sd, σ_mad, 1e-6)
end

"""
    truncate_weights(H; trunc=10, q_lo=0.01, q_hi=0.99) -> Vector{Float64}

Adaptive truncation: clamp to the intersection of a hard cap and empirical quantiles.
"""
function truncate_weights(
    H::AbstractVector{<:Real};
    trunc::Real = 10.0,
    q_lo::Real = 0.01,
    q_hi::Real = 0.99,
)
    h = Float64.(H)
    length(h) < 4 && return clamp.(h, 1 / trunc, trunc)
    lo = max(quantile(h, q_lo), 1 / trunc)
    hi = min(quantile(h, q_hi), trunc)
    lo > hi && return clamp.(h, 1 / trunc, trunc)
    return clamp.(h, lo, hi)
end

"""
    stabilize_weights(H) -> Vector{Float64}

Rescale so `mean(H) == 1` (stabilised IPTW / density-ratio weights).
"""
function stabilize_weights(H::AbstractVector{<:Real})
    h = Float64.(H)
    m = mean(h)
    m <= 1e-12 && return h
    return h ./ m
end

"""
    prepare_clever_covariate(H; trunc=10, stabilize=true) -> Vector{Float64}

Truncate, optionally stabilise, then re-truncate so weights stay in `[1/trunc, trunc]`.
"""
function prepare_clever_covariate(
    H::AbstractVector{<:Real};
    trunc::Real = 10.0,
    stabilize::Bool = true,
)
    h = truncate_weights(H; trunc = trunc)
    if stabilize
        h = stabilize_weights(h)
        h = truncate_weights(h; trunc = trunc)
    end
    return h
end

"""
    requested_shift(delta, a, stratum_mask, shift_scale) -> Float64

Map grid `delta` to realised shift (R `mtp_requested_shift`).
"""
function requested_shift(
    delta::Real,
    a::AbstractVector{<:Real},
    stratum_mask::BitVector,
    shift_scale::String,
)
    shift_scale in ("z", "raw") && return Float64(delta)
    if shift_scale == "global_sd"
        s = std(Float64.(a))
    elseif shift_scale == "stratum_sd"
        sa = Float64.(a)[stratum_mask]
        s = std(sa)
    else
        error("Unknown shift_scale: $shift_scale")
    end
    (!isfinite(s) || s <= 0) && return NaN
    return Float64(delta) * s
end

"""
    support_diagnostics(df, trt, stratum, stratify_by, lower_q, upper_q, delta, shift_scale; kwargs...) -> NamedTuple
"""
function support_diagnostics(
    df::DataFrame,
    trt::Symbol,
    stratum::AbstractString,
    stratify_by,
    lower_q::Real,
    upper_q::Real,
    delta::Real,
    shift_scale::String;
    min_stratum_n::Int = 10,
    max_stratum_clamp_prop::Real = 0.25,
    min_shift_retention::Real = 0.5,
    stratum_col::Symbol = :STRAT,
)
    a = Float64.(df[!, trt])
    idx = BitVector(.!ismissing.(df[!, stratum_col]) .& (string.(df[!, stratum_col]) .== stratum))
    stratum_a = a[idx]
    L, U = exposure_bounds(a, lower_q, upper_q)
    req = requested_shift(delta, a, idx, shift_scale)
    shifted = copy(a)
    unclamped = copy(a)
    if isfinite(req)
        unclamped[idx] .= a[idx] .+ req
        shifted[idx] .= clamp.(unclamped[idx], L, U)
    end
    a_nat = clamp.(a, L, U)
    achieved = shifted[idx] .- a_nat[idx]
    req_nz = isfinite(req) && !isapprox(req, 0; atol = 1e-12)
    eff_mean = mean(achieved)
    retention = req_nz ? eff_mean / req : NaN
    stratum_clamp = any(idx) && isfinite(req) ? mean((unclamped[idx] .< L) .| (unclamped[idx] .> U)) : NaN
    global_clamp = isfinite(req) ? mean((unclamped .< L) .| (unclamped .> U)) : NaN
    stratum_n = sum(idx)
    valid = stratum_n >= 2 && std(stratum_a) > 0
    support_status = "ok"
    if !valid ||
       (!isnan(stratum_clamp) && stratum_clamp >= 0.75) ||
       (req_nz && (!isfinite(retention) || abs(retention) < 0.10))
        support_status = "unsupported_shift"
    elseif stratum_n < min_stratum_n ||
           (!isnan(stratum_clamp) && stratum_clamp > max_stratum_clamp_prop) ||
           (req_nz && isfinite(retention) && abs(retention) < min_shift_retention)
        support_status = "weak_support"
    end
    return (
        requested_shift = req,
        stratum_clamp_prop = stratum_clamp,
        global_clamp_prop = global_clamp,
        effective_shift_mean = eff_mean,
        shift_retention = retention,
        support_status = support_status,
        pi_s = mean(idx),
        stratum_n = stratum_n,
        L = L,
        U = U,
    )
end

"""
    get_target_strata(df; stratum_col=:STRAT) -> Vector{String}
"""
function get_target_strata(df::DataFrame; stratum_col::Symbol = :STRAT)
    return sort(unique(string.(df[!, stratum_col])))
end

export exposure_bounds, clamp_exposure, shifted_exposure, clamp_diagnostics
export additive_clamp_diagnostics, targeting_weight_from_clamp
export make_analysis_strata, crossfit_indices, wald_ci
export robust_residual_sd, truncate_weights, stabilize_weights, prepare_clever_covariate
export requested_shift, support_diagnostics, get_target_strata
