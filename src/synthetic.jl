"""Synthetic DGPs with known MTP and mediation effects for unit tests."""

using DataFrames
using Random
using StableRNGs
using Distributions
using Statistics

"""
    simulate_linear_mtp(n; β_a=0.5, β_w=1.0, σ_y=0.5, σ_a=1.0, rng) -> (df, truth)

Linear DGP: `W ~ N(0,1)`, `A = W + ε_a`, `Y = β_a A + β_w W + ε_y`.
Under an additive shift of `δ` on clamped A (no clamp if unbounded),
`E[Y^{A+δ} - Y^{A}] = β_a * δ` when clamp is inactive.
"""
function simulate_linear_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_w::Real = 1.0,
    σ_y::Real = 0.5,
    σ_a::Real = 1.0,
    rng = StableRNG(1),
)
    W = randn(rng, n)
    A = W .+ σ_a .* randn(rng, n)
    Y = β_a .* A .+ β_w .* W .+ σ_y .* randn(rng, n)
    df = DataFrame(W = W, A = A, Y = Y)
    truth = (β_a = Float64(β_a), β_w = Float64(β_w), shift_effect = δ -> Float64(β_a) * δ)
    return df, truth
end

"""
    simulate_mediation(n; β_a=0.4, β_m=0.6, γ_a=0.5, rng) -> (df, truth)

Simple mediation: `W ~ N(0,1)`, `A ~ Bern(logit^{-1}(W))`,
`M = γ_a A + W + ε_m`, `Y = β_a A + β_m M + W + ε_y`.

Interventional effects for binary A (d1=1 vs d0=0):
- TE = β_a + β_m * γ_a
- NDE = β_a
- NIE = β_m * γ_a
"""
function simulate_mediation(
    n::Int;
    β_a::Real = 0.4,
    β_m::Real = 0.6,
    γ_a::Real = 0.5,
    σ_m::Real = 0.5,
    σ_y::Real = 0.5,
    rng = StableRNG(2),
)
    W = randn(rng, n)
    p = 1 ./ (1 .+ exp.(-W))
    A = Float64.(rand.(rng, Bernoulli.(p)))
    M = γ_a .* A .+ W .+ σ_m .* randn(rng, n)
    Y = β_a .* A .+ β_m .* M .+ W .+ σ_y .* randn(rng, n)
    df = DataFrame(W = W, A = A, M = M, Y = Y)
    truth = (
        nde = Float64(β_a),
        nie = Float64(β_m * γ_a),
        te = Float64(β_a + β_m * γ_a),
    )
    return df, truth
end

"""
    simulate_continuous_mtp_mediation(n; ...) -> (df, truth)

Continuous-A mediation DGP:

- `W ~ N(0,1)`
- `A = W + ε_a`
- `M = γ_a A + γ_w W + ε_m`
- `Y = β_a A + β_m M + β_w W + ε_y`

For additive MTP shift `δ` (inactive clamp), interventional effects scale linearly:

- `NDE(δ) = β_a · δ`
- `NIE(δ) = β_m · γ_a · δ`
- `TE(δ) = (β_a + β_m · γ_a) · δ`

`truth.effects(δ)` returns `(nde, nie, te)`.
"""
function simulate_continuous_mtp_mediation(
    n::Int;
    β_a::Real = 0.35,
    β_m::Real = 0.55,
    β_w::Real = 0.4,
    γ_a::Real = 0.7,
    γ_w::Real = 0.5,
    σ_a::Real = 1.0,
    σ_m::Real = 0.5,
    σ_y::Real = 0.5,
    rng = StableRNG(3),
)
    W = randn(rng, n)
    A = W .+ σ_a .* randn(rng, n)
    M = γ_a .* A .+ γ_w .* W .+ σ_m .* randn(rng, n)
    Y = β_a .* A .+ β_m .* M .+ β_w .* W .+ σ_y .* randn(rng, n)
    df = DataFrame(W = W, A = A, M = M, Y = Y)
    β_a_f, β_m_f, γ_a_f = Float64(β_a), Float64(β_m), Float64(γ_a)
    truth = (
        β_a = β_a_f,
        β_m = β_m_f,
        γ_a = γ_a_f,
        effects = δ -> begin
            d = Float64(δ)
            nde = β_a_f * d
            nie = β_m_f * γ_a_f * d
            (nde = nde, nie = nie, te = nde + nie)
        end,
    )
    return df, truth
end

"""
    truth_shift_effect(truth, δ) -> Float64

Evaluate closed-form additive MTP contrast for a linear DGP.
"""
truth_shift_effect(truth, δ::Real) = truth.shift_effect(δ)

export simulate_linear_mtp, simulate_mediation, simulate_continuous_mtp_mediation, truth_shift_effect
