"""Synthetic DGPs with known MTP / mediation effects for recovery benchmarks.

Closed-form truths are used when the structural equations are linear and clamp
is inactive (or when effects scale with *effective* mean shift under clamp).
Stress DGPs expose weak positivity, intermediate confounding, and nuisance
misspecification; interventional contrasts are recovered by shared-noise
potential-outcome oracles.
"""

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
    truth = (
        name = "linear_mtp",
        β_a = Float64(β_a),
        β_w = Float64(β_w),
        shift_effect = δ -> Float64(β_a) * δ,
        effects = δ -> begin
            te = Float64(β_a) * Float64(δ)
            (nde = te, nie = 0.0, te = te)
        end,
    )
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
        name = "binary_mediation",
        nde = Float64(β_a),
        nie = Float64(β_m * γ_a),
        te = Float64(β_a + β_m * γ_a),
        effects = δ -> (  # δ unused; binary A=0/1 contrast
            nde = Float64(β_a),
            nie = Float64(β_m * γ_a),
            te = Float64(β_a + β_m * γ_a),
        ),
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

For additive MTP shift `δ` in **SD units of A** (inactive clamp), interventional
effects scale with the *mean achieved shift* `eff = E[d₁(A) − d₀(A)]`:

- `NDE = β_a · eff`
- `NIE = β_m · γ_a · eff`
- `TE = (β_a + β_m · γ_a) · eff`

`truth.effects(eff)` takes the *effective* mean shift (not the nominal δ).
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
        name = "continuous_mtp_mediation",
        β_a = β_a_f,
        β_m = β_m_f,
        γ_a = γ_a_f,
        effects = δ_eff -> begin
            d = Float64(δ_eff)
            nde = β_a_f * d
            nie = β_m_f * γ_a_f * d
            (nde = nde, nie = nie, te = nde + nie)
        end,
    )
    return df, truth
end

"""
    simulate_weak_positivity_mtp(n; ...) -> (df, truth)

Strong A–W dependence so additive SD-shifts hit clamps heavily.
Truth uses `effects(eff)` with effective mean shift.
"""
function simulate_weak_positivity_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_w::Real = 1.0,
    σ_a::Real = 0.15,
    σ_y::Real = 0.5,
    rng = StableRNG(11),
)
    W = randn(rng, n)
    A = 2.0 .* W .+ σ_a .* randn(rng, n)
    Y = β_a .* A .+ β_w .* W .+ σ_y .* randn(rng, n)
    df = DataFrame(W = W, A = A, Y = Y)
    β = Float64(β_a)
    truth = (
        name = "weak_positivity_mtp",
        β_a = β,
        shift_effect = δ -> β * δ,
        effects = δ_eff -> (nde = β * δ_eff, nie = 0.0, te = β * δ_eff),
    )
    return df, truth
end

"""
    simulate_intermediate_confounding_mediation(n; ...) -> (df, truth)

Mediation with intermediate confounder `L` on the A→M→Y pathway:

- `L = α_a A + α_w W + ε_l`
- `M = γ_a A + γ_l L + γ_w W + ε_m`
- `Y = β_a A + β_m M + β_l L + β_w W + ε_y`

Shared-noise oracle in `truth.oracle(δ_sd; lower_q, upper_q)` returns
`(nde, nie, te)` for an SD-unit shift policy (matches Julia/R mediation grids).
Closed-form `effects(eff)` ignores intermediate confounding (baseline-only
linear path through A) and is *not* the interventional truth when L is present.
"""
function simulate_intermediate_confounding_mediation(
    n::Int;
    β_a::Real = 0.3,
    β_m::Real = 0.5,
    β_l::Real = 0.4,
    β_w::Real = 0.3,
    γ_a::Real = 0.4,
    γ_l::Real = 0.6,
    γ_w::Real = 0.3,
    α_a::Real = 0.5,
    α_w::Real = 0.4,
    σ_a::Real = 1.0,
    σ_l::Real = 0.4,
    σ_m::Real = 0.4,
    σ_y::Real = 0.4,
    rng = StableRNG(12),
)
    W = randn(rng, n)
    Ua = randn(rng, n)
    Ul = randn(rng, n)
    Um = randn(rng, n)
    Uy = randn(rng, n)
    A = W .+ σ_a .* Ua
    L = α_a .* A .+ α_w .* W .+ σ_l .* Ul
    M = γ_a .* A .+ γ_l .* L .+ γ_w .* W .+ σ_m .* Um
    Y = β_a .* A .+ β_m .* M .+ β_l .* L .+ β_w .* W .+ σ_y .* Uy
    df = DataFrame(W = W, A = A, L = L, M = M, Y = Y)

    β_a_f, β_m_f, β_l_f = Float64(β_a), Float64(β_m), Float64(β_l)
    γ_a_f, γ_l_f, α_a_f = Float64(γ_a), Float64(γ_l), Float64(α_a)
    # Path-only formula (misses interventional M / L structure) — diagnostic only
    path_te = β_a_f + β_m_f * (γ_a_f + γ_l_f * α_a_f) + β_l_f * α_a_f

    function oracle(δ_sd::Real; lower_q::Real = 0.01, upper_q::Real = 0.99)
        sdA = std(A)
        Lq, Uq = quantile(A, lower_q), quantile(A, upper_q)
        a0 = clamp.(A, Lq, Uq)
        a1 = clamp.(A .+ Float64(δ_sd) * sdA, Lq, Uq)
        # Interventional ψ(a_t, a_m): L follows treatment a_t; M drawn under a_m (with L(a_m)).
        L_of = a -> α_a_f .* a .+ α_w .* W .+ σ_l .* Ul
        M_of = a -> begin
            La = L_of(a)
            γ_a_f .* a .+ γ_l_f .* La .+ γ_w .* W .+ σ_m .* Um
        end
        Y_of = (a_t, a_m) -> begin
            Lt = L_of(a_t)
            Mm = M_of(a_m)
            β_a_f .* a_t .+ β_m_f .* Mm .+ β_l_f .* Lt .+ β_w .* W .+ σ_y .* Uy
        end
        Y00 = Y_of(a0, a0)
        Y10 = Y_of(a1, a0)
        Y11 = Y_of(a1, a1)
        nde = mean(Y10 .- Y00)
        nie = mean(Y11 .- Y10)
        return (nde = nde, nie = nie, te = nde + nie, eff = mean(a1 .- a0))
    end

    truth = (
        name = "intermediate_confounding_mediation",
        path_te_per_unit = path_te,
        effects = δ_eff -> begin
            d = Float64(δ_eff)
            (nde = (β_a_f + β_l_f * α_a_f) * d, nie = β_m_f * (γ_a_f + γ_l_f * α_a_f) * d, te = path_te * d)
        end,
        oracle = oracle,
    )
    return df, truth
end

"""
    simulate_misspecified_nuisance_mtp(n; ...) -> (df, truth)

Nonlinear outcome in W so a GLM-only Super Learner is misspecified, while the
A→Y structural coefficient remains linear (`β_a`). Truth still scales with
effective mean shift: `TE = β_a · eff`.
"""
function simulate_misspecified_nuisance_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_w2::Real = 1.2,
    σ_a::Real = 1.0,
    σ_y::Real = 0.5,
    rng = StableRNG(13),
)
    W = randn(rng, n)
    A = W .+ σ_a .* randn(rng, n)
    Y = β_a .* A .+ β_w2 .* (W .^ 2) .+ σ_y .* randn(rng, n)
    df = DataFrame(W = W, A = A, Y = Y)
    β = Float64(β_a)
    truth = (
        name = "misspecified_nuisance_mtp",
        β_a = β,
        effects = δ_eff -> (nde = β * δ_eff, nie = 0.0, te = β * δ_eff),
    )
    return df, truth
end

"""
    truth_shift_effect(truth, δ) -> Float64

Evaluate closed-form additive MTP contrast for a linear DGP (`truth.shift_effect`).
"""
truth_shift_effect(truth, δ::Real) = truth.shift_effect(δ)

"""
    effective_sd_shift(a, δ_sd; lower_q=0.01, upper_q=0.99) -> Float64

Mean achieved shift under the SD-unit clamp policy used by mediation grids.
"""
function effective_sd_shift(
    a::AbstractVector{<:Real},
    δ_sd::Real;
    lower_q::Real = 0.01,
    upper_q::Real = 0.99,
)
    av = Float64.(a)
    sdA = std(av)
    Lq, Uq = quantile(av, lower_q), quantile(av, upper_q)
    a0 = clamp.(av, Lq, Uq)
    a1 = clamp.(av .+ Float64(δ_sd) * sdA, Lq, Uq)
    return mean(a1 .- a0)
end

"""
    effective_raw_shift(a, δ_raw; lower_q=0.01, upper_q=0.99) -> Float64

Mean achieved shift under LMTP `shift_scale=\"z\"` (raw additive δ then clamp).
When exposures are standardised upstream this coincides with an SD-unit shift.
"""
function effective_raw_shift(
    a::AbstractVector{<:Real},
    δ_raw::Real;
    lower_q::Real = 0.01,
    upper_q::Real = 0.99,
)
    av = Float64.(a)
    Lq, Uq = quantile(av, lower_q), quantile(av, upper_q)
    a0 = clamp.(av, Lq, Uq)
    a1 = clamp.(av .+ Float64(δ_raw), Lq, Uq)
    return mean(a1 .- a0)
end

"""
    simulate_nonlinear_interaction_mtp(n; ...) -> (df, truth)

Nonlinear outcome with treatment–covariate interaction:

- `W1, W2 ~ N(0,1)`
- `A = W1 + W2 + ε_a`
- `Y = β_a·A + β_int·A·W1 + β_w2·W2² + ε_y`

Truth: `TE = (β_a + β_int·E[W1]) · eff` where `eff` is the effective mean shift.
GLM-only SuperLearner is misspecified; learners with interaction features should
recover the effect.
"""
function simulate_nonlinear_interaction_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_int::Real = 0.4,
    β_w2::Real = 0.8,
    σ_a::Real = 1.0,
    σ_y::Real = 0.5,
    rng = StableRNG(20),
)
    W1 = randn(rng, n)
    W2 = randn(rng, n)
    A = W1 .+ W2 .+ σ_a .* randn(rng, n)
    Y = β_a .* A .+ β_int .* A .* W1 .+ β_w2 .* (W2 .^ 2) .+ σ_y .* randn(rng, n)
    df = DataFrame(W1 = W1, W2 = W2, A = A, Y = Y)
    β_a_f, β_int_f = Float64(β_a), Float64(β_int)
    truth = (
        name = "nonlinear_interaction_mtp",
        β_a = β_a_f,
        β_int = β_int_f,
        effects = δ_eff -> (nde = (β_a_f + β_int_f * mean(W1)) * δ_eff, nie = 0.0,
                            te = (β_a_f + β_int_f * mean(W1)) * δ_eff),
    )
    return df, truth
end

"""
    simulate_smooth_nonlinear_mtp(n; ...) -> (df, truth)

Smooth high-frequency outcome nuisance that shallow piecewise models struggle
with, while a small MLP can approximate:

- `W1, W2, W3 ~ N(0,1)`
- `A = (W1 + W2 + W3)/√3 + ε_a`
- `Y = β_a·A + ∑ⱼ sin(πω Wⱼ) + β_cross·sin(πω W₁ W₂) + ε_y`

Truth: `TE = β_a · eff` (additive treatment effect; nonlinear terms are
confounding / outcome nuisance only). Intended as an optional stress test for
`:mlj_mlp` alongside GLM/EvoTree libraries — not part of small-*n* defaults.
"""
function simulate_smooth_nonlinear_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_cross::Real = 0.8,
    ω::Real = 2.0,
    σ_a::Real = 1.0,
    σ_y::Real = 0.4,
    rng = StableRNG(21),
)
    W1 = randn(rng, n)
    W2 = randn(rng, n)
    W3 = randn(rng, n)
    A = (W1 .+ W2 .+ W3) ./ sqrt(3) .+ σ_a .* randn(rng, n)
    ωf = Float64(ω)
    nuis = sin.(ωf * π .* W1) .+ sin.(ωf * π .* W2) .+ sin.(ωf * π .* W3) .+
           Float64(β_cross) .* sin.(ωf * π .* W1 .* W2)
    Y = Float64(β_a) .* A .+ nuis .+ σ_y .* randn(rng, n)
    df = DataFrame(W1 = W1, W2 = W2, W3 = W3, A = A, Y = Y)
    β = Float64(β_a)
    truth = (
        name = "smooth_nonlinear_mtp",
        β_a = β,
        effects = δ_eff -> (nde = β * δ_eff, nie = 0.0, te = β * δ_eff),
    )
    return df, truth
end

"""
    simulate_missing_outcome_mtp(n; miss_rate=0.2, ...) -> (df, truth)

Linear MTP where `Y` is set to `missing` under a MAR mechanism:
`P(R=0 | W) = logistic(α_w·W)`, calibrated to achieve approximately `miss_rate`
overall. The complete-data truth is unchanged.
"""
function simulate_missing_outcome_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_w::Real = 1.0,
    σ_y::Real = 0.5,
    σ_a::Real = 1.0,
    miss_rate::Real = 0.2,
    rng = StableRNG(30),
)
    W = randn(rng, n)
    A = W .+ σ_a .* randn(rng, n)
    Y_full = β_a .* A .+ β_w .* W .+ σ_y .* randn(rng, n)
    # MAR mechanism: calibrate intercept so E[miss] ≈ miss_rate
    α_w = 0.8
    α_0 = log(miss_rate / (1 - miss_rate))
    p_miss = 1.0 ./ (1.0 .+ exp.(-(α_0 .+ α_w .* W)))
    R = [rand(rng) > p for p in p_miss]
    Y = Vector{Union{Float64, Missing}}(Y_full)
    Y[.!R] .= missing
    df = DataFrame(W = W, A = A, Y = Y)
    β = Float64(β_a)
    truth = (
        name = "missing_outcome_mtp",
        β_a = β,
        miss_rate_actual = 1.0 - mean(R),
        effects = δ_eff -> (nde = β * δ_eff, nie = 0.0, te = β * δ_eff),
    )
    return df, truth
end

"""
    simulate_missing_covariate_mtp(n; miss_rate=0.15, ...) -> (df, truth)

Linear MTP where `W` has MCAR missingness at approximately `miss_rate`.
The complete-data truth is unchanged.
"""
function simulate_missing_covariate_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_w::Real = 1.0,
    σ_y::Real = 0.5,
    σ_a::Real = 1.0,
    miss_rate::Real = 0.15,
    rng = StableRNG(31),
)
    W_full = randn(rng, n)
    A = W_full .+ σ_a .* randn(rng, n)
    Y = β_a .* A .+ β_w .* W_full .+ σ_y .* randn(rng, n)
    W = Vector{Union{Float64, Missing}}(W_full)
    R = [rand(rng) > miss_rate for _ in 1:n]
    W[.!R] .= missing
    df = DataFrame(W = W, A = A, Y = Y)
    β = Float64(β_a)
    truth = (
        name = "missing_covariate_mtp",
        β_a = β,
        miss_rate_actual = 1.0 - mean(R),
        effects = δ_eff -> (nde = β * δ_eff, nie = 0.0, te = β * δ_eff),
    )
    return df, truth
end

"""
    simulate_did_2x2(n; τ=1.0, ...) -> (df, truth)

Classic 2×2 difference-in-differences panel data.

- `n` units, 2 periods (`t ∈ {0, 1}`)
- Treatment group assignment: `D_i ~ Bernoulli(p_treat)`
- `Y_{it} = α_i + λ_t + τ · D_{it} + ε_{it}`
- `D_{it} = D_i · 1{t=1}` (treated units receive treatment in period 1 only)

Parallel trends hold by construction. Truth: `ATT = τ`.
Returns long-format DataFrame with columns `:unit`, `:time`, `:treat`, `:Y`.
"""
function simulate_did_2x2(
    n::Int;
    τ::Real = 1.0,
    λ::Real = 0.5,
    p_treat::Real = 0.5,
    σ_α::Real = 1.0,
    σ_ε::Real = 0.5,
    rng = StableRNG(40),
)
    D = [rand(rng) < p_treat ? 1.0 : 0.0 for _ in 1:n]
    α = σ_α .* randn(rng, n)
    rows = Dict{String, Any}[]
    for t in 0:1
        ε = σ_ε .* randn(rng, n)
        Dit = D .* Float64(t)
        Yit = α .+ λ * t .+ τ .* Dit .+ ε
        for i in 1:n
            push!(rows, Dict{String, Any}(
                "unit" => i, "time" => t, "treat" => D[i], "Y" => Yit[i],
            ))
        end
    end
    df = DataFrame(rows)
    truth = (
        name = "did_2x2",
        att = Float64(τ),
    )
    return df, truth
end

"""
    simulate_did_staggered(n; n_periods=4, ...) -> (df, truth)

Staggered difference-in-differences with heterogeneous treatment effects by cohort.

- `n` units, `n_periods` periods
- 3 cohorts: never-treated, early adopters (treat at t=2), late adopters (treat at t=3)
- `Y_{it} = α_i + λ_t + τ_g · D_{it} + ε` where `τ_g` is cohort-specific
- `τ_early = 1.0`, `τ_late = 0.5`

Returns long-format DataFrame and truth with cohort-specific ATTs.
"""
function simulate_did_staggered(
    n::Int;
    n_periods::Int = 4,
    τ_early::Real = 1.0,
    τ_late::Real = 0.5,
    σ_α::Real = 1.0,
    σ_ε::Real = 0.5,
    rng = StableRNG(41),
)
    # Assign cohorts: ~1/3 each
    cohort = [rand(rng) < 1/3 ? :never : (rand(rng) < 0.5 ? :early : :late) for _ in 1:n]
    treat_time = Dict(:never => n_periods + 1, :early => 2, :late => 3)
    τ_map = Dict(:never => 0.0, :early => Float64(τ_early), :late => Float64(τ_late))
    α = σ_α .* randn(rng, n)
    λ = range(0.0, 1.0, length = n_periods)

    rows = Dict{String, Any}[]
    for t in 1:n_periods
        ε = σ_ε .* randn(rng, n)
        for i in 1:n
            Dit = t >= treat_time[cohort[i]] ? 1.0 : 0.0
            τ_i = τ_map[cohort[i]]
            Yit = α[i] + λ[t] + τ_i * Dit + ε[i]
            push!(rows, Dict{String, Any}(
                "unit" => i, "time" => t,
                "cohort" => String(cohort[i]),
                "treat" => Dit, "Y" => Yit,
            ))
        end
    end
    df = DataFrame(rows)
    truth = (
        name = "did_staggered",
        att_early = Float64(τ_early),
        att_late = Float64(τ_late),
        att_aggregate = (Float64(τ_early) + Float64(τ_late)) / 2,
    )
    return df, truth
end

"""
    simulate_gcomp_nonlinear(n; ...) -> (df, truth)

Binary treatment with treatment–covariate interaction for g-computation testing:

- `W ~ N(0,1)`
- `A ~ Bernoulli(logistic(0.5·W))`
- `Y = β_a·A + β_w·W + β_aw·A·W + ε`

Truth computed by shared-noise oracle: `ATE = E[Y(1) - Y(0)]`.
"""
function simulate_gcomp_nonlinear(
    n::Int;
    β_a::Real = 1.0,
    β_w::Real = 0.5,
    β_aw::Real = 0.4,
    σ_y::Real = 0.5,
    rng = StableRNG(50),
)
    W = randn(rng, n)
    p_a = 1.0 ./ (1.0 .+ exp.(-0.5 .* W))
    A = Float64.([rand(rng) < p for p in p_a])
    ε = σ_y .* randn(rng, n)
    Y = β_a .* A .+ β_w .* W .+ β_aw .* A .* W .+ ε
    df = DataFrame(W = W, A = A, Y = Y)

    β_a_f, β_w_f, β_aw_f = Float64(β_a), Float64(β_w), Float64(β_aw)
    # Shared-noise oracle: ATE = E[Y(1) - Y(0)] = β_a + β_aw·E[W]
    Y1 = β_a_f .* 1.0 .+ β_w_f .* W .+ β_aw_f .* 1.0 .* W .+ ε
    Y0 = β_w_f .* W .+ ε
    ate_oracle = mean(Y1 .- Y0)  # = β_a + β_aw·mean(W), close to β_a for large n

    truth = (
        name = "gcomp_nonlinear",
        ate = ate_oracle,
        β_a = β_a_f,
        β_aw = β_aw_f,
    )
    return df, truth
end

"""
    simulate_discrete_survival_mtp(n; T=3, α=-1.2, β_a=-0.5, β_w=0.3, σ_a=1.0, rng)

Discrete-time survival under longitudinal continuous treatments.

For `t = 1…T`:

- `A_t = W + σ_a ε_t`
- hazard `λ_t = logit⁻¹(α + β_a A_t + β_w W)` among units still event-free
- `S_t = 1` if still event-free after occasion `t` (monotone: once 0, stays 0)

`truth.survival(δ)` is a shared-noise Monte Carlo oracle for
``E[S_T^{A+δ}]`` under a raw additive shift (no clamp) of every `A_t` by `δ`.
With `β_a < 0`, larger `δ` raises survival.
"""
function simulate_discrete_survival_mtp(
    n::Int;
    T::Int = 3,
    α::Real = -1.2,
    β_a::Real = -0.5,
    β_w::Real = 0.3,
    σ_a::Real = 1.0,
    rng = StableRNG(7),
)
    T >= 1 || throw(ArgumentError("T must be ≥ 1"))
    α_f, β_a_f, β_w_f, σ_a_f = Float64(α), Float64(β_a), Float64(β_w), Float64(σ_a)
    W = randn(rng, n)
    ε = [σ_a_f .* randn(rng, n) for _ in 1:T]
    A = [W .+ ε[t] for t in 1:T]

    function _draw_surv(A_path)
        S = [zeros(n) for _ in 1:T]
        at_risk = trues(n)
        for t in 1:T
            η = α_f .+ β_a_f .* A_path[t] .+ β_w_f .* W
            λ = 1.0 ./ (1.0 .+ exp.(-η))
            fail = at_risk .& (rand(rng, n) .< λ)
            at_risk = at_risk .& .!fail
            S[t] .= Float64.(at_risk)
        end
        return S
    end

    S = _draw_surv(A)
    df = DataFrame(:W => W)
    for t in 1:T
        df[!, Symbol("A$t")] = A[t]
        df[!, Symbol("S$t")] = S[t]
    end

    function survival_oracle(δ::Real; n_mc::Int = 20_000, oracle_rng = StableRNG(701))
        δ_f = Float64(δ)
        W_mc = randn(oracle_rng, n_mc)
        ε_mc = [σ_a_f .* randn(oracle_rng, n_mc) for _ in 1:T]
        A_pol = [W_mc .+ ε_mc[t] .+ δ_f for t in 1:T]
        at_risk = trues(n_mc)
        for t in 1:T
            η = α_f .+ β_a_f .* A_pol[t] .+ β_w_f .* W_mc
            λ = 1.0 ./ (1.0 .+ exp.(-η))
            fail = at_risk .& (rand(oracle_rng, n_mc) .< λ)
            at_risk = at_risk .& .!fail
        end
        return mean(Float64.(at_risk))
    end

    truth = (
        name = "discrete_survival_mtp",
        T = T,
        α = α_f,
        β_a = β_a_f,
        β_w = β_w_f,
        survival = survival_oracle,
        treatments = [Symbol("A$t") for t in 1:T],
        surv = [Symbol("S$t") for t in 1:T],
    )
    return df, truth
end

"""
    simulate_mixed_baseline_mtp(n; β_a=0.5, β_w=1.0, σ_y=0.5, σ_a=1.0, rng) -> (df, truth)

Linear MTP DGP with mixed-type baseline covariates for `CovariateSchema` stress:

- `W::Float64`, `site::String`, `vaccinated::Bool`, `breed::String`
- Outcome still depends only on `A` and `W` (site / vaccinated / breed are
  adjustment noise), so under an additive shift
  `E[Y^{A+δ} - Y^{A}] = β_a * δ` when clamp is inactive (same as
  [`simulate_linear_mtp`](@ref)).
"""
function simulate_mixed_baseline_mtp(
    n::Int;
    β_a::Real = 0.5,
    β_w::Real = 1.0,
    σ_y::Real = 0.5,
    σ_a::Real = 1.0,
    rng = StableRNG(1),
)
    df, truth = simulate_linear_mtp(n; β_a = β_a, β_w = β_w, σ_y = σ_y, σ_a = σ_a, rng = rng)
    sites = ("A", "B", "C")
    breeds = ("lab", "collie", "cross")
    df.site = [sites[rand(rng, 1:3)] for _ in 1:n]
    df.vaccinated = rand(rng, Bool, n)
    # Rare level \"other\" (~5%) exercises DummyCoding with sparse columns
    df.breed = [rand(rng) < 0.05 ? "other" : breeds[rand(rng, 1:3)] for _ in 1:n]
    truth = (
        name = "mixed_baseline_mtp",
        β_a = truth.β_a,
        β_w = truth.β_w,
        shift_effect = truth.shift_effect,
        effects = truth.effects,
        baseline = [:W, :site, :vaccinated, :breed],
    )
    return df, truth
end

"""
    simulate_binomial_mtp(n; β_a=0.8, β_w=1.0, σ_a=1.0, rng) -> (df, truth)

Logistic outcome, additive shift on continuous `A`. Sample oracle
`effects(δ)` is the mean difference of `σ(β_a (A+δ) + β_w W)` versus the
factual conditional mean on the same `(A, W)`.
"""
function simulate_binomial_mtp(
    n::Int;
    β_a::Real = 0.8,
    β_w::Real = 1.0,
    σ_a::Real = 1.0,
    rng = StableRNG(1),
)
    W = randn(rng, n)
    A = W .+ σ_a .* randn(rng, n)
    η = Float64(β_a) .* A .+ Float64(β_w) .* W
    p = 1.0 ./ (1.0 .+ exp.(-η))
    Y = Float64.(rand(rng, n) .< p)
    df = DataFrame(W = W, A = A, Y = Y)
    truth = (
        name = "binomial_mtp",
        β_a = Float64(β_a),
        β_w = Float64(β_w),
        effects = δ -> begin
            η1 = Float64(β_a) .* (A .+ Float64(δ)) .+ Float64(β_w) .* W
            η0 = Float64(β_a) .* A .+ Float64(β_w) .* W
            te = mean((1.0 ./ (1.0 .+ exp.(-η1))) .- (1.0 ./ (1.0 .+ exp.(-η0))))
            (nde = te, nie = 0.0, te = te)
        end,
    )
    return df, truth
end

"""
    simulate_multinomial_outcome(n; K=3, rng) -> (df, truth)

Softmax class probabilities of a linear index in `W`. `truth.P` is the
`n × K` probability matrix; `Y` is a sampled integer label in `1:K`.
"""
function simulate_multinomial_outcome(
    n::Int;
    K::Int = 3,
    rng = StableRNG(1),
)
    K >= 2 || throw(ArgumentError("K must be at least 2"))
    W = randn(rng, n)
    β = [(k - (K + 1) / 2) for k in 1:K]
    η = W * transpose(β)
    eη = exp.(η .- maximum(η; dims = 2))
    P = eη ./ sum(eη; dims = 2)
    Y = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        u = rand(rng)
        c = 0.0
        Y[i] = K
        for k in 1:K
            c += P[i, k]
            if u <= c
                Y[i] = k
                break
            end
        end
    end
    df = DataFrame(W = W, Y = Y)
    truth = (name = "multinomial_outcome", K = K, P = Matrix{Float64}(P), levels = collect(1:K))
    return df, truth
end

"""
    simulate_categorical_treatment_mtp(n; rng) -> (df, truth)

Three-level string exposure `A ∈ {0,1,2}`, `Y = β₁ 1{A=1} + β₂ 1{A=2} + β_w W + ε`.
Default policy recodes `2 → 1`. Oracle TE is `(β₁ - β₂) P(A=2)` on the sample.
"""
function simulate_categorical_treatment_mtp(
    n::Int;
    β1::Real = 1.0,
    β2::Real = -0.5,
    β_w::Real = 0.8,
    σ_y::Real = 0.4,
    rng = StableRNG(1),
)
    W = randn(rng, n)
    # Softmax propensity with a W slope so A is confounded.
    scores = hcat(0.2 .* W, 0.1 .+ 0.4 .* W, -0.2 .- 0.3 .* W)
    e = exp.(scores .- maximum(scores; dims = 2))
    pr = e ./ sum(e; dims = 2)
    A = Vector{String}(undef, n)
    @inbounds for i in 1:n
        u = rand(rng)
        c = 0.0
        A[i] = "2"
        for (k, lab) in enumerate(("0", "1", "2"))
            c += pr[i, k]
            if u <= c
                A[i] = lab
                break
            end
        end
    end
    Y = Float64(β1) .* (A .== "1") .+ Float64(β2) .* (A .== "2") .+
        Float64(β_w) .* W .+ Float64(σ_y) .* randn(rng, n)
    df = DataFrame(W = W, A = A, Y = Y)
    p2 = mean(A .== "2")
    te = (Float64(β1) - Float64(β2)) * p2
    truth = (
        name = "categorical_treatment_mtp",
        β1 = Float64(β1),
        β2 = Float64(β2),
        β_w = Float64(β_w),
        recode = Dict("2" => "1"),
        effects = _ -> (nde = te, nie = 0.0, te = te),
        te = te,
        p2 = p2,
    )
    return df, truth
end

"""
    _softmax_string_labels(scores, labels, rng) -> Vector{String}

Sample one label per row from a softmax of `scores` (`n × K`).
"""
function _softmax_string_labels(scores::AbstractMatrix, labels, rng)
    n, K = size(scores)
    length(labels) == K || throw(ArgumentError("labels length must match score columns"))
    e = exp.(scores .- maximum(scores; dims = 2))
    pr = e ./ sum(e; dims = 2)
    out = Vector{String}(undef, n)
    @inbounds for i in 1:n
        u = rand(rng)
        c = 0.0
        out[i] = labels[K]
        for k in 1:K
            c += pr[i, k]
            if u <= c
                out[i] = labels[k]
                break
            end
        end
    end
    return out
end

"""
    _sequential_factor_structural_mean(A1, A2, W; β1, β2, α1, α2, β_w) -> Vector{Float64}

Linear dummy structural mean for the T=2 factor DGP.
"""
function _sequential_factor_structural_mean(
    A1::AbstractVector, A2::AbstractVector, W::AbstractVector;
    β1::Real, β2::Real, α1::Real, α2::Real, β_w::Real,
)
    return Float64(β1) .* (A1 .== "1") .+ Float64(β2) .* (A1 .== "2") .+
        Float64(α1) .* (A2 .== "1") .+ Float64(α2) .* (A2 .== "2") .+
        Float64(β_w) .* W
end

"""Apply a string recode map, leaving unmapped levels unchanged."""
function _recode_string_vector(A::AbstractVector{<:AbstractString}, recode::AbstractDict)
    return [haskey(recode, a) ? string(recode[a]) : a for a in A]
end

"""
    simulate_sequential_factor_mtp(n; rng) -> (df, truth)

T=2 string exposures `A1, A2 ∈ {0,1,2}`, time-varying `L1`, continuous `W`.
Default MTP recodes `2 → 1` at both times. Oracle `psi` is the sample
structural mean under that recode (g-computation with known coefficients).
"""
function simulate_sequential_factor_mtp(
    n::Int;
    β1::Real = 1.0,
    β2::Real = -0.5,
    α1::Real = 0.8,
    α2::Real = -0.4,
    β_w::Real = 0.6,
    γ_l::Real = 0.5,
    σ_y::Real = 0.4,
    rng = StableRNG(1),
)
    W = randn(rng, n)
    A1 = _softmax_string_labels(
        hcat(0.2 .* W, 0.1 .+ 0.4 .* W, -0.2 .- 0.3 .* W),
        ("0", "1", "2"),
        rng,
    )
    L1 = Float64(γ_l) .* (A1 .== "2") .+ 0.4 .* W .+ 0.3 .* randn(rng, n)
    A2 = _softmax_string_labels(
        hcat(0.1 .* W .+ 0.35 .* L1, 0.15 .+ 0.25 .* W, -0.15 .- 0.3 .* L1),
        ("0", "1", "2"),
        rng,
    )
    Y = _sequential_factor_structural_mean(
        A1, A2, W; β1 = β1, β2 = β2, α1 = α1, α2 = α2, β_w = β_w,
    ) .+ Float64(σ_y) .* randn(rng, n)
    recode = Dict("2" => "1")
    A1d = _recode_string_vector(A1, recode)
    A2d = _recode_string_vector(A2, recode)
    μd = _sequential_factor_structural_mean(
        A1d, A2d, W; β1 = β1, β2 = β2, α1 = α1, α2 = α2, β_w = β_w,
    )
    μ = _sequential_factor_structural_mean(
        A1, A2, W; β1 = β1, β2 = β2, α1 = α1, α2 = α2, β_w = β_w,
    )
    psi = mean(μd)
    te = psi - mean(μ)
    df = DataFrame(W = W, A1 = A1, L1 = L1, A2 = A2, Y = Y)
    truth = (
        name = "sequential_factor_mtp",
        β1 = Float64(β1),
        β2 = Float64(β2),
        α1 = Float64(α1),
        α2 = Float64(α2),
        β_w = Float64(β_w),
        recode = recode,
        psi = psi,
        te = te,
        p_A1_2 = mean(A1 .== "2"),
        p_A2_2 = mean(A2 .== "2"),
        effects = _ -> (nde = te, nie = 0.0, te = te),
    )
    return df, truth
end

# Book / README DGPs; remaining simulators stay available as CausalTargeted.simulate_*
export simulate_linear_mtp, simulate_mediation, simulate_discrete_survival_mtp
export simulate_mixed_baseline_mtp
export simulate_binomial_mtp, simulate_multinomial_outcome, simulate_categorical_treatment_mtp
export simulate_sequential_factor_mtp
export truth_shift_effect, effective_sd_shift, effective_raw_shift
