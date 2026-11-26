# ---------------------------------------------------------------------------
# Exported book / README synthetic DGPs
# ---------------------------------------------------------------------------

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
    # Rare level "other" (~5%) exercises DummyCoding with sparse columns
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

"""
    simulate_repeated_outcome_ate(n; T=4, β_a, β_w, ρ, σ_y, rng) -> (df, truth)

Binary point treatment with ``T`` correlated outcomes (wide layout):

- `W ~ N(0,1)`
- `A ~ Bernoulli(logistic(0.5·W))`
- shared factor `U ~ N(0,1)` and independent `ε_t`
- `Y_t = β_a[t]·A + β_w·W + √ρ·U + √(1-ρ)·ε_t`

Truth: `tau = β_a` (ATE profile; no treatment–covariate interaction).
"""
function simulate_repeated_outcome_ate(
    n::Int;
    T::Int = 4,
    β_a::AbstractVector{<:Real} = [0.1, 0.7, 0.9, 0.3],
    β_w::Real = 1.0,
    ρ::Real = 0.5,
    σ_y::Real = 0.5,
    rng::AbstractRNG = StableRNG(1),
)
    T >= 1 || throw(ArgumentError("T must be ≥ 1"))
    β = collect(Float64, β_a)
    if length(β) == 1
        β = fill(β[1], T)
    elseif length(β) != T
        throw(ArgumentError("β_a must have length 1 or T=$T; got $(length(β))"))
    end
    0 <= ρ <= 1 || throw(ArgumentError("ρ must lie in [0, 1]"))

    W = randn(rng, n)
    A = Float64.(rand(rng, n) .< (1 ./ (1 .+ exp.(-0.5 .* W))))
    U = randn(rng, n)
    cols = Dict{Symbol, Any}(:W => W, :A => A)
    sqrt_ρ = sqrt(ρ)
    sqrt_1mρ = sqrt(1 - ρ)
    for t in 1:T
        ε = randn(rng, n)
        cols[Symbol("Y", t)] = β[t] .* A .+ Float64(β_w) .* W .+
            σ_y .* (sqrt_ρ .* U .+ sqrt_1mρ .* ε)
    end
    df = DataFrame(cols)
    # Stable column order: W, A, Y1…YT
    select!(df, :W, :A, [Symbol("Y", t) for t in 1:T]...)
    truth = (
        name = "repeated_outcome_ate",
        tau = β,
        β_w = Float64(β_w),
        ρ = Float64(ρ),
        σ_y = Float64(σ_y),
        T = T,
    )
    return df, truth
end
