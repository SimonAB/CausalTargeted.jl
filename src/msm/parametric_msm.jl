"""Parametric treatment×time MSM via GLS projection of unstructured IF estimates.

Projects the unstructured profile ``\\widehat{\\tau}`` (or stacked means
``\\widehat{\\mu}(t,a)``) onto a design matrix with covariance-weighted GLS,
following the Rosenblum–van der Laan (2010) MSM spirit.

Designs for `target=:tau` (default):

- `:constant` — ``\\tau(t)=\\beta_0``
- `:linear_time` — ``\\tau(t)=\\beta_0+\\beta_1(t-1)``
- `:factor_time` — saturated (identity); recovers unstructured ``\\tau``
- custom `AbstractMatrix` — ``T \\times p`` design for ``\\tau``

Design for `target=:mean`:

- `:mean_treatment_time` — ``m(t,a)=\\beta_0+\\beta_A a+\\sum_{t'\\ge 2}
  \\beta_{t'}I(t=t')+\\sum_{t'\\ge 2}\\beta_{A t'} a\\,I(t=t')``
"""

"""
    ParametricRepeatedOutcomeMSM(trt, outcomes, adjustment; design=:linear_time, target=:tau)

Parametric MSM estimand. Engine [`estimand_engine`](@ref) is `:parametric_msm`.
"""
struct ParametricRepeatedOutcomeMSM <: Estimand
    trt::Symbol
    outcomes::Vector{Symbol}
    adjustment::Vector{Symbol}
    design::Any
    target::Symbol
end

function ParametricRepeatedOutcomeMSM(
    trt::Symbol,
    outcomes::AbstractVector{Symbol},
    adjustment::AbstractVector{Symbol};
    design = :linear_time,
    target::Symbol = :tau,
)
    target in (:tau, :mean) || throw(ArgumentError(
        "target must be :tau or :mean; got :$target",
    ))
    return ParametricRepeatedOutcomeMSM(
        trt, collect(Symbol, outcomes), collect(Symbol, adjustment), design, target,
    )
end

estimand_engine(::ParametricRepeatedOutcomeMSM) = :parametric_msm

"""
    _tau_msm_design(T, design) -> (X, names, design_symbol)

Build a ``T \\times p`` design for projecting ``\\tau(t)``.
"""
function _tau_msm_design(T::Int, design::Symbol)
    if design === :constant
        return ones(T, 1), [:intercept], :constant
    elseif design === :linear_time
        X = hcat(ones(T), Float64.(0:(T - 1)))
        return X, [:intercept, :time], :linear_time
    elseif design === :factor_time
        return Matrix{Float64}(I, T, T), [Symbol("tau_", t) for t in 1:T], :factor_time
    else
        throw(ArgumentError(
            "unknown τ design :$design; use :constant, :linear_time, :factor_time, or a matrix",
        ))
    end
end

function _tau_msm_design(T::Int, design::AbstractMatrix{<:Real})
    X = Float64.(design)
    size(X, 1) == T || throw(ArgumentError(
        "design matrix must have $T rows (one per unstructured parameter; K*T when stratified); got $(size(X, 1))",
    ))
    p = size(X, 2)
    names = [Symbol("β", j) for j in 1:p]
    return X, names, :custom
end

"""
    _mean_treatment_time_design(T) -> (X, names)

Build a ``2T \\times 2T`` saturated treatment×time design for stacked
``(\\mu(t,0), \\mu(t,1))_{t=1}^T``.
"""
function _mean_treatment_time_design(T::Int)
    T >= 1 || throw(ArgumentError("T must be ≥ 1"))
    p = 2 + 2 * max(T - 1, 0)
    X = zeros(2T, p)
    names = Symbol[:intercept, :A]
    for t in 2:T
        push!(names, Symbol("time_", t))
    end
    for t in 2:T
        push!(names, Symbol("A_time_", t))
    end
    for t in 1:T
        for (ia, a) in enumerate((0.0, 1.0))
            r = 2 * (t - 1) + ia
            X[r, 1] = 1.0
            X[r, 2] = a
            if t >= 2
                X[r, 1 + t] = 1.0          # time_t column
                X[r, T + t] = a            # A_time_t column
            end
        end
    end
    return X, names
end

"""
    _gls_project(θ, Σ, X) -> (β, V, fitted)

Covariance-weighted GLS: ``\\hat\\beta=(X^\\top\\Sigma^{-1}X)^{-1}X^\\top\\Sigma^{-1}\\theta``.
`Σ` is the sampling covariance of ``\\hat\\theta`` (same scale as unstructured MSM).
"""
function _gls_project(
    θ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
)
    m = length(θ)
    size(Σ) == (m, m) || throw(ArgumentError("Σ must be $(m)×$(m)"))
    size(X, 1) == m || throw(ArgumentError("X must have $m rows"))
    p = size(X, 2)
    p >= 1 || throw(ArgumentError("design must have at least one column"))
    p > m && throw(ArgumentError("overparameterised design: p=$p > m=$m"))

    Σreg = Matrix{Float64}(Σ) + 1e-10 * I(m)
    W = inv(Σreg)
    XtW = X' * W
    XtWX = Symmetric(XtW * X)
    β = XtWX \ (XtW * Float64.(θ))
    V = Matrix{Float64}(inv(XtWX))
    V = Symmetric(0.5 .* (V .+ V'))
    fitted = X * β
    return β, Matrix{Float64}(V), fitted
end

"""
    _stack_mu_moments(un) -> (μ, Σ, psi_like)

Stack ``(\\mu(t,0), \\mu(t,1))`` with joint IF covariance from unit-level curves.
"""
function _stack_mu_moments(un::NamedTuple)
    μ0 = un.mu0
    μ1 = un.mu1
    ic0 = un.ic_mu0
    ic1 = un.ic_mu1
    n, T = size(ic0)
    μ = Vector{Float64}(undef, 2T)
    ic = Matrix{Float64}(undef, n, 2T)
    for t in 1:T
        μ[2t - 1] = μ0[t]
        μ[2t] = μ1[t]
        ic[:, 2t - 1] = ic0[:, t]
        ic[:, 2t] = ic1[:, t]
    end
    Σ, _ = _msm_covariance_from_ic(ic; cluster = un.cluster)
    return μ, Σ, ic
end

"""
    _fitted_tau_from_mean(β, T) -> Vector

Implied ``\\tau(t)=m(t,1)-m(t,0)`` under `:mean_treatment_time`.
"""
function _fitted_tau_from_mean(β::AbstractVector{<:Real}, T::Int)
    X, _ = _mean_treatment_time_design(T)
    m = X * β
    return [m[2t] - m[2t - 1] for t in 1:T]
end

"""
    run_parametric_repeated_msm(df, treatment, outcomes; baseline, design, target, ...)
        -> NamedTuple

Estimate a parametric MSM by GLS projection of the unstructured repeated-outcome
IF estimates from [`run_repeated_outcome_msm`](@ref).

`strata` and `propensity` are passed through to the unstructured fit. A
stratified projection requires an explicit `K*T × p` design for `target=:tau`
or `2*K*T × p` for `target=:mean`; symbolic T-row designs are rejected.

Returns `(coefficients, se, covariance, coef_names, fitted_tau, design, target,
outcomes, n, positivity, missingness, …)`.
"""
function run_parametric_repeated_msm(
    df::DataFrame,
    treatment::Symbol,
    outcomes::AbstractVector{Symbol};
    baseline::Vector{Symbol} = Symbol[],
    design = :linear_time,
    target::Symbol = :tau,
    folds::Int = 3,
    learners = DEFAULT_SL_LEARNERS,
    learners_trt = (:logistic, :mean),
    rng::AbstractRNG = StableRNG(1),
    handle_missing::Symbol = :drop,
    estimator::Symbol = :tmle,
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
    strata = nothing,
    propensity = nothing,
)
    target in (:tau, :mean) || throw(ArgumentError(
        "target must be :tau or :mean; got :$target",
    ))
    if target === :mean
        design === :mean_treatment_time || design isa AbstractMatrix || throw(ArgumentError(
            "target=:mean requires design=:mean_treatment_time or a custom 2T×p matrix",
        ))
    elseif design === :mean_treatment_time
        throw(ArgumentError("design=:mean_treatment_time requires target=:mean"))
    end
    if strata !== nothing && design isa Symbol
        required_rows = target === :tau ? "K*T" : "2*K*T"
        throw(ArgumentError(
            "stratified parametric MSMs require a custom $required_rows × p design matrix; " *
            "symbolic design :$design is ambiguous for a stacked stratum profile",
        ))
    end

    un = run_repeated_outcome_msm(
        df, treatment, outcomes;
        baseline = baseline,
        folds = folds,
        learners = learners,
        learners_trt = learners_trt,
        rng = rng,
        handle_missing = handle_missing,
        estimator = estimator,
        cluster = cluster,
        strata = strata,
        propensity = propensity,
    )
    T = length(un.outcomes)
    P = length(un.estimates)

    if target === :tau
        X, coef_names, design_sym = _tau_msm_design(P, design)
        β, V, fitted = _gls_project(un.estimates, un.covariance, X)
        fitted_tau = fitted
    else
        μ, Σμ, _ = _stack_mu_moments(un)
        if design === :mean_treatment_time
            X, coef_names = _mean_treatment_time_design(T)
            design_sym = :mean_treatment_time
        else
            X = Float64.(design)
            size(X, 1) == 2P || throw(ArgumentError(
                "mean design must have $(2P) rows; got $(size(X, 1))",
            ))
            coef_names = [Symbol("β", j) for j in 1:size(X, 2)]
            design_sym = :custom
        end
        β, V, fitted_mean = _gls_project(μ, Σμ, X)
        fitted_tau = [fitted_mean[2p] - fitted_mean[2p - 1] for p in 1:P]
    end

    se = sqrt.(diag(V))
    return with_missingness((
        coefficients = β,
        se = se,
        covariance = V,
        coef_names = coef_names,
        fitted_tau = fitted_tau,
        unstructured_tau = un.estimates,
        unstructured_covariance = un.covariance,
        design = design_sym,
        target = target,
        outcomes = un.outcomes,
        n = un.n,
        positivity = un.positivity,
        propensity_metadata = un.propensity_metadata,
        parameter_index = [(
            position = j, coefficient = coef_names[j], estimand = :msm_coefficient,
        ) for j in eachindex(coef_names)],
        unstructured_parameter_index = un.parameter_index,
        strata = un.strata,
        cluster = un.cluster,
        covariance_kind = un.covariance_kind,
    ), un.missingness)
end

"""
    simulate_mean_treatment_time_msm(n; T=3, β, β_w, ρ, σ_y, rng) -> (df, truth)

Binary treatment with outcomes whose conditional means follow a treatment×time MSM:

```math
m(t,a)=\\beta_0+\\beta_A a+\\sum_{t'\\ge 2}\\beta_{t'}I(t=t')
+\\sum_{t'\\ge 2}\\beta_{A t'} a\\,I(t=t')
```

and ``Y_t=m(t,A)+\\beta_w W+\\sigma_y(\\sqrt{\\rho}U+\\sqrt{1-\\rho}\\varepsilon_t)``.

Default ``\\beta`` for ``T=3``: ``[1.0, 0.5, 0.2, 0.35, 0.15, 0.25]``.
Truth includes `beta`, `tau` (implied ATE profile), and `m`.
"""
function simulate_mean_treatment_time_msm(
    n::Int;
    T::Int = 3,
    β::AbstractVector{<:Real} = Float64[],
    β_w::Real = 1.0,
    ρ::Real = 0.5,
    σ_y::Real = 0.5,
    rng::AbstractRNG = StableRNG(1),
)
    T >= 1 || throw(ArgumentError("T must be ≥ 1"))
    p = 2 + 2 * max(T - 1, 0)
    βv = if isempty(β)
        # Mild treatment and time structure for recovery tests
        v = zeros(p)
        v[1] = 1.0
        v[2] = 0.5
        for t in 2:T
            v[1 + t] = 0.1 * (t - 1)           # time_t
            v[T + t] = 0.12 * (t - 1)          # A_time_t
        end
        # T=3 → [1, 0.5, 0.1, 0.2, 0.12, 0.24]
        v
    else
        collect(Float64, β)
    end
    length(βv) == p || throw(ArgumentError(
        "β must have length $p for T=$T; got $(length(βv))",
    ))
    0 <= ρ <= 1 || throw(ArgumentError("ρ must lie in [0, 1]"))

    X, _ = _mean_treatment_time_design(T)
    m_cells = X * βv
    m = Dict{Tuple{Int, Int}, Float64}()
    for t in 1:T
        m[(t, 0)] = m_cells[2t - 1]
        m[(t, 1)] = m_cells[2t]
    end
    tau = [m[(t, 1)] - m[(t, 0)] for t in 1:T]

    W = randn(rng, n)
    A = Float64.(rand(rng, n) .< (1 ./ (1 .+ exp.(-0.5 .* W))))
    U = randn(rng, n)
    cols = Dict{Symbol, Any}(:W => W, :A => A)
    sqrt_ρ = sqrt(ρ)
    sqrt_1mρ = sqrt(1 - ρ)
    for t in 1:T
        ε = randn(rng, n)
        mt = [m[(t, Int(a))] for a in A]
        cols[Symbol("Y", t)] = mt .+ Float64(β_w) .* W .+
            σ_y .* (sqrt_ρ .* U .+ sqrt_1mρ .* ε)
    end
    df = DataFrame(cols)
    select!(df, :W, :A, [Symbol("Y", t) for t in 1:T]...)
    truth = (
        name = "mean_treatment_time_msm",
        beta = βv,
        tau = tau,
        m = m,
        β_w = Float64(β_w),
        ρ = Float64(ρ),
        σ_y = Float64(σ_y),
        T = T,
    )
    return df, truth
end

export ParametricRepeatedOutcomeMSM, run_parametric_repeated_msm
export simulate_mean_treatment_time_msm
