"""TMLE targeting diagnostics (score norms, AD-assisted fluctuation checks)."""

using Statistics

"""
    tmle_score_diagnostics(components; λ=1.0) -> NamedTuple

Compute targeting equation diagnostics from shared-fold LMTP components.

Returns score norms for policy 1 and 0, relative to `√n`, plus a combined
score that should be near zero after successful TMLE.
"""
function tmle_score_diagnostics(
    components::NamedTuple;
    λ::Real = 1.0,
)
    resid = components.y .- components.Q_obs
    s1 = sum(λ .* components.H1 .* resid)
    s0 = sum(λ .* components.H0 .* resid)
    n = components.n
    rn = sqrt(n)
    return (
        score_policy1 = s1,
        score_policy0 = s0,
        score_norm_policy1 = abs(s1) / rn,
        score_norm_policy0 = abs(s0) / rn,
        score_combined_norm = sqrt(s1^2 + s0^2) / rn,
        mean_abs_residual = mean(abs.(resid)),
        mean_h1 = mean(components.H1),
        mean_h0 = mean(components.H0),
        targeting_ok = abs(s1) / rn < 0.05 && abs(s0) / rn < 0.05,
    )
end

"""
    tmle_fluctuation_objective(ε, Q1, Q0, resid, H1, H0; λ) -> Float64

Sum of squared targeting scores after a joint ε fluctuation on both policies.
Used for AD-assisted one-dimensional root finding.
"""
function tmle_fluctuation_objective(
    ε::Real,
    Q1::AbstractVector{<:Real},
    Q0::AbstractVector{<:Real},
    resid::AbstractVector{<:Real},
    H1::AbstractVector{<:Real},
    H0::AbstractVector{<:Real};
    λ::Real = 1.0,
)
    r = copy(resid)
    r .-= λ .* ε .* (H1 .+ H0)
    s1 = sum(λ .* H1 .* r)
    s0 = sum(λ .* H0 .* r)
    return s1^2 + s0^2
end

"""
    optimise_tmle_fluctuation(Q1, Q0, resid, H1, H0; λ, ε_bounds) -> NamedTuple

Find ε minimising targeting scores via ForwardDiff gradient descent on the
one-dimensional fluctuation objective (diagnostic / polish helper).
"""
function optimise_tmle_fluctuation(
    Q1::AbstractVector{<:Real},
    Q0::AbstractVector{<:Real},
    resid::AbstractVector{<:Real},
    H1::AbstractVector{<:Real},
    H0::AbstractVector{<:Real};
    λ::Real = 1.0,
    ε_bounds::Tuple{Real, Real} = (-2.0, 2.0),
)
    f(ε) = tmle_fluctuation_objective(ε, Q1, Q0, resid, H1, H0; λ = λ)
    ε_grid = collect(range(ε_bounds[1], ε_bounds[2]; length = 41))
    scores = [f(ε) for ε in ε_grid]
    ε = ε_grid[argmin(scores)]
    Q1_opt = Q1 .+ λ .* ε .* H1
    Q0_opt = Q0 .+ λ .* ε .* H0
    resid_opt = resid .- λ .* ε .* (H1 .+ H0)
    diag = tmle_score_diagnostics((
        y = Q1_opt .+ resid_opt,
        Q_obs = Q1_opt,
        Q1 = Q1_opt,
        Q0 = Q0_opt,
        H1 = H1,
        H0 = H0,
        n = length(Q1),
    ); λ = λ)
    return (
        epsilon = ε,
        psi1 = mean(Q1_opt),
        psi0 = mean(Q0_opt),
        estimate = mean(Q1_opt) - mean(Q0_opt),
        diagnostics = diag,
    )
end

"""
    diagnostics_dict(diag) -> Dict{String, Any}
"""
function diagnostics_dict(diag::NamedTuple)
    return Dict{String, Any}(
        "diag_score_norm_p1" => diag.score_norm_policy1,
        "diag_score_norm_p0" => diag.score_norm_policy0,
        "diag_score_combined" => diag.score_combined_norm,
        "diag_targeting_ok" => diag.targeting_ok,
        "diag_mean_abs_resid" => diag.mean_abs_residual,
    )
end

export tmle_score_diagnostics, tmle_fluctuation_objective, optimise_tmle_fluctuation
export diagnostics_dict
