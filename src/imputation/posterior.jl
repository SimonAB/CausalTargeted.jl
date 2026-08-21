"""Posterior / nested-MC imputation for incomplete outcomes (Observable L2).

Phase 4 ships a Gaussian MAR draw for continuous `Y` given a conditioning set
(and optional treatment). Draws complete tables without inventing a single
point fill; [`run_lmtp_grid`](@ref) may average estimators across draws.
Turing / RxInfer weakdeps are deferred; this path uses GLM residuals already
in the package.
"""

using DataFrames
using Random
using StableRNGs
using Statistics

"""
    ImputationDraws

Collection of completed tables from [`impute_posterior`](@ref).

# Fields
- `draws`: length-`n_draws` completed `DataFrame`s (no `missing` in `outcome`)
- `outcome`: imputed column
- `n_draws`, `method`, `mar_set`, `meta`
"""
struct ImputationDraws
    draws::Vector{DataFrame}
    outcome::Symbol
    n_draws::Int
    method::Symbol
    mar_set::Vector{Symbol}
    meta::NamedTuple
end

function _certificate_mar_set(certificate)
    certificate === nothing && return nothing
    if hasproperty(certificate, :missingness)
        miss = certificate.missingness
        miss === nothing && return nothing
        return miss
    end
    return certificate
end

function _assert_certificate_allows_imputation(certificate)
    cert = _certificate_mar_set(certificate)
    cert === nothing && return nothing
    if hasproperty(cert, :identifiable) && !cert.identifiable
        throw(ArgumentError(
            "missingness certificate is unidentified (status=$(cert.status)); " *
            "refusing posterior imputation under this claim",
        ))
    end
    return hasproperty(cert, :mar_set) ? collect(Symbol, cert.mar_set) : nothing
end

"""
    impute_posterior(data, outcome, covariates; treatment, certificate, n_draws, rng)
        -> ImputationDraws

Draw completed outcomes under a **Gaussian MAR** model:

1. Fit `Y ~ covariates (+ treatment)` by OLS on rows with observed `Y`.
2. For each draw, replace missing `Y` with ``N(\\hat\\mu(x), \\hat\\sigma^2)``;
   observed `Y` are left unchanged.

`covariates` should be the MAR conditioning set (or pass `certificate` /
`IdentificationResult` with `missingness` so `mar_set` is taken from the
certificate when nonempty). Does not default into estimators — opt in via
`run_lmtp_grid(...; imputation=draws)`.
"""
function impute_posterior(
    data::DataFrame,
    outcome::Symbol,
    covariates::Vector{Symbol};
    treatment::Union{Nothing, Symbol} = nothing,
    certificate = nothing,
    n_draws::Int = 20,
    rng::AbstractRNG = StableRNG(1),
)
    n_draws >= 1 || throw(ArgumentError("n_draws must be ≥ 1"))
    hasproperty(data, outcome) || throw(ArgumentError("outcome :$outcome not in data"))
    cert_mar = _assert_certificate_allows_imputation(certificate)
    preds = if cert_mar !== nothing && !isempty(cert_mar)
        unique(cert_mar)
    else
        unique(covariates)
    end
    for c in preds
        hasproperty(data, c) || throw(ArgumentError("covariate :$c not in data"))
        any(ismissing, data[!, c]) && throw(ArgumentError(
            "covariate :$c contains missing; impute or drop covariates before outcome draws",
        ))
    end
    if treatment !== nothing
        hasproperty(data, treatment) || throw(ArgumentError("treatment :$treatment not in data"))
        any(ismissing, data[!, treatment]) && throw(ArgumentError(
            "treatment :$treatment contains missing",
        ))
    end

    obs = .!ismissing.(data[!, outcome])
    count(obs) >= 2 || throw(ArgumentError(
        "need at least 2 observed :$outcome rows to fit the imputation model",
    ))
    design_cols = treatment === nothing ? preds : unique(vcat(preds, [treatment]))
    X_obs = design_matrix(data[obs, :], design_cols)
    y_obs = Float64.(data[obs, outcome])
    model = _fit_glm_safe(X_obs, y_obs)
    μ_obs = _predict_glm(model, X_obs)
    σ = max(std(y_obs .- μ_obs; corrected = true), 1e-6)

    miss_idx = findall(.!obs)
    X_miss = isempty(miss_idx) ? zeros(0, size(X_obs, 2)) :
        design_matrix(data[miss_idx, :], design_cols)
    μ_miss = isempty(miss_idx) ? Float64[] : _predict_glm(model, X_miss)

    completed = DataFrame[]
    for d in 1:n_draws
        df = copy(data)
        y = Vector{Float64}(undef, nrow(df))
        @inbounds for i in 1:nrow(df)
            if obs[i]
                y[i] = Float64(df[i, outcome])
            end
        end
        for (j, i) in enumerate(miss_idx)
            y[i] = μ_miss[j] + σ * randn(rng)
        end
        df[!, outcome] = y
        push!(completed, df)
    end

    meta = (
        strategy = :posterior_gaussian_mar,
        rung = :L2,
        time_indexed = false,
        n_draws = n_draws,
        n_missing = length(miss_idx),
        residual_sd = σ,
        predictors = design_cols,
    )
    return ImputationDraws(
        completed, outcome, n_draws, :gaussian_mar, collect(Symbol, preds), meta,
    )
end

"""
    pool_lmtp_grids(grids; rubin=true) -> DataFrame

Average LMTP grid rows across imputation draws. Within-draw `se` and
between-draw variance of `est` combine via Rubin's rule when `rubin=true`:

``T = \\bar U + (1 + 1/m) B``, ``\\mathrm{se} = \\sqrt{T}``.
"""
function pool_lmtp_grids(grids::Vector{<:DataFrame}; rubin::Bool = true)
    isempty(grids) && throw(ArgumentError("pool_lmtp_grids requires at least one grid"))
    m = length(grids)
    template = grids[1]
    for g in grids
        nrow(g) == nrow(template) || throw(ArgumentError("imputation grids must share row count"))
    end
    rows = NamedTuple[]
    for i in 1:nrow(template)
        ests = Float64[grids[k].est[i] for k in 1:m]
        ses = Float64[grids[k].se[i] for k in 1:m]
        Q̄ = mean(ests)
        if rubin && m > 1
            Ū = mean(abs2, ses)
            B = var(ests; corrected = true)
            T = Ū + (1 + 1 / m) * B
            se = sqrt(max(T, 0.0))
        else
            se = mean(ses)
        end
        lwr, upr = wald_ci(Q̄, se)
        row = (
            delta = Float64(template.delta[i]),
            estimand = string(template.estimand[i]),
            est = Q̄,
            se = se,
            lwr = lwr,
            upr = upr,
            clamp = Float64(template.clamp[i]),
            severity = Float64(template.severity[i]),
            effective_shift = Float64(template.effective_shift[i]),
            shift_retention = Float64(template.shift_retention[i]),
            lower_q = Float64(template.lower_q[i]),
            upper_q = Float64(template.upper_q[i]),
            sd_exposure = Float64(template.sd_exposure[i]),
            support_status = string(template.support_status[i]),
            stratum = string(template.stratum[i]),
            lwr_sim = lwr,
            upr_sim = upr,
            crit_sim = NaN,
        )
        push!(rows, row)
    end
    return DataFrame(rows)
end

export ImputationDraws, impute_posterior, pool_lmtp_grids
