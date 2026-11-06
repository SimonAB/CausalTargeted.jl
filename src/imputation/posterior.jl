"""Posterior / nested-MC imputation for incomplete outcomes (Observable L2).

Gaussian MAR draws for continuous `Y` given a conditioning set (and optional
treatment) draw both regression coefficients and residual variance under the
normal-inverse-chi-square reference posterior. Draws complete tables without a
single point fill; [`run_lmtp_grid`](@ref) may average estimators across draws.
The model assumes a full-rank Gaussian linear imputation design.
"""

using DataFrames
using Random
using StableRNGs
using Statistics
using LinearAlgebra
using Distributions: Chisq, TDist, quantile

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
2. For each draw, sample residual variance from its scaled inverse-chi-square
   posterior and coefficients from their conditional multivariate normal.
3. Replace missing `Y` with a normal draw at the sampled conditional mean;
   observed `Y` are left unchanged.

Categorical predictors use one schema fitted to the outcome-observed rows for
both model fitting and missing-row prediction. A category absent from that
schema is reported as an unseen-level error.

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
    observed_data = data[obs, :]
    schema = fit_covariate_schema(observed_data, design_cols)
    X_obs = design_matrix(schema, observed_data)
    y_obs = Float64.(data[obs, outcome])
    n_obs, n_parameters = size(X_obs)
    residual_df = n_obs - n_parameters
    residual_df > 0 || throw(ArgumentError(
        "Gaussian MAR imputation needs positive residual degrees of freedom",
    ))
    factor = try
        cholesky(Symmetric(transpose(X_obs) * X_obs))
    catch
        throw(ArgumentError("Gaussian MAR imputation design must have full column rank"))
    end
    β_hat = X_obs \ y_obs
    residuals = y_obs - X_obs * β_hat
    rss = max(sum(abs2, residuals), eps(Float64))

    miss_idx = findall(.!obs)
    X_miss = isempty(miss_idx) ? zeros(0, size(X_obs, 2)) :
        design_matrix(schema, data[miss_idx, :])
    completed = DataFrame[]
    sigma_draws = Float64[]
    for d in 1:n_draws
        σ² = rss / rand(rng, Chisq(residual_df))
        β = β_hat .+ sqrt(σ²) .* (factor.L' \ randn(rng, n_parameters))
        μ_miss = isempty(miss_idx) ? Float64[] : X_miss * β
        df = copy(data)
        y = Vector{Float64}(undef, nrow(df))
        @inbounds for i in 1:nrow(df)
            if obs[i]
                y[i] = Float64(df[i, outcome])
            end
        end
        for (j, i) in enumerate(miss_idx)
            y[i] = μ_miss[j] + sqrt(σ²) * randn(rng)
        end
        df[!, outcome] = y
        push!(completed, df)
        push!(sigma_draws, sqrt(σ²))
    end

    meta = (
        strategy = :posterior_gaussian_mar,
        rung = :L2,
        time_indexed = false,
        n_draws = n_draws,
        n_missing = length(miss_idx),
        residual_sd = sqrt(rss / residual_df),
        posterior_sigma_mean = mean(sigma_draws),
        posterior_sigma_sd = std(sigma_draws; corrected = n_draws > 1),
        residual_df = residual_df,
        parameter_draws = :normal_inverse_chisq,
        predictors = design_cols,
    )
    return ImputationDraws(
        completed, outcome, n_draws, :gaussian_mar, collect(Symbol, preds), meta,
    )
end

"""
    pool_lmtp_grids(grids; rubin=true) -> DataFrame

Average LMTP grid rows across imputation draws. Within-draw `se` and
between-draw variance of `est` combine via Rubin's rule when `rubin=true`,
with finite-imputation t limits; Rubin pooling requires at least two grids.
Input grids must have identical `delta`, `estimand`, and `stratum` keys in the
same row order. Because imputed grids do not retain a joint
influence-curve covariance, simultaneous-band columns are `NaN` rather than
pointwise limits copied into a simultaneous field:

``T = \\bar U + (1 + 1/m) B``, ``\\mathrm{se} = \\sqrt{T}``.
"""
function pool_lmtp_grids(grids::Vector{<:DataFrame}; rubin::Bool = true)
    isempty(grids) && throw(ArgumentError("pool_lmtp_grids requires at least one grid"))
    m = length(grids)
    rubin && m < 2 && throw(ArgumentError(
        "Rubin pooling requires at least two imputation grids",
    ))
    template = grids[1]
    for g in grids
        nrow(g) == nrow(template) || throw(ArgumentError("imputation grids must share row count"))
    end
    key_columns = (:delta, :estimand, :stratum)
    for column in key_columns
        all(g -> column in propertynames(g), grids) || throw(ArgumentError(
            "imputation grids must all contain row-key column :$column",
        ))
    end
    for g in grids[2:end], i in 1:nrow(template), column in key_columns
        isequal(g[i, column], template[i, column]) || throw(ArgumentError(
            "imputation grids must have identical :delta, :estimand, and :stratum keys in the same row order",
        ))
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
            df = B > 0 ? (m - 1) * (1 + Ū / ((1 + 1 / m) * B))^2 : Inf
        else
            se = mean(ses)
            df = Inf
        end
        critical = isfinite(df) ? quantile(TDist(df), 0.975) : quantile(TDist(1e12), 0.975)
        lwr, upr = Q̄ - critical * se, Q̄ + critical * se
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
            lwr_sim = NaN,
            upr_sim = NaN,
            crit_sim = NaN,
        )
        push!(rows, row)
    end
    return DataFrame(rows)
end

export ImputationDraws, impute_posterior, pool_lmtp_grids
