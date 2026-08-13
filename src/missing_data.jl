"""Missing data handling: IPCW for outcome missingness and mean-imputation for covariates.

Follows the tlverse convention: model P(observed | W) via SuperLearner, then
reweight estimating equations by stabilised inverse-probability-of-censoring weights.
Covariate missingness is handled by column-wise mean/mode imputation with added
missingness indicator columns.

# References

- van der Laan & Rose (2011) — IPCW-TMLE
- Weberpals et al. (2024) — bias–coverage trade-offs for missing-data methods in TMLE
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    impute_covariates_mean!(df, cols) -> Vector{Symbol}

Replace `missing` values in each column of `cols` with the column mean (continuous)
or mode (categorical). Adds binary indicator columns `col_miss` for each imputed
column. Returns the names of the added indicator columns.

Modifies `df` in place.
"""
function impute_covariates_mean!(df::DataFrame, cols::Vector{Symbol})
    indicators = Symbol[]
    for c in cols
        col = df[!, c]
        miss_mask = ismissing.(col)
        any(miss_mask) || continue

        ind_name = Symbol(string(c) * "_miss")
        df[!, ind_name] = Float64.(miss_mask)
        push!(indicators, ind_name)

        non_miss = skipmissing(col)
        if eltype(non_miss) <: Number
            fill_val = mean(non_miss)
            df[!, c] = coalesce.(col, fill_val)
        else
            # Mode for non-numeric
            counts = Dict{Any,Int}()
            for v in non_miss
                counts[v] = get(counts, v, 0) + 1
            end
            isempty(counts) && throw(ArgumentError(
                "cannot impute covariate :$c because it has no observed values",
            ))
            fill_val = first(sort(collect(counts), by = x -> -x[2]))[1]
            df[!, c] = coalesce.(col, fill_val)
        end
    end
    return indicators
end

"""
    ipcw_weights(df, outcome, covariates; learners, folds, rng) -> Vector{Float64}

Fit P(R=1 | W) where R = !ismissing(outcome), and return stabilised
inverse-probability-of-censoring weights `P(R=1) / P(R=1 | W)` for observed units.
Missing-outcome units receive weight 0.

Uses the existing SuperLearner for the censoring model.
"""
function ipcw_weights(
    df::DataFrame,
    outcome::Symbol,
    covariates::Vector{Symbol};
    learners = (:logistic, :mean),
    folds::Int = 3,
    rng::AbstractRNG = StableRNG(1),
)
    n = nrow(df)
    R = Float64.(.!ismissing.(df[!, outcome]))
    marginal_p = mean(R)

    if marginal_p >= 1.0 - 1e-10
        return ones(n)
    end

    X = design_matrix(df, covariates)
    sl = fit_super_learner(X, R; learners = learners, family = :binomial,
                           metalearner = :invmse, rng = rng)
    p_obs = clamp.(predict_super_learner(sl, X), 1e-3, 1.0 - 1e-3)
    w = marginal_p ./ p_obs
    w[R .== 0.0] .= 0.0
    return w
end

"""
    handle_missing_data(df, outcome, covariates, strategy; learners, folds, rng)
        -> (df_clean, weights, extra_cols)

Dispatch missing-data handling according to `strategy`:

- `:drop` — complete-case analysis (default, preserves current behaviour)
- `:ipcw` — IPCW for outcome missingness, drop covariate-missing rows
- `:impute` — mean-impute covariates (with indicators), drop outcome-missing rows
- `:ipcw_impute` — both: impute covariates, then IPCW for outcome missingness
"""
function handle_missing_data(
    df::DataFrame,
    outcome::Symbol,
    covariates::Vector{Symbol},
    strategy::Symbol;
    learners = (:logistic, :mean),
    folds::Int = 3,
    rng::AbstractRNG = StableRNG(1),
)
    n = nrow(df)
    extra_cols = Symbol[]

    if strategy == :drop
        df_clean = dropmissing(df, vcat([outcome], covariates))
        return df_clean, ones(nrow(df_clean)), extra_cols

    elseif strategy == :ipcw
        df_cc = dropmissing(df, covariates)
        w = ipcw_weights(df_cc, outcome, covariates; learners = learners, folds = folds, rng = rng)
        obs_mask = .!ismissing.(df_cc[!, outcome])
        df_clean = df_cc[obs_mask, :]
        w_clean = w[obs_mask]
        return df_clean, w_clean, extra_cols

    elseif strategy == :impute
        df_copy = copy(df)
        indicators = impute_covariates_mean!(df_copy, covariates)
        extra_cols = indicators
        df_clean = dropmissing(df_copy, [outcome])
        return df_clean, ones(nrow(df_clean)), extra_cols

    elseif strategy == :ipcw_impute
        df_copy = copy(df)
        indicators = impute_covariates_mean!(df_copy, covariates)
        extra_cols = indicators
        all_covars = vcat(covariates, indicators)
        w = ipcw_weights(df_copy, outcome, all_covars; learners = learners, folds = folds, rng = rng)
        obs_mask = .!ismissing.(df_copy[!, outcome])
        df_clean = df_copy[obs_mask, :]
        w_clean = w[obs_mask]
        return df_clean, w_clean, extra_cols
    else
        error("Unknown missing-data strategy: $strategy. Use :drop, :ipcw, :impute, or :ipcw_impute.")
    end
end

export impute_covariates_mean!, ipcw_weights, handle_missing_data
