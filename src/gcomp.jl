"""Cross-fitted plug-in g-computation estimator with bootstrap confidence intervals.

G-computation estimates `E[Y^{d(A)}]` by fitting an outcome model `Q(A, W)` on
training folds, predicting `Q(d(A), W)` on test folds, and averaging. This is the
plug-in (no targeting) counterpart of TMLE — simpler but relies more heavily on
correct outcome model specification.

Bootstrap CIs use the percentile method over `n_boot` resamples.

# References

- Robins (1986) — g-computation formula
- Snowden, Rose & Mortimer (2011) — implementation of g-computation
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    run_gcomp(df, treatment, outcome; covariates, delta, folds, learners, rng, n_boot)
        -> NamedTuple

Cross-fitted plug-in g-computation for a binary or continuous treatment.

For binary `treatment`: estimates `ATE = E[Q(1, W) - Q(0, W)]`.
For continuous `treatment` with `delta`: estimates `E[Q(A + δ, W) - Q(A, W)]`.

Returns `(estimate, se, ci_lower, ci_upper, n)`.
"""
function run_gcomp(
    df::DataFrame,
    treatment::Symbol,
    outcome::Symbol;
    covariates::Vector{Symbol} = Symbol[],
    delta::Union{Nothing, Float64} = nothing,
    folds::Int = 3,
    learners = DEFAULT_SL_LEARNERS,
    rng::AbstractRNG = StableRNG(1),
    n_boot::Int = 200,
    handle_missing::Symbol = :drop,
)
    # Handle missing data
    all_cols = vcat(covariates, [treatment])
    df_clean, _, extra_cols = handle_missing_data(df, outcome, all_cols, handle_missing; rng = rng)
    adj_covars = vcat(covariates, extra_cols)

    n = nrow(df_clean)
    y = Float64.(df_clean[!, outcome])
    a = Float64.(df_clean[!, treatment])
    adjust = unique(vcat(adj_covars, [treatment]))
    binary_a = all(x -> x == 0.0 || x == 1.0, a)

    # Cross-fitted plugin predictions
    psi = zeros(n)
    fold_sets = crossfit_indices(n, folds, rng)
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df_clean[train_idx, :]
        test = df_clean[test_idx, :]
        y_tr = y[train_idx]

        sl = _fit_sl_outcome(train, adj_covars, y_tr; treatment = treatment, learners = learners, rng = rng)

        if binary_a
            Q1 = _predict_sl(sl, test, adj_covars; treatment = treatment,
                             treatment_values = ones(length(test_idx)))
            Q0 = _predict_sl(sl, test, adj_covars; treatment = treatment,
                             treatment_values = zeros(length(test_idx)))
            psi[test_idx] = Q1 .- Q0
        else
            δ = something(delta, std(a))
            Lq, Uq = exposure_bounds(a, 0.01, 0.99)
            a_te = a[test_idx]
            a_shifted = clamp.(a_te .+ δ, Lq, Uq)
            Q1 = _predict_sl(sl, test, adj_covars; treatment = treatment,
                             treatment_values = a_shifted)
            Q0 = _predict_sl(sl, test, adj_covars; treatment = treatment,
                             treatment_values = a_te)
            psi[test_idx] = Q1 .- Q0
        end
    end

    estimate = mean(psi)

    # Bootstrap CIs
    boot_estimates = Float64[]
    for b in 1:n_boot
        boot_idx = rand(rng, 1:n, n)
        push!(boot_estimates, mean(psi[boot_idx]))
    end
    se = std(boot_estimates)
    ci_lower = quantile(boot_estimates, 0.025)
    ci_upper = quantile(boot_estimates, 0.975)

    return (estimate = estimate, se = se, ci_lower = ci_lower, ci_upper = ci_upper, n = n)
end

export run_gcomp
