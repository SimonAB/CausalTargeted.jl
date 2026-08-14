"""Cross-fitted plug-in g-computation estimator with bootstrap confidence intervals.

G-computation estimates `E[Y^{d(A)}]` by fitting an outcome model `Q(A, W)` on
training folds, predicting `Q(d(A), W)` on test folds, and averaging. This is the
plug-in (no targeting) counterpart of TMLE — simpler but relies more heavily on
correct outcome model specification.

Bootstrap CIs use the percentile method over `n_boot` **refitting** resamples
(outcome model and folds recomputed each draw). Set `n_boot = 0` for a fast
influence-function SE with normal-based Wald intervals.

# References

- Robins (1986) — g-computation formula
- Snowden, Rose & Mortimer (2011) — implementation of g-computation
"""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    _gcomp_crossfit_psi(df, treatment, outcome; ...) -> (psi, estimate)

Cross-fitted unit-level plug-in contrasts and their IPCW-weighted mean.
`df` must already be cleaned (no `Missing` in outcome / covariates).
"""
function _gcomp_crossfit_psi(
    df::DataFrame,
    treatment::Symbol,
    outcome::Symbol,
    adj_covars::Vector{Symbol},
    ipcw_w::AbstractVector{<:Real};
    delta::Union{Nothing, Float64} = nothing,
    folds::Int = 3,
    learners = DEFAULT_SL_LEARNERS,
    rng::AbstractRNG = StableRNG(1),
)
    n = nrow(df)
    length(ipcw_w) == n || throw(ArgumentError("ipcw_w length must match nrow(df)"))
    y = Float64.(df[!, outcome])
    a = Float64.(df[!, treatment])
    binary_a = all(x -> x == 0.0 || x == 1.0, a)
    covariate_schema = fit_covariate_schema(df, adj_covars)

    psi = zeros(n)
    fold_sets = crossfit_indices(n, folds, rng)
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        y_tr = y[train_idx]

        sl = _fit_sl_outcome(
            train, adj_covars, y_tr;
            treatment = treatment,
            learners = learners,
            rng = rng,
            schema = covariate_schema,
        )

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

    return psi, transport_weighted_mean(psi, ipcw_w)
end

"""
    run_gcomp(df, treatment, outcome; covariates, delta, folds, learners, rng, n_boot)
        -> NamedTuple

Cross-fitted plug-in g-computation for a binary or continuous treatment.

For binary `treatment`: estimates `ATE = E[Q(1, W) - Q(0, W)]`.
For continuous `treatment` with `delta`: estimates `E[Q(A + δ, W) - Q(A, W)]`
with `A + δ` clamped to the 1%–99% range of observed `A`.

`n_boot > 0` (default 200): percentile CI from a **refitting** bootstrap that
redraws rows and recomputes the cross-fitted plug-in (and carries IPCW weights
with the resampled rows). `n_boot = 0`: influence-function SE via
[`weighted_influence_summary`](@ref) and normal Wald CI.

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
    validate_contrast_learners(learners; context = "run_gcomp")
    n_boot >= 0 || throw(ArgumentError("n_boot must be ≥ 0 (0 = influence-function SE)"))
    all_cols = vcat(covariates, [treatment])
    df_clean, ipcw_w, extra_cols = handle_missing_data(df, outcome, all_cols, handle_missing; rng = rng)
    adj_covars = vcat(covariates, extra_cols)

    n = nrow(df_clean)
    psi, estimate = _gcomp_crossfit_psi(
        df_clean, treatment, outcome, adj_covars, ipcw_w;
        delta = delta, folds = folds, learners = learners, rng = rng,
    )

    if n_boot == 0
        s = weighted_influence_summary(psi, ipcw_w)
        z = 1.96
        return (
            estimate = estimate,
            se = s.se,
            ci_lower = estimate - z * s.se,
            ci_upper = estimate + z * s.se,
            n = n,
        )
    end

    boot_estimates = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        boot_idx = rand(rng, 1:n, n)
        boot_df = df_clean[boot_idx, :]
        boot_w = ipcw_w[boot_idx]
        _, boot_estimates[b] = _gcomp_crossfit_psi(
            boot_df, treatment, outcome, adj_covars, boot_w;
            delta = delta, folds = folds, learners = learners, rng = rng,
        )
    end
    se = std(boot_estimates)
    ci_lower = quantile(boot_estimates, 0.025)
    ci_upper = quantile(boot_estimates, 0.975)

    return (estimate = estimate, se = se, ci_lower = ci_lower, ci_upper = ci_upper, n = n)
end

export run_gcomp
