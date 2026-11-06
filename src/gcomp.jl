"""Cross-fitted plug-in g-computation estimator with bootstrap confidence intervals.

G-computation estimates `E[Y^{d(A)}]` by fitting an outcome model `Q(A, W)` on
training folds, predicting `Q(d(A), W)` on test folds, and averaging. This is the
plug-in (no targeting) counterpart of TMLE — simpler but relies more heavily on
correct outcome model specification.

Bootstrap CIs use the percentile method over `n_boot` **refitting** resamples
(outcome model and folds recomputed each draw). A plug-in estimate with
`n_boot = 0` has no inferential SE. The explicit binary-treatment
`estimator = :onestep` route adds the outcome-residual correction and uses its
cross-fitted influence curve, without bootstrap refits.

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

"""Cross-fit the binary ATE outcome and propensity models on common folds."""
function _gcomp_binary_onestep(df, treatment, outcome, covariates;
    folds, learners, propensity, rng,
)
    n = nrow(df)
    n >= 4 || throw(ArgumentError("one-step g-computation needs at least four rows"))
    a = Float64.(df[!, treatment])
    all(x -> x == 0 || x == 1, a) || throw(ArgumentError(
        "one-step g-computation currently supports binary 0/1 treatment only",
    ))
    length(unique(a)) == 2 || throw(ArgumentError("both treatment arms are required"))
    p = propensity === nothing ? nothing :
        propensity isa Real ? fill(Float64(propensity), n) : Float64.(propensity)
    if p !== nothing
        length(p) == n || throw(ArgumentError("propensity length must match the data"))
        all(x -> isfinite(x) && 0 < x < 1, p) || throw(ArgumentError(
            "known propensities must lie strictly between zero and one",
        ))
    end
    y = Float64.(df[!, outcome])
    schema = fit_covariate_schema(df, covariates)
    scores, plugin = zeros(n), zeros(n)
    truncated = 0
    propensity_metalearner = :known
    for test_idx in crossfit_indices(n, folds, rng)
        train_idx = setdiff(1:n, test_idx)
        train, test = df[train_idx, :], df[test_idx, :]
        length(unique(a[train_idx])) == 2 || throw(ArgumentError(
            "a training fold lacks a treatment arm; use fewer folds or more observations",
        ))
        fit = _fit_sl_outcome(train, covariates, y[train_idx];
            treatment = treatment, learners = learners, rng = rng, schema = schema)
        q1 = _predict_sl(fit, test, covariates; treatment = treatment,
                         treatment_values = ones(length(test_idx)))
        q0 = _predict_sl(fit, test, covariates; treatment = treatment,
                         treatment_values = zeros(length(test_idx)))
        if p === nothing
            fit_p = fit_super_learner(design_matrix(schema, train), a[train_idx];
                learners = (:logistic, :mean), family = :binomial,
                rng = rng)
            propensity_metalearner = fit_p.metalearner
            raw = predict_super_learner(fit_p, design_matrix(schema, test))
            truncated += count(x -> x < 1e-3 || x > 1 - 1e-3, raw)
            e = clamp.(raw, 1e-3, 1 - 1e-3)
        else
            e = p[test_idx]
        end
        at = a[test_idx]
        plugin[test_idx] = q1 .- q0
        scores[test_idx] = q1 .- q0 .+ at ./ e .* (y[test_idx] .- q1) .-
                           (1 .- at) ./ (1 .- e) .* (y[test_idx] .- q0)
    end
    estimate = mean(scores)
    ic = scores .- estimate
    se = std(ic) / sqrt(n)
    lower, upper = wald_ci(estimate, se)
    return (estimate = estimate, se = se, ci_lower = lower, ci_upper = upper,
        n = n, estimator = :onestep, uncertainty = :influence_function,
        plugin_estimate = mean(plugin), ic = ic,
        propensity_source = p === nothing ? :estimated : :known,
        propensity_metalearner = propensity_metalearner,
        propensity_truncation = truncated / n,
        bootstrap_refits_missingness = false)
end

const _GCOMP_BOOTSTRAP_MISSINGNESS_STRATEGIES = (:ipcw, :impute, :ipcw_impute)

"""Resample original rows and rerun the selected missing-data preprocessing."""
function _gcomp_refit_missingness_bootstrap(
    original_df::DataFrame,
    outcome::Symbol,
    missingness_covariates::Vector{Symbol},
    covariates::Vector{Symbol},
    strategy::Symbol,
    boot_idx::AbstractVector{<:Integer};
    folds::Int,
    rng::AbstractRNG,
)
    strategy in _GCOMP_BOOTSTRAP_MISSINGNESS_STRATEGIES || throw(ArgumentError(
        "missingness bootstrap helper does not support strategy :$strategy",
    ))
    resampled_input = original_df[boot_idx, :]
    miss = handle_missing_data(
        resampled_input, outcome, missingness_covariates, strategy;
        folds = folds, rng = rng,
    )
    data, weights, extra_cols = miss
    return (
        data = data,
        weights = weights,
        covariates = unique(vcat(covariates, extra_cols)),
        n_input = nrow(resampled_input),
        missingness = miss.meta,
    )
end

"""
    run_gcomp(df, treatment, outcome; covariates, delta, folds, learners, rng, n_boot)
        -> NamedTuple

Cross-fitted plug-in g-computation for a binary or continuous treatment.

For binary `treatment`: estimates `ATE = E[Q(1, W) - Q(0, W)]`.
For continuous `treatment` with `delta`: estimates `E[Q(A + δ, W) - Q(A, W)]`
with `A + δ` clamped to the 1%–99% range of observed `A`.

`estimator=:plugin` (default), `n_boot > 0` (default 200): percentile CI from a **refitting** bootstrap that
redraws rows and recomputes the cross-fitted plug-in. For `:ipcw`, `:impute`,
and `:ipcw_impute`, each draw resamples the original incomplete rows and reruns
missing-data preprocessing, including imputation and response-model fitting;
`:drop` resamples the complete-case analysis rows. `n_boot = 0` returns `NaN` inferential SE/limits and
`uncertainty=:unavailable`; `prediction_se` describes fitted-contrast variation
only, excluding outcome-model estimation uncertainty.

`estimator=:onestep` estimates a binary 0/1 ATE using cross-fitted AIPW scores
and their normal-Wald interval, with no bootstrap by default. Optional
`propensity` is a known scalar or row-aligned vector of treatment probabilities;
otherwise a cross-fitted logistic Super Learner is fitted. This route requires
complete data and independent rows; continuous clamped shifts and missing-data
corrections need different scores. Propensity truncation is reported and can
introduce bias. Cross-fitting does not remove nuisance-rate or overlap assumptions.

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
    n_boot::Union{Nothing, Int} = nothing,
    handle_missing::Symbol = :drop,
    estimator::Symbol = :plugin,
    propensity = nothing,
)
    validate_contrast_learners(learners; context = "run_gcomp")
    estimator in (:plugin, :onestep) || throw(ArgumentError("estimator must be :plugin or :onestep"))
    n_boot = something(n_boot, estimator === :onestep ? 0 : 200)
    (n_boot == 0 || n_boot >= 2) || throw(ArgumentError("n_boot must be 0 or at least 2"))
    if estimator === :onestep
        delta === nothing || throw(ArgumentError("one-step binary ATE does not accept a treatment shift delta"))
        n_boot == 0 || throw(ArgumentError("one-step inference uses n_boot=0; use plug-in bootstrap as a comparator"))
        columns = unique(vcat(covariates, [treatment, outcome]))
        any(c -> any(ismissing, df[!, c]), columns) && throw(ArgumentError(
            "one-step g-computation requires complete analysis columns; missingness needs a joint score",
        ))
        handle_missing === :drop || throw(ArgumentError("one-step g-computation requires handle_missing=:drop"))
        return _gcomp_binary_onestep(df, treatment, outcome, covariates;
            folds = folds, learners = learners, propensity = propensity, rng = rng)
    end
    propensity === nothing || throw(ArgumentError("propensity is used only by estimator=:onestep"))
    all_cols = vcat(covariates, [treatment])
    miss = handle_missing_data(
        df, outcome, all_cols, handle_missing;
        rng = rng, rung = :L2, time_indexed = false,
    )
    df_clean, ipcw_w, extra_cols = miss
    adj_covars = vcat(covariates, extra_cols)

    n = nrow(df_clean)
    psi, estimate = _gcomp_crossfit_psi(
        df_clean, treatment, outcome, adj_covars, ipcw_w;
        delta = delta, folds = folds, learners = learners, rng = rng,
    )

    if n_boot == 0
        s = weighted_influence_summary(psi, ipcw_w)
        return with_missingness((
            estimate = estimate,
            se = NaN,
            ci_lower = NaN,
            ci_upper = NaN,
            n = n,
            estimator = :plugin,
            uncertainty = :unavailable,
            prediction_se = s.se,
            bootstrap_refits_missingness = false,
        ), miss.meta)
    end

    bootstrap_refits_missingness = handle_missing in _GCOMP_BOOTSTRAP_MISSINGNESS_STRATEGIES
    boot_estimates = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        if bootstrap_refits_missingness
            boot_idx = rand(rng, 1:nrow(df), nrow(df))
            bootstrap = _gcomp_refit_missingness_bootstrap(
                df, outcome, all_cols, covariates, handle_missing, boot_idx;
                folds = folds, rng = rng,
            )
            boot_df = bootstrap.data
            boot_w = bootstrap.weights
            boot_covars = bootstrap.covariates
        else
            boot_idx = rand(rng, 1:n, n)
            boot_df = df_clean[boot_idx, :]
            boot_w = ipcw_w[boot_idx]
            boot_covars = adj_covars
        end
        _, boot_estimates[b] = _gcomp_crossfit_psi(
            boot_df, treatment, outcome, boot_covars, boot_w;
            delta = delta, folds = folds, learners = learners, rng = rng,
        )
    end
    se = std(boot_estimates)
    ci_lower = quantile(boot_estimates, 0.025)
    ci_upper = quantile(boot_estimates, 0.975)

    return with_missingness((
        estimate = estimate, se = se, ci_lower = ci_lower, ci_upper = ci_upper, n = n,
        estimator = :plugin, uncertainty = :refitting_bootstrap,
        bootstrap_refits_missingness = bootstrap_refits_missingness,
    ), miss.meta)
end

export run_gcomp
