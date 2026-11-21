"""SuperLearner-style nuisance models for MTP estimators.

Default metalearner is cross-validated nonnegative least squares (`:nnls`),
with package-specific simplex normalisation and a squared-loss safeguard; it is
not equivalent to R `method.NNLS` or sl3 `Lrnr_nnls`. Binary outcomes default
to `:nnloglik` (R `method.NNloglik`) at `fit_super_learner`. The discrete
Super Learner (Phillips dSL / sl3 `Lrnr_cv_selector`) is `:cv_selector`.
"""

using DataFrames
using Statistics
using GLM
using Distributions
using Random
using StableRNGs
using Logging

const DEFAULT_SL_LEARNERS = (:glm, :mean)
const COUNT_SL_LEARNERS = (:zeroinflated_nb, :glm_nb, :glm_poisson, :mean)
const SMALL_COUNT_SL_LEARNERS = (:glm_nb, :zeroinflated_nb, :mean)
const DEFAULT_NB_THETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
const COUNT_OUTCOME_FAMILIES = (:poisson, :negbin, :zeroinflated_nb)
const RICH_SL_LEARNERS = (
    :glm,
    :glm_interact,
    :glm_quad,
    :glmnet,
    :glmnet_lasso,
    :glmnet_ridge,
    :randomforest,
    :evotree,
    :evotree_deep,
    :mean,
)
const NNLOGLIK_TRIM = 1e-5

"""
    validate_contrast_learners(learners; context) -> nothing

Require at least one treatment-dependent Super Learner candidate. `:mean` alone
predicts a constant and cannot identify binary or shift contrasts.
"""
function validate_contrast_learners(learners; context::AbstractString = "contrast estimation")
    candidate_keys = map(_candidate_key, learners)
    if length(candidate_keys) == 1 && only(candidate_keys) === :mean
        throw(ArgumentError(
            "learners=(:mean,) cannot identify treatment contrasts; include at least " *
            "one treatment-dependent learner (e.g. :glm). Context: $context",
        ))
    end
    return nothing
end

"""
    SuperLearnerFit

Typed SuperLearner ensemble: candidate fits, metalearner weights, library names,
and outcome `family` (`:gaussian`, `:binomial`, `:multinomial`, `:poisson`, or
`:negbin`).
"""
struct SuperLearnerFit
    fits::Dict{Symbol, Any}
    weights::Vector{Float64}
    learners::Vector{Symbol}
    metalearner::Symbol
    family::Symbol
    levels::Vector{Any}
end

SuperLearnerFit(
    fits::Dict{Symbol, Any},
    weights::AbstractVector{<:Real},
    learners::AbstractVector{Symbol},
    metalearner::Symbol,
) = SuperLearnerFit(fits, collect(Float64, weights), collect(Symbol, learners),
                     metalearner, :gaussian, Any[])

const NNLS_METALEARNERS = (:nnls, :discrete)
const CV_SELECTOR_METALEARNERS = (:cv_selector, :winner)
const _DISCRETE_METALEARNER_WARNED = Ref(false)

"""Default metalearner for an outcome family (R SuperLearner / sl3)."""
function _default_metalearner(family::Symbol)
    family === :binomial && return :nnloglik
    family === :multinomial && return :nnloglik
    return :nnls
end

"""
    _canonical_metalearner(metalearner; family) -> Symbol

Map aliases: `:discrete` → `:nnls` (deprecated), `:winner` → `:cv_selector`.
"""
function _canonical_metalearner(metalearner::Symbol; family::Symbol = :gaussian)
    if metalearner === :discrete
        if !_DISCRETE_METALEARNER_WARNED[]
            @warn "metalearner=:discrete currently aliases :nnls (cross-validated nonnegative least squares). In CausalTargeted 0.4 it will become the discrete Super Learner (:cv_selector). Pass :nnls or :cv_selector explicitly."
            _DISCRETE_METALEARNER_WARNED[] = true
        end
        return :nnls
    elseif metalearner === :winner
        return :cv_selector
    end
    return metalearner
end

"""
    design_matrix(schema, df; treatment=nothing, treatment_values=nothing) -> Matrix{Float64}

Build `[intercept | optional numeric treatment | encoded covariates]` using a
fitted [`CovariateSchema`](@ref). Treatment is never encoded by the schema.
"""
function design_matrix(
    schema::CovariateSchema,
    df::AbstractDataFrame;
    treatment::Union{Symbol, Nothing} = nothing,
    treatment_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = nrow(df)
    encoded = transform_covariates(schema, df)
    p = 1 + Int(treatment !== nothing) + size(encoded, 2)
    X = Matrix{Float64}(undef, n, p)
    X[:, 1] .= 1.0
    col = 2
    if treatment !== nothing
        treatment_values === nothing && _validate_requested_columns(df, [treatment])
        values = treatment_values === nothing ? df[!, treatment] : treatment_values
        length(values) == n || throw(DimensionMismatch(
            "treatment_values length $(length(values)) does not match $n rows",
        ))
        any(ismissing, values) && throw(ArgumentError(
            "treatment :$treatment contains missing values; apply the existing missing-data handling first",
        ))
        @inbounds for i in 1:n
            X[i, col] = try
                Float64(values[i])
            catch
                throw(ArgumentError(
                    "treatment :$treatment must be numeric; categorical treatment is not supported",
                ))
            end
        end
        col += 1
    end
    if !isempty(schema.feature_names)
        X[:, col:end] .= encoded
    end
    return X
end

"""
    design_matrix(df, covariates; treatment=nothing, treatment_values=nothing) -> Matrix{Float64}

Convenience form that fits a covariate schema on `df` and immediately encodes
it. As in earlier releases, requested covariates absent from `df` are ignored.
Cross-fitting code should fit once on the cleaned analysis data and call the
schema-aware method for every fold.
"""
function design_matrix(
    df::AbstractDataFrame,
    covariates::AbstractVector{Symbol};
    treatment::Union{Symbol, Nothing} = nothing,
    treatment_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    present = [covariate for covariate in covariates if hasproperty(df, covariate)]
    fitted_schema = fit_covariate_schema(df, present)
    return design_matrix(
        fitted_schema,
        df;
        treatment = treatment,
        treatment_values = treatment_values,
    )
end

"""
    covariate_design_matrix(df, covariates) -> Matrix{Float64}

Intercept plus covariates only (no treatment column). Reused across δ / counterfactual
predictions via [`outcome_design_matrix`](@ref).
"""
function covariate_design_matrix(df::AbstractDataFrame, covariates::AbstractVector{Symbol})
    return design_matrix(df, covariates; treatment = nothing)
end

function covariate_design_matrix(schema::CovariateSchema, df::AbstractDataFrame)
    return design_matrix(schema, df; treatment = nothing)
end

"""
    outcome_design_matrix(W, a) -> Matrix{Float64}

Assemble `[1 | A | covariates]` from a cached covariate design `W` (intercept +
covariates) and a treatment vector `a` of matching length.
"""
function outcome_design_matrix(W::AbstractMatrix{<:Real}, a::AbstractVector{<:Real})
    n, p = size(W)
    length(a) == n || throw(DimensionMismatch(
        "treatment length $(length(a)) does not match design rows $n",
    ))
    X = Matrix{Float64}(undef, n, p + 1)
    @inbounds for i in 1:n
        X[i, 1] = Float64(W[i, 1])
        X[i, 2] = Float64(a[i])
    end
    if p > 1
        @inbounds for j in 2:p
            for i in 1:n
                X[i, j + 1] = Float64(W[i, j])
            end
        end
    end
    return X
end

"""
    sparse_exposure_diagnostic(a; modal_prop=0.5, min_nonmodal=20) -> NamedTuple
"""
function sparse_exposure_diagnostic(
    a::AbstractVector{<:Real};
    modal_prop::Real = 0.5,
    min_nonmodal::Int = 20,
)
    x = collect(skipmissing(Float64.(a)))
    n = length(x)
    isempty(x) && return (sparse = false, modal_prop = 0.0, modal_count = 0, n = n, n_unique = 0)
    counts = Dict{Float64, Int}()
    for v in x
        counts[v] = get(counts, v, 0) + 1
    end
    modal_count = maximum(values(counts))
    prop = modal_count / n
    nonmodal = n - modal_count
    return (
        sparse = prop >= modal_prop && nonmodal <= min_nonmodal,
        modal_prop = prop,
        modal_count = modal_count,
        n = n,
        n_unique = length(counts),
    )
end

include("mtp_learner_adapters.jl")

include("mtp_metalearners.jl")

"""
    TabularSuperLearnerFit

Super Learner fit paired with the fitted covariate schema used to construct its
numeric design matrix.
"""
struct TabularSuperLearnerFit
    fit::SuperLearnerFit
    schema::CovariateSchema
    covariates::Vector{Symbol}
    treatment::Union{Nothing, Symbol}
end

"""
    _fit_sl_outcome(df, cols, y; treatment, learners, rng, schema) -> TabularSuperLearnerFit

Fit a Super Learner on `design_matrix(df, cols; treatment)`. Used by g-computation,
sequential LMTP, and (via CausalMediation) mediation nuisances.
"""
function _fit_sl_outcome(
    df::DataFrame,
    cols::Vector{Symbol},
    y::AbstractVector{<:Real};
    treatment = nothing,
    learners = DEFAULT_SL_LEARNERS,
    rng = StableRNG(1),
    schema::Union{Nothing, CovariateSchema} = nothing,
)
    fitted_schema = schema === nothing ? fit_covariate_schema(df, cols) : schema
    fitted_schema.covariates == cols || throw(ArgumentError(
        "provided schema covariates $(repr(fitted_schema.covariates)) do not match $(repr(cols))",
    ))
    X = design_matrix(fitted_schema, df; treatment = treatment)
    fit = fit_super_learner(X, Float64.(y); learners = learners, rng = rng)
    return TabularSuperLearnerFit(fit, fitted_schema, copy(cols), treatment)
end

"""Predict from a Super Learner fit on a design matrix for `cols`."""
function _predict_sl(
    sl::TabularSuperLearnerFit,
    df::DataFrame,
    cols::Vector{Symbol};
    treatment = nothing,
    treatment_values = nothing,
)
    cols == sl.covariates || throw(ArgumentError(
        "prediction covariates $(repr(cols)) do not match fitted covariates $(repr(sl.covariates))",
    ))
    treatment == sl.treatment || throw(ArgumentError(
        "prediction treatment $(repr(treatment)) does not match fitted treatment $(repr(sl.treatment))",
    ))
    X = design_matrix(
        sl.schema,
        df;
        treatment = treatment,
        treatment_values = treatment_values,
    )
    return predict_super_learner(sl.fit, X)
end

"""
    fit_super_learner(X, y; learners, metalearner, folds, rng, family) -> SuperLearnerFit

Fit candidate learners and combine with a metalearner.

- `:nnls` — cross-validated nonnegative least squares, normalised to simplex
  weights with a same-prediction squared-loss check that falls back to the best
  single candidate when normalisation worsens the blend (default for
  `family=:gaussian`). This package-specific version is not equivalent to R
  `SuperLearner::method.NNLS` or sl3 `Lrnr_nnls`.
- `:nnloglik` — nonnegative Bernoulli (or multinomial) log-likelihood on the
  logit / probability simplex (default for `family=:binomial`; R `method.NNloglik`)
- `:cv_selector` — discrete Super Learner: one-hot on the CV-best candidate
  (Phillips dSL / sl3 `Lrnr_cv_selector`). Alias: `:winner`. Candidates may
  include [`nested_sl_candidate`](@ref) ensembles (eSL-inside-dSL).
- `:invmse` — inverse training MSE weights (fast fallback; LMTP classifiers)
- `:discrete` — **deprecated** alias of `:nnls` (will become `:cv_selector` in 0.4)

The NNloglik prediction rule for binary outcomes is
`logistic(sum(w[j] * logit(p[j])))`, not an arithmetic mean of probabilities.
It is independently implemented from the same statistical construction as R
`SuperLearner::method.NNloglik`, which can serve as an external numerical QC
benchmark.

Squared error remains a proper scoring rule for binary probabilities; NNloglik
is an alternative rather than a universally superior criterion. Its stronger
penalty for overconfident errors can be useful when fitted probabilities feed
propensity scores, density ratios, censoring or missingness models, or other
odds-sensitive calculations.
"""
function fit_super_learner(
    X::Matrix{Float64},
    y::AbstractVector;
    learners = DEFAULT_SL_LEARNERS,
    metalearner::Union{Symbol, Nothing} = nothing,
    folds::Int = 3,
    rng = StableRNG(42),
    family::Symbol = :gaussian,
    levels = nothing,
)
    family in (:gaussian, :binomial, :multinomial, :poisson, :negbin, :zeroinflated_nb) || throw(ArgumentError(
        "unknown family $family; expected :gaussian, :binomial, :multinomial, :poisson, :negbin, or :zeroinflated_nb",
    ))
    requested = metalearner === nothing ? _default_metalearner(family) : metalearner
    canon = _canonical_metalearner(requested; family = family)
    canon in (:nnls, :invmse, :nnloglik, :cv_selector) || throw(ArgumentError(
        "unknown metalearner $requested; expected :nnls, :nnloglik, :cv_selector, :invmse, or deprecated :discrete",
    ))
    if family === :multinomial
        return _fit_multinomial_super_learner(
            X, y; learners = learners, metalearner = canon,
            folds = folds, rng = rng, levels = levels,
        )
    end
    yf = collect(Float64, y)
    if canon === :nnloglik
        family === :binomial || throw(ArgumentError(
            "metalearner=:nnloglik requires family=:binomial or family=:multinomial",
        ))
        _validate_binary_outcome(yf)
    end
    n = length(yf)
    candidates = collect(learners)
    keys = [_candidate_key(c) for c in candidates]
    length(unique(keys)) == length(keys) || throw(ArgumentError(
        "Super Learner candidate names must be unique; got $keys",
    ))
    if any(c -> c isa NestedSLCandidate, candidates)
        canon === :cv_selector || throw(ArgumentError(
            "nested_sl_candidate is only valid under metalearner=:cv_selector " *
            "(Phillips eSL-inside-dSL); got $canon",
        ))
    end
    k = length(candidates)
    Z = zeros(n, k)
    use_cv = canon in (:nnls, :nnloglik, :cv_selector) && n >= 2 * folds
    if use_cv
        fold_sets = crossfit_indices(n, folds, rng)
        for test_idx in fold_sets
            train_idx = setdiff(1:n, test_idx)
            Xtr = X[train_idx, :]
            ytr = yf[train_idx]
            Xte = X[test_idx, :]
            for (j, c) in enumerate(candidates)
                m = _fit_any_candidate(c, Xtr, ytr; family = family)
                Z[test_idx, j] = _predict_learner(m, Xte)
            end
        end
    else
        for (j, c) in enumerate(candidates)
            m = _fit_any_candidate(c, X, yf; family = family)
            Z[:, j] = _predict_learner(m, X)
        end
    end
    weights = if canon === :nnloglik
        _nnloglik_weights(Z, yf)
    elseif canon === :cv_selector && use_cv
        _cv_selector_weights(Z, yf)
    elseif canon === :nnls && use_cv
        _nonneg_ls_weights(Z, yf)
    else
        _invmse_weights(Z, yf)
    end
    fits = Dict{Symbol, Any}()
    for (key, c) in zip(keys, candidates)
        fits[key] = _fit_any_candidate(c, X, yf; family = family)
    end
    return SuperLearnerFit(fits, weights, keys, canon, family, Any[])
end

"""
    predict_super_learner(sl, X) -> Vector{Float64}
"""
function predict_super_learner(sl::SuperLearnerFit, X::Matrix{Float64})
    sl.family === :multinomial && return _predict_multinomial_super_learner(sl, X)
    n = size(X, 1)
    if sl.metalearner == :nnloglik
        Z = Matrix{Float64}(undef, n, length(sl.learners))
        for (j, lrn) in enumerate(sl.learners)
            Z[:, j] = _predict_learner(sl.fits[lrn], X)
        end
        return _predict_nnloglik(Z, sl.weights)
    end
    out = zeros(n)
    for (j, lrn) in enumerate(sl.learners)
        out .+= sl.weights[j] .* _predict_learner(sl.fits[lrn], X)
    end
    if sl.family in COUNT_OUTCOME_FAMILIES
        out .= max.(out, 0.0)
    end
    return out
end

# Compat for NamedTuple fits from older call sites / notebooks
function predict_super_learner(sl::NamedTuple, X::Matrix{Float64})
    family = hasproperty(sl, :family) ? sl.family : :gaussian
    levels = hasproperty(sl, :levels) ? collect(Any, sl.levels) : Any[]
    return predict_super_learner(
        SuperLearnerFit(
            sl.fits, collect(Float64, sl.weights), collect(Symbol, sl.learners),
            sl.metalearner, family, levels,
        ),
        X,
    )
end

"""
    _crossfit_super_learner(df, target, covariates, folds, rng; kwargs...) -> Vector{Float64}

Cross-fit a Super Learner for `target` and return out-of-fold predictions.

The covariate schema is fitted once on the full frame; each fold then trains on
its complement and predicts the held-out rows. `treatment`, when given, enters
the design matrix as a numeric column. `treatment_values` substitutes
counterfactual exposures on the test rows only, leaving training on the observed
exposure. `fold_sets` reuses an existing partition instead of drawing a new one,
and `clamp_to` bounds the predictions (used for propensities).
"""
function _crossfit_super_learner(
    df::DataFrame,
    target::AbstractVector{<:Real},
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    treatment::Union{Symbol, Nothing} = nothing,
    treatment_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
    learners = DEFAULT_SL_LEARNERS,
    family::Symbol = :gaussian,
    metalearner::Union{Symbol, Nothing} = nothing,
    clamp_to::Union{Nothing, Tuple{Real, Real}} = nothing,
    fold_sets = nothing,
)
    n = nrow(df)
    y = Float64.(target)
    preds = zeros(n)
    schema = fit_covariate_schema(df, covariates)
    sets = fold_sets === nothing ? crossfit_indices(n, folds, rng) : fold_sets
    for test_idx in sets
        train_idx = setdiff(1:n, test_idx)
        Xtr = design_matrix(schema, df[train_idx, :]; treatment = treatment)
        Xte = design_matrix(
            schema, df[test_idx, :];
            treatment = treatment,
            treatment_values = treatment_values === nothing ?
                nothing : treatment_values[test_idx],
        )
        sl = fit_super_learner(
            Xtr, y[train_idx];
            learners = learners, family = family,
            metalearner = metalearner, rng = rng,
        )
        raw = predict_super_learner(sl, Xte)
        preds[test_idx] = clamp_to === nothing ? raw : clamp.(raw, clamp_to[1], clamp_to[2])
    end
    return preds
end

"""
    crossfit_outcome_predictions(df, outcome, treatment, covariates, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_outcome_predictions(
    df::DataFrame,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    return _crossfit_super_learner(
        df, Float64.(df[!, outcome]), covariates, folds, rng;
        treatment = treatment, learners = learners,
    )
end

"""
    crossfit_predict_outcome(df, outcome, treatment, covariates, treatment_values, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_predict_outcome(
    df::DataFrame,
    outcome::Symbol,
    treatment::Symbol,
    covariates::Vector{Symbol},
    treatment_values::AbstractVector{<:Real},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    return _crossfit_super_learner(
        df, Float64.(df[!, outcome]), covariates, folds, rng;
        treatment = treatment, treatment_values = Float64.(treatment_values),
        learners = learners,
    )
end

"""
    crossfit_treatment_mean(df, treatment, covariates, folds, rng; learners) -> Vector{Float64}
"""
function crossfit_treatment_mean(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    learners = DEFAULT_SL_LEARNERS,
)
    return _crossfit_super_learner(
        df, Float64.(df[!, treatment]), covariates, folds, rng; learners = learners,
    )
end

"""
    crossfit_propensity(df, treatment, covariates, folds, rng; learners, fold_sets) -> Vector{Float64}

Cross-fitted propensity scores, combined with inverse-MSE weights and clamped to
`[1e-3, 1 - 1e-3]`. Pass `fold_sets` to share a partition with other nuisances
fitted in the same estimator.
"""
function crossfit_propensity(
    df::DataFrame,
    treatment::Symbol,
    covariates::Vector{Symbol},
    folds::Int,
    rng;
    learners = (:logistic, :mean),
    fold_sets = nothing,
)
    return _crossfit_super_learner(
        df, Float64.(df[!, treatment]), covariates, folds, rng;
        learners = learners, family = :binomial,
        metalearner = :invmse, clamp_to = (1e-3, 1 - 1e-3),
        fold_sets = fold_sets,
    )
end

"""
    columns_present(df, cols) -> Vector{Symbol}
"""
function columns_present(df::DataFrame, cols)
    return [column for column in cols if hasproperty(df, column)]
end

export DEFAULT_SL_LEARNERS, RICH_SL_LEARNERS
export validate_contrast_learners
export SuperLearnerFit
export design_matrix, covariate_design_matrix, outcome_design_matrix, sparse_exposure_diagnostic
export fit_super_learner, predict_super_learner
export crossfit_outcome_predictions, crossfit_predict_outcome
export crossfit_treatment_mean, crossfit_propensity, columns_present
