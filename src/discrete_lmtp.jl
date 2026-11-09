"""Categorical-treatment LMTP via Díaz–Williams 2n density-ratio classification."""

"""
    DiscreteTreatmentPolicy

Modified treatment policy on a finite exposure. Recode maps, a static
`do(A = a*)`, or a user function `(a, wrow) -> a'` (R `lmtp` `shift`).
`mtp=true` when the rule depends on the natural value of `A`.
"""
struct DiscreteTreatmentPolicy
    levels::Vector{Any}
    recode::Dict{Any, Any}
    static::Any
    shift::Union{Nothing, Function}
    mtp::Bool
end

function DiscreteTreatmentPolicy(;
    levels = Any[],
    recode = Dict{Any, Any}(),
    static = nothing,
    shift = nothing,
    mtp::Bool = true,
)
    return DiscreteTreatmentPolicy(collect(Any, levels), Dict{Any, Any}(recode),
                                   static, shift, mtp)
end

"""
    discrete_recode_policy(recode; levels=nothing, mtp=true) -> DiscreteTreatmentPolicy

Map observed levels onto new levels (e.g. `Dict("high" => "low")`). Unmapped
values are left unchanged.
"""
function discrete_recode_policy(recode; levels = nothing, mtp::Bool = true)
    rec = Dict{Any, Any}(recode)
    lev = levels === nothing ? unique(vcat(collect(keys(rec)), collect(values(rec)))) :
        collect(Any, levels)
    return DiscreteTreatmentPolicy(; levels = lev, recode = rec, mtp = mtp)
end

"""
    discrete_static_policy(value; levels=nothing) -> DiscreteTreatmentPolicy

Static assignment `do(A = value)` (R `lmtp` with `mtp=false`).
"""
function discrete_static_policy(value; levels = nothing)
    lev = levels === nothing ? Any[value] : collect(Any, levels)
    return DiscreteTreatmentPolicy(; levels = lev, static = value, mtp = false)
end

"""
    discrete_shift_policy(f; levels=nothing, mtp=true) -> DiscreteTreatmentPolicy

User function `f(a, wrow) -> a′` analogous to R `shift(data, trt)`.
"""
function discrete_shift_policy(f::Function; levels = nothing, mtp::Bool = true)
    lev = levels === nothing ? Any[] : collect(Any, levels)
    return DiscreteTreatmentPolicy(; levels = lev, shift = f, mtp = mtp)
end

"""
    DiscreteInterventionalMean(trt, outcome, adjustment, policy)

L2 mean under a [`DiscreteTreatmentPolicy`](@ref), contrasted with the natural
policy.
"""
struct DiscreteInterventionalMean <: Estimand
    trt::Symbol
    outcome::Symbol
    adjustment::Vector{Symbol}
    policy::DiscreteTreatmentPolicy
end

estimand_engine(::DiscreteInterventionalMean) = :discrete_lmtp

"""
    _canonical_treatment_levels(a, policy) -> Vector

Union of observed treatment values in `a`, policy-declared `levels`, any `static` level, and keys/values of
`recode`, with duplicates removed and order preserved (sorted when `sort` succeeds). Used for dummy coding and
2n density-ratio training so every policy arm appears in the schema.
"""
function _canonical_treatment_levels(a::AbstractVector, policy::DiscreteTreatmentPolicy)
    observed = collect(unique(a))
    extra = copy(policy.levels)
    if policy.static !== nothing
        push!(extra, policy.static)
    end
    append!(extra, collect(keys(policy.recode)))
    append!(extra, collect(values(policy.recode)))
    merged = Any[]
    seen = Set{Any}()
    for v in vcat(observed, extra)
        v in seen && continue
        push!(merged, v)
        push!(seen, v)
    end
    try
        return sort(merged)
    catch
        return merged
    end
end

"""
    _as_policy_level(value, levels)

Return `value` if it matches an element of `levels` by equality or by `string` equality (so `"1"` and `1`
align); otherwise return `value` unchanged. Used when building treatment frames for consistent factor levels.
"""
function _as_policy_level(value, levels::AbstractVector)
    value in levels && return value
    # Allow integer/string interchange ("1" vs 1) without silent Float64 coercion.
    for lev in levels
        string(lev) == string(value) && return lev
    end
    return value
end

"""
    apply_discrete_policy(a, W, policy) -> Vector

Intervened exposure under `policy`. `W` is an `n × p` numeric matrix whose
rows are passed to a user `shift` function.
"""
function apply_discrete_policy(
    a::AbstractVector,
    W::AbstractMatrix,
    policy::DiscreteTreatmentPolicy,
)
    n = length(a)
    size(W, 1) == n || throw(DimensionMismatch(
        "covariate rows $(size(W, 1)) do not match treatment length $n",
    ))
    out = Vector{Any}(undef, n)
    if policy.shift !== nothing
        @inbounds for i in 1:n
            out[i] = policy.shift(a[i], view(W, i, :))
        end
        return out
    end
    if policy.static !== nothing
        fill!(out, policy.static)
        return out
    end
    @inbounds for i in 1:n
        out[i] = _recode_lookup(policy.recode, a[i])
    end
    return out
end

"""Map `a` through `recode` by key equality or by string-matched keys; if no match, return `a`."""
function _recode_lookup(recode::AbstractDict, a)
    haskey(recode, a) && return recode[a]
    sa = string(a)
    for (k, v) in recode
        string(k) == sa && return v
    end
    return a
end

function apply_discrete_policy(a::AbstractVector, policy::DiscreteTreatmentPolicy)
    n = length(a)
    return apply_discrete_policy(a, zeros(n, 0), policy)
end

"""Row-wise support of a deterministic policy's treatment distribution."""
function _discrete_policy_support(
    a_observed::AbstractVector,
    W::AbstractMatrix{<:Real},
    policy::DiscreteTreatmentPolicy;
    natural_levels::Union{Nothing, AbstractVector} = nothing,
)
    n = length(a_observed)
    size(W, 1) == n || throw(DimensionMismatch(
        "covariate rows $(size(W, 1)) do not match treatment length $n",
    ))
    possible_natural_levels = natural_levels === nothing ?
        _canonical_treatment_levels(a_observed, policy) :
        unique(vcat(collect(natural_levels), collect(unique(a_observed))))
    support = falses(n)
    for i in 1:n
        wrow = view(W, i:i, :)
        for natural_level in possible_natural_levels
            assigned = only(apply_discrete_policy([natural_level], wrow, policy))
            if string(assigned) == string(a_observed[i])
                support[i] = true
                break
            end
        end
    end
    return support
end

"""
    _treatment_frame(a, levels) -> DataFrame

Single-column `DataFrame` `:A` with each entry coerced via `_as_policy_level` for schema fitting.
"""
function _treatment_frame(a::AbstractVector, levels::AbstractVector)
    return DataFrame(A = [_as_policy_level(v, levels) for v in a])
end

"""
    _fit_treatment_schema(a_obs, a_policy, levels) -> CovariateSchema

Stack observed and policy treatment vectors into one frame and fit dummy coding for `:A`, ensuring both laws
share the same level set (`levels` drive `_treatment_frame`).
"""
function _fit_treatment_schema(a_obs, a_policy, levels)
    stacked = vcat(_treatment_frame(a_obs, levels), _treatment_frame(a_policy, levels))
    return fit_covariate_schema(stacked, [:A])
end

"""
    _treatment_dummies(schema, a, levels)

Encode observed or policy `a` through `schema` (dummy columns for `A` via `transform_covariates`).
"""
function _treatment_dummies(schema::CovariateSchema, a, levels)
    return transform_covariates(schema, _treatment_frame(a, levels))
end

"""
    _discrete_ratio_design(schema, a, W, levels) -> Matrix{Float64}

Design matrix for the 2n classifier: intercept, dummy-coded `A` (`_treatment_dummies`), and baseline covariates
`W` (rows aligned with `a`).
"""
function _discrete_ratio_design(schema::CovariateSchema, a, W, levels)
    n = size(W, 1)
    Aenc = _treatment_dummies(schema, a, levels)
    return hcat(ones(n), Aenc, W)
end

"""
    _fit_discrete_density_ratio(a_obs, a_policy, W, levels; learners, rng) -> (fit=sl, schema=...)

Díaz–Williams 2n construction: stack natural and policy arms with label `S ∈ {1,0}`, train a binomial Super
Learner on `[intercept, A dummies, W]`, padding `a_obs`/`a_policy` so every `levels` arm appears in the training
frame (stable cross-fold schemas).
"""
function _fit_discrete_density_ratio(
    a_obs,
    a_policy,
    W::Matrix{Float64},
    levels;
    learners = (:logistic, :mean),
    rng = StableRNG(1),
)
    n = length(a_obs)
    # Pad with canonical levels so cross-fold schemas include unseen arms.
    padded_obs = vcat(collect(a_obs), string.(levels))
    padded_pol = vcat(collect(a_policy), string.(levels))
    schema = _fit_treatment_schema(padded_obs, padded_pol, levels)
    W_tr = vcat(W, W)
    S_tr = vcat(ones(n), zeros(n))
    X_tr = _discrete_ratio_design(
        schema,
        vcat(collect(a_obs), collect(a_policy)),
        W_tr,
        levels,
    )
    sl = fit_super_learner(
        X_tr, S_tr;
        learners = learners,
        family = :binomial,
        metalearner = :invmse,
        rng = rng,
    )
    return (fit = sl, schema = schema)
end

"""
    _discrete_policy_ratio(p_observed; trunc, support) -> Vector{Float64}

Map fitted probabilities that the label is “policy” into inverse-odds weights ``(1-p)/p``, clamped to
`[1/trunc, trunc]`. Optional `support` zeroes weights where the policy arm is not followed (static-arm path).
"""
function _discrete_policy_ratio(
    p_observed::AbstractVector{<:Real};
    trunc::Real = 10.0,
    support::Union{Nothing, AbstractVector{Bool}} = nothing,
    probability_floor::Real = 1e-4,
)
    trunc > 0 || throw(ArgumentError("trunc must be positive"))
    0 < probability_floor < 0.5 || throw(ArgumentError("probability_floor must be in (0, 0.5)"))
    p = clamp.(Float64.(p_observed), probability_floor, 1 - probability_floor)
    raw_ratio = (1 .- p) ./ p
    ratio = isfinite(trunc) ? clamp.(raw_ratio, 1 / trunc, trunc) : raw_ratio
    if support !== nothing
        length(support) == length(ratio) || throw(ArgumentError(
            "support length must match classifier probabilities",
        ))
        ratio[.!support] .= 0.0
    end
    return ratio
end

"""
    _ratio_from_discrete_classifier(clf, a, W, levels; trunc, mtp, a_policy, support_mask) -> Vector{Float64}

Evaluate the fitted 2n classifier on test rows (`a`, `W`) via `_discrete_ratio_design`, predict Pr(policy|`A`,`W`),
and map to Díaz–Williams weights with `_discrete_policy_ratio`. When `mtp` is false and `a_policy` is given (static
contrast), weights are masked to rows where observed `a` equals the assigned policy arm.
"""
function _ratio_from_discrete_classifier(
    clf,
    a,
    W::Matrix{Float64},
    levels;
    trunc::Real = 10.0,
    mtp::Bool = true,
    a_policy = nothing,
    support_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    weight_truncation::Bool = true,
)
    X = _discrete_ratio_design(clf.schema, a, W, levels)
    probability_floor = weight_truncation ? 1e-4 : eps(Float64)
    p = clamp.(predict_super_learner(clf.fit, X), probability_floor, 1 - probability_floor)
    support = support_mask
    if !mtp && a_policy !== nothing
        followed = [string(a[i]) == string(a_policy[i]) for i in eachindex(a)]
        support = support === nothing ? followed : (followed .& support)
    end
    ratio_bound = weight_truncation ? trunc : Inf
    return _discrete_policy_ratio(
        p; trunc = ratio_bound, support = support, probability_floor = probability_floor,
    )
end

function _discrete_inference_status(
    positivity,
    weight_truncation::Bool,
    ipcw::Bool,
    imputed_covariates::Bool,
)
    !positivity.ok && return :positivity_violation
    imputed_covariates && return :unsupported_covariate_imputation
    ipcw && return :unsupported_ipcw
    weight_truncation && return :truncated_weights
    return :available
end

function _discrete_lmtp_fold_sets(
    n::Int,
    folds::Int,
    rng;
    cluster_ids::Union{Nothing, AbstractVector} = nothing,
    fold_groups::Union{Nothing, AbstractVector} = nothing,
)
    cluster_ids === nothing || length(cluster_ids) == n || throw(ArgumentError(
        "cluster IDs must align with the analysis rows",
    ))
    fold_groups === nothing || length(fold_groups) == n || throw(ArgumentError(
        "fold_groups must align with the analysis rows",
    ))
    if cluster_ids !== nothing && fold_groups !== nothing
        for cluster_id in unique(cluster_ids)
            rows = findall(value -> isequal(value, cluster_id), cluster_ids)
            length(unique(fold_groups[rows])) == 1 || throw(ArgumentError(
                "fold_groups must keep each covariance cluster within one fold group",
            ))
        end
    end
    grouping = fold_groups === nothing ? cluster_ids : fold_groups
    return crossfit_indices(n, folds, rng; cluster = grouping)
end

"""
    _discrete_influence_interval(estimate, ic; cluster, alpha) -> NamedTuple

Asymptotic SE and equal-tailed interval from the mean influence curve (`ic`), or a cluster-robust sandwich with
``df = G - 1`` when `cluster` labels are provided (`cluster_robust_variance`).
"""
function _discrete_influence_interval(
    estimate::Real,
    ic::AbstractVector{<:Real};
    cluster::Union{Nothing, AbstractVector} = nothing,
    alpha::Real = 0.05,
)
    n = length(ic)
    n > 0 || throw(ArgumentError("influence curve must be non-empty"))
    if cluster === nothing
        se = n > 1 ? std(ic) / sqrt(n) : 0.0
        critical = quantile(Distributions.Normal(), 1 - alpha / 2)
        return (;
            se, lower = estimate - critical * se, upper = estimate + critical * se,
            df = Inf, n_clusters = n,
        )
    end
    length(cluster) == n || throw(ArgumentError("cluster length must match influence curve"))
    n_clusters = length(unique(cluster))
    n_clusters >= 2 || throw(ArgumentError(
        "cluster inference requires at least two observed clusters",
    ))
    variance = cluster_robust_variance(ic; cluster = cluster) * n_clusters / (n_clusters - 1)
    se = sqrt(max(variance, 0.0))
    df = n_clusters - 1
    critical = quantile(Distributions.TDist(df), 1 - alpha / 2)
    return (;
        se, lower = estimate - critical * se, upper = estimate + critical * se,
        df, n_clusters,
    )
end

"""Shallow copy of `df` with column `trt` replaced by counterfactual assignments `a_cf` (same length as `df`)."""
function _counterfactual_frame(df::DataFrame, trt::Symbol, a_cf)
    out = copy(df)
    out[!, trt] = a_cf
    return out
end

"""
    discrete_positivity(a_obs, a_policy) -> NamedTuple

Levels assigned by the policy that never occur in the observed treatment.
"""
function discrete_positivity(a_obs::AbstractVector, a_policy::AbstractVector)
    observed = Set(string.(a_obs))
    assigned = unique(string.(a_policy))
    missing_support = [lev for lev in assigned if !(lev in observed)]
    return (
        n_observed_levels = length(observed),
        n_policy_levels = length(assigned),
        empty_support = missing_support,
        ok = isempty(missing_support),
    )
end

"""
    _shared_fold_discrete_lmtp(df, trt, outcome, covariates, a_policy, folds, rng; ...) -> NamedTuple

Cross-fitted nuisance components for one discrete policy vs natural `a_ref`: outcome Super Learner `Q_obs`,
counterfactual `Q1` under `a_policy`, reference `Q0` under observed `a`, and treatment odds `H1` from
`_fit_discrete_density_ratio` / `_ratio_from_discrete_classifier`; `H0` is ones. Pads the schema with synthetic
rows so policy levels appear in `outcome_schema`. Optional `cluster`-aligned `fold_sets` preserve cluster integrity.

Returns `n`, `y`, `Q_*`, `H_*`, `trunc`, `discrete_positivity`, `density_ratio`, and outcome `family`.
"""
function _shared_fold_discrete_lmtp(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector,
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    learners_trt = (:logistic, :mean),
    trunc::Real = 10.0,
    weight_truncation::Bool = true,
    mtp::Bool = true,
    fold_sets = nothing,
    cluster_ids::Union{Nothing, AbstractVector} = nothing,
    policy_support::Union{Nothing, AbstractVector{Bool}} = nothing,
)
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = collect(df[!, trt])
    levels = unique(vcat(collect(a), collect(a_policy)))
    try
        levels = sort(levels)
    catch
    end
    outcome_cols = unique(vcat([trt], covariates))
    schema_df = copy(df)
    if !isempty(a_policy)
        template = df[1:1, :]
        extras = DataFrame[]
        for lev in unique(a_policy)
            row = copy(template)
            row[1, trt] = lev
            push!(extras, row)
        end
        schema_df = vcat(df, extras...)
    end
    outcome_schema = fit_covariate_schema(schema_df, outcome_cols)
    W = _covariate_matrix(fit_covariate_schema(df, covariates), df)
    Q_obs = zeros(n)
    Q1 = zeros(n)
    Q0 = zeros(n)
    H1 = ones(n)
    H0 = ones(n)
    a_ref = a
    identity_policy = all(string.(a_policy) .== string.(a))
    fold_sets === nothing && (fold_sets = crossfit_indices(n, folds, rng; cluster = cluster_ids))
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(outcome_schema, train)
        sl_y = fit_super_learner(
            Xtr, y[train_idx];
            learners = learners_outcome, family = family_outcome, rng = rng,
        )
        Q_obs[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test))
        test0 = _counterfactual_frame(test, trt, a_ref[test_idx])
        Q0[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test0))
        if identity_policy
            Q1[test_idx] = Q0[test_idx]
            H1[test_idx] .= 1.0
        else
            test1 = _counterfactual_frame(test, trt, a_policy[test_idx])
            Q1[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test1))
            W_tr = Matrix{Float64}(W[train_idx, :])
            clf1 = _fit_discrete_density_ratio(
                a[train_idx], a_policy[train_idx], W_tr, levels;
                learners = learners_trt, rng = rng,
            )
            H1[test_idx] = _ratio_from_discrete_classifier(
                clf1, a[test_idx], Matrix{Float64}(W[test_idx, :]), levels;
                trunc = trunc, mtp = mtp, a_policy = a_policy[test_idx],
                support_mask = policy_support === nothing ? nothing : policy_support[test_idx],
                weight_truncation = weight_truncation,
            )
        end
        H0[test_idx] .= 1.0
    end
    pos = discrete_positivity(a, a_policy)
    return (
        n = n, y = y, Q_obs = Q_obs, Q1 = Q1, Q0 = Q0, H1 = H1, H0 = H0,
        trunc = weight_truncation ? trunc : Inf, positivity = pos, density_ratio = :classification,
        clamp_aware = false, family_outcome = family_outcome,
        inference_status = identity_policy ? :trivial :
            _discrete_inference_status(pos, weight_truncation, false, false),
    )
end

"""
    _shared_fold_discrete_lmtp_policies(df, trt, outcome, covariates, policies, folds, rng; ...) -> Vector

One cross-fitted outcome Super Learner shared by every counterfactual vector in `policies`; for each policy `j`,
`Q1 = Q_policies[j]` and `H1 = H_policies[j]` from that policy's 2n density ratio, with `Q0 = Q_obs`. Pads levels
for the outcome schema. Returns a `Vector` of component named tuples in policy order (used by `run_discrete_lmtp_contrast`).
"""
function _shared_fold_discrete_lmtp_policies(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    policies::Vector{<:AbstractVector},
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    learners_trt = (:logistic, :mean),
    trunc::Real = 10.0,
    weight_truncation::Bool = true,
    mtp::Bool = true,
    fold_sets = nothing,
)
    isempty(policies) && throw(ArgumentError("at least one policy is required"))
    n = nrow(df)
    y = Float64.(df[!, outcome])
    a = collect(df[!, trt])
    levels = unique(vcat(a, reduce(vcat, policies)))
    try
        sort!(levels)
    catch
    end
    schema_df = copy(df)
    extras = DataFrame[]
    template = df[1:1, :]
    for lev in unique(reduce(vcat, policies))
        row = copy(template)
        row[1, trt] = lev
        push!(extras, row)
    end
    !isempty(extras) && (schema_df = vcat(df, extras...))
    outcome_schema = fit_covariate_schema(schema_df, unique(vcat([trt], covariates)))
    W = isempty(covariates) ? zeros(n, 0) :
        _covariate_matrix(fit_covariate_schema(df, covariates), df)
    Q_obs = zeros(n)
    Q_policies = [zeros(n) for _ in policies]
    H_policies = [ones(n) for _ in policies]
    fold_sets === nothing && (fold_sets = crossfit_indices(n, folds, rng))
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        sl_y = fit_super_learner(
            design_matrix(outcome_schema, train), y[train_idx];
            learners = learners_outcome, family = family_outcome, rng = rng,
        )
        Q_obs[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test))
        W_tr = Matrix{Float64}(W[train_idx, :])
        W_te = Matrix{Float64}(W[test_idx, :])
        for j in eachindex(policies)
            policy = policies[j]
            duplicate_policy = findfirst(
                k -> k < j && all(string.(policies[k]) .== string.(policy)), eachindex(policies),
            )
            if duplicate_policy !== nothing
                Q_policies[j][test_idx] = Q_policies[duplicate_policy][test_idx]
                H_policies[j][test_idx] = H_policies[duplicate_policy][test_idx]
                continue
            end
            test_policy = _counterfactual_frame(test, trt, policy[test_idx])
            Q_policies[j][test_idx] = predict_super_learner(
                sl_y, design_matrix(outcome_schema, test_policy),
            )
            clf = _fit_discrete_density_ratio(
                a[train_idx], policy[train_idx], W_tr, levels;
                learners = learners_trt, rng = rng,
            )
            H_policies[j][test_idx] = _ratio_from_discrete_classifier(
                clf, a[test_idx], W_te, levels;
                trunc = trunc, mtp = mtp, a_policy = policy[test_idx],
                weight_truncation = weight_truncation,
            )
        end
    end
    return [(
        n = n, y = y, Q_obs = Q_obs, Q1 = Q_policies[j], Q0 = copy(Q_obs),
        H1 = H_policies[j], H0 = ones(n), trunc = weight_truncation ? trunc : Inf,
        positivity = discrete_positivity(a, policies[j]),
        density_ratio = :classification, clamp_aware = false,
        family_outcome = family_outcome,
        inference_status = _discrete_inference_status(
            discrete_positivity(a, policies[j]), weight_truncation, false, false,
        ),
    ) for j in eachindex(policies)]
end

"""
    _factorise_treatment(a) -> Vector

Coerce treatment `a` to `String` codes: pass through `AbstractString`; for `Integer` or `Bool`, or at most 12
distinct values, stringify; otherwise raise (likely continuous exposure). Used before discrete LMTP and mediation.
"""
function _factorise_treatment(a::AbstractVector)
    values = collect(a)
    T = Base.nonmissingtype(eltype(values))
    T <: AbstractString && return values
    unique_values = unique(values)
    if T <: Integer || length(unique_values) <= 12
        return string.(values)
    end
    throw(ArgumentError(
        "treatment looks continuous ($(length(unique_values)) unique values); " *
        "use ShiftPolicy / run_lmtp_grid, or pass a String / categorical exposure",
    ))
end

"""
    run_discrete_lmtp(df, trt, outcome; policy, baseline, folds, rng, ...) -> NamedTuple

Point-treatment LMTP for a categorical exposure. Density ratios use the 2n
classification construction of Díaz et al. (2021), with dummy-coded `A`.
Gaussian location-scale ratios are rejected. Missing data follow the same
`handle_missing` strategies as [`run_lmtp_grid`](@ref). When `cluster` is set,
folds keep clusters intact and intervals use a finite-cluster correction with
`G − 1` degrees of freedom. A failed positivity check leaves the point estimate
available but marks its interval unavailable (`NaN`). The default finite ratio
cap also leaves the point estimate available but reports `:truncated_weights`
and unavailable EIF uncertainty. Set `weight_truncation=false` for uncapped
ratios when finite-support policy inference is required; positivity, IPCW, and
covariate-imputation gates still apply.
"""
function run_discrete_lmtp(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    policy::DiscreteTreatmentPolicy,
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    rng = StableRNG(1),
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    learners_trt = (:logistic, :mean),
    density_ratio::Symbol = :classification,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    weight_truncation::Bool = true,
    epochs::Int = 3,
    handle_missing::Symbol = :drop,
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
    fold_groups::Union{Nothing, AbstractVector} = nothing,
)
    density_ratio === :classification || throw(ArgumentError(
        "categorical treatment LMTP requires density_ratio=:classification " *
        "(Díaz–Williams 2n classifier); got $density_ratio",
    ))
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "estimator must be one of :tmle, :itmle, :eif, :aipw, :sdr; got :$estimator",
    ))
    validate_contrast_learners(learners_outcome; context = "discrete LMTP outcome")
    fold_groups === nothing || length(fold_groups) == nrow(df) || throw(ArgumentError(
        "fold_groups must align with the original input rows",
    ))
    all_cols = unique(vcat(baseline, [trt]))
    input_df = copy(df)
    row_index_col = Symbol("_ct_input_row")
    while row_index_col in propertynames(input_df)
        row_index_col = Symbol(string(row_index_col), "_")
    end
    input_df[!, row_index_col] = collect(1:nrow(df))
    miss = handle_missing_data(
        input_df, outcome, all_cols, handle_missing;
        rng = rng, rung = :L2, time_indexed = false,
    )
    data_clean, ipcw_w, extra_cols = miss
    input_rows = collect(data_clean[!, row_index_col])
    analysis_fold_groups = fold_groups === nothing ? nothing : fold_groups[input_rows]
    select!(data_clean, Not(row_index_col))
    if !isempty(extra_cols)
        baseline = unique(vcat(baseline, extra_cols))
    end
    validate_family_outcome(data_clean[!, outcome], family_outcome)
    a_raw = collect(data_clean[!, trt])
    a = _factorise_treatment(a_raw)
    analysis = copy(data_clean)
    analysis[!, trt] = a
    n = nrow(analysis)
    W = isempty(baseline) ? zeros(n, 0) :
        Matrix{Float64}(_covariate_matrix(fit_covariate_schema(analysis, baseline), analysis))
    a_policy = apply_discrete_policy(a, W, policy)
    a_policy = _factorise_treatment(a_policy)
    policy_support = _discrete_policy_support(a, W, policy)
    cluster_ids = resolve_cluster_ids(
        data_clean, cluster, n; input_n = nrow(df), input_rows = input_rows,
    )
    if all(string.(a_policy) .== string.(a))
        zero_score = zeros(n)
        cluster_result = cluster_ids === nothing ? NamedTuple() : (
            covariance_kind = :cluster,
            cluster = cluster_ids,
            inference_df = length(unique(cluster_ids)) - 1,
            n_clusters = length(unique(cluster_ids)),
        )
        return with_missingness(merge((;
            estimate = 0.0, se = 0.0, lower = 0.0, upper = 0.0,
            ic = zero_score, point_score = copy(zero_score),
            estimator = estimator, targeting_weight = 1.0, epochs = 0,
            psi1 = mean(Float64.(data_clean[!, outcome])),
            psi0 = mean(Float64.(data_clean[!, outcome])),
            trunc = weight_truncation ? trunc : Inf,
            inference_status = :trivial,
            targeting_diagnostics = (
                score_policy1 = 0.0, score_policy0 = 0.0,
                score_norm_policy1 = 0.0, score_norm_policy0 = 0.0,
                score_combined_norm = 0.0, mean_abs_residual = NaN,
                mean_h1 = 1.0, mean_h0 = 1.0, targeting_ok = true,
            ),
            positivity = discrete_positivity(a, a_policy),
            policy = policy,
            n_changed = 0,
        ), cluster_result), miss.meta)
    end
    fold_sets = _discrete_lmtp_fold_sets(
        n, folds, rng; cluster_ids = cluster_ids, fold_groups = analysis_fold_groups,
    )
    components = _shared_fold_discrete_lmtp(
        analysis, trt, outcome, baseline, a_policy, folds, rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        learners_trt = learners_trt,
        trunc = trunc,
        weight_truncation = weight_truncation,
        mtp = policy.mtp,
        fold_sets = fold_sets,
        cluster_ids = cluster_ids,
        policy_support = policy_support,
    )
    component_status = components.inference_status === :trivial ? :trivial :
        _discrete_inference_status(
            components.positivity, weight_truncation, _uses_ipcw_weights(ipcw_w), !isempty(extra_cols),
        )
    components = merge(components, (; inference_status = component_status))
    result = lmtp_tmle_from_components(
        components;
        estimator = estimator,
        targeting_weight = 1.0,
        epochs = epochs,
        family_outcome = family_outcome,
    )
    if _uses_ipcw_weights(ipcw_w)
        estimate = sum(ipcw_w .* result.point_score) / sum(ipcw_w)
        point_score = ipcw_w .* result.point_score .* (length(ipcw_w) / sum(ipcw_w))
        result = merge(result, (; estimate, point_score))
    end
    inference_status = hasproperty(result, :inference_status) ? result.inference_status : component_status
    if cluster_ids !== nothing && inference_status in (:available, :trivial)
        interval = _discrete_influence_interval(
            result.estimate, result.ic; cluster = cluster_ids,
        )
        result = merge(result, (;
            se = interval.se, lower = interval.lower, upper = interval.upper,
            covariance_kind = :cluster,
            cluster = cluster_ids,
            inference_df = interval.df,
            n_clusters = interval.n_clusters,
        ))
    elseif cluster_ids !== nothing
        result = merge(result, (;
            covariance_kind = :cluster, cluster = cluster_ids,
            inference_df = length(unique(cluster_ids)) - 1,
            n_clusters = length(unique(cluster_ids)),
        ))
    end
    pos = components.positivity
    if inference_status ∉ (:available, :trivial)
        result = merge(result, (; se = NaN, lower = NaN, upper = NaN))
    end
    if inference_status ∉ (:available, :trivial)
        result = merge(result, (; ic = fill(NaN, length(result.ic))))
    end
    return with_missingness(merge(result, (;
        positivity = pos,
        policy = policy,
        n_changed = count(string.(a) .!= string.(a_policy)),
        inference_status = inference_status,
    )), miss.meta)
end

"""
    run_discrete_lmtp_contrast(
        df, trt, outcome;
        arm_hi, arm_ref, levels, kwargs...
    ) -> NamedTuple

Difference `E[Y|do(arm_hi)] - E[Y|do(arm_ref)]` from two static discrete LMTP
policies. The standard error uses the row-aligned difference of their influence
curves, preserving covariance between policy estimates. Each fold's outcome
model is shared across both policies; policy-specific treatment models remain
separate. Cluster intervals use `G − 1` degrees of freedom. Finite default ratio
caps gate EIF uncertainty; set `weight_truncation=false` for supported
uncapped inference. Remaining `kwargs` are forwarded to
[`run_discrete_lmtp`](@ref).
"""
function run_discrete_lmtp_contrast(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    arm_hi,
    arm_ref,
    levels,
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    rng = StableRNG(1),
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_outcome::Symbol = :gaussian,
    learners_trt = (:logistic, :mean),
    density_ratio::Symbol = :classification,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    weight_truncation::Bool = true,
    epochs::Int = 3,
    handle_missing::Symbol = :drop,
    cluster::Union{Nothing, Symbol, AbstractVector} = nothing,
    fold_groups::Union{Nothing, AbstractVector} = nothing,
)
    density_ratio === :classification || throw(ArgumentError(
        "categorical treatment LMTP requires density_ratio=:classification " *
        "(Díaz–Williams 2n classifier); got $density_ratio",
    ))
    estimator in (:tmle, :itmle, :eif, :aipw, :sdr) || throw(ArgumentError(
        "estimator must be one of :tmle, :itmle, :eif, :aipw, :sdr; got :$estimator",
    ))
    levels_vec = collect(levels)
    pol_hi = discrete_static_policy(arm_hi; levels = levels_vec)
    pol_ref = discrete_static_policy(arm_ref; levels = levels_vec)
    validate_contrast_learners(learners_outcome; context = "discrete LMTP outcome")
    fold_groups === nothing || length(fold_groups) == nrow(df) || throw(ArgumentError(
        "fold_groups must align with the original input rows",
    ))
    row_index_col = Symbol("_ct_input_row")
    while row_index_col in propertynames(df)
        row_index_col = Symbol(string(row_index_col), "_")
    end
    input_df = copy(df)
    input_df[!, row_index_col] = collect(1:nrow(df))
    miss = handle_missing_data(
        input_df, outcome, unique(vcat(baseline, [trt])), handle_missing;
        rng = rng, rung = :L2, time_indexed = false,
    )
    data_clean, ipcw_w, extra_cols = miss
    input_rows = collect(data_clean[!, row_index_col])
    analysis_fold_groups = fold_groups === nothing ? nothing : fold_groups[input_rows]
    select!(data_clean, Not(row_index_col))
    !isempty(extra_cols) && (baseline = unique(vcat(baseline, extra_cols)))
    validate_family_outcome(data_clean[!, outcome], family_outcome)
    a = _factorise_treatment(collect(data_clean[!, trt]))
    analysis = copy(data_clean)
    analysis[!, trt] = a
    n = nrow(analysis)
    W = isempty(baseline) ? zeros(n, 0) :
        Matrix{Float64}(_covariate_matrix(fit_covariate_schema(analysis, baseline), analysis))
    a_hi = _factorise_treatment(apply_discrete_policy(a, W, pol_hi))
    a_ref = _factorise_treatment(apply_discrete_policy(a, W, pol_ref))
    same_policy = all(string.(a_hi) .== string.(a_ref))
    cluster_ids = resolve_cluster_ids(
        data_clean, cluster, n; input_n = nrow(df), input_rows = input_rows,
    )
    if same_policy
        zero_score = zeros(n)
        pos_hi = discrete_positivity(a, a_hi)
        pos_ref = discrete_positivity(a, a_ref)
        unfit_component(pos, policy) = (
            estimate = NaN, se = NaN, lower = NaN, upper = NaN,
            ic = fill(NaN, n), point_score = fill(NaN, n),
            inference_status = :not_evaluated, positivity = pos, policy = policy,
        )
        hi = unfit_component(pos_hi, pol_hi)
        ref = unfit_component(pos_ref, pol_ref)
        cluster_result = cluster_ids === nothing ? NamedTuple() : (
            covariance_kind = :cluster,
            cluster = cluster_ids,
            inference_df = length(unique(cluster_ids)) - 1,
            n_clusters = length(unique(cluster_ids)),
        )
        return merge((;
            estimate = 0.0, se = 0.0, lower = 0.0, upper = 0.0,
            arm_hi = arm_hi, arm_ref = arm_ref,
            contrast = string(arm_hi, "_vs_", arm_ref),
            ic = zero_score, point_score = copy(zero_score),
            covariance_kind = cluster_ids === nothing ? :unit : :cluster,
            nuisance_fit_counts = (outcome_model_fits = 0, treatment_model_fits = 0),
            inference_status = :trivial,
            inference_df = cluster_ids === nothing ? Inf : length(unique(cluster_ids)) - 1,
            n_clusters = cluster_ids === nothing ? n : length(unique(cluster_ids)),
            hi = hi, ref = ref,
            positivity = (ok = pos_hi.ok && pos_ref.ok, hi = pos_hi, ref = pos_ref),
        ), cluster_result)
    end
    seed = CausalDynamics.stable_rng_seed(rng)
    fit_rng = StableRNG(seed)
    fold_sets = _discrete_lmtp_fold_sets(
        n, folds, fit_rng; cluster_ids = cluster_ids, fold_groups = analysis_fold_groups,
    )
    components = _shared_fold_discrete_lmtp_policies(
        analysis, trt, outcome, baseline, [a_hi, a_ref], folds, fit_rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        learners_trt = learners_trt,
        trunc = trunc,
        weight_truncation = weight_truncation,
        mtp = pol_hi.mtp,
        fold_sets = fold_sets,
    )
    function finalise(components, policy, assigned)
        component_status = _discrete_inference_status(
            components.positivity, weight_truncation, _uses_ipcw_weights(ipcw_w), !isempty(extra_cols),
        )
        components = merge(components, (; inference_status = component_status))
        result = lmtp_tmle_from_components(
            components; estimator = estimator, targeting_weight = 1.0,
            epochs = epochs, family_outcome = family_outcome,
        )
        if _uses_ipcw_weights(ipcw_w)
            estimate = sum(ipcw_w .* result.point_score) / sum(ipcw_w)
            point_score = ipcw_w .* result.point_score .* (length(ipcw_w) / sum(ipcw_w))
            result = merge(result, (; estimate, point_score))
        end
        inference_status = hasproperty(result, :inference_status) ? result.inference_status : component_status
        if cluster_ids !== nothing && inference_status in (:available, :trivial)
            interval = _discrete_influence_interval(
                result.estimate, result.ic; cluster = cluster_ids,
            )
            result = merge(result, (;
                se = interval.se, lower = interval.lower, upper = interval.upper,
                covariance_kind = :cluster, cluster = cluster_ids,
                inference_df = interval.df, n_clusters = interval.n_clusters,
            ))
        elseif cluster_ids !== nothing
            result = merge(result, (;
                covariance_kind = :cluster, cluster = cluster_ids,
                inference_df = length(unique(cluster_ids)) - 1,
                n_clusters = length(unique(cluster_ids)),
            ))
        end
        if inference_status ∉ (:available, :trivial)
            result = merge(result, (;
                se = NaN, lower = NaN, upper = NaN, ic = fill(NaN, length(result.ic)),
            ))
        end
        return with_missingness(merge(result, (;
            positivity = components.positivity, policy = policy,
            n_changed = count(string.(a) .!= string.(assigned)),
            inference_status = inference_status,
        )), miss.meta)
    end
    res_hi = finalise(components[1], pol_hi, a_hi)
    res_ref = finalise(components[2], pol_ref, a_ref)
    est = res_hi.estimate - res_ref.estimate
    point_score = same_policy ? zeros(n) : res_hi.point_score .- res_ref.point_score
    length(res_hi.ic) == length(res_ref.ic) || throw(ArgumentError(
        "arm influence curves must have identical row alignment",
    ))
    contrast_ic = same_policy ? zeros(n) : res_hi.ic .- res_ref.ic
    cluster_ids = hasproperty(res_hi, :cluster) ? res_hi.cluster : nothing
    component_status = _discrete_inference_status(
        (ok = res_hi.positivity.ok && res_ref.positivity.ok,), weight_truncation,
        _uses_ipcw_weights(ipcw_w), !isempty(extra_cols),
    )
    base_statuses = (res_hi.inference_status, res_ref.inference_status)
    inference_status = if same_policy
        :trivial
    elseif component_status ∉ (:available, :trivial)
        component_status
    else
        unsupported = findfirst(status -> status ∉ (:available, :trivial), base_statuses)
        unsupported === nothing ? :available : base_statuses[unsupported]
    end
    est = same_policy ? 0.0 : est
    interval = inference_status in (:available, :trivial) ?
        _discrete_influence_interval(est, contrast_ic; cluster = cluster_ids) :
        (se = NaN, lower = NaN, upper = NaN, df = NaN, n_clusters = cluster_ids === nothing ? length(contrast_ic) : length(unique(cluster_ids)))
    se, lwr, upr = interval.se, interval.lower, interval.upper
    pos_ok = res_hi.positivity.ok && res_ref.positivity.ok
    if inference_status ∉ (:available, :trivial)
        se, lwr, upr = NaN, NaN, NaN
        contrast_ic = fill(NaN, length(contrast_ic))
    end
    return (;
        estimate = est,
        se = se,
        lower = lwr,
        upper = upr,
        arm_hi = arm_hi,
        arm_ref = arm_ref,
        contrast = string(arm_hi, "_vs_", arm_ref),
        ic = contrast_ic,
        point_score = point_score,
        covariance_kind = cluster_ids === nothing ? :unit : :cluster,
        nuisance_fit_counts = (
            outcome_model_fits = length(fold_sets),
            treatment_model_fits = 2 * length(fold_sets),
        ),
        inference_status = inference_status,
        inference_df = interval.df,
        n_clusters = interval.n_clusters,
        hi = res_hi,
        ref = res_ref,
        positivity = (ok = pos_ok, hi = res_hi.positivity, ref = res_ref.positivity),
    )
end

export DiscreteTreatmentPolicy, DiscreteInterventionalMean
export discrete_recode_policy, discrete_static_policy, discrete_shift_policy
export apply_discrete_policy, run_discrete_lmtp, run_discrete_lmtp_contrast, discrete_positivity
