# ---------------------------------------------------------------------------
# Cross-fitted discrete LMTP estimation, contrasts & cluster inference
# ---------------------------------------------------------------------------

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
    _shared_fold_discrete_lmtp_policies(df, trt, outcome, covariates, policies, folds, rng; ...) -> Vector

One cross-fitted outcome Super Learner shared by every counterfactual vector in `policies`; for each policy `j`,
`Q1 = Q_policies[j]` and `H1 = H_policies[j]` from that policy's 2n density ratio, with `Q0 = Q_obs`. Pads levels
for the outcome schema. Returns a `Vector` of component named tuples in policy order (used by
`run_discrete_lmtp` and `run_discrete_lmtp_contrast`).
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
    cluster_ids::Union{Nothing, AbstractVector} = nothing,
    policy_supports::Union{Nothing, AbstractVector} = nothing,
)
    isempty(policies) && throw(ArgumentError("at least one policy is required"))
    n = nrow(df)
    y = collect(Float64, df[!, outcome])::Vector{Float64}
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
    a_str = string.(a)
    identity_flags = [all(string.(p) .== a_str) for p in policies]
    fold_sets === nothing && (fold_sets = crossfit_indices(n, folds, rng; cluster = cluster_ids))
    for test_idx in fold_sets
        train_idx = _complement_indices(n, test_idx)
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
            if identity_flags[j]
                Q_policies[j][test_idx] = Q_obs[test_idx]
                H_policies[j][test_idx] .= 1.0
                continue
            end
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
            sup = policy_supports === nothing ? nothing : policy_supports[j]
            H_policies[j][test_idx] = _ratio_from_discrete_classifier(
                clf, a[test_idx], W_te, levels;
                trunc = trunc, mtp = mtp, a_policy = policy[test_idx],
                support_mask = sup === nothing ? nothing : sup[test_idx],
                weight_truncation = weight_truncation,
            )
        end
    end
    return [let pos = discrete_positivity(a, policies[j])
        (
            n = n, y = y, Q_obs = Q_obs, Q1 = Q_policies[j], Q0 = copy(Q_obs),
            H1 = H_policies[j], H0 = ones(n), trunc = weight_truncation ? trunc : Inf,
            positivity = pos, density_ratio = :classification, clamp_aware = false,
            family_outcome = family_outcome,
            inference_status = identity_flags[j] ? :trivial :
                _discrete_inference_status(pos, weight_truncation, false, false),
        )
    end for j in eachindex(policies)]
end

"""
    _shared_fold_discrete_lmtp(df, trt, outcome, covariates, a_policy, folds, rng; ...) -> NamedTuple

Cross-fitted nuisance components for one discrete policy vs natural `a_ref`. Delegates to
[`_shared_fold_discrete_lmtp_policies`](@ref) with a single policy arm.
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
    supports = policy_support === nothing ? nothing : Any[policy_support]
    return only(_shared_fold_discrete_lmtp_policies(
        df, trt, outcome, covariates, [a_policy], folds, rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        learners_trt = learners_trt,
        trunc = trunc,
        weight_truncation = weight_truncation,
        mtp = mtp,
        fold_sets = fold_sets,
        cluster_ids = cluster_ids,
        policy_supports = supports,
    ))
end

function _prepare_discrete_lmtp_data(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    baseline::Vector{Symbol};
    learners_outcome,
    family_outcome::Symbol,
    density_ratio::Symbol,
    estimator::Symbol,
    handle_missing::Symbol,
    cluster::Union{Nothing, Symbol, AbstractVector},
    fold_groups::Union{Nothing, AbstractVector},
    rng,
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
    a = _factorise_treatment(collect(data_clean[!, trt]))
    analysis = copy(data_clean)
    analysis[!, trt] = a
    n = nrow(analysis)
    W = isempty(baseline) ? zeros(n, 0) :
        Matrix{Float64}(_covariate_matrix(fit_covariate_schema(analysis, baseline), analysis))
    cluster_ids = resolve_cluster_ids(
        data_clean, cluster, n; input_n = nrow(df), input_rows = input_rows,
    )
    return (;
        data_clean, analysis, a, W, n, baseline, ipcw_w, extra_cols,
        miss_meta = miss.meta, cluster_ids, analysis_fold_groups,
    )
end

function _finalise_discrete_lmtp_arm(
    components,
    policy::DiscreteTreatmentPolicy,
    a::AbstractVector,
    assigned::AbstractVector;
    estimator::Symbol,
    epochs::Int,
    family_outcome::Symbol,
    weight_truncation::Bool,
    ipcw_w,
    extra_cols,
    cluster_ids::Union{Nothing, AbstractVector},
    miss_meta,
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
            covariance_kind = :cluster,
            cluster = cluster_ids,
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
        positivity = components.positivity,
        policy = policy,
        n_changed = count(string.(a) .!= string.(assigned)),
        inference_status = inference_status,
    )), miss_meta)
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
    prep = _prepare_discrete_lmtp_data(
        df, trt, outcome, baseline;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        density_ratio = density_ratio,
        estimator = estimator,
        handle_missing = handle_missing,
        cluster = cluster,
        fold_groups = fold_groups,
        rng = rng,
    )
    a_policy = _factorise_treatment(apply_discrete_policy(prep.a, prep.W, policy))
    policy_support = _discrete_policy_support(prep.a, prep.W, policy)
    if all(string.(a_policy) .== string.(prep.a))
        zero_score = zeros(prep.n)
        cluster_result = prep.cluster_ids === nothing ? NamedTuple() : (
            covariance_kind = :cluster,
            cluster = prep.cluster_ids,
            inference_df = length(unique(prep.cluster_ids)) - 1,
            n_clusters = length(unique(prep.cluster_ids)),
        )
        y_mean = mean(Float64.(prep.data_clean[!, outcome]))
        return with_missingness(merge((;
            estimate = 0.0, se = 0.0, lower = 0.0, upper = 0.0,
            ic = zero_score, point_score = copy(zero_score),
            estimator = estimator, targeting_weight = 1.0, epochs = 0,
            psi1 = y_mean, psi0 = y_mean,
            trunc = weight_truncation ? trunc : Inf,
            inference_status = :trivial,
            targeting_diagnostics = (
                score_policy1 = 0.0, score_policy0 = 0.0,
                score_norm_policy1 = 0.0, score_norm_policy0 = 0.0,
                score_combined_norm = 0.0, mean_abs_residual = NaN,
                mean_h1 = 1.0, mean_h0 = 1.0, targeting_ok = true,
            ),
            positivity = discrete_positivity(prep.a, a_policy),
            policy = policy,
            n_changed = 0,
        ), cluster_result), prep.miss_meta)
    end
    fold_sets = _discrete_lmtp_fold_sets(
        prep.n, folds, rng;
        cluster_ids = prep.cluster_ids, fold_groups = prep.analysis_fold_groups,
    )
    components = _shared_fold_discrete_lmtp(
        prep.analysis, trt, outcome, prep.baseline, a_policy, folds, rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        learners_trt = learners_trt,
        trunc = trunc,
        weight_truncation = weight_truncation,
        mtp = policy.mtp,
        fold_sets = fold_sets,
        cluster_ids = prep.cluster_ids,
        policy_support = policy_support,
    )
    return _finalise_discrete_lmtp_arm(
        components, policy, prep.a, a_policy;
        estimator = estimator,
        epochs = epochs,
        family_outcome = family_outcome,
        weight_truncation = weight_truncation,
        ipcw_w = prep.ipcw_w,
        extra_cols = prep.extra_cols,
        cluster_ids = prep.cluster_ids,
        miss_meta = prep.miss_meta,
    )
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
    levels_vec = collect(levels)
    pol_hi = discrete_static_policy(arm_hi; levels = levels_vec)
    pol_ref = discrete_static_policy(arm_ref; levels = levels_vec)
    prep = _prepare_discrete_lmtp_data(
        df, trt, outcome, baseline;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        density_ratio = density_ratio,
        estimator = estimator,
        handle_missing = handle_missing,
        cluster = cluster,
        fold_groups = fold_groups,
        rng = rng,
    )
    a_hi = _factorise_treatment(apply_discrete_policy(prep.a, prep.W, pol_hi))
    a_ref = _factorise_treatment(apply_discrete_policy(prep.a, prep.W, pol_ref))
    if all(string.(a_hi) .== string.(a_ref))
        zero_score = zeros(prep.n)
        pos_hi = discrete_positivity(prep.a, a_hi)
        pos_ref = discrete_positivity(prep.a, a_ref)
        unfit_component(pos, policy) = (
            estimate = NaN, se = NaN, lower = NaN, upper = NaN,
            ic = fill(NaN, prep.n), point_score = fill(NaN, prep.n),
            inference_status = :not_evaluated, positivity = pos, policy = policy,
        )
        hi = unfit_component(pos_hi, pol_hi)
        ref = unfit_component(pos_ref, pol_ref)
        cluster_result = prep.cluster_ids === nothing ? NamedTuple() : (
            covariance_kind = :cluster,
            cluster = prep.cluster_ids,
            inference_df = length(unique(prep.cluster_ids)) - 1,
            n_clusters = length(unique(prep.cluster_ids)),
        )
        return merge((;
            estimate = 0.0, se = 0.0, lower = 0.0, upper = 0.0,
            arm_hi = arm_hi, arm_ref = arm_ref,
            contrast = string(arm_hi, "_vs_", arm_ref),
            ic = zero_score, point_score = copy(zero_score),
            covariance_kind = prep.cluster_ids === nothing ? :unit : :cluster,
            nuisance_fit_counts = (outcome_model_fits = 0, treatment_model_fits = 0),
            inference_status = :trivial,
            inference_df = prep.cluster_ids === nothing ? Inf : length(unique(prep.cluster_ids)) - 1,
            n_clusters = prep.cluster_ids === nothing ? prep.n : length(unique(prep.cluster_ids)),
            hi = hi, ref = ref,
            positivity = (ok = pos_hi.ok && pos_ref.ok, hi = pos_hi, ref = pos_ref),
        ), cluster_result)
    end
    seed = CausalDynamics.stable_rng_seed(rng)
    fit_rng = StableRNG(seed)
    fold_sets = _discrete_lmtp_fold_sets(
        prep.n, folds, fit_rng;
        cluster_ids = prep.cluster_ids, fold_groups = prep.analysis_fold_groups,
    )
    components = _shared_fold_discrete_lmtp_policies(
        prep.analysis, trt, outcome, prep.baseline, [a_hi, a_ref], folds, fit_rng;
        learners_outcome = learners_outcome,
        family_outcome = family_outcome,
        learners_trt = learners_trt,
        trunc = trunc,
        weight_truncation = weight_truncation,
        mtp = pol_hi.mtp,
        fold_sets = fold_sets,
    )
    arm_kw = (;
        estimator = estimator,
        epochs = epochs,
        family_outcome = family_outcome,
        weight_truncation = weight_truncation,
        ipcw_w = prep.ipcw_w,
        extra_cols = prep.extra_cols,
        cluster_ids = prep.cluster_ids,
        miss_meta = prep.miss_meta,
    )
    res_hi = _finalise_discrete_lmtp_arm(components[1], pol_hi, prep.a, a_hi; arm_kw...)
    res_ref = _finalise_discrete_lmtp_arm(components[2], pol_ref, prep.a, a_ref; arm_kw...)
    est = res_hi.estimate - res_ref.estimate
    point_score = res_hi.point_score .- res_ref.point_score
    length(res_hi.ic) == length(res_ref.ic) || throw(ArgumentError(
        "arm influence curves must have identical row alignment",
    ))
    contrast_ic = res_hi.ic .- res_ref.ic
    cluster_ids = prep.cluster_ids
    pos_ok = res_hi.positivity.ok && res_ref.positivity.ok
    component_status = _discrete_inference_status(
        (ok = pos_ok,), weight_truncation, _uses_ipcw_weights(prep.ipcw_w), !isempty(prep.extra_cols),
    )
    base_statuses = (res_hi.inference_status, res_ref.inference_status)
    inference_status = if component_status ∉ (:available, :trivial)
        component_status
    else
        unsupported = findfirst(status -> status ∉ (:available, :trivial), base_statuses)
        unsupported === nothing ? :available : base_statuses[unsupported]
    end
    interval = inference_status in (:available, :trivial) ?
        _discrete_influence_interval(est, contrast_ic; cluster = cluster_ids) :
        (
            se = NaN, lower = NaN, upper = NaN, df = NaN,
            n_clusters = cluster_ids === nothing ? length(contrast_ic) : length(unique(cluster_ids)),
        )
    se, lwr, upr = interval.se, interval.lower, interval.upper
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
