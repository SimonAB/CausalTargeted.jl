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

function _treatment_frame(a::AbstractVector, levels::AbstractVector)
    return DataFrame(A = [_as_policy_level(v, levels) for v in a])
end

function _fit_treatment_schema(a_obs, a_policy, levels)
    stacked = vcat(_treatment_frame(a_obs, levels), _treatment_frame(a_policy, levels))
    return fit_covariate_schema(stacked, [:A])
end

function _treatment_dummies(schema::CovariateSchema, a, levels)
    return transform_covariates(schema, _treatment_frame(a, levels))
end

function _discrete_ratio_design(schema::CovariateSchema, a, W, levels)
    n = size(W, 1)
    Aenc = _treatment_dummies(schema, a, levels)
    return hcat(ones(n), Aenc, W)
end

function _fit_discrete_density_ratio(
    a_obs,
    a_policy,
    W::Matrix{Float64},
    levels;
    learners = (:logistic, :mean),
    rng = StableRNG(1),
)
    n = length(a_obs)
    schema = _fit_treatment_schema(a_obs, a_policy, levels)
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

function _ratio_from_discrete_classifier(
    clf,
    a,
    W::Matrix{Float64},
    levels;
    trunc::Real = 10.0,
    mtp::Bool = true,
    a_policy = nothing,
)
    X = _discrete_ratio_design(clf.schema, a, W, levels)
    p = clamp.(predict_super_learner(clf.fit, X), 1e-4, 1 - 1e-4)
    if !mtp && a_policy !== nothing
        followed = [string(a[i]) == string(a_policy[i]) for i in eachindex(a)]
        for i in eachindex(p)
            if followed[i]
                p[i] = max(p[i], 0.5)
            else
                p[i] = 0.0
            end
        end
    end
    r = similar(p)
    @inbounds for i in eachindex(p)
        r[i] = p[i] <= 0 ? 0.0 : clamp(p[i] / (1 - min(p[i], 0.999)), 1 / trunc, trunc)
    end
    return r
end

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

function _shared_fold_discrete_lmtp(
    df::DataFrame,
    trt::Symbol,
    outcome::Symbol,
    covariates::Vector{Symbol},
    a_policy::AbstractVector,
    folds::Int,
    rng;
    learners_outcome = DEFAULT_SL_LEARNERS,
    learners_trt = (:logistic, :mean),
    trunc::Real = 10.0,
    mtp::Bool = true,
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
    fold_sets = crossfit_indices(n, folds, rng)
    for test_idx in fold_sets
        train_idx = setdiff(1:n, test_idx)
        train = df[train_idx, :]
        test = df[test_idx, :]
        Xtr = design_matrix(outcome_schema, train)
        sl_y = fit_super_learner(Xtr, y[train_idx]; learners = learners_outcome, rng = rng)
        Q_obs[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test))
        test1 = _counterfactual_frame(test, trt, a_policy[test_idx])
        test0 = _counterfactual_frame(test, trt, a_ref[test_idx])
        Q1[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test1))
        Q0[test_idx] = predict_super_learner(sl_y, design_matrix(outcome_schema, test0))
        W_tr = Matrix{Float64}(W[train_idx, :])
        clf1 = _fit_discrete_density_ratio(
            a[train_idx], a_policy[train_idx], W_tr, levels;
            learners = learners_trt, rng = rng,
        )
        H1[test_idx] = _ratio_from_discrete_classifier(
            clf1, a[test_idx], Matrix{Float64}(W[test_idx, :]), levels;
            trunc = trunc, mtp = mtp, a_policy = a_policy[test_idx],
        )
        H0[test_idx] .= 1.0
    end
    pos = discrete_positivity(a, a_policy)
    return (
        n = n, y = y, Q_obs = Q_obs, Q1 = Q1, Q0 = Q0, H1 = H1, H0 = H0,
        trunc = trunc, positivity = pos, density_ratio = :classification,
        clamp_aware = false,
    )
end

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
`handle_missing` strategies as [`run_lmtp_grid`](@ref).
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
    learners_trt = (:logistic, :mean),
    density_ratio::Symbol = :classification,
    estimator::Symbol = :tmle,
    trunc::Real = 10.0,
    epochs::Int = 3,
    handle_missing::Symbol = :drop,
)
    density_ratio === :classification || throw(ArgumentError(
        "categorical treatment LMTP requires density_ratio=:classification " *
        "(Díaz–Williams 2n classifier); got $density_ratio",
    ))
    validate_contrast_learners(learners_outcome; context = "discrete LMTP outcome")
    all_cols = unique(vcat(baseline, [trt]))
    miss = handle_missing_data(
        df, outcome, all_cols, handle_missing;
        rng = rng, rung = :L2, time_indexed = false,
    )
    data_clean, ipcw_w, extra_cols = miss
    if !isempty(extra_cols)
        baseline = unique(vcat(baseline, extra_cols))
    end
    a_raw = collect(data_clean[!, trt])
    a = _factorise_treatment(a_raw)
    analysis = copy(data_clean)
    analysis[!, trt] = a
    n = nrow(analysis)
    W = isempty(baseline) ? zeros(n, 0) :
        Matrix{Float64}(_covariate_matrix(fit_covariate_schema(analysis, baseline), analysis))
    a_policy = apply_discrete_policy(a, W, policy)
    a_policy = _factorise_treatment(a_policy)
    components = _shared_fold_discrete_lmtp(
        analysis, trt, outcome, baseline, a_policy, folds, rng;
        learners_outcome = learners_outcome,
        learners_trt = learners_trt,
        trunc = trunc,
        mtp = policy.mtp,
    )
    result = lmtp_tmle_from_components(
        components;
        estimator = estimator,
        targeting_weight = 1.0,
        epochs = epochs,
    )
    if _uses_ipcw_weights(ipcw_w)
        ic_uncent = result.ic .+ result.estimate
        s = weighted_influence_summary(ic_uncent, ipcw_w)
        lwr, upr = wald_ci(s.estimate, s.se)
        result = merge(result, (;
            estimate = s.estimate, se = s.se, lower = lwr, upper = upr, ic = s.ic,
        ))
    end
    pos = components.positivity
    return with_missingness(merge(result, (;
        positivity = pos,
        policy = policy,
        n_changed = count(string.(a) .!= string.(a_policy)),
    )), miss.meta)
end

export DiscreteTreatmentPolicy, DiscreteInterventionalMean
export discrete_recode_policy, discrete_static_policy, discrete_shift_policy
export apply_discrete_policy, run_discrete_lmtp, discrete_positivity
