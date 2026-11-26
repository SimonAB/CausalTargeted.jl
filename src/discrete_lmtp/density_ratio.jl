# ---------------------------------------------------------------------------
# Treatment dummy encoding & Díaz–Williams 2n density-ratio classification
# ---------------------------------------------------------------------------

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
