"""Two-part hurdle LMTP: binomial presence + Gaussian intensity among positives."""

"""
    TwoPartInterventionalMean(trt, presence, intensity, adjustment, policy)

Hurdle estimand pairing a binary presence outcome with a continuous intensity
outcome observed only when presence is positive. Presence is estimated with
[`run_discrete_lmtp`](@ref) (`family_outcome=:binomial`); intensity is estimated
on the subsample with observed positive intensity (conditional on ``Y_{\\mathrm{pres}} > 0``).
"""
struct TwoPartInterventionalMean <: Estimand
    trt::Symbol
    presence::Symbol
    intensity::Symbol
    adjustment::Vector{Symbol}
    policy::DiscreteTreatmentPolicy
end

estimand_engine(::TwoPartInterventionalMean) = :two_part_discrete_lmtp

"""
    run_two_part_discrete_lmtp(
        df, trt;
        presence, intensity, policy, baseline, folds, rng, ...
    ) -> NamedTuple

Fit presence and conditional-intensity discrete LMTPs under one
[`DiscreteTreatmentPolicy`](@ref). The intensity part conditions on observed
positive presence (`presence > 0`); the estimand is
``E[Y_{\\mathrm{int}} \\mid Y_{\\mathrm{pres}} > 0, \\mathrm{do}(A)]``.
"""
function run_two_part_discrete_lmtp(
    df::DataFrame,
    trt::Symbol;
    presence::Symbol,
    intensity::Symbol,
    policy::DiscreteTreatmentPolicy,
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    rng = StableRNG(1),
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_presence::Symbol = :binomial,
    family_intensity::Symbol = :gaussian,
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
    fold_groups === nothing || length(fold_groups) == nrow(df) || throw(ArgumentError(
        "fold_groups must align with the original input rows",
    ))
    shared = (;
        baseline, folds, rng, learners_outcome, learners_trt,
        density_ratio, estimator, trunc, weight_truncation, epochs, handle_missing, cluster, fold_groups,
    )
    pres = run_discrete_lmtp(
        df, trt, presence;
        policy = policy,
        family_outcome = family_presence,
        shared...,
    )
    intensity_mask = BitVector(undef, nrow(df))
    pres_values = df[!, presence]
    for i in eachindex(pres_values)
        value = pres_values[i]
        intensity_mask[i] = !ismissing(value) && coalesce(value, 0) > 0
    end
    int_df = df[intensity_mask, :]
    intensity_cluster = if cluster isa AbstractVector && length(cluster) == nrow(df)
        cluster[intensity_mask]
    else
        cluster
    end
    intensity_groups = fold_groups === nothing ? nothing : fold_groups[intensity_mask]
    intensity_shared = merge(shared, (; cluster = intensity_cluster, fold_groups = intensity_groups))
    intens = run_discrete_lmtp(
        int_df, trt, intensity;
        policy = policy,
        family_outcome = family_intensity,
        intensity_shared...,
    )
    return (;
        presence = pres,
        intensity = intens,
        n_presence = nrow(df),
        n_intensity = nrow(int_df),
    )
end

"""
    run_two_part_discrete_lmtp_contrast(
        df, trt;
        presence, intensity, arm_hi, arm_ref, levels, kwargs...
    ) -> NamedTuple

Arm contrast for a two-part hurdle outcome. Returns `.presence` and
`.intensity` summaries (`estimate`, `se`, `lower`, `upper`, `n`). The
intensity contrast is conditional on ``Y_{\\mathrm{pres}} > 0`` among rows
retained after missing-data handling on the full panel. Each component uses the
paired arm-score covariance; the intensity component remains a conditional
positive-only contrast, not a marginal hurdle mean effect.
"""
function run_two_part_discrete_lmtp_contrast(
    df::DataFrame,
    trt::Symbol;
    presence::Symbol,
    intensity::Symbol,
    arm_hi,
    arm_ref,
    levels,
    baseline::Vector{Symbol} = Symbol[],
    folds::Int = 3,
    rng = StableRNG(1),
    learners_outcome = DEFAULT_SL_LEARNERS,
    family_presence::Symbol = :binomial,
    family_intensity::Symbol = :gaussian,
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
    fold_groups === nothing || length(fold_groups) == nrow(df) || throw(ArgumentError(
        "fold_groups must align with the original input rows",
    ))
    levels_vec = collect(levels)
    pres_contrast = run_discrete_lmtp_contrast(
        df, trt, presence;
        arm_hi = arm_hi, arm_ref = arm_ref, levels = levels_vec,
        baseline = baseline, folds = folds, rng = rng,
        learners_outcome = learners_outcome, family_outcome = family_presence,
        learners_trt = learners_trt, density_ratio = density_ratio,
        estimator = estimator, trunc = trunc, weight_truncation = weight_truncation, epochs = epochs,
        handle_missing = handle_missing, cluster = cluster, fold_groups = fold_groups,
    )
    pres_values = df[!, presence]
    intensity_mask = BitVector([
        !ismissing(value) && coalesce(value, 0) > 0 for value in pres_values
    ])
    int_df = df[intensity_mask, :]
    intensity_cluster = if cluster isa AbstractVector && length(cluster) == nrow(df)
        cluster[intensity_mask]
    else
        cluster
    end
    int_contrast = run_discrete_lmtp_contrast(
        int_df, trt, intensity;
        arm_hi = arm_hi, arm_ref = arm_ref, levels = levels_vec,
        baseline = baseline, folds = folds, rng = rng,
        learners_outcome = learners_outcome, family_outcome = family_intensity,
        learners_trt = learners_trt, density_ratio = density_ratio,
        estimator = estimator, trunc = trunc, weight_truncation = weight_truncation, epochs = epochs,
        handle_missing = handle_missing, cluster = intensity_cluster,
        fold_groups = fold_groups === nothing ? nothing : fold_groups[intensity_mask],
    )
    pres_est = pres_contrast.estimate
    pres_se = pres_contrast.se
    pres_lwr, pres_upr = pres_contrast.lower, pres_contrast.upper
    int_est = int_contrast.estimate
    int_se = int_contrast.se
    int_lwr, int_upr = int_contrast.lower, int_contrast.upper
    return (;
        presence = (;
            estimate = pres_est,
            inference_status = pres_contrast.inference_status,
            se = pres_se,
            lower = pres_lwr,
            upper = pres_upr,
            n = nrow(df),
            contrast = string(arm_hi, "_vs_", arm_ref),
            hi = pres_contrast.hi,
            ref = pres_contrast.ref,
        ),
        intensity = (;
            estimate = int_est,
            inference_status = int_contrast.inference_status,
            se = int_se,
            lower = int_lwr,
            upper = int_upr,
            n = nrow(int_df),
            contrast = string(arm_hi, "_vs_", arm_ref),
            hi = int_contrast.hi,
            ref = int_contrast.ref,
            conditional_on = presence,
        ),
        arm_hi = arm_hi,
        arm_ref = arm_ref,
        contrast = string(arm_hi, "_vs_", arm_ref),
    )
end

export TwoPartInterventionalMean
export run_two_part_discrete_lmtp, run_two_part_discrete_lmtp_contrast
