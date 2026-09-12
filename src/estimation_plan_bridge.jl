"""Execute a CausalDynamics `EstimationPlan` on a wide panel."""

"""
    _estimation_plan_runner_opts(df, est_plan) -> NamedTuple

Merge [`recommend_run_options`](@ref) with planner `family_outcome` when set.
"""
function _estimation_plan_runner_opts(df::DataFrame, est_plan::CausalDynamics.EstimationPlan)
    outcome_col = est_plan.outcome
    outcome_vec = outcome_col in propertynames(df) ? df[!, outcome_col] : nothing
    opts = if outcome_vec !== nothing
        recommend_run_options(nrow(df); engine = est_plan.engine, outcome = outcome_vec)
    else
        recommend_run_options(nrow(df); engine = est_plan.engine)
    end
    if hasproperty(est_plan, :family_outcome) && est_plan.family_outcome !== nothing
        fam = est_plan.family_outcome
        if fam in COUNT_OUTCOME_FAMILIES
            learners = recommend_count_learners(nrow(df); family = fam)
            opts = merge(opts, (;
                family_outcome = fam,
                learners = learners,
                learners_outcome = learners,
            ))
        else
            opts = merge(opts, (family_outcome = fam,))
        end
    end
    return opts
end

"""
    run_estimation_plan(df, plan; arm_hi, arm_ref, levels, policy, kwargs...) -> NamedTuple

Run the CausalTargeted engine named in `plan` using column names from
[`plan_targeted_estimation`](@ref).

For `:discrete_lmtp`, pass `arm_hi`, `arm_ref`, and `levels` to call
[`run_discrete_lmtp_contrast`](@ref), or pass `policy` for a single
[`run_discrete_lmtp`](@ref) fit.

For `:two_part_discrete_lmtp`, pass `arm_hi`, `arm_ref`, and `levels`; uses
`plan.presence_col` and `plan.intensity_col`.

Remaining `kwargs` are forwarded to the runner (e.g. `folds`, `rng`,
`learners_outcome`, `cluster`). When omitted, [`recommend_run_options`](@ref)
supplies defaults from `nrow(df)` and `plan.engine`.

# Identification discipline

A plan whose `identifiable` flag is `false` is **refused** by default
(`on_unidentified = :refuse`): the estimator would otherwise compute a number
that the certificate does not license as a causal contrast. Pass
`on_unidentified = :exploratory` to run anyway; the returned NamedTuple then
carries `identifiable = false` and
`identification_status = :not_identified_by_procedure` so the status travels
with the estimate instead of living only in a log line. Identified plans return
`identification_status = :identified` (CausalDynamics
`IDENTIFICATION_STATUSES`).
"""
function run_estimation_plan(
    df::DataFrame,
    est_plan::CausalDynamics.EstimationPlan;
    arm_hi = nothing,
    arm_ref = nothing,
    levels = nothing,
    policy = nothing,
    folds = nothing,
    learners_outcome = nothing,
    rng = nothing,
    on_unidentified::Symbol = :refuse,
    kwargs...,
)
    on_unidentified in (:refuse, :exploratory) || throw(ArgumentError(
        "on_unidentified must be :refuse or :exploratory; got :$on_unidentified",
    ))
    if !est_plan.identifiable
        if on_unidentified === :refuse
            throw(ArgumentError(
                "EstimationPlan is not graphically identifiable (strategy=" *
                "$(est_plan.strategy)); refusing to estimate. Pass " *
                "`on_unidentified = :exploratory` to compute a descriptive contrast " *
                "whose result is tagged identification_status = :not_identified_by_procedure.",
            ))
        end
        @warn("EstimationPlan not graphically identifiable; running as exploratory: $est_plan")
    end
    identification_status = est_plan.identifiable ? :identified : :not_identified_by_procedure
    tag = result -> merge(result, (;
        identification_status = identification_status,
        identifiable = est_plan.identifiable,
    ))
    if est_plan.estimability === :underpowered
        @warn("Empirical support below threshold: min_complete_n=$(est_plan.min_complete_n)")
    elseif est_plan.estimability === :structural_skip
        throw(ArgumentError(
            "EstimationPlan structural_skip: $(est_plan.missingness_note)",
        ))
    elseif est_plan.estimability === :empty
        throw(ArgumentError(
            "EstimationPlan has no complete cases across analysis columns " *
            "(min_complete_n=$(something(est_plan.min_complete_n, 0)))",
        ))
    end
    if !isempty(est_plan.missing_columns)
        @warn("Adjustment columns missing from panel: $(est_plan.missing_columns)")
    end

    opts = _estimation_plan_runner_opts(df, est_plan)
    folds = something(folds, opts.folds)
    learners_outcome = something(learners_outcome, opts.learners_outcome)
    rng = something(rng, StableRNG(1))
    family_kw = if hasproperty(opts, :family_outcome)
        (family_outcome = opts.family_outcome,)
    else
        NamedTuple()
    end
    shared = (;
        baseline = est_plan.baseline,
        folds,
        learners_outcome,
        rng,
    )
    discrete_shared = merge(shared, family_kw)

    if est_plan.engine === :discrete_lmtp
        if arm_hi !== nothing && arm_ref !== nothing
            levels === nothing && throw(ArgumentError(
                "run_estimation_plan for :discrete_lmtp contrast requires levels",
            ))
            return tag(run_discrete_lmtp_contrast(
                df, est_plan.treatment, est_plan.outcome;
                arm_hi = arm_hi,
                arm_ref = arm_ref,
                levels = levels,
                discrete_shared...,
                kwargs...,
            ))
        elseif policy !== nothing
            return tag(run_discrete_lmtp(
                df, est_plan.treatment, est_plan.outcome;
                policy = policy,
                discrete_shared...,
                kwargs...,
            ))
        else
            throw(ArgumentError(
                "run_estimation_plan for :discrete_lmtp requires arm_hi/arm_ref/levels " *
                "or policy",
            ))
        end
    elseif est_plan.engine === :two_part_discrete_lmtp
        est_plan.presence_col === nothing && throw(ArgumentError(
            "two-part plan missing presence_col; set OutcomeKind.hurdle in outcome_specs",
        ))
        est_plan.intensity_col === nothing && throw(ArgumentError(
            "two-part plan missing intensity_col; set OutcomeKind.hurdle in outcome_specs",
        ))
        arm_hi === nothing && throw(ArgumentError(
            "run_estimation_plan for :two_part_discrete_lmtp requires arm_hi, arm_ref, levels",
        ))
        arm_ref === nothing && throw(ArgumentError(
            "run_estimation_plan for :two_part_discrete_lmtp requires arm_hi, arm_ref, levels",
        ))
        levels === nothing && throw(ArgumentError(
            "run_estimation_plan for :two_part_discrete_lmtp requires arm_hi, arm_ref, levels",
        ))
        return tag(run_two_part_discrete_lmtp_contrast(
            df, est_plan.treatment;
            presence = est_plan.presence_col,
            intensity = est_plan.intensity_col,
            arm_hi = arm_hi,
            arm_ref = arm_ref,
            levels = levels,
            shared...,
            family_presence = :binomial,
            family_intensity = :gaussian,
            kwargs...,
        ))
    elseif est_plan.engine === :sequential_lmtp
        throw(ArgumentError(
            "run_estimation_plan does not run :sequential_lmtp; use " *
            "sequential_spec_from_identification + run_sequential_lmtp",
        ))
    elseif est_plan.engine === :lmtp_grid
        throw(ArgumentError(
            "run_estimation_plan does not run :lmtp_grid; use run_lmtp_grid with " *
            "est_plan.treatment, est_plan.outcome, and est_plan.baseline",
        ))
    else
        throw(ArgumentError("unsupported EstimationPlan engine=$(est_plan.engine)"))
    end
end

export run_estimation_plan
