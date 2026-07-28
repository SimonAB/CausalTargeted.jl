"""tmle3-style interventional direct effect (NDE) for binary treatment."""

using DataFrames
using Statistics
using StableRNGs

"""
    run_tmle3_nde(data, treatment, outcome; baseline, mediators, folds, rng) -> DataFrame

Cross-fitted SuperLearner interventional NDE with one-step TMLE targeting.
"""
function run_tmle3_nde(
    data::DataFrame,
    treatment::Symbol,
    outcome::Symbol;
    baseline::Vector{Symbol},
    mediators::Vector{Symbol} = Symbol[],
    folds::Int = mtp_settings().folds,
    rng = StableRNG(42),
)
    df = dropmissing(data[:, unique(vcat([treatment, outcome], baseline, mediators))])
    n = nrow(df)
    A = Float64.(df[!, treatment])
    Y = Float64.(df[!, outcome])
    covar = columns_present(df, baseline)

    e_hat = crossfit_propensity(df, treatment, covar, folds, rng)
    adjust = isempty(mediators) ? covar : unique(vcat(covar, mediators))

    Q1 = crossfit_predict_outcome(
        df, outcome, treatment, adjust, ones(n), folds, rng,
    )
    Q0 = crossfit_predict_outcome(
        df, outcome, treatment, adjust, zeros(n), folds, rng,
    )
    Q_obs = crossfit_outcome_predictions(df, outcome, treatment, adjust, folds, rng)

    H1 = A ./ e_hat
    H0 = (1 .- A) ./ (1 .- e_hat)
    resid = Y .- Q_obs
    denom = sum((H1 .- H0) .^ 2)
    ε = denom > 1e-12 ? sum((H1 .- H0) .* resid) / denom : 0.0
    ic = (Q1 .- Q0) .+ ε .* (H1 .- H0)
    est = mean(ic)
    se = std(ic) / sqrt(n)

    return DataFrame(
        treatment = [string(treatment)],
        outcome = [string(outcome)],
        estimate = [est],
        se = [se],
    )
end

"""
    run_antibodies_tasks(data, tasks=load_antibodies_auc_tasks()) -> DataFrame
"""
function run_antibodies_tasks(data::DataFrame, tasks = load_antibodies_auc_tasks())
    isempty(tasks) && return DataFrame(
        treatment = String[], outcome = String[], estimate = Float64[], se = Float64[],
        lower = Float64[], upper = Float64[],
    )
    rows = Dict{String, Any}[]
    for task in tasks
        trt = hasproperty(data, task.trt) ? task.trt : Symbol(string(task.trt))
        out = hasproperty(data, task.outcome) ? task.outcome : Symbol(string(task.outcome))
        baseline = [hasproperty(data, b) ? b : Symbol(string(b)) for b in task.baseline]
        baseline = [b for b in baseline if hasproperty(data, b)]
        hasproperty(data, trt) || continue
        hasproperty(data, out) || continue
        res = run_lmtp_contrast(data, trt, out; baseline = baseline)
        push!(rows, Dict(
            "treatment" => string(trt),
            "outcome" => string(out),
            "estimate" => res.estimate,
            "se" => res.se,
            "lower" => res.lower,
            "upper" => res.upper,
        ))
    end
    return DataFrame(rows)
end

export run_tmle3_nde, run_antibodies_tasks
