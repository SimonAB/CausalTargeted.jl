"""Monte Carlo recovery scores for synthetic MTP / mediation DGPs."""

using DataFrames
using Statistics
using Random
using StableRNGs

"""
    recovery_row(; kwargs...) -> Dict{String,Any}

One estimand×δ recovery record (estimate vs truth).
"""
function recovery_row(;
    scenario::AbstractString,
    stack::AbstractString,
    estimand::AbstractString,
    delta::Real,
    truth::Real,
    estimate::Real,
    se::Real = NaN,
    n::Int = 0,
    notes::AbstractString = "",
)
    err = Float64(estimate) - Float64(truth)
    z = isfinite(se) && se > 0 ? abs(err) / se : NaN
    cover = isfinite(se) && se > 0 ? abs(err) <= 1.96 * se : missing
    return Dict{String, Any}(
        "scenario" => String(scenario),
        "stack" => String(stack),
        "estimand" => String(estimand),
        "delta" => Float64(delta),
        "truth" => Float64(truth),
        "estimate" => Float64(estimate),
        "se" => Float64(se),
        "bias" => err,
        "abs_error" => abs(err),
        "z_score" => z,
        "cover_95" => cover,
        "sign_agree" => sign(estimate) == sign(truth) || (abs(estimate) < 0.02 && abs(truth) < 0.02),
        "n" => Int(n),
        "notes" => String(notes),
    )
end

"""
    run_julia_synthetic_once(scenario; n, delta, folds, rng) -> DataFrame

Fit Julia LMTP / mediation on one draw and score against truth.
"""
function run_julia_synthetic_once(
    scenario::Symbol;
    n::Int = 400,
    delta::Float64 = 1.0,
    folds::Int = 3,
    rng::AbstractRNG = StableRNG(42),
    learners = RICH_SL_LEARNERS,
    n_mc::Int = 32,
)
    rows = Dict{String, Any}[]
    if scenario === :linear_mtp
        df, truth = simulate_linear_mtp(n; rng = rng)
        # LMTP shift_scale=z: raw δ on A after standardising? grids use requested_shift
        # Match run_lmtp_grid: shift_scale \"z\" → requested_shift = delta (on z-scored A in R).
        # Julia lmtp_grid uses std(A) only for mediation-like paths; for LMTP `requested_shift`
        # from shift_scale z returns delta directly — see requested_shift(). Data are not
        # pre-standardised here, so use δ * sd(A) as raw additive to match SD-unit intent.
        sdA = std(df.A)
        Lq, Uq = exposure_bounds(df.A, 0.01, 0.99)
        eff = effective_raw_shift(df.A, delta * sdA; lower_q = 0.01, upper_q = 0.99)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W],
            deltas = [delta * sdA],  # raw additive ≈ one SD
            folds = folds,
            learners_outcome = learners,
            learners_trt = learners,
            parallel = false,
            simultaneous = false,
            cache_nuisances = false,
            rng = rng,
            shift_scale = "raw",
        )
        est = only(grid.est)
        se = only(grid.se)
        push!(rows, recovery_row(;
            scenario = "linear_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = est, se = se, n = n,
            notes = "eff=$(round(eff; digits=4))",
        ))
    elseif scenario === :mixed_baseline_mtp
        df, truth = simulate_mixed_baseline_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(df.A, delta * sdA; lower_q = 0.01, upper_q = 0.99)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = truth.baseline,
            deltas = [delta * sdA],
            folds = folds,
            learners_outcome = learners,
            learners_trt = learners,
            parallel = false,
            simultaneous = false,
            cache_nuisances = false,
            rng = rng,
            shift_scale = "raw",
        )
        push!(rows, recovery_row(;
            scenario = "mixed_baseline_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "eff=$(round(eff; digits=4));mixed_baseline",
        ))
    elseif scenario === :binary_mediation
        df, truth = simulate_mediation(n; rng = rng)
        t = truth.effects(1.0)
        res = run_mediation_scalar(
            df, :A, :Y;
            mediators = [:M], covar = [:W],
            folds = folds, epochs = 1, learners = learners, n_mc = n_mc, rng = rng,
        )
        for lab in ("NDE", "NIE", "TE")
            r = only(eachrow(filter(row -> row.effect == lab, res)))
            truth_v = lab == "NDE" ? t.nde : lab == "NIE" ? t.nie : t.te
            push!(rows, recovery_row(;
                scenario = "binary_mediation", stack = "julia", estimand = lab,
                delta = 1.0, truth = truth_v, estimate = r.estimate, se = r.se, n = n,
            ))
        end
    elseif scenario === :continuous_mtp_mediation
        df, truth = simulate_continuous_mtp_mediation(n; rng = rng)
        eff = effective_sd_shift(df.A, delta)
        t = truth.effects(eff)
        grid = run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [delta], folds = folds, n_mc = n_mc,
            learners = learners, parallel = false, cache_nuisances = false, rng = rng,
        )
        for lab in ("NDE", "NIE", "TE")
            r = only(eachrow(filter(row -> row.estimand == lab && abs(row.delta - delta) < 1e-12, grid)))
            truth_v = lab == "NDE" ? t.nde : lab == "NIE" ? t.nie : t.te
            push!(rows, recovery_row(;
                scenario = "continuous_mtp_mediation", stack = "julia", estimand = lab,
                delta = delta, truth = truth_v, estimate = r.est, se = r.se, n = n,
                notes = "eff=$(round(eff; digits=4))",
            ))
        end
    elseif scenario === :weak_positivity_mtp
        df, truth = simulate_weak_positivity_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(df.A, delta * sdA)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W], deltas = [delta * sdA], folds = folds,
            learners_outcome = learners, learners_trt = learners,
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = rng, shift_scale = "raw",
        )
        push!(rows, recovery_row(;
            scenario = "weak_positivity_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "eff=$(round(eff; digits=4));clamp=$(round(only(grid.clamp); digits=3))",
        ))
    elseif scenario === :misspecified_nuisance_mtp
        df, truth = simulate_misspecified_nuisance_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(df.A, delta * sdA)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W], deltas = [delta * sdA], folds = folds,
            learners_outcome = learners, learners_trt = learners,
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = rng, shift_scale = "raw",
        )
        push!(rows, recovery_row(;
            scenario = "misspecified_nuisance_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "glm_only_misspec;eff=$(round(eff; digits=4))",
        ))
    elseif scenario === :intermediate_confounding_mediation
        df, truth = simulate_intermediate_confounding_mediation(n; rng = rng)
        ora = truth.oracle(delta)
        grid = run_mediation_grid(
            df, :A, :Y;
            covar = [:W], mediators = [:M],
            deltas = [delta], folds = folds, n_mc = n_mc,
            learners = learners, parallel = false, cache_nuisances = false, rng = rng,
        )
        for lab in ("NDE", "NIE", "TE")
            r = only(eachrow(filter(row -> row.estimand == lab && abs(row.delta - delta) < 1e-12, grid)))
            truth_v = lab == "NDE" ? ora.nde : lab == "NIE" ? ora.nie : ora.te
            push!(rows, recovery_row(;
                scenario = "intermediate_confounding_mediation", stack = "julia", estimand = lab,
                delta = delta, truth = truth_v, estimate = r.est, se = r.se, n = n,
                notes = "oracle_interventional;eff=$(round(ora.eff; digits=4))",
            ))
        end
    elseif scenario === :nonlinear_interaction_mtp
        df, truth = simulate_nonlinear_interaction_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(df.A, delta * sdA)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W1, :W2], deltas = [delta * sdA], folds = folds,
            learners_outcome = learners, learners_trt = learners,
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = rng, shift_scale = "raw",
        )
        push!(rows, recovery_row(;
            scenario = "nonlinear_interaction_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "interact+quad;eff=$(round(eff; digits=4))",
        ))
    elseif scenario === :smooth_nonlinear_mtp
        df, truth = simulate_smooth_nonlinear_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(df.A, delta * sdA)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W1, :W2, :W3], deltas = [delta * sdA], folds = folds,
            learners_outcome = learners, learners_trt = learners,
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = rng, shift_scale = "raw",
        )
        push!(rows, recovery_row(;
            scenario = "smooth_nonlinear_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "smooth_sin;eff=$(round(eff; digits=4))",
        ))
    elseif scenario === :missing_outcome_mtp
        df, truth = simulate_missing_outcome_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(Float64.(df.A), delta * sdA)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W], deltas = [delta * sdA], folds = folds,
            learners_outcome = learners, learners_trt = learners,
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = rng, shift_scale = "raw", handle_missing = :ipcw,
        )
        push!(rows, recovery_row(;
            scenario = "missing_outcome_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "ipcw;miss=$(round(truth.miss_rate_actual; digits=2))",
        ))
    elseif scenario === :missing_covariate_mtp
        df, truth = simulate_missing_covariate_mtp(n; rng = rng)
        sdA = std(df.A)
        eff = effective_raw_shift(Float64.(df.A), delta * sdA)
        t = truth.effects(eff)
        grid = run_lmtp_grid(
            df, :A, :Y;
            baseline = [:W], deltas = [delta * sdA], folds = folds,
            learners_outcome = learners, learners_trt = learners,
            parallel = false, simultaneous = false, cache_nuisances = false,
            rng = rng, shift_scale = "raw", handle_missing = :impute,
        )
        push!(rows, recovery_row(;
            scenario = "missing_covariate_mtp", stack = "julia", estimand = "TE",
            delta = delta, truth = t.te, estimate = only(grid.est), se = only(grid.se), n = n,
            notes = "impute;miss=$(round(truth.miss_rate_actual; digits=2))",
        ))
    elseif scenario === :categorical_treatment_mtp
        df, truth = simulate_categorical_treatment_mtp(n; rng = rng)
        policy = discrete_recode_policy(truth.recode)
        res = run_discrete_lmtp(
            df, :A, :Y;
            policy = policy,
            baseline = [:W],
            folds = folds,
            learners_outcome = learners,
            rng = rng,
        )
        push!(rows, recovery_row(;
            scenario = "categorical_treatment_mtp", stack = "julia", estimand = "TE",
            delta = 0.0, truth = truth.te, estimate = res.estimate, se = res.se, n = n,
            notes = "recode 2→1; classification ratios",
        ))
    elseif scenario === :sequential_factor_mtp
        df, truth = simulate_sequential_factor_mtp(n; rng = rng)
        policy = discrete_recode_policy(truth.recode)
        res = run_sequential_lmtp(
            df, [:A1, :A2], :Y;
            baseline = [:W],
            time_vary = [Symbol[], [:L1]],
            policies = [policy],
            folds = folds,
            learners = learners,
            rng = rng,
        )
        push!(rows, recovery_row(;
            scenario = "sequential_factor_mtp", stack = "julia", estimand = "policy_mean",
            delta = 0.0, truth = truth.psi, estimate = res.estimate, se = res.se, n = n,
            notes = "T=2 recode 2→1; classification ratios at t=1",
        ))
    elseif scenario === :repeated_outcome_ate
        df, truth = simulate_repeated_outcome_ate(n; rng = rng)
        res = run_repeated_outcome_msm(
            df, :A, [Symbol("Y", t) for t in 1:truth.T];
            baseline = [:W],
            folds = folds,
            learners = learners,
            rng = rng,
        )
        for t in 1:truth.T
            push!(rows, recovery_row(;
                scenario = "repeated_outcome_ate", stack = "julia",
                estimand = "tau_t$t",
                delta = 0.0, truth = truth.tau[t],
                estimate = res.estimates[t], se = res.se[t], n = n,
                notes = "unstructured MSM; joint IF Σ",
            ))
        end
    else
        error("Unknown scenario: $scenario")
    end
    return DataFrame(rows)
end

"""
    julia_synthetic_scenarios() -> Vector{Symbol}
"""
julia_synthetic_scenarios() = [
    :linear_mtp,
    :binary_mediation,
    :continuous_mtp_mediation,
    :weak_positivity_mtp,
    :misspecified_nuisance_mtp,
    :intermediate_confounding_mediation,
    :nonlinear_interaction_mtp,
    :smooth_nonlinear_mtp,
    :missing_outcome_mtp,
    :missing_covariate_mtp,
    :categorical_treatment_mtp,
    :sequential_factor_mtp,
    :repeated_outcome_ate,
]

# Available as CausalTargeted.run_julia_synthetic_once / julia_synthetic_scenarios (not exported).
