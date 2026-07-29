"""Small-n-first run profiles for continuous MTP and mediation.

Conservation, ecology, and early biomedical cohorts often have *n* in the tens
to low hundreds. These helpers choose folds, learner libraries, and Monte Carlo
depth that favour stability over asymptotic richness.

# References

- Díaz et al. (2023), *JASA* — LMTP positivity-aware policies at continuous exposures
- van der Laan, Polley & Hubbard (2007) — Super Learner library must be estimable at *n*
- van der Laan & Rose (2011) — cross-fitted TMLE practice
"""

"""
Lean SuperLearner library for small samples (avoids deep trees / multi-α glmnet /
neural nets).
"""
const SMALL_N_SL_LEARNERS = (:glm, :glmnet, :mean)

"""
    recommend_folds(n) -> Int

Cross-fitting folds as a function of sample size.
- `n < 40` → 2
- `40 ≤ n < 120` → 3
- otherwise → 5 (capped; never exceed `max(2, n ÷ 5)`)
"""
function recommend_folds(n::Integer)
    n = Int(n)
    n < 1 && throw(ArgumentError("n must be ≥ 1, got $n"))
    raw = if n < 40
        2
    elseif n < 120
        3
    else
        5
    end
    return clamp(raw, 2, max(2, n ÷ 5))
end

"""
    recommend_learners(n; rich=false) -> Tuple

Choose SuperLearner library. Small *n* defaults to [`SMALL_N_SL_LEARNERS`](@ref);
`rich=true` and `n ≥ 80` upgrades to [`RICH_SL_LEARNERS`](@ref).
"""
function recommend_learners(n::Integer; rich::Bool = false)
    n = Int(n)
    if rich && n >= 80
        return RICH_SL_LEARNERS
    elseif n < 40
        return SMALL_N_SL_LEARNERS
    else
        return DEFAULT_SL_LEARNERS
    end
end

"""
    recommend_run_options(n; engine, n_mediators, rich) -> NamedTuple

Kwargs suitable for `run_lmtp_grid` / `run_mediation_grid` /
`execute_estimand` under a small-n-first policy.

| *n* | folds | learners | density_ratio | n_mc (mediation) | parallel |
|-----|-------|----------|---------------|------------------|----------|
| < 40 | 2 | small | gaussian | 64–128 | false |
| 40–79 | 3 | default | gaussian | 64 | false |
| ≥ 80 | 3–5 | default/rich | hybrid if rich | 32–64 | false* |

\\* Parallel remains off by default for memory safety; set `parallel=true` explicitly.
"""
function recommend_run_options(
    n::Integer;
    engine::Symbol = :lmtp,
    n_mediators::Integer = 0,
    rich::Bool = false,
)
    n = Int(n)
    engine = normalize_engine(engine)
    folds = recommend_folds(n)
    learners = recommend_learners(n; rich = rich)
    density_ratio = (rich && n >= 80) ? :hybrid : :gaussian
    n_mc = if is_mediation_engine(engine) || n_mediators > 0
        n < 40 ? 128 : (n < 80 ? 64 : 32)
    else
        32
    end
    positivity = n < 120
    return (
        folds = folds,
        learners = learners,
        learners_outcome = learners,
        learners_trt = learners,
        density_ratio = density_ratio,
        estimator = :tmle,
        n_mc = n_mc,
        parallel = false,
        cache_nuisances = true,
        cv_trunc = true,
        simultaneous = n >= 40,
        positivity = positivity,
        profile = n < 40 ? :small_n : (n < 80 ? :moderate_n : :large_n),
    )
end

"""
    warn_if_folds_too_large(n, folds) -> Nothing

Emit a warning when `folds > n/5` (unstable cross-fitting at small *n*).
"""
function warn_if_folds_too_large(n::Integer, folds::Integer)
    n = Int(n)
    folds = Int(folds)
    max_ok = max(2, n ÷ 5)
    if folds > max_ok
        @warn "folds=$folds exceeds n/5=$max_ok for n=$n; prefer recommend_folds(n)=$max_ok" folds n max_ok
    end
    return nothing
end

export SMALL_N_SL_LEARNERS
export recommend_folds, recommend_learners, recommend_run_options, warn_if_folds_too_large
