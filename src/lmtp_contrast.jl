"""Scalar LMTP contrasts (quantile shift policies)."""

using DataFrames
using Statistics
using StableRNGs

"""
    run_lmtp_contrast(data, trt, outcome; baseline, q_hi, q_lo, folds, rng) -> NamedTuple

LMTP TMLE contrast: fixed exposure at `q_hi` vs `q_lo` quantiles (antibodies / AUC notebooks).
"""
function run_lmtp_contrast(
    data::DataFrame,
    trt::Symbol,
    outcome::Symbol;
    baseline::Vector{Symbol},
    q_hi::Real = 0.975,
    q_lo::Real = 0.025,
    folds::Int = mtp_settings().folds,
    learners_outcome = DEFAULT_SL_LEARNERS,
    learners_trt = DEFAULT_SL_LEARNERS,
    density_ratio::Symbol = :gaussian,
    estimator::Symbol = :tmle,
    rng = StableRNG(42),
)
    cols = unique(vcat([trt, outcome], baseline))
    df = dropmissing(data[:, cols])
    a = Float64.(df[!, trt])
    a_hi = fill(quantile(a, q_hi), nrow(df))
    a_lo = fill(quantile(a, q_lo), nrow(df))
    adjust = columns_present(df, baseline)
    return lmtp_tmle_contrast(
        df, trt, outcome, adjust, a_hi, a_lo, folds, rng;
        learners_outcome = learners_outcome,
        learners_trt = learners_trt,
        density_ratio = density_ratio,
        estimator = estimator,
    )
end

export run_lmtp_contrast
