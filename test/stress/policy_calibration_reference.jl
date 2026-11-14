# CDF-at-truth checks for continuous bivariate logistic policy distributions.
# A positive cohort multiplier preserves their PIT and central-interval coverage.
isdefined(@__MODULE__, :LogisticTailReference) ||
    include(joinpath(@__DIR__, "..", "logistic_tail_reference.jl"))

"""Assess independent CDF/survival integrations without rounding ambiguous coverage."""
function assess_policy_cdf(working, tight; cdf_tol = 5e-5,
        integration_tol = 5e-5, mass_tol = 5e-5)
    all(t -> isfinite(t) && t >= 0, (cdf_tol, integration_tol, mass_tol)) ||
        throw(ArgumentError("CDF tolerances must be finite and nonnegative"))
    values = (working.cdf, working.survival, working.error, working.log_mass,
        tight.cdf, tight.survival, tight.error, tight.log_mass)
    finite = all(isfinite, values) && working.error >= 0 && tight.error >= 0
    cdf_change = max(abs(working.cdf - tight.cdf), abs(working.survival - tight.survival))
    closure_error = max(abs(working.cdf + working.survival - 1),
        abs(tight.cdf + tight.survival - 1))
    tail_error = max(working.error, tight.error, closure_error)
    mass_change = abs(expm1(working.log_mass - tight.log_mass))
    # Use the lighter tail; do not clamp a probability or conceal a failed rule.
    pit = tight.cdf <= tight.survival ? tight.cdf : 1 - tight.survival
    probabilities = (working.cdf, working.survival, tight.cdf, tight.survival)
    valid = finite && isfinite(pit) && 0 <= pit <= 1 &&
        all(p -> -integration_tol <= p <= 1 + integration_tol, probabilities)
    status = !valid ? :invalid_probability : mass_change > mass_tol ? :mass_disagreement :
        cdf_change > cdf_tol ? :cdf_disagreement : tail_error > integration_tol ? :integration_error : :ok
    candidates = (working.cdf, 1 - working.survival, tight.cdf, 1 - tight.survival)
    decision_lower = minimum(candidates) - tail_error
    decision_upper = maximum(candidates) + tail_error
    inside = decision_lower >= 0.025 && decision_upper <= 0.975
    outside = decision_upper < 0.025 || decision_lower > 0.975
    status == :ok && !(inside || outside) && (status = :boundary_ambiguous)
    available = status == :ok
    return (; available, pit, covered = available ? inside : missing, status,
        cdf_change, tail_error, mass_change, decision_lower, decision_upper)
end

"""Retain both independently integrated tails and the unscaled normalising mass."""
function policy_cdf_probe(rule, contrast)
    reference = LogisticTailReference
    lower = reference.policy_tail_details(rule, contrast)
    upper = reference.policy_tail_details(rule, contrast; upper_tail = true)
    return (; cdf = lower.probability, survival = upper.probability,
        error = max(lower.error, upper.error), log_mass = log(rule.mass) + rule.log_peak)
end

"""
    checked_policy_cdf(X, y, μ, V, arm_truth; weights=nothing, gaussian=false)

Validate coverage using the CDF at the known unscaled risk-difference truth.
`gaussian=true` requires an empty likelihood and checks a selected Gaussian law;
it does not assert that this law is the original posterior. A cohort target
with positive untreated fraction has the same PIT after scaling.

If numerical uncertainty overlaps a coverage boundary, repeat once with both
rules tightened by 100. Acceptance gates are unchanged. An unresolved decision
is unavailable. Integration error estimates and tolerance refinement are
numerical diagnostics, not rigorous probability enclosures.
"""
function checked_policy_cdf(X, y, μ, V, arm_truth; weights = nothing, gaussian = false)
    isfinite(arm_truth) || throw(ArgumentError("policy truth must be finite"))
    gaussian && !isempty(y) && throw(ArgumentError("Gaussian-law checks require an empty likelihood"))
    rtol, atol = gaussian ? (1e-7, 1e-10) : (1e-5, 1e-8)
    gate = gaussian ? 5e-6 : 5e-5
    for refinement in 0:1
        factor = 100.0^refinement
        working = LogisticTailReference.bivariate_policy_reference(X, y, μ, V;
            weights, rtol = rtol / factor, atol = atol / factor)
        tight = LogisticTailReference.bivariate_policy_reference(X, y, μ, V;
            weights, rtol = rtol / (100factor), atol = atol / (100factor))
        result = assess_policy_cdf(policy_cdf_probe(working, arm_truth),
            policy_cdf_probe(tight, arm_truth); cdf_tol = gate,
            integration_tol = gate, mass_tol = gate)
        (result.status != :boundary_ambiguous || refinement == 1) && return result
    end
    error("unreachable CDF refinement state")
end
