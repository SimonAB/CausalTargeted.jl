"""Synthetic DGPs with known MTP / mediation effects for recovery benchmarks.

Closed-form truths are used when the structural equations are linear and clamp
is inactive (or when effects scale with *effective* mean shift under clamp).
Stress DGPs expose weak positivity, intermediate confounding, and nuisance
misspecification; interventional contrasts are recovered by shared-noise
potential-outcome oracles.
"""

include("public_dgps.jl")
include("internal_dgps.jl")
