"""Bounded exponential-tilting stochastic interventions with calibration-balanced TMLE.

Defines exponential-tilting policies ``g_\\delta(a \\mid W) = w_\\delta(a) g(a \\mid W) / m_0(W)``
with ``w_\\delta(a) = \\exp(\\delta \\phi(a))``, where the tilt score ``\\phi(a)`` is either
linear (``a - c``) or smoothly bounded via ``M \\tanh((a - c) / M)``. When bounded by ``M``,
the true density ratio satisfies ``\\exp(-2|\\delta|M) \\le H_\\delta(A, W) \\le \\exp(2|\\delta|M)``
almost surely without ad-hoc weight truncation.

The normaliser ``\\log m_0(W) = \\beta^\\top \\psi(W)`` is estimated via a strictly convex
calibration-balancing objective that enforces exact finite-sample covariate balance
``\\frac{1}{n}\\sum_i H_i \\psi(W_i) = \\frac{1}{n}\\sum_i \\psi(W_i)`` at ``\\lambda = 0``.
Targeting reuses `_weighted_intercept_tilt`, yielding canonical-link TMLE estimates and
dual-axis EIF inference for both the outcome contrast ``\\tau(\\delta)`` and the realised
mean exposure shift ``\\Delta(\\delta)``.

# References

- Díaz & Hejazi (2020), *JRSS-B* — incremental and exponential-tilting stochastic interventions
- Schindl, Shen & Kennedy (2024) — bounded exponential-tilting interventions
- Tan (2020), *Biometrika* — regularised calibrated estimation of propensity/normaliser weights
- Mammen (1993), *Annals of Statistics* — wild / multiplier bootstrap for skewed functionals
"""

using DataFrames
using Distributions
using GLM
using LinearAlgebra
using Random
using StableRNGs
using Statistics

include("exponential_tilt/policy.jl")
include("exponential_tilt/normalizer.jl")
include("exponential_tilt/weighted_glm.jl")
include("exponential_tilt/estimator.jl")
