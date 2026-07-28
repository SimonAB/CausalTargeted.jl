"""Adaptive SuperLearner libraries that drop learners failing at small *n*."""

"""
    adaptive_learners(n; rich=false, include_sieve=false) -> Tuple

Cross-validated library selection by sample size:

- Always includes `:mean`
- Adds `:glm` when `n ≥ 12`
- Adds `:glmnet` when `n ≥ 20`
- Adds `:evotree` when `n ≥ 40`
- Adds rich glmnet variants / deep trees when `rich && n ≥ 80`
- Optional `:sieve` placeholder flag (`include_sieve`) maps to extra glmnet α=0.25/0.75
  when `n ≥ 100` (lightweight HAL-like expansion without a HAL dependency)
"""
function adaptive_learners(
    n::Integer;
    rich::Bool = false,
    include_sieve::Bool = false,
)
    n = Int(n)
    libs = Symbol[:mean]
    n >= 12 && push!(libs, :glm)
    n >= 20 && push!(libs, :glmnet)
    n >= 40 && push!(libs, :evotree)
    if rich && n >= 80
        push!(libs, :glmnet_lasso, :glmnet_ridge, :evotree_deep)
    end
    if include_sieve && n >= 100
        # Reuse existing glmnet variants as a sieve-style expansion
        for s in (:glmnet_lasso, :glmnet_ridge)
            s in libs || push!(libs, s)
        end
    end
    return Tuple(unique(libs))
end

export adaptive_learners
