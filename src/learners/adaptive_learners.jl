"""Adaptive SuperLearner libraries that drop learners failing at small *n*."""

"""
    adaptive_learners(n; rich=false, include_sieve=false) -> Tuple

Cross-validated library selection by sample size:

- Always includes `:mean`
- Adds `:glm` when `n ≥ 12`
- Adds `:glmnet` when `n ≥ 20` and MLJ / MLJLinearModels are loaded
- Adds `:evotree` when `n ≥ 40` and EvoTrees is loaded
- Adds rich glmnet variants / deep trees when `rich && n ≥ 80` (if extensions loaded)
- Optional `:sieve` placeholder flag (`include_sieve`) maps to extra glmnet α variants
  when `n ≥ 100` (lightweight HAL-like expansion without a HAL dependency)

Neural learners (`:mlj_mlp`, `:mlj_nn_binary`) and optional MLJ trees
(`:randomforest`, `:xgboost`) are **never** included here — request them
explicitly (or pass `RICH_SL_LEARNERS`, which includes `:randomforest`) after
loading the matching weakdeps. `:xgboost` remains opt-in even for rich grids
because EvoTrees already supplies boosting.
"""
function adaptive_learners(
    n::Integer;
    rich::Bool = false,
    include_sieve::Bool = false,
)
    n = Int(n)
    libs = Symbol[:mean]
    n >= 12 && push!(libs, :glm)
    has_mlj = Base.get_extension(@__MODULE__, :CausalTargetedMLJExt) !== nothing
    has_evotree = Base.get_extension(@__MODULE__, :CausalTargetedEvoTreesExt) !== nothing
    if has_mlj && n >= 20
        push!(libs, :glmnet)
    end
    if has_evotree && n >= 40
        push!(libs, :evotree)
    end
    if rich && n >= 80
        has_mlj && push!(libs, :glmnet_lasso, :glmnet_ridge)
        has_evotree && push!(libs, :evotree_deep)
    end
    if include_sieve && has_mlj && n >= 100
        for s in (:glmnet_lasso, :glmnet_ridge)
            s in libs || push!(libs, s)
        end
    end
    return Tuple(unique(libs))
end

export adaptive_learners
