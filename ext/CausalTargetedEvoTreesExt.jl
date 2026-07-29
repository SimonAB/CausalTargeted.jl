"""
Optional EvoTrees integration for tree SuperLearner candidates.

Activated by `using EvoTrees`. Provides `:evotree` / `:evotree_deep` via
`_fit_evotree_safe` / `_predict_evotree`.
"""
module CausalTargetedEvoTreesExt

using CausalTargeted
using EvoTrees
using Statistics

function CausalTargeted._fit_evotree_safe(
    X::Matrix{Float64},
    y::Vector{Float64};
    max_depth::Int = 2,
    nrounds::Int = 100,
)
    n = size(X, 1)
    if n < 8 || std(y) < 1e-12
        return (:mean, CausalTargeted._safe_mean(y))
    end
    try
        mw = n < 50 ? 5.0 : max(3.0, ceil(n / 100))
        cfg = EvoTreeRegressor(
            nrounds = nrounds,
            max_depth = max_depth,
            eta = max_depth >= 4 ? 0.05 : 0.1,
            min_weight = mw,
            rowsample = 0.8,
            colsample = 0.8,
            lambda = 0.5,
            gamma = 0.0,
            seed = 42,
        )
        m = EvoTrees.fit(cfg, X, y)
        return (:evotree, m)
    catch
        return (:mean, CausalTargeted._safe_mean(y))
    end
end

function CausalTargeted._predict_evotree(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return vec(EvoTrees.predict(obj, X))
end

end # module
