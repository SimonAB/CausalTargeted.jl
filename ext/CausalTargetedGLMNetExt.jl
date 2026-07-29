"""
Optional GLMNet integration for elastic-net SuperLearner candidates.

Activated by `using GLMNet`. Provides `:glmnet`, `:glmnet_lasso`, and
`:glmnet_ridge` via `_fit_glmnet_safe` / `_predict_glmnet`.
"""
module CausalTargetedGLMNetExt

using CausalTargeted
using GLMNet
using Statistics

function CausalTargeted._fit_glmnet_safe(
    X::Matrix{Float64},
    y::Vector{Float64};
    α = 0.5,
    nfolds = 3,
)
    n, p = size(X)
    pred_sd = p >= 2 ? std(X[:, 2:end]; dims = 1) : [0.0]
    if n < 5 || p < 2 || std(y) < 1e-12 || all(pred_sd .< 1e-12)
        return (:mean, CausalTargeted._safe_mean(y))
    end
    try
        cv = glmnetcv(X, y; α = α, nfolds = min(nfolds, n))
        λ = cv.lambda[cv.index_1se]
        return (:glmnet, (cv = cv, λ = λ))
    catch
        return (:mean, CausalTargeted._safe_mean(y))
    end
end

function CausalTargeted._predict_glmnet(model, X::Matrix{Float64})
    typ, obj = model
    typ == :mean && return fill(Float64(obj), size(X, 1))
    return vec(GLMNet.predict(obj.cv.path, X, obj.λ))
end

end # module
