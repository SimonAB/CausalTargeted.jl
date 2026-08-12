"""
Optional Random Forest integration through MLJ and DecisionTree.jl.

Activated by `using MLJ, MLJDecisionTreeInterface`. The public Super Learner
symbol is `:randomforest`; MLJ and its interface package remain implementation
details. A leading artificial intercept is removed and features are not scaled.
"""
module CausalTargetedMLJDecisionTreeExt

using CausalTargeted
using DataFrames: DataFrame
using Logging: NullLogger, with_logger
using MLJ: categorical, classes, fit!, machine, pdf, predict
using MLJDecisionTreeInterface
using Random: MersenneTwister

const RANDOM_FOREST_SEED = 42

"""Construct the package's deterministic Random Forest model for `p` predictors."""
function randomforest_model(family::Symbol, p::Integer)
    family in (:gaussian, :binomial) || throw(ArgumentError(
        ":randomforest supports family=:gaussian or family=:binomial, got $family",
    ))
    p > 0 || throw(ArgumentError(":randomforest requires at least one predictor"))
    Model = family == :gaussian ? RandomForestRegressor : RandomForestClassifier
    return Model(
        n_trees = 500,
        n_subfeatures = max(1, floor(Int, sqrt(p))),
        min_samples_leaf = 5,
        sampling_fraction = 1.0,
        rng = MersenneTwister(RANDOM_FOREST_SEED),
    )
end

function CausalTargeted._fit_mlj_tree(
    ::Val{:randomforest},
    X::Matrix{Float64},
    y::Vector{Float64};
    family::Symbol,
)
    Xt = CausalTargeted._prepare_mlj_tree_features(X)
    if size(Xt, 2) == 0 || length(y) < 2 || all(==(first(y)), y)
        μ = family == :binomial ? clamp(CausalTargeted._safe_mean(y), 1e-6, 1 - 1e-6) :
            CausalTargeted._safe_mean(y)
        return (mach = nothing, family = family, mean = μ)
    end
    model = randomforest_model(family, size(Xt, 2))
    target = family == :binomial ? categorical(round.(Int, clamp.(y, 0.0, 1.0))) : y
    mach = machine(model, DataFrame(Xt, :auto), target)
    with_logger(NullLogger()) do
        fit!(mach, verbosity = 0)
    end
    return (mach = mach, family = family, mean = nothing)
end

function CausalTargeted._predict_mlj_tree(
    ::Val{:randomforest},
    fit,
    X::Matrix{Float64},
)
    fit.mach === nothing && return fill(Float64(fit.mean), size(X, 1))
    Xt = CausalTargeted._prepare_mlj_tree_features(X)
    pred = predict(fit.mach, DataFrame(Xt, :auto))
    if fit.family == :gaussian
        return vec(Float64.(pred))
    end
    levels = collect(classes(first(pred)))
    positive = 1 in levels ? 1 : levels[end]
    return clamp.(vec(Float64.(pdf.(pred, positive))), 1e-6, 1 - 1e-6)
end

end # module
