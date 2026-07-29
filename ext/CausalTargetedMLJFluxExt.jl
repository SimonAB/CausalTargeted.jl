"""
Optional MLJFlux integration for tiny neural nuisance learners.

Activated by `using MLJFlux`. Provides fit/predict helpers for the
`:mlj_mlp` (regression) and `:mlj_nn_binary` (binomial classifier) SuperLearner
symbols. Defaults are deliberately small (few hidden units, few epochs) for
tabular targeted-inference use, not large-scale deep learning.
"""
module CausalTargetedMLJFluxExt

using CausalTargeted
using DataFrames: DataFrame
using Distributions: pdf
using MLJ: machine, fit!, predict, categorical, classes
using MLJFlux

"""
    fit_mlp(X, y; hidden, epochs, batch_size, rng) -> machine

Fit a tiny MLP regressor via MLJFlux (`NeuralNetworkRegressor`).
"""
function fit_mlp(
    X::Matrix{Float64},
    y::Vector{Float64};
    hidden = (16,),
    epochs::Int = 40,
    batch_size::Int = 16,
    rng::Int = 1,
)
    Xdf = DataFrame(X, :auto)
    NeuralNetworkRegressor = MLJFlux.NeuralNetworkRegressor
    model = NeuralNetworkRegressor(
        builder = MLJFlux.MLP(hidden = hidden),
        epochs = epochs,
        batch_size = batch_size,
        rng = rng,
    )
    mach = machine(model, Xdf, y)
    fit!(mach; verbosity = 0)
    return mach
end

"""
    predict_mlp(mach, X) -> Vector{Float64}

Deterministic MLP predictions as `Float64`.
"""
function predict_mlp(mach, X::Matrix{Float64})
    Xdf = DataFrame(X, :auto)
    return vec(Float64.(predict(mach, Xdf)))
end

"""
    fit_nn_binary(X, y; n_hidden, epochs, batch_size, rng) -> machine

Fit a tiny binary neural classifier via MLJFlux (`NeuralNetworkBinaryClassifier`).
Target `y` is expected in `{0,1}` (Float64); coerced to categorical levels.
"""
function fit_nn_binary(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_hidden::Int = 16,
    epochs::Int = 40,
    batch_size::Int = 16,
    rng::Int = 1,
)
    Xdf = DataFrame(X, :auto)
    ycat = categorical(round.(Int, clamp.(y, 0.0, 1.0)))
    NeuralNetworkBinaryClassifier = MLJFlux.NeuralNetworkBinaryClassifier
    model = NeuralNetworkBinaryClassifier(
        builder = MLJFlux.Short(n_hidden = n_hidden, dropout = 0.0),
        epochs = epochs,
        batch_size = batch_size,
        rng = rng,
    )
    mach = machine(model, Xdf, ycat)
    fit!(mach; verbosity = 0)
    return mach
end

"""
    predict_nn_binary(mach, X) -> Vector{Float64}

Class-1 probabilities from a binary neural classifier.
"""
function predict_nn_binary(mach, X::Matrix{Float64})
    Xdf = DataFrame(X, :auto)
    pred_dist = predict(mach, Xdf)
    # Prefer level 1 (positive class); fall back to the last support level
    lv = collect(classes(first(pred_dist)))
    pos = 1 in lv ? 1 : lv[end]
    return clamp.(vec(Float64.(pdf.(pred_dist, pos))), 1e-6, 1 - 1e-6)
end

end # module
