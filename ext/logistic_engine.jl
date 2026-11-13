GraphPPL.@model function _logistic_gaussian_model(x, z, w, μ₀, V₀)
    β ~ MvNormal(mean = μ₀, covariance = V₀)
    for i in eachindex(z)
        z[i] ~ Normal(mean = dot(x[i], β), precision = w[i])
    end
end

function _logistic_weights(r)
    return map(r) do v
        v >= 0 && isfinite(v) || throw(ArgumentError("Logit second moments must be finite and non-negative"))
        v < 1e-6 && return 0.25 - v / 48 + v^2 / 480
        s = sqrt(v)
        return tanh(s / 2) / (2s)
    end
end

function _logistic_weight_derivative(r)
    r < 1e-4 && return -1 / 48 + r / 240 - 17r^2 / 26880
    s = sqrt(r)
    t = tanh(s / 2)
    return (s * (1 - t^2) / 2 - t) / (4s^3)
end

function _validate_logistic_inputs(X, y, μ₀, V₀, tilt, maxiter, tol, covariance)
    ndims(X) == 2 || throw(ArgumentError("X must be a matrix"))
    n, p = size(X)
    p > 0 || throw(ArgumentError("X must contain at least one coefficient column"))
    length(y) == n || throw(DimensionMismatch("X and y have different row counts"))
    length(μ₀) == p || throw(DimensionMismatch("Prior mean does not match X columns"))
    size(V₀) == (p, p) || throw(DimensionMismatch("Prior covariance does not match X columns"))
    length(tilt) == p || throw(DimensionMismatch("Tilt does not match X columns"))
    all(a -> all(v -> v isa Real && isfinite(v), a), (X, y, μ₀, V₀, tilt)) ||
        throw(ArgumentError("Data, prior and tilt must contain finite real numbers"))
    all(v -> v == 0 || v == 1, y) || throw(ArgumentError("y must contain only 0 and 1"))
    issymmetric(V₀) && isposdef(Symmetric(V₀)) ||
        throw(ArgumentError("Prior covariance must be symmetric positive definite"))
    maxiter isa Integer && maxiter > 0 || throw(ArgumentError("maxiter must be a positive integer"))
    tol isa Real && isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    covariance in (:variational, :linear_response) ||
        throw(ArgumentError("covariance must be :variational or :linear_response"))
    return nothing
end

function _logistic_relative_change(next, previous)
    isempty(next) && return 0.0
    return maximum(abs.(next .- previous) ./ max.(1.0, abs.(next), abs.(previous)))
end

function _logistic_response_diagnostics(; valid = false, reason, relative_residual = NaN,
        condition_number = NaN, spectral_radius = NaN, symmetry_error = NaN,
        minimum_eigenvalue = NaN)
    return (; valid, reason, relative_residual, condition_number, spectral_radius,
        symmetry_error, minimum_eigenvalue)
end

"""Differentiate the converged, full Gaussian JJ fixed point with respect to a linear tilt."""
function _logistic_linear_response(X, m, S, r)
    n = size(X, 1)
    η = X * m
    K = X * S * X'
    g = _logistic_weight_derivative.(r)
    H = K.^2 + 2 .* (η .* K .* η')
    rootD = sqrt.(-g)
    similar_J = Symmetric(rootD .* H .* rootD')
    spectral_radius = eigmax(similar_J)
    J = H .* (-g)'
    A = Matrix{Float64}(I, n, n) - J
    condition_number = cond(A)
    if !isfinite(condition_number) || condition_number > 1e10 ||
            !isfinite(spectral_radius) || spectral_radius >= 1 - 1e-8
        diagnostics = _logistic_response_diagnostics(; reason = :unstable_fixed_point,
            condition_number, spectral_radius)
        return (covariance = nothing, diagnostics = diagnostics)
    end
    B = 2 .* (η .* (X * S))
    R = A \ B
    relative_residual = LinearAlgebra.norm(A * R - B) / max(eps(Float64), LinearAlgebra.norm(B))
    corrected = S - S * X' * ((η .* g) .* R)
    symmetry_error = LinearAlgebra.norm(corrected - corrected') / max(eps(Float64), LinearAlgebra.norm(corrected))
    if !all(isfinite, corrected) || !isfinite(relative_residual) || relative_residual > 1e-8 ||
            !isfinite(symmetry_error) || symmetry_error > 1e-6
        diagnostics = _logistic_response_diagnostics(; reason = :invalid_response,
            relative_residual, condition_number, spectral_radius, symmetry_error)
        return (covariance = nothing, diagnostics = diagnostics)
    end
    corrected = Matrix(Symmetric((corrected + corrected') / 2))
    minimum_eigenvalue = eigmin(Symmetric(corrected))
    valid = minimum_eigenvalue > 0 && isposdef(Symmetric(corrected))
    diagnostics = _logistic_response_diagnostics(; valid,
        reason = valid ? :valid : :nonpositive_response, relative_residual,
        condition_number, spectral_radius, symmetry_error, minimum_eigenvalue)
    return (covariance = valid ? corrected : nothing, diagnostics = diagnostics)
end

"""
Fit Bernoulli logistic regression with a Gaussian prior by full Gaussian JJ updates.

Each coefficient update uses RxInfer Gaussian pseudo-observations. The optional
linear-response covariance differentiates this variational fixed point; neither
covariance is claimed to be the exact logistic posterior covariance. A failed
convergence or response check withholds the correction without substituting a
different covariance. The returned raw object is the final Gaussian update, so
its free energy must not be interpreted as the Bernoulli model's ELBO.
"""
function _fit_logistic_engine(X, y, μ₀, V₀; maxiter = 1000, tol = 1e-8,
        covariance = :linear_response, tilt = zeros(size(X, 2)))
    _validate_logistic_inputs(X, y, μ₀, V₀, tilt, maxiter, tol, covariance)
    X, y = Matrix{Float64}(X), Vector{Float64}(y)
    μ₀, V₀, tilt = Vector{Float64}(μ₀), Matrix{Float64}(V₀), Vector{Float64}(tilt)
    n = size(X, 1)
    if n == 0
        lr = _logistic_response_diagnostics(; valid = covariance == :linear_response,
            reason = covariance == :linear_response ? :prior_only : :not_requested)
        diagnostics = (prior_only = true, max_relative_change = 0.0,
            relative_changes = (mean = 0.0, covariance = 0.0, second_moment = 0.0),
            weights = Float64[], second_moment = Float64[], linear_response = lr)
        return (mean = μ₀ + V₀ * tilt, variational_covariance = copy(V₀),
            corrected_covariance = covariance == :linear_response ? copy(V₀) : nothing,
            n_iterations = 0, converged = true, status = :converged, diagnostics, raw = nothing)
    end
    m, S = copy(μ₀), copy(V₀)
    r = vec(sum((X * S) .* X; dims = 2)) + (X * m).^2
    raw = nothing
    converged = false
    n_iterations = 0
    changes = (mean = Inf, covariance = Inf, second_moment = Inf)
    for iteration in 1:maxiter
        w = _logistic_weights(r)
        update = _logistic_gaussian_update(X, y, μ₀, V₀, w; tilt)
        mnext, Snext = update.mean, update.covariance
        rnext = vec(sum((X * Snext) .* X; dims = 2)) + (X * mnext).^2
        changes = (mean = _logistic_relative_change(mnext, m),
            covariance = _logistic_relative_change(Snext, S),
            second_moment = _logistic_relative_change(rnext, r))
        m, S, r, raw = mnext, Snext, rnext, update.raw
        n_iterations = iteration
        if maximum(changes) <= tol
            converged = true
            break
        end
    end
    response = if covariance == :linear_response && converged
        _logistic_linear_response(X, m, S, r)
    else
        (covariance = nothing, diagnostics = _logistic_response_diagnostics(;
            reason = covariance == :variational ? :not_requested : :not_converged))
    end
    status = !converged ? :maxiter : covariance == :linear_response && !response.diagnostics.valid ?
        :invalid_linear_response : :converged
    diagnostics = (prior_only = false, max_relative_change = maximum(changes),
        relative_changes = changes, weights = _logistic_weights(r), second_moment = r,
        linear_response = response.diagnostics)
    return (mean = m, variational_covariance = S, corrected_covariance = response.covariance,
        n_iterations, converged, status, diagnostics, raw)
end

"""Independent Gaussian algebra used to check the message-passing update."""
function _logistic_gaussian_algebra(X, y, μ₀, V₀, w; tilt = zeros(length(μ₀)))
    p = length(μ₀)
    Q₀ = cholesky(Symmetric(V₀)) \ Matrix{Float64}(I, p, p)
    Q = cholesky(Symmetric(Q₀ + X' * (w .* X)))
    S = Matrix(Symmetric(Q \ Matrix{Float64}(I, p, p)))
    m = Q \ (Q₀ * μ₀ + X' * (y .- 0.5) + tilt)
    return (mean = m, covariance = S)
end

"""Gaussian pseudo-observations, with the full coefficient covariance retained."""
function _logistic_gaussian_update(X, y, μ₀, V₀, w; tilt = zeros(length(μ₀)))
    z = (y .- 0.5) ./ w
    x = [collect(row) for row in eachrow(X)]
    raw = RxInfer.infer(
        model = _logistic_gaussian_model(x = x, w = w, μ₀ = μ₀ + V₀ * tilt, V₀ = V₀),
        data = (z = z,),
        returnvars = (β = RxInfer.KeepLast(),),
        iterations = 1,
    )
    posterior = raw.posteriors[:β]
    return (mean = Vector{Float64}(mean(posterior)), covariance = Matrix{Float64}(cov(posterior)), raw = raw)
end
