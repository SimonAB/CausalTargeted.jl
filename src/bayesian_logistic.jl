"""
    GaussianCoefficientPrior(; mean, covariance)

An explicit multivariate normal prior on logistic regression coefficients,
independent of the outcome. The finite, symmetric covariance must be positive
definite; coefficients follow the frozen design column order.
"""
struct GaussianCoefficientPrior
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    function GaussianCoefficientPrior(; mean::AbstractVector, covariance::AbstractMatrix)
        μ, V = Vector{Float64}(mean), Matrix{Float64}(covariance)
        p = length(μ)
        p > 0 || throw(ArgumentError("prior mean must contain at least one coefficient"))
        size(V) == (p, p) || throw(DimensionMismatch("prior covariance must be $p × $p"))
        all(isfinite, μ) && all(isfinite, V) || throw(ArgumentError("prior parameters must be finite"))
        issymmetric(V) || throw(ArgumentError("prior covariance must be symmetric"))
        isposdef(Symmetric(V)) || throw(ArgumentError("prior covariance must be positive definite"))
        new(μ, V)
    end
end

"""Variational logistic coefficient moments, a frozen design, and explicit assumptions.

`variational_covariance` is the raw approximation. `corrected_covariance` stores
linear-response moments when available. `posterior_covariance` contains the
requested usable moments, or NaNs when convergence or correction fails. These
moments do not define a corrected posterior density.
"""
struct BayesianLogisticFit
    posterior_mean::Vector{Float64}
    variational_covariance::Matrix{Float64}
    corrected_covariance::Union{Nothing, Matrix{Float64}}
    posterior_covariance::Matrix{Float64}
    prior::GaussianCoefficientPrior
    outcome::Symbol
    treatment::Symbol
    covariates::Vector{Symbol}
    schema::CovariateSchema
    covariate_schema::CovariateSchema
    basis::Any
    design_metadata::NamedTuple
    metadata::NamedTuple
    diagnostics::NamedTuple
    status::Symbol
end

_logistic_valid_covariance(V, p) = V isa AbstractMatrix && size(V) == (p, p) &&
    all(isfinite, V) && issymmetric(V) && isposdef(Symmetric(V))

"""
    fit_bayesian_logistic(data, outcome, treatment, covariates; prior,
        basis=nothing, identification=nothing, handle_missing=:error,
        maxiter=200, tol=1e-8, covariance=:linear_response)

Fit Bernoulli logistic regression with the optional RxInfer extension; load it
with `using RxInfer, GraphPPL`. A proper coefficient prior must be specified.
The default design and categorical levels are fitted once. A custom
`basis(predictor_data, treatment_values)` receives only declared predictors and
must preserve its column meanings. Missing predictors are rejected. Missing
outcomes require explicit `handle_missing=:mar`; only observed outcomes enter
the likelihood, under user-assumed ignorable MAR. With no observed outcomes,
the coefficient posterior remains the specified prior.

Graph identification and outcome-model assumptions are recorded separately.
A certificate does not certify a custom basis or regression adequacy. Neither
variational moments nor linear-response corrections are exact posterior
inference. Failed convergence or invalid requested correction withholds usable
coefficient covariance; raw covariance is never silently substituted.
"""
function fit_bayesian_logistic(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
    covariates::AbstractVector{Symbol}; prior::GaussianCoefficientPrior, basis = nothing,
    identification = nothing, handle_missing::Symbol = :error, maxiter::Integer = 200,
    tol::Real = 1e-8, covariance::Symbol = :linear_response)
    handle_missing in (:error, :mar) || throw(ArgumentError("handle_missing must be :error or :mar"))
    covariance in (:linear_response, :variational) || throw(ArgumentError("covariance must be :linear_response or :variational"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    covariates = collect(covariates)
    length(unique(covariates)) == length(covariates) || throw(ArgumentError("covariates must be unique"))
    treatment ∉ covariates && outcome ∉ [treatment; covariates] || throw(ArgumentError("treatment, outcome and covariates must be distinct"))
    _validate_requested_columns(data, [outcome])
    frame = _bayesian_predictor_frame(data, treatment, covariates)
    nrow(frame) > 0 || throw(ArgumentError("fitting data must contain at least one row"))
    observed = findall(!ismissing, data[!, outcome])
    length(observed) == nrow(data) || handle_missing == :mar || throw(ArgumentError("missing outcomes require explicit handle_missing=:mar"))
    values = data[observed, outcome]
    all(value -> value isa Real && isfinite(value) && (value == 0 || value == 1), values) ||
        throw(ArgumentError("observed outcomes must be binary numeric values zero or one"))
    y = Vector{Float64}(values)
    graph_status = _bayesian_identification(identification, treatment, outcome, covariates)
    schema = fit_covariate_schema(frame, [treatment; covariates])
    covariate_schema = fit_covariate_schema(frame, covariates)
    p = length(prior.mean)
    X = _bayesian_design(schema, basis, frame, treatment, covariates, frame[!, treatment], p)
    extension = Base.get_extension(CausalTargeted, :CausalTargetedRxInferExt)
    extension === nothing && throw(ArgumentError("Bayesian logistic regression requires the optional extension; load it with `using RxInfer, GraphPPL`"))
    engine = extension._fit_logistic_engine(X[observed, :], y, prior.mean, prior.covariance; maxiter, tol, covariance)
    μ = Vector{Float64}(engine.mean)
    raw_covariance = Matrix{Float64}(engine.variational_covariance)
    corrected = engine.corrected_covariance === nothing ? nothing : Matrix{Float64}(engine.corrected_covariance)
    selected = covariance == :linear_response ? corrected : raw_covariance
    usable = engine.converged && _logistic_valid_covariance(selected, p)
    posterior_covariance = usable ? copy(selected) : fill(NaN, p, p)
    feature_names = basis === nothing ? ["(Intercept)"; schema.feature_names] : ["basis_$j" for j in 1:p]
    design_metadata = (; feature_names, n_columns = p,
        basis_source = basis === nothing ? :covariate_schema : :user_function,
        adjustment = copy(covariates), categorical_levels = _bayesian_categorical_levels(frame),
        treatment_levels = collect(unique(frame[!, treatment])), outcome_scaling = :none)
    status = usable ? :converged : (engine.status == :converged ? :covariance_unavailable : engine.status)
    metadata = (; identification_status = basis === nothing ? graph_status : :user_assumed,
        graph_identification_status = graph_status, outcome_model_status = :user_assumed, identification,
        missingness_status = handle_missing == :mar ? :user_assumed_mar : :complete,
        n_observed = length(observed), n_missing = nrow(data) - length(observed),
        likelihood = :bernoulli_logit, posterior = :variational, covariance_method = covariance,
        uncertainty_available = usable, converged = engine.converged, n_iterations = engine.n_iterations)
    diagnostics = (; engine.diagnostics..., raw = engine.raw)
    return BayesianLogisticFit(μ, raw_covariance, corrected, posterior_covariance,
        prior, outcome, treatment, covariates, schema, covariate_schema, basis,
        design_metadata, metadata, diagnostics, status)
end

"""
    bayesian_policy_contrast(fit::BayesianLogisticFit, target, policy;
        deltas, level=0.95, n_draws=2000, rng=Random.Xoshiro(42))

Approximate fixed-cohort posterior risk differences against observed treatment.
One set of joint normal coefficient draws, using the selected fitted moments,
is shared over the entire policy grid. Return Monte Carlo means, covariance,
pointwise quantiles and mean MCSE. This normal moment approximation is an
additional approximation: linear-response moments are not a corrected
posterior density. No outcome noise or covariate-population sampling is added.

Use fixed-bound smooth or discrete policies. A deterministic discrete callback
is evaluated once per target row before drawing, independently of grid labels.
Factual and assigned discrete levels must occur in the fitting data.
When fitted uncertainty is unavailable, `mean` is the plug-in contrast at the
coefficient mean, and covariance, intervals and MCSE are NaN.
"""
function bayesian_policy_contrast(fit::BayesianLogisticFit, target::AbstractDataFrame, policy;
    deltas, level::Real = 0.95, n_draws::Integer = 2000, rng::AbstractRNG = Random.Xoshiro(42))
    policy isa Union{SmoothBoundedLogitShiftPolicy, DiscreteTreatmentPolicy} ||
        throw(ArgumentError("use a fixed-bound smooth policy or DiscreteTreatmentPolicy; clamped ShiftPolicy is unsupported"))
    isfinite(level) && 0 < level < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
    n_draws >= 2 || throw(ArgumentError("n_draws must be at least two"))
    deltas = Float64.(collect(deltas))
    !isempty(deltas) && all(isfinite, deltas) || throw(ArgumentError("deltas must be non-empty and finite"))
    frame = _bayesian_predictor_frame(target, fit.treatment, fit.covariates)
    nrow(frame) > 0 || throw(ArgumentError("target cohort must contain at least one row"))
    a = frame[!, fit.treatment]
    p, k = length(fit.posterior_mean), length(deltas)
    categorical_levels = fit.design_metadata.categorical_levels
    X = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, a, p; categorical_levels)
    W = transform_covariates(fit.covariate_schema, frame)
    discrete_values = policy isa DiscreteTreatmentPolicy ?
        _bayesian_discrete_values(a, W, policy, fit.design_metadata.treatment_levels) : nothing
    designs = [_bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates,
        policy isa DiscreteTreatmentPolicy ? discrete_values : apply_policy_values(Float64.(a), δ, policy),
        p; categorical_levels) for δ in deltas]
    usable = fit.metadata.uncertainty_available && _logistic_valid_covariance(fit.posterior_covariance, p)
    point_estimate = [mean(_policy_logistic.(Xp * fit.posterior_mean) - _policy_logistic.(X * fit.posterior_mean)) for Xp in designs]
    μ, covariance, lower, upper, mcse = if usable
        β = fit.posterior_mean .+ cholesky(Symmetric(fit.posterior_covariance)).L * randn(rng, p, n_draws)
        factual = _policy_logistic.(X * β)
        draws = Matrix{Float64}(undef, k, n_draws)
        for j in 1:k
            draws[j, :] = vec(mean(_policy_logistic.(designs[j] * β) - factual; dims = 1))
        end
        C = Matrix{Float64}(Statistics.cov(draws; dims = 2))
        α = (1 - Float64(level)) / 2
        (vec(mean(draws; dims = 2)), C,
            [quantile(view(draws, j, :), α) for j in 1:k],
            [quantile(view(draws, j, :), 1 - α) for j in 1:k], sqrt.(max.(diag(C), 0) ./ n_draws))
    else
        (point_estimate, fill(NaN, k, k), fill(NaN, k), fill(NaN, k), fill(NaN, k))
    end
    policy_constants = policy isa SmoothBoundedLogitShiftPolicy ?
        (; kind = :smooth_bounded_logit_shift, lower = policy.lower, upper = policy.upper, deltas = copy(deltas)) :
        (; kind = :discrete, policy = deepcopy(policy), deltas = copy(deltas))
    metadata = (; fit.metadata..., functional = :policy_vs_observed, factual_reference = :observed,
        target = :fixed_cohort, n_target = nrow(frame), level = Float64(level),
        uncertainty = usable ? :gaussian_moment_approximation : :unavailable,
        method = fit.metadata.covariance_method, n_draws = usable ? n_draws : 0,
        mean_method = usable ? :monte_carlo : :plugin_at_coefficient_mean,
        intervals = :pointwise, corrected_posterior_density = false,
        policy_constants, design = fit.design_metadata)
    return (; deltas, mean = μ, covariance, lower, upper, mcse, point_estimate, metadata)
end
