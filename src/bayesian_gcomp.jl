using Distributions

"""
    ConjugateGaussianPrior(; mean, covariance, shape, scale)

Explicit normal–inverse-gamma prior: `β | σ² ~ Normal(mean, σ² * covariance)`
and `σ² ~ InverseGamma(shape, scale)`. `covariance` is the conditional scale
matrix, not the marginal covariance of β. Require positive definite covariance,
`shape > 1` and `scale > 0`, so marginal coefficient variances exist. No
outcome-dependent rescaling is performed. With known observation variance,
condition the normal prior on that variance; inverse-gamma parameters are unused.
"""
struct ConjugateGaussianPrior
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    shape::Float64
    scale::Float64

    function ConjugateGaussianPrior(; mean::AbstractVector, covariance::AbstractMatrix,
        shape::Real, scale::Real)
        μ, V = Vector{Float64}(mean), Matrix{Float64}(covariance)
        a, b = Float64(shape), Float64(scale)
        p = length(μ)
        p > 0 || throw(ArgumentError("prior mean must contain at least one coefficient"))
        size(V) == (p, p) || throw(DimensionMismatch("prior covariance must be $p × $p"))
        all(isfinite, μ) && all(isfinite, V) || throw(ArgumentError("prior parameters must be finite"))
        issymmetric(V) || throw(ArgumentError("prior covariance must be symmetric"))
        isposdef(Symmetric(V)) || throw(ArgumentError("prior covariance must be positive definite"))
        isfinite(a) && a > 1 || throw(ArgumentError("prior shape must be finite and greater than one"))
        isfinite(b) && b > 0 || throw(ArgumentError("prior scale must be finite and positive"))
        new(μ, V, a, b)
    end
end

"""Exact coefficient posterior, fitted design, and assumptions for Gaussian g-computation.

`posterior_scale` is the Student-t scale matrix; `posterior_covariance` is its
covariance. They coincide for known observation variance (`posterior_df = Inf`).
`Vn`, `shape`, and `scale` retain the conjugate parameters. `design_metadata`
records column order; `variance_diagnostics` reports prior and posterior marginal
variances and their ratios. These ratios describe marginal contraction or
expansion relative to the prior; sensitivity requires refitting under other
priors. They do not establish causal identification or adequate model fit.
"""
struct BayesianGCompFit
    posterior_mean::Vector{Float64}
    posterior_covariance::Matrix{Float64}
    posterior_scale::Matrix{Float64}
    posterior_df::Float64
    Vn::Matrix{Float64}
    shape::Float64
    scale::Float64
    observation_variance::Union{Nothing, Float64}
    prior::ConjugateGaussianPrior
    outcome::Symbol
    treatment::Symbol
    covariates::Vector{Symbol}
    schema::CovariateSchema
    covariate_schema::CovariateSchema
    basis::Any
    design_metadata::NamedTuple
    metadata::NamedTuple
    variance_diagnostics::NamedTuple
end

function _bayesian_gaussian_posterior(X::AbstractMatrix, y::AbstractVector,
    prior::ConjugateGaussianPrior; observation_variance = nothing)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("outcome length must match design rows"))
    length(prior.mean) == p || throw(DimensionMismatch("prior mean must match $p design columns"))
    all(isfinite, X) && all(isfinite, y) || throw(ArgumentError("Gaussian likelihood inputs must be finite"))
    if observation_variance !== nothing
        observation_variance isa Real && isfinite(observation_variance) && observation_variance > 0 ||
            throw(ArgumentError("observation_variance must be finite and positive"))
    end
    X, y = Matrix{Float64}(X), Vector{Float64}(y)
    prior_factor = cholesky(Symmetric(prior.covariance))
    # Pₙ = V₀⁻¹ + X'X; solve both systems through Cholesky factors.
    prior_precision = prior_factor \ Matrix{Float64}(I, p, p)
    posterior_factor = cholesky(Symmetric(prior_precision + X' * X))
    Vn = Matrix(Symmetric(posterior_factor \ Matrix{Float64}(I, p, p)))
    μn = posterior_factor \ (prior_factor \ prior.mean + X' * y)
    an = prior.shape + n / 2
    residual = y - X * μn
    departure = μn - prior.mean
    # This residual-plus-prior-penalty expression avoids subtracting large sums.
    bn = prior.scale + (sum(abs2, residual) + dot(departure, prior_factor \ departure)) / 2
    σ² = observation_variance === nothing ? nothing : Float64(observation_variance)
    scale_matrix = (σ² === nothing ? bn / an : σ²) .* Vn
    covariance = σ² === nothing ? bn / (an - 1) .* Vn : copy(scale_matrix)
    return (; posterior_mean = μn, posterior_covariance = covariance,
        posterior_scale = scale_matrix, posterior_df = σ² === nothing ? 2an : Inf,
        Vn, shape = an, scale = bn, observation_variance = σ²)
end

function _bayesian_identification(identification, treatment, outcome, covariates)
    identification === nothing && return :user_assumed
    identification isa IdentificationCertificate || throw(ArgumentError(
        "identification must be an IdentificationCertificate; graph-to-column mapping must be explicit"))
    cert = identification
    cert.trt == treatment && cert.outcome == outcome || throw(ArgumentError(
        "identification certificate treatment and outcome must match the fitted columns"))
    Set(cert.adjustment) == Set(covariates) && length(cert.adjustment) == length(covariates) ||
        throw(ArgumentError("certificate adjustment must equal the actual covariate set; extra adjustment is not certified"))
    result = cert.result
    result.identifiable && result.identification_status == :identified || throw(ArgumentError(
        "identification certificate does not establish identification"))
    result.strategy == :backdoor || throw(ArgumentError("Gaussian g-computation requires a backdoor certificate"))
    # Integer-labelled graphs rely on the explicit certificate's column mapping.
    if all(x -> x isa Symbol || x isa AbstractString, result.adjustment)
        Set(Symbol.(result.adjustment)) == Set(covariates) || throw(ArgumentError(
            "certificate columns disagree with the underlying graph adjustment set"))
    else
        length(result.adjustment) == length(covariates) || throw(ArgumentError(
            "certificate graph adjustment and column mapping have different lengths"))
    end
    query = result.query
    query isa Union{TotalEffectQuery, InterventionalPolicyQuery} || throw(ArgumentError(
        "certificate must concern a static total effect or interventional policy"))
    for (node, column) in ((query.treatment, treatment), (query.outcome, outcome))
        if node isa Symbol || node isa AbstractString
            Symbol(node) == column || throw(ArgumentError("underlying graph query disagrees with certificate columns"))
        end
    end
    return :identified
end

function _bayesian_predictor_frame(data, treatment, covariates)
    columns = [treatment; covariates]
    _validate_requested_columns(data, columns)
    _validate_covariate_values(data, columns)
    for column in columns, value in data[!, column]
        value isa Real && !isfinite(value) && throw(ArgumentError("predictor :$column contains non-finite values"))
    end
    return select(data, columns; copycols = true)
end

function _bayesian_categorical_levels(frame)
    return Dict{Symbol, Vector{Any}}(column => collect(Any, StatsModels.levels(frame[!, column]))
        for column in propertynames(frame)
        if !(Base.nonmissingtype(eltype(frame[!, column])) <: Real))
end

function _bayesian_design(schema, basis, data, treatment, covariates, treatment_values, p;
    categorical_levels = ())
    frame = _bayesian_predictor_frame(data, treatment, covariates)
    length(treatment_values) == nrow(frame) || throw(DimensionMismatch("treatment override length must match target rows"))
    for (column, levels) in categorical_levels
        values = column == treatment ? treatment_values : frame[!, column]
        all(value -> value in levels, values) || throw(ArgumentError(
            "predictor :$column contains an unseen categorical level"))
    end
    X = if basis === nothing
        frame[!, treatment] = collect(treatment_values)
        design_matrix(schema, frame)
    else
        Matrix{Float64}(basis(frame, collect(treatment_values)))
    end
    size(X) == (nrow(frame), p) || throw(DimensionMismatch("basis must retain $p columns in the fitted order"))
    all(isfinite, X) || throw(ArgumentError("basis contains non-finite values"))
    return X
end

"""
    fit_bayesian_gcomp(data, outcome, treatment, covariates; prior,
        basis=nothing, identification=nothing, handle_missing=:error,
        observation_variance=nothing)

Fit a Gaussian outcome regression with an explicit conjugate prior. The default
design is `[intercept | treatment | covariates]`, with categorical dummy columns
and their order retained by a single fitted `CovariateSchema`. Alternatively,
`basis(predictor_data, treatment_values)` returns a matrix in a prespecified,
unchanging column order, including any intercept and interactions. Only the
treatment and declared covariate columns are passed to this function; the caller
is responsible for keeping its transformations and column meanings unchanged.

Treatment and covariates must be fully observed and finite. Missing outcomes
require `handle_missing=:mar`, explicitly recording user-assumed ignorable MAR;
only observed outcomes enter the likelihood. Missing outcomes do not add
information to the posterior. Without an identification certificate, causal
identification is recorded as `:user_assumed`; a proper posterior is not a graph
identification argument. A supplied certificate must certify the actual
adjustment set, rather than a subset of additional regressors.
`graph_identification_status` records that graph claim independently of the
basis. `outcome_model_status` is always `:user_assumed`, including for the default
linear basis: a graph certificate does not establish regression adequacy.
The conservative summary `identification_status` remains `:user_assumed` for a
user-supplied basis; the checked certificate and its graph status are retained.
"""
function fit_bayesian_gcomp(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
    covariates::AbstractVector{Symbol}; prior::ConjugateGaussianPrior, basis = nothing,
    identification = nothing, handle_missing::Symbol = :error, observation_variance = nothing)
    handle_missing in (:error, :mar) || throw(ArgumentError("handle_missing must be :error or :mar"))
    covariates = collect(covariates)
    length(unique(covariates)) == length(covariates) || throw(ArgumentError("covariates must be unique"))
    treatment ∉ covariates && outcome ∉ [treatment; covariates] || throw(ArgumentError("treatment, outcome and covariates must be distinct"))
    _validate_requested_columns(data, [outcome])
    frame = _bayesian_predictor_frame(data, treatment, covariates)
    nrow(frame) > 0 || throw(ArgumentError("fitting data must contain at least one row"))
    observed = findall(!ismissing, data[!, outcome])
    length(observed) == nrow(data) || handle_missing == :mar || throw(ArgumentError(
        "missing outcomes require explicit handle_missing=:mar"))
    y = try
        Vector{Float64}(data[observed, outcome])
    catch
        throw(ArgumentError("Gaussian outcome must be real numeric"))
    end
    all(isfinite, y) || throw(ArgumentError("observed outcomes must be finite"))
    graph_status = _bayesian_identification(identification, treatment, outcome, covariates)
    # Preserve the conservative summary without discarding graph provenance.
    status = basis === nothing ? graph_status : :user_assumed
    schema = fit_covariate_schema(frame, [treatment; covariates])
    covariate_schema = fit_covariate_schema(frame, covariates)
    p = length(prior.mean)
    X = _bayesian_design(schema, basis, frame, treatment, covariates, frame[!, treatment], p)
    posterior = _bayesian_gaussian_posterior(X[observed, :], y, prior; observation_variance)
    prior_variance = (observation_variance === nothing ? prior.scale / (prior.shape - 1) : observation_variance) .* diag(prior.covariance)
    posterior_variance = diag(posterior.posterior_covariance)
    variance_diagnostics = (; prior_variance, posterior_variance,
        posterior_to_prior_ratio = posterior_variance ./ prior_variance)
    feature_names = basis === nothing ? ["(Intercept)"; schema.feature_names] : ["basis_$j" for j in 1:p]
    design_metadata = (; feature_names, n_columns = p,
        basis_source = basis === nothing ? :covariate_schema : :user_function,
        adjustment = copy(covariates), categorical_levels = _bayesian_categorical_levels(frame),
        treatment_levels = collect(unique(frame[!, treatment])),
        outcome_scaling = :none)
    metadata = (; identification_status = status,
        graph_identification_status = graph_status,
        outcome_model_status = :user_assumed, identification,
        missingness_status = handle_missing == :mar ? :user_assumed_mar : :complete,
        n_observed = length(observed), n_missing = nrow(data) - length(observed),
        likelihood = :gaussian, posterior = observation_variance === nothing ? :student_t : :normal)
    return BayesianGCompFit(posterior.posterior_mean, posterior.posterior_covariance,
        posterior.posterior_scale, posterior.posterior_df, posterior.Vn,
        posterior.shape, posterior.scale, posterior.observation_variance, prior,
        outcome, treatment, covariates, schema, covariate_schema, basis,
        design_metadata, metadata, variance_diagnostics)
end

"""
    bayesian_policy_contrast(fit, target, policy; deltas, level=0.95)

Exact joint posterior for the fixed-cohort `:policy_vs_observed` contrast.
Each row of `contrast_matrix` averages `design(policy(A)) - design(A)` over
the supplied target rows. The returned mean, covariance and scale retain
cross-policy dependence. Intervals are pointwise equal-tailed credible
intervals for a conditional mean contrast, without future-outcome noise.
Supported policies are fixed-bound `SmoothBoundedLogitShiftPolicy` and
`DiscreteTreatmentPolicy`; a discrete policy is independent of the grid labels
in `deltas`. Its callback must be deterministic and is evaluated once for the
target cohort. Sample-clamped `ShiftPolicy` is rejected. No treatment density is
estimated; causal interpretation requires identification, positivity and the
specified outcome model, which a credible interval does not establish.
"""
function bayesian_policy_contrast(fit::BayesianGCompFit, target::AbstractDataFrame, policy;
    deltas, level::Real = 0.95)
    policy isa Union{SmoothBoundedLogitShiftPolicy, DiscreteTreatmentPolicy} ||
        throw(ArgumentError("use a fixed-bound smooth policy or DiscreteTreatmentPolicy; clamped ShiftPolicy is unsupported"))
    isfinite(level) && 0 < level < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
    deltas = Float64.(collect(deltas))
    !isempty(deltas) && all(isfinite, deltas) || throw(ArgumentError("deltas must be non-empty and finite"))
    frame = _bayesian_predictor_frame(target, fit.treatment, fit.covariates)
    nrow(frame) > 0 || throw(ArgumentError("target cohort must contain at least one row"))
    a = frame[!, fit.treatment]
    p = length(fit.posterior_mean)
    categorical_levels = fit.design_metadata.categorical_levels
    X = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, a, p; categorical_levels)
    W = transform_covariates(fit.covariate_schema, frame)
    discrete_values = if policy isa DiscreteTreatmentPolicy
        # Preserve the package's equality/string-equality policy level semantics.
        natural_levels = fit.design_metadata.treatment_levels
        [_as_policy_level(value, natural_levels) for value in apply_discrete_policy(a, W, policy)]
    else
        nothing
    end
    C = Matrix{Float64}(undef, length(deltas), p)
    for (j, δ) in enumerate(deltas)
        shifted = policy isa DiscreteTreatmentPolicy ? discrete_values :
            apply_policy_values(Float64.(a), δ, policy)
        Xpolicy = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates, shifted, p; categorical_levels)
        C[j, :] = vec(mean(Xpolicy - X; dims = 1))
    end
    μ = C * fit.posterior_mean
    covariance = Matrix(Symmetric(C * fit.posterior_covariance * C'))
    scale = Matrix(Symmetric(C * fit.posterior_scale * C'))
    reference = isinf(fit.posterior_df) ? Normal() : TDist(fit.posterior_df)
    q = quantile(reference, (1 + Float64(level)) / 2)
    halfwidth = q .* sqrt.(max.(diag(scale), 0))
    policy_constants = policy isa SmoothBoundedLogitShiftPolicy ?
        (; kind = :smooth_bounded_logit_shift, lower = policy.lower, upper = policy.upper, deltas = copy(deltas)) :
        (; kind = :discrete, policy = deepcopy(policy), deltas = copy(deltas))
    metadata = (; fit.metadata..., functional = :policy_vs_observed, factual_reference = :observed,
        target = :fixed_cohort, n_target = nrow(frame), level = Float64(level),
        uncertainty = :coefficient_posterior, intervals = :pointwise,
        policy_constants, design = fit.design_metadata)
    return (; deltas, mean = μ, covariance, scale, df = fit.posterior_df,
        lower = μ - halfwidth, upper = μ + halfwidth, contrast_matrix = C, metadata)
end

"""
    predictive_missing_outcomes(fit, data; rng)

Draw missing outcomes conditional on one shared joint posterior draw of β and
σ². Return `(rows, values, beta, sigma2)`, leaving data and fit unchanged.
Outcome-predictive noise belongs here, separately from policy mean uncertainty.
This is a posterior predictive draw, not another likelihood contribution.
"""
function predictive_missing_outcomes(fit::BayesianGCompFit, data::AbstractDataFrame;
    rng::AbstractRNG)
    _validate_requested_columns(data, [fit.outcome])
    frame = _bayesian_predictor_frame(data, fit.treatment, fit.covariates)
    rows = findall(ismissing, data[!, fit.outcome])
    X = _bayesian_design(fit.schema, fit.basis, frame, fit.treatment, fit.covariates,
        frame[!, fit.treatment], length(fit.posterior_mean);
        categorical_levels = fit.design_metadata.categorical_levels)
    σ² = fit.observation_variance === nothing ? rand(rng, InverseGamma(fit.shape, fit.scale)) : fit.observation_variance
    β = fit.posterior_mean + sqrt(σ²) .* (cholesky(Symmetric(fit.Vn)).L * randn(rng, length(fit.posterior_mean)))
    values = X[rows, :] * β + sqrt(σ²) .* randn(rng, length(rows))
    return (; rows, values, beta = β, sigma2 = σ²)
end
