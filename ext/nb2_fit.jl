"""Dedicated subject-integrated NB2/log/random-intercept fitting and diagnostics."""

function _optimize_nb2(
    fitdata::_NB2FitData,
    initial::Vector{Float64};
    fixed_logtheta::Union{Nothing, Float64} = nothing,
    theta_bounds::Tuple{Float64, Float64} = (1.0e-3, 1.0e3),
    sigma_bounds::Tuple{Float64, Float64} = (1.0e-6, 1.0e2),
    maxeval::Int = 2_000,
)
    dimension = length(initial)
    optimizer = NLopt.Opt(:LD_LBFGS, dimension)
    lower = fill(-Inf, dimension)
    upper = fill(Inf, dimension)
    p = size(fitdata.X, 2)
    lower[p + 1] = log(sigma_bounds[1])
    upper[p + 1] = log(sigma_bounds[2])
    if fixed_logtheta === nothing
        lower[p + 2] = log(theta_bounds[1])
        upper[p + 2] = log(theta_bounds[2])
    end
    NLopt.lower_bounds!(optimizer, lower)
    NLopt.upper_bounds!(optimizer, upper)
    NLopt.ftol_rel!(optimizer, 1.0e-10)
    NLopt.xtol_rel!(optimizer, 1.0e-8)
    NLopt.maxeval!(optimizer, maxeval)
    evaluations = Ref(0)
    objective_function = function (parameters, gradient_storage)
        evaluations[] += 1
        loglikelihood_value, score = _nb2_likelihood(
            fitdata, parameters; fixed_logtheta, gradient = !isempty(gradient_storage),
        )
        if !isempty(gradient_storage)
            gradient_storage .= .-score
        end
        return -loglikelihood_value
    end
    NLopt.min_objective!(optimizer, objective_function)
    minimum_value, minimizer, return_code = NLopt.optimize(optimizer, initial)
    final_loglikelihood, final_score = _nb2_likelihood(
        fitdata, minimizer; fixed_logtheta, gradient = true,
    )
    success = return_code in (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED)
    return (
        parameters = Vector{Float64}(minimizer),
        objective = Float64(minimum_value),
        loglikelihood = Float64(final_loglikelihood),
        return_code,
        converged = success,
        evaluations = evaluations[],
        gradient_norm = maximum(abs, final_score),
    )
end

function _prepare_nb2_data(
    formula_term,
    data::DataFrame,
    id::Symbol,
    quadrature_points::Int;
    treatment::Union{Nothing, Symbol} = nothing,
    quadrature_method::Symbol = :adaptive_gauss_hermite,
)
    quadrature_points >= 5 || throw(ArgumentError(
        "quadrature_points must be at least 5; got $quadrature_points",
    ))
    quadrature_method in (:adaptive_gauss_hermite, :gauss_hermite) ||
        throw(ArgumentError(
            "quadrature_method must be :adaptive_gauss_hermite or :gauss_hermite",
        ))
    id in propertynames(data) || throw(ArgumentError("grouping column :$id is missing"))
    nrow(data) > 0 || throw(ArgumentError("data must contain at least one row"))
    any(ismissing, data[!, id]) && throw(ArgumentError("grouping column :$id contains missing values"))
    occursin("offset", lowercase(string(formula_term))) && throw(ArgumentError(
        "offsets are not supported by the dedicated NB2 random-intercept fitter",
    ))

    template = try
        LinearMixedModel(formula_term, data)
    catch error
        throw(ArgumentError("could not construct the mixed-model formula/design: $error"))
    end
    grouping = collect(fnames(template))
    grouping == [id] || throw(ArgumentError(
        "exactly one random-intercept grouping factor :$id is supported; detected $grouping",
    ))
    applied_formula = formula(template)
    rhs_terms = applied_formula.rhs isa Tuple ? applied_formula.rhs : (applied_formula.rhs,)
    random_terms = [term for term in rhs_terms if term isa MixedModels.RandomEffectsTerm]
    length(random_terms) == 1 || throw(ArgumentError(
        "exactly one random-effects term is supported; detected $(length(random_terms))",
    ))
    random_names = MixedModels.StatsModels.coefnames(only(random_terms).lhs)
    random_names = random_names isa AbstractString ? [String(random_names)] : String.(random_names)
    random_names == ["(Intercept)"] || throw(ArgumentError(
        "random slopes are unsupported; detected random-effect columns $random_names",
    ))
    fixed_terms = [term for term in rhs_terms if !(term isa MixedModels.RandomEffectsTerm)]
    length(fixed_terms) == 1 || throw(ArgumentError(
        "could not identify one fixed-effect matrix term in the formula",
    ))
    fixed_term = only(fixed_terms)
    X = Matrix{Float64}(MixedModels.StatsModels.modelcols(fixed_term, data))
    all(isfinite, X) || throw(ArgumentError("fixed-effect design contains non-finite values"))
    matrix_rank = rank(X)
    matrix_rank == size(X, 2) || throw(ArgumentError(
        "fixed-effect design is rank deficient (rank $matrix_rank for $(size(X, 2)) columns)",
    ))

    outcome = Symbol(responsename(template))
    outcome in propertynames(data) || throw(ArgumentError("outcome column :$outcome is missing"))
    any(ismissing, data[!, outcome]) && throw(ArgumentError(
        "outcome column :$outcome contains missing values",
    ))
    y = Float64.(data[!, outcome])
    all(isfinite, y) || throw(ArgumentError("outcome column :$outcome contains non-finite values"))
    minimum(y) >= 0 || throw(ArgumentError("NB2 outcomes must be nonnegative counts"))
    integer_tolerance = 64eps(Float64) .* max.(1.0, abs.(y))
    all(abs.(y .- round.(y)) .<= integer_tolerance) || throw(ArgumentError(
        "NB2 outcomes must be integer-valued counts",
    ))

    treatment === nothing || _validate_static_treatment(data, treatment, id)
    identifiers = unique(data[!, id])
    length(identifiers) >= 2 || throw(ArgumentError(
        "at least two nonempty :$id groups are required",
    ))
    groups = [findall(value -> isequal(value, identifier), data[!, id]) for identifier in identifiers]
    any(isempty, groups) && throw(ArgumentError("empty grouping levels are unsupported"))
    nodes, weights = gausshermite(quadrature_points)
    all(weights .> 0) || throw(ArgumentError("Gauss-Hermite weights must be positive"))
    fitdata = _NB2FitData(
        y,
        X,
        groups,
        Float64.(nodes),
        log.(Float64.(weights)),
        loggamma.(y .+ 1),
        quadrature_method,
    )
    return fitdata, applied_formula, fixed_term, outcome, String.(coefnames(template))
end

function _initial_nb2_parameters(
    fitdata::_NB2FitData;
    theta_initial::Union{Nothing, Real} = nothing,
)
    beta = fitdata.X \ log.(fitdata.y .+ 0.5)
    all(isfinite, beta) || fill!(beta, 0.0)
    response_mean = sum(fitdata.y) / length(fitdata.y)
    response_variance = sum((fitdata.y .- response_mean) .^ 2) /
        max(1, length(fitdata.y) - 1)
    moment_theta = response_variance > response_mean ?
        response_mean^2 / (response_variance - response_mean) : 10.0
    initial_theta = theta_initial === nothing ? clamp(moment_theta, 0.1, 100.0) :
        Float64(theta_initial)
    isfinite(initial_theta) && initial_theta > 0 || throw(ArgumentError(
        "theta_initial must be positive and finite; got $theta_initial",
    ))
    return vcat(Vector{Float64}(beta), log(0.5), log(initial_theta))
end

function CausalTargeted.fit_profiled_nb2(
    formula_term,
    data::DataFrame;
    id::Symbol,
    family::Symbol = :nb2,
    link = LogLink(),
    treatment::Union{Nothing, Symbol} = nothing,
    weights = nothing,
    offset = nothing,
    theta_initial::Union{Nothing, Real} = nothing,
    theta_bounds = (1.0e-3, 1.0e3),
    quadrature_points::Int = 25,
    quadrature_method::Symbol = :adaptive_gauss_hermite,
    optimizer = nothing,
    progress::Bool = true,
    maxeval::Int = 2_000,
    multiple_starts::Int = 3,
    profile::Bool = true,
    uncertainty::Symbol = :delta_fixed,
    initial_parameters::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    family == :nb2 || throw(ArgumentError(
        "unsupported family $(repr(family)); only NB2 is supported",
    ))
    link isa LogLink || throw(ArgumentError(
        "unsupported NB2 link $(typeof(link)); only LogLink is supported",
    ))
    weights === nothing || throw(ArgumentError(
        "prior or frequency weights are not supported by the dedicated NB2 fitter",
    ))
    offset === nothing || throw(ArgumentError(
        "offsets are not supported by the dedicated NB2 fitter",
    ))
    optimizer === nothing || throw(ArgumentError(
        "custom optimizers are not yet supported; the dedicated fitter uses NLopt L-BFGS",
    ))
    uncertainty in (:delta_fixed, :parametric_bootstrap) || throw(ArgumentError(
        "uncertainty must be :delta_fixed or :parametric_bootstrap",
    ))
    multiple_starts >= 1 || throw(ArgumentError("multiple_starts must be positive"))
    bounds = Tuple(Float64.(theta_bounds))
    length(bounds) == 2 && 0 < bounds[1] < bounds[2] || throw(ArgumentError(
        "theta_bounds must be two positive increasing values",
    ))
    fitdata, applied_formula, fixed_term, outcome, coefficient_names =
        _prepare_nb2_data(
            formula_term,
            data,
            id,
            quadrature_points;
            treatment,
            quadrature_method,
        )
    initial = if initial_parameters === nothing
        _initial_nb2_parameters(fitdata; theta_initial)
    else
        length(initial_parameters) == size(fitdata.X, 2) + 2 || throw(ArgumentError(
            "initial_parameters must contain beta, log(sigma_b), and log(theta)",
        ))
        Float64.(initial_parameters)
    end
    all(isfinite, initial) || throw(ArgumentError("initial_parameters must be finite"))
    initial[end] = clamp(initial[end], log(bounds[1]) + 0.1, log(bounds[2]) - 0.1)
    starts = Vector{Vector{Float64}}([copy(initial)])
    start_multipliers = ((0.5, 0.5), (2.0, 2.0), (0.5, 2.0), (2.0, 0.5))
    for (sigma_multiplier, theta_multiplier) in start_multipliers
        length(starts) >= multiple_starts && break
        candidate = copy(initial)
        candidate[end - 1] += log(sigma_multiplier)
        candidate[end] = clamp(
            candidate[end] + log(theta_multiplier),
            log(bounds[1]) + 0.1,
            log(bounds[2]) - 0.1,
        )
        push!(starts, candidate)
    end

    runs = map(starts) do start
        _optimize_nb2(fitdata, start; theta_bounds = bounds, maxeval)
    end
    successful = findall(run -> run.converged && isfinite(run.objective), runs)
    isempty(successful) && throw(ErrorException(
        "NB2 optimization failed from all $multiple_starts starting values; return codes " *
        "were $([run.return_code for run in runs])",
    ))
    best_index = successful[argmin([runs[index].objective for index in successful])]
    best = runs[best_index]
    p = size(fitdata.X, 2)
    beta = best.parameters[1:p]
    logsigma = best.parameters[p + 1]
    logtheta = best.parameters[p + 2]

    profile_offsets = profile ? [-0.5, -0.25, 0.0, 0.25, 0.5] : [0.0]
    profile_rows = NamedTuple[]
    fixed_start = vcat(beta, logsigma)
    for offset in profile_offsets
        candidate_logtheta = clamp(logtheta + offset, log(bounds[1]), log(bounds[2]))
        profile_fit = if offset == 0
            (
                objective = best.objective,
                loglikelihood = best.loglikelihood,
                converged = best.converged,
                parameters = fixed_start,
                gradient_norm = best.gradient_norm,
                return_code = best.return_code,
            )
        else
            _optimize_nb2(
                fitdata,
                fixed_start;
                fixed_logtheta = candidate_logtheta,
                theta_bounds = bounds,
                maxeval,
            )
        end
        push!(profile_rows, (;
            log_theta = candidate_logtheta,
            theta = exp(candidate_logtheta),
            objective = profile_fit.objective,
            loglikelihood = profile_fit.loglikelihood,
            converged = profile_fit.converged,
            random_intercept_variance = exp(2profile_fit.parameters[p + 1]),
            gradient_norm = profile_fit.gradient_norm,
        ))
    end
    theta_profile = DataFrame(profile_rows)
    profile_curvature = if profile
        center = findfirst(==(0.0), profile_offsets)
        h = 0.25
        (theta_profile.objective[center - 1] - 2theta_profile.objective[center] +
            theta_profile.objective[center + 1]) / h^2
    else
        NaN
    end
    objective_spread = maximum(run.objective for run in runs[successful]) -
        minimum(run.objective for run in runs[successful])
    theta_spread = maximum(run.parameters[end] for run in runs[successful]) -
        minimum(run.parameters[end] for run in runs[successful])
    starts_agree = length(successful) == length(runs) &&
        objective_spread <= 1.0e-7 * (1 + abs(best.objective)) &&
        theta_spread <= 1.0e-3
    interior = logtheta > log(bounds[1]) + 1.0e-4 &&
        logtheta < log(bounds[2]) - 1.0e-4
    profile_identified = !profile || (
        all(theta_profile.converged) && isfinite(profile_curvature) && profile_curvature > 0 &&
        all(theta_profile.objective[[1, 2, 4, 5]] .> theta_profile.objective[3])
    )
    beta_vcov = _conditional_beta_information(fitdata, beta, logsigma, logtheta)
    diagnostics = (;
        backend = :dedicated_subject_integrated_likelihood,
        optimizer = :NLopt_LD_LBFGS,
        return_code = best.return_code,
        evaluations = best.evaluations,
        gradient_norm = best.gradient_norm,
        multiple_starts,
        successful_starts = length(successful),
        objective_spread,
        logtheta_spread = theta_spread,
        starts_agree,
        theta_bounds = bounds,
        theta_interior = interior,
        profile_curvature,
        profile_identified,
        random_effect_boundary = exp(logsigma) <= 1.01e-6,
        quadrature_method,
        quadrature_points,
        response = outcome,
    )
    gradient_tolerance = max(1.0e-3, 1.0e-5 * length(fitdata.y))
    diagnostics = merge(diagnostics, (; gradient_tolerance))
    passed = best.converged && interior && profile_identified && starts_agree &&
        isfinite(best.gradient_norm) && best.gradient_norm <= gradient_tolerance
    progress && @info(
        "fitted dedicated NB2/log random-intercept model",
        theta = exp(logtheta),
        random_intercept_variance = exp(2logsigma),
        loglikelihood = best.loglikelihood,
        gradient_norm = best.gradient_norm,
        converged = passed,
    )
    return NB2RandomInterceptModel(
        applied_formula,
        fixed_term,
        id,
        beta,
        coefficient_names,
        exp(logtheta),
        exp(2logsigma),
        best.loglikelihood,
        passed,
        diagnostics,
        theta_profile,
        beta_vcov,
        uncertainty,
        quadrature_method,
        quadrature_points,
        length(fitdata.y),
        copy(data),
    )
end

MixedModels.StatsAPI.coef(model::NB2RandomInterceptModel) = copy(model.coefficients)
MixedModels.fixef(model::NB2RandomInterceptModel) = copy(model.coefficients)
MixedModels.StatsAPI.vcov(model::NB2RandomInterceptModel) = copy(model.beta_vcov)
MixedModels.StatsModels.formula(model::NB2RandomInterceptModel) = model.formula
MixedModels.StatsAPI.loglikelihood(model::NB2RandomInterceptModel) = model.loglikelihood
MixedModels.StatsAPI.deviance(model::NB2RandomInterceptModel) = -2model.loglikelihood
MixedModels.StatsAPI.nobs(model::NB2RandomInterceptModel) = model.observations
MixedModels.StatsAPI.coefnames(model::NB2RandomInterceptModel) = copy(model.coefficient_names)
MixedModels.StatsAPI.responsename(model::NB2RandomInterceptModel) =
    Symbol(model.formula.lhs.sym)
MixedModels.fnames(model::NB2RandomInterceptModel) = (model.id,)
MixedModels.issingular(model::NB2RandomInterceptModel) =
    model.optimizer_diagnostics.random_effect_boundary

_random_intercept_variance(model::NB2RandomInterceptModel) =
    model.random_intercept_variance

function _prediction_components(
    model::NB2RandomInterceptModel,
    prediction_data::DataFrame,
    design::Matrix{Float64},
    random_effects::Symbol,
)
    eta = design * model.coefficients
    all(isfinite, eta) || throw(ArgumentError(
        "dedicated NB2 link-scale predictions contain non-finite values",
    ))
    predictions = exp.(eta)
    all(isfinite, predictions) || throw(ArgumentError(
        "dedicated NB2 response-scale predictions overflowed",
    ))
    if random_effects == :marginal
        predictions .*= exp(0.5model.random_intercept_variance)
    end
    return predictions, true
end

function _refit_with_quadrature(model::NB2RandomInterceptModel, points::Int)
    return CausalTargeted.fit_profiled_nb2(
        model.formula,
        model.fitted_data;
        id = model.id,
        theta_initial = model.theta,
        theta_bounds = model.optimizer_diagnostics.theta_bounds,
        quadrature_points = points,
        quadrature_method = model.quadrature_method,
        progress = false,
        multiple_starts = 2,
        profile = false,
    )
end

function CausalTargeted.quadrature_diagnostics(
    model::NB2RandomInterceptModel;
    points = (15, 25, 41),
)
    rows = NamedTuple[]
    for count in points
        fitted = count == model.quadrature_points ? model : _refit_with_quadrature(model, count)
        push!(rows, (;
            quadrature_points = count,
            loglikelihood = fitted.loglikelihood,
            theta = fitted.theta,
            random_intercept_variance = fitted.random_intercept_variance,
            coefficient_maximum_difference = maximum(abs.(
                fitted.coefficients .- model.coefficients
            )),
            converged = fitted.converged,
        ))
    end
    return DataFrame(rows)
end

function CausalTargeted.validate_fixed_theta_nb2_likelihood(
    formula_term,
    data::DataFrame;
    id::Symbol,
    theta_grid = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    quadrature_points::Int = 41,
    tolerance::Float64 = 1.0e-4,
)
    fitdata, _, _, _, _ = _prepare_nb2_data(
        formula_term,
        data,
        id,
        quadrature_points;
        quadrature_method = :adaptive_gauss_hermite,
    )
    rows = NamedTuple[]
    for theta_value in theta_grid
        fixed_model = fit(
            MixedModel,
            formula_term,
            data,
            MixedModels.Distributions.NegativeBinomial(theta_value),
            LogLink();
            progress = false,
        )
        variance = _random_intercept_variance(fixed_model)
        parameters = vcat(Float64.(coef(fixed_model)), 0.5log(variance))
        independent, score = _nb2_likelihood(
            fitdata,
            parameters;
            fixed_logtheta = log(theta_value),
            gradient = true,
        )
        mixedmodels_loglikelihood = Float64(loglikelihood(fixed_model))
        difference = independent - mixedmodels_loglikelihood
        push!(rows, (;
            theta = Float64(theta_value),
            mixedmodels_loglikelihood,
            independent_loglikelihood = independent,
            difference,
            absolute_difference = abs(difference),
            independent_gradient_norm = maximum(abs, score),
            agrees = abs(difference) <= tolerance,
        ))
    end
    result = DataFrame(rows)
    return (validated = all(result.agrees), tolerance, comparisons = result)
end
