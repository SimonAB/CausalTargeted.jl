"""Compatibility façades: mediation APIs live in CausalMediation.jl."""

function _require_causal_mediation()
    ext = Base.get_extension(@__MODULE__, :CausalTargetedCausalMediationExt)
    if ext === nothing
        error(
            "Mediation estimation moved to CausalMediation.jl. " *
            "Add CausalMediation to your environment and `using CausalMediation`.",
        )
    end
    return ext
end

"""
    run_mediation_grid(args...; kwargs...)

Deprecated façade — prefer `CausalMediation.run_mediation_grid`.
"""
function run_mediation_grid(args...; kwargs...)
    return _require_causal_mediation().run_mediation_grid(args...; kwargs...)
end

function run_mediation_scalar(args...; kwargs...)
    return _require_causal_mediation().run_mediation_scalar(args...; kwargs...)
end

function run_mediation_scalar_ppl(args...; kwargs...)
    return _require_causal_mediation().run_mediation_scalar_ppl(args...; kwargs...)
end

function run_tmle3_nde(args...; kwargs...)
    return _require_causal_mediation().run_tmle3_nde(args...; kwargs...)
end

function build_mediation_fold_cache(args...; kwargs...)
    return _require_causal_mediation().build_mediation_fold_cache(args...; kwargs...)
end

function prepare_ppl_mediation_spec(args...; kwargs...)
    return _require_causal_mediation().prepare_ppl_mediation_spec(args...; kwargs...)
end

function conjugate_mediation_bootstrap(args...; kwargs...)
    return _require_causal_mediation().conjugate_mediation_bootstrap(args...; kwargs...)
end

function mediation_n_mc_sweep(args...; kwargs...)
    return _require_causal_mediation().mediation_n_mc_sweep(args...; kwargs...)
end

function mediation_stability_summary(args...; kwargs...)
    return _require_causal_mediation().mediation_stability_summary(args...; kwargs...)
end

function mediation_stability_markdown(args...; kwargs...)
    return _require_causal_mediation().mediation_stability_markdown(args...; kwargs...)
end

# Soft type alias when extension loaded
function __init__()
    # Deprecations registered after CausalMediation may load
    return nothing
end

Base.@deprecate run_crumble_grid(args...; kwargs...) run_mediation_grid(args...; kwargs...)
Base.@deprecate run_crumble_scalar(args...; kwargs...) run_mediation_scalar(args...; kwargs...)
Base.@deprecate run_crumble_scalar_ppl(args...; kwargs...) run_mediation_scalar_ppl(args...; kwargs...)
Base.@deprecate build_crumble_fold_cache(args...; kwargs...) build_mediation_fold_cache(args...; kwargs...)
