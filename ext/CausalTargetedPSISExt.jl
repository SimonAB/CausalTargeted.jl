module CausalTargetedPSISExt

using CausalTargeted
using PSIS

function _run_psis(log_ratios::AbstractVector{Float64})
    res = PSIS.psis(collect(Float64, log_ratios), 1.0)
    return (; weights = Vector{Float64}(res.weights),
        khat = Float64(res.pareto_shape),
        ess = Float64(res.ess))
end

function __init__()
    CausalTargeted._PSIS_SMOOTHER[] = _run_psis
end

end
