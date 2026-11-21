"""
    MTPSettings

Configuration container for default MTP shift-grid settings (shift grid `deltas`,
support quantiles, cross-fitting `folds`, training `epochs`, `shift_scale`, and
stratum retention thresholds). Construct the package defaults with
[`mtp_settings`](@ref).
"""
struct MTPSettings
    deltas::Vector{Float64}
    lower_q::Float64
    upper_q::Float64
    folds::Int
    epochs::Int
    shift_scale::String
    min_stratum_n::Int
    max_stratum_clamp_prop::Float64
    min_shift_retention::Float64
end

"""
    mtp_settings() -> MTPSettings
"""
function mtp_settings()
    MTPSettings(
        collect(-2.0:0.1:2.0),
        0.01,
        0.99,
        3,
        30,
        "z",
        10,
        0.25,
        0.5,
    )
end

"""
    default_deltas() -> Vector{Float64}

Return the default shift grid `deltas` from [`mtp_settings`](@ref).
"""
default_deltas() = mtp_settings().deltas

"""
    resolved_stratify_by() -> Union{Nothing,Symbol}

Read `MTP_STRATIFY_BY` environment variable (`age`, `immunisation`, `sex`, or unset).
"""
function resolved_stratify_by()
    raw = strip(get(ENV, "MTP_STRATIFY_BY", ""))
    isempty(raw) && return nothing
    lowercase(raw) in ("null", "none") && return nothing
    sym = Symbol(lowercase(raw))
    sym in (:age, :immunisation, :sex) || error("Unexpected MTP_STRATIFY_BY: $raw")
    return sym
end

export mtp_settings, default_deltas, MTPSettings, resolved_stratify_by
