# Hurdle-aware conditional independence testing (two-part GLM).
using Distributions: Chisq, Normal, cdf
using GLM
using Graphs: inneighbors, nv, outneighbors
using CategoricalArrays: categorical
using StatsModels: Term, term
using CausalDynamics: TemporalUnrolling, d_separated, temporal_node_label

"""
    IndependenceStatement

Conditional independence claim ``X ⟂ Y | Z`` with optional graph indices and
human-readable labels (e.g. `"fec[1]"`, `"grid_type[2]"`).
"""
struct IndependenceStatement
    x::Int
    y::Int
    z::Vector{Int}
    label_x::String
    label_y::String
    label_z::Vector{String}
    implied_by_dag::Bool
end

"""
    local_markov_statements(unrolling::TemporalUnrolling) -> Vector{IndependenceStatement}

Enumerate pairwise local Markov conditions on an unrolled temporal DAG.
"""
function local_markov_statements(unrolling::TemporalUnrolling)
    g = unrolling.graph
    n = nv(g)
    statements = IndependenceStatement[]
    for v in 1:n
        pa = sort(collect(inneighbors(g, v)))
        desc = _hurdle_descendants(g, v)
        nondesc = [w for w in 1:n if w != v && w ∉ desc && w ∉ Set(pa)]
        for w in nondesc
            implied = d_separated(g, v, w, pa)
            push!(
                statements,
                IndependenceStatement(
                    v,
                    w,
                    copy(pa),
                    temporal_node_label(unrolling, v),
                    temporal_node_label(unrolling, w),
                    [temporal_node_label(unrolling, z) for z in pa],
                    implied,
                ),
            )
        end
    end
    return statements
end

function _hurdle_descendants(g, v::Int)
    seen = Set{Int}()
    stack = collect(outneighbors(g, v))
    while !isempty(stack)
        u = pop!(stack)
        u in seen && continue
        push!(seen, u)
        append!(stack, outneighbors(g, u))
    end
    return seen
end

"""Map temporal label `"var[t]"` to column `var_t` (or `var` when `t == "1"`)."""
function default_hurdle_label_to_col(label::String)
    m = match(r"^([a-z_]+)\[(\d+)\]$", label)
    m === nothing && throw(ArgumentError("expected var[t] label, got $label"))
    var = Symbol(m.captures[1])
    t = m.captures[2]
    return t == "1" ? var : Symbol(string(var, "_t", t))
end

function _hurdle_base_node(
    label::String,
    node_parts::Dict{Symbol, Tuple{Symbol, Symbol}},
)
    m = match(r"^([a-z_]+)\[\d+\]$", label)
    m === nothing && return nothing
    base = Symbol(m.captures[1])
    return haskey(node_parts, base) ? base : nothing
end

function _resolve_col(
    label::String,
    colmap::Union{Nothing, Dict{String, Symbol}},
)
    colmap === nothing && return default_hurdle_label_to_col(label)
    haskey(colmap, label) || throw(ArgumentError("label $label missing from colmap"))
    return colmap[label]
end

function _statement_involves_hurdle(
    st::IndependenceStatement,
    node_parts::Dict{Symbol, Tuple{Symbol, Symbol}},
)
    return _hurdle_base_node(st.label_x, node_parts) !== nothing ||
           _hurdle_base_node(st.label_y, node_parts) !== nothing
end

function _response_side(
    st::IndependenceStatement,
    node_parts::Dict{Symbol, Tuple{Symbol, Symbol}},
)
    _hurdle_base_node(st.label_y, node_parts) !== nothing && return :y
    _hurdle_base_node(st.label_x, node_parts) !== nothing && return :x
    return :y
end

function _prepare_predictor_frame(df::DataFrame, cols::Vector{Symbol})
    sub = copy(df)
    for col in cols
        column = sub[!, col]
        value_type = Base.nonmissingtype(eltype(column))
        if value_type <: AbstractString
            sub[!, col] = categorical(column)
        end
    end
    return sub
end

function _is_categorical_predictor(column)
    value_type = Base.nonmissingtype(eltype(column))
    return value_type <: AbstractString || StatsModels.DataAPI.refpool(column) !== nothing
end

function _wald_p(model, x_col::Symbol)
    ct = coeftable(model)
    prefix = string(x_col)
    pvals = Float64[]
    for (i, name) in enumerate(ct.rownms)
        string(name) == prefix || startswith(string(name), prefix * ":") || continue
        se = ct.cols[2][i]
        se <= 0 && continue
        z = abs(ct.cols[1][i] / se)
        push!(pvals, 2 * (1 - cdf(Normal(), z)))
    end
    return isempty(pvals) ? NaN : minimum(pvals)
end

function _addition_pvalue(model_red, model_full, x_col::Symbol, sub::DataFrame)
    _is_categorical_predictor(sub[!, x_col]) && return _likelihood_ratio_p(model_full, model_red)
    return _wald_p(model_full, x_col)
end

function _likelihood_ratio_p(model_full, model_reduced)
    Δdev = deviance(model_reduced) - deviance(model_full)
    Δdev = max(0.0, Δdev)
    Δdf = length(coef(model_full)) - length(coef(model_reduced))
    Δdf <= 0 && return NaN
    return ccdf(Chisq(Δdf), Δdev)
end

function _regression_ci_test(
    df::DataFrame,
    y_col::Symbol,
    x_col::Symbol,
    z_cols::Vector{Symbol};
    family = Normal(),
    min_n::Int = 10,
    α::Float64 = 0.05,
)
    cols = vcat([y_col, x_col], z_cols)
    use = completecases(select(df, cols))
    sub = _prepare_predictor_frame(df[use, :], vcat([x_col], z_cols))
    n = nrow(sub)
    n < min_n && return (; independent=true, p=NaN, n=n, skipped=true, part="")
    z_terms = Term.(z_cols)
    form_red = isempty(z_cols) ? Term(y_col) ~ term(1) : Term(y_col) ~ sum(z_terms)
    form_full = Term(y_col) ~ sum(vcat([Term(x_col)], z_terms))
    try
        model_red = glm(form_red, sub, family)
        model_full = glm(form_full, sub, family)
        p = _addition_pvalue(model_red, model_full, x_col, sub)
        isnan(p) && return (; independent=true, p=NaN, n=n, skipped=true, part="")
        return (; independent=p >= α, p=p, n=n, skipped=false, part="")
    catch
        return (; independent=true, p=NaN, n=n, skipped=true, part="")
    end
end

"""
    test_implied_hurdle_independences(statements, df, node_parts; α=0.05, colmap=nothing, min_n=10)

Test implied CIs for hurdle-split nodes: binomial presence GLM and Gaussian
intensity GLM among positives. String predictors use StatsModels `DummyCoding`.
"""
function test_implied_hurdle_independences(
    statements::Vector{IndependenceStatement},
    df::DataFrame,
    node_parts::Dict{Symbol, Tuple{Symbol, Symbol}};
    α::Float64 = 0.05,
    colmap::Union{Nothing, Dict{String, Symbol}} = nothing,
    min_n::Int = 10,
)
    rows = NamedTuple[]
    for st in statements
        st.implied_by_dag || continue
        _statement_involves_hurdle(st, node_parts) || continue

        side = _response_side(st, node_parts)
        y_lab = side == :y ? st.label_y : st.label_x
        x_lab = side == :y ? st.label_x : st.label_y
        z_labs = st.label_z

        y_base = _hurdle_base_node(y_lab, node_parts)
        y_base === nothing && continue
        y_pres, y_int = node_parts[y_base]
        x_col = _resolve_col(x_lab, colmap)
        z_cols = Symbol[_resolve_col(z, colmap) for z in z_labs]

        needed = vcat([y_pres, x_col], z_cols)
        all(c -> hasproperty(df, c), needed) || continue

        pres = _regression_ci_test(
            df, y_pres, x_col, z_cols;
            family = Binomial(), min_n = min_n, α = α,
        )
        pres = merge(pres, (part = "presence",))

        int_cols = vcat([y_int, x_col], z_cols)
        if all(hasproperty(df, c) for c in int_cols)
            pos = df[coalesce.(df[!, y_pres], 0.0) .> 0, :]
            int = _regression_ci_test(
                pos, y_int, x_col, z_cols;
                family = Normal(), min_n = min_n, α = α,
            )
            int = merge(int, (part = "intensity",))
        else
            int = (; independent = true, p = NaN, n = 0, skipped = true, part = "intensity")
        end

        for res in (pres, int)
            push!(
                rows,
                (;
                    x = st.label_x,
                    y = st.label_y,
                    z = join(st.label_z, ", "),
                    part = res.part,
                    p = res.p,
                    independent = res.independent,
                    n = res.n,
                    skipped = res.skipped,
                ),
            )
        end
    end
    return rows
end


"""Wide-column symbol for hurdle part at occasion `t` (`1` → no suffix)."""
function _hurdle_wide_col(base::Symbol, part::AbstractString, t::Integer)
    if t == 1
        return Symbol(string(base, "_", part))
    end
    return Symbol(string(base, "_", part, "_t", t))
end

"""
    hurdle_colmap_presence_intensity(base; time=1, lag=false) -> Dict{String, Symbol}

Colmap preset mapping temporal labels `"base[t]"` to wide presence/intensity
columns (e.g. `fec_bin_t2`, `fec_intensity_t2`). With `lag=true`, occasion `t`
maps to columns at `t+1` for lagged outcomes.
"""
function hurdle_colmap_presence_intensity(
    base::Symbol;
    time::Integer = 1,
    lag::Bool = false,
)
    t = Int(time)
    if lag
        pres = Symbol(string(base, "_bin_t", t))
        int = Symbol(string(base, "_intensity_t", t))
    else
        pres = _hurdle_wide_col(base, "bin", t)
        int = _hurdle_wide_col(base, "intensity", t)
    end
    return Dict{String, Symbol}(
        "$(base)[$t]" => pres,
        "$(base)_presence[$t]" => pres,
        "$(base)_intensity[$t]" => int,
    )
end

"""
    hurdle_colmap_grid_arm(; time=1, source=:grid_type, col=nothing) -> Dict{String, Symbol}

Map `"grid_type[t]"` (or another `source` label) to wide arm columns for
categorical arm coding on lag panels.
"""
function hurdle_colmap_grid_arm(;
    time::Integer = 1,
    source::Symbol = :grid_type,
    col::Union{Nothing, Symbol} = nothing,
)
    t = Int(time)
    arm_col = something(col, t == 1 ? :grid_arm : Symbol("grid_arm_t", t))
    return Dict{String, Symbol}("$(source)[$t]" => arm_col)
end

"""
    hurdle_colmap_lag_panel(parts; occasions=(1, 2)) -> Dict{String, Symbol}

Merge colmaps for a two-occasion lag panel. `parts` is a `Dict` of
`base => (presence_suffix, intensity_suffix)` or use standard `"bin"` /
`"intensity"` parts via [`hurdle_colmap_presence_intensity`](@ref).
"""
function hurdle_colmap_lag_panel(
    bases::AbstractVector{Symbol};
    occasions::Tuple{Integer, Integer} = (1, 2),
    unit_level::AbstractVector{Symbol} = Symbol[],
)
    colmap = Dict{String, Symbol}()
    for base in bases
        for (i, t) in enumerate(occasions)
            merge!(colmap, hurdle_colmap_presence_intensity(base; time = t, lag = i == 2))
        end
    end
    for u in unit_level
        for t in occasions
            col = t == 1 ? u : Symbol(string(u, "_t", t))
            colmap["$(u)[$t]"] = col
        end
    end
    return colmap
end

