using Documenter
using CausalTargeted
using CausalDynamics
using Graphs
using DAGMakie
using CairoMakie

# Prefer PNG MIME so Documenter writes figure files instead of huge inline HTML
# (same convention as DAGMakie.jl / CausalDynamics.jl docs).
CairoMakie.activate!(type = "png")
CairoMakie.enable_only_mime!("png")

makedocs(
    sitename = "CausalTargeted.jl",
    authors = "Simon A. Babayan",
    modules = [CausalTargeted],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://simonab.github.io/CausalTargeted.jl",
        assets = String[],
        example_size_threshold = 0,  # always write @example figures to files
    ),
    pages = [
        "Home" => "index.md",
        "Comparison" => "comparison.md",
        "Methods and literature" => "methods.md",
        "Small-n checklist" => "small_n.md",
        "Stress validation" => "stress_validation.md",
        "API overview" => "api.md",
        "References" => "references.md",
    ],
    checkdocs = :none,
    warnonly = [:missing_docs, :cross_references],
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/SimonAB/CausalTargeted.jl.git",
        devbranch = "main",
        push_preview = true,
    )
end
