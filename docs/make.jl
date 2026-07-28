using Documenter
using CausalTargeted

makedocs(
    sitename = "CausalTargeted.jl",
    authors = "Simon A. Babayan",
    modules = [CausalTargeted],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://simonab.github.io/CausalTargeted.jl",
        assets = String[],
        example_size_threshold = 0,
    ),
    pages = [
        "Home" => "index.md",
        "Methods and literature" => "methods.md",
        "Small-n checklist" => "small_n.md",
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
