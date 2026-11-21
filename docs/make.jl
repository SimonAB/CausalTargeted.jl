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
        "Choosing an approach" => "orientation.md",
        "Getting Started" => "getting-started.md",
        "Guides" => [
            "Methods & estimators" => "methods.md",
            "Bayesian Gaussian & cluster" => "bayesian.md",
            "Bayesian logistic (RxInfer)" => "bayesian_logistic.md",
            "Missingness" => "missingness.md",
            "Small-n checklist" => "small_n.md",
        ],
        "API Reference" => [
            "Overview" => "api.md",
            "LMTP, sequential & survival" => "api/lmtp.md",
            "Discrete, hurdle, MSM & DiD" => "api/discrete_msm.md",
            "Parametric & Bayesian g-comp" => "api/gcomp_bayesian.md",
            "Super Learner & schemas" => "api/super_learner.md",
            "Missingness & diagnostics" => "api/diagnostics_missingness.md",
            "Planning, certificates & plotting" => "api/planning_plotting.md",
        ],
        "Architecture & Validation" => [
            "Ecosystem comparison" => "comparison.md",
            "Stress validation" => "stress_validation.md",
        ],
        "References" => "references.md",
    ],
    checkdocs = :exports,
    warnonly = false,
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/SimonAB/CausalTargeted.jl.git",
        devbranch = "main",
        push_preview = true,
    )
end
