project_text = read(joinpath(@__DIR__, "..", "Project.toml"), String)

@testset "package source portability" begin
    @test !occursin(r"(?ms)^\[sources\].*?^CausalDynamics\s*=\s*\{path\s*=", project_text)
end
