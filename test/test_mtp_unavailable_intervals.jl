using Test, CausalTargeted, CairoMakie

@testset "Point curves remain plottable without inferential intervals" begin
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    handles = CausalTargeted.mtp_curve!(ax, [-0.5, 0.5], [-0.2, 0.2], fill(NaN, 2), fill(NaN, 2))
    @test handles.series[1].estimate == [-0.2, 0.2]
    @test handles.series[1].ribbon === nothing
    @test handles.series[1].lower_edge === nothing
    @test handles.series[1].upper_edge === nothing
    @test isfinite(handles.geometry.y_span)
    @test_throws ArgumentError CausalTargeted.mtp_curve!(ax, [0.0], [0.1], [NaN], [0.2])
    @test_throws ArgumentError CausalTargeted.mtp_curve!(ax, [0.0], [0.1], [-Inf], [Inf])
end
