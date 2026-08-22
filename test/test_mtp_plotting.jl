@testset "MTP plotting" begin
    @testset "optional extension façade" begin
        @test !CausalTargeted.has_makie()
        error_message = try
            CausalTargeted.plot_mtp_curve(
                [-0.5, 0.0], [0.1, 0.0], [0.0, 0.0], [0.2, 0.0],
            )
            ""
        catch error
            sprint(showerror, error)
        end
        @test occursin("Makie.jl is required", error_message)
        @test occursin("using CairoMakie", error_message)
    end

    import CairoMakie

    @test CausalTargeted.has_makie()
    @test Base.get_extension(CausalTargeted, :CausalTargetedMakieExt) !== nothing

    function png_dimensions(path)
        bytes = read(path)
        @test bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        bigendian(offset) = foldl(
            (value, byte) -> (value << 8) | Int(byte),
            bytes[offset:(offset + 3)];
            init = 0,
        )
        return bigendian(17), bigendian(21)
    end

    shift = [0.5, -0.5, 0.0]
    estimate = [0.10, -0.08, 0.0]
    lower = [0.04, -0.14, -0.03]
    upper = [0.16, -0.02, 0.03]

    @testset "one TE curve and input validation" begin
        original = deepcopy((shift, estimate, lower, upper))
        fig, ax = CausalTargeted.plot_mtp_curve(
            shift, estimate, lower, upper; title = "Exposure -> Outcome",
        )
        @test fig isa CairoMakie.Figure
        @test ax isa CairoMakie.Axis
        @test shift == original[1]
        @test estimate == original[2]
        @test lower == original[3]
        @test upper == original[4]
        @test !any(item -> item isa CairoMakie.Colorbar, fig.content)

        scratch = CairoMakie.Figure()
        scratch_axis = CairoMakie.Axis(scratch[1, 1])
        handles = CausalTargeted.mtp_curve!(scratch_axis, shift, estimate, lower, upper)
        @test handles.estimands == [:TE]
        @test handles.series[1].shift == sort(shift)
        permutation = sortperm(shift)
        @test handles.series[1].lower == lower[permutation]
        @test handles.series[1].upper == upper[permutation]
        @test handles.series[1].estimate_line.linewidth[] ≈ 0.9 * 120 / 25.4
        @test handles.series[1].lower_edge.linewidth[] ≈ 0.6 * 120 / 25.4
        marker_size = handles.series[1].points.markersize[]
        @test all(value -> value ≈ 2.0 * 120 / 25.4, Tuple(marker_size))
        @test handles.series[1].points.strokewidth[] ≈ 0.28 * 120 / 25.4

        @test_throws DimensionMismatch CausalTargeted.plot_mtp_curve(
            shift, estimate[1:2], lower, upper,
        )
        @test_throws ArgumentError CausalTargeted.plot_mtp_curve(
            [0.0, Inf], estimate[1:2], lower[1:2], upper[1:2],
        )
        @test_throws ArgumentError CausalTargeted.plot_mtp_curve(
            shift, estimate, [0.2, -0.1, 0.0], [0.1, 0.0, 0.1],
        )
        @test_throws ArgumentError CausalTargeted.plot_mtp_curve(
            shift, estimate, lower, upper; estimand = ["TE", "bad", "NIE"],
        )
    end

    @testset "TE/NDE/NIE styling, sorting, and clamp strip" begin
        all_shift = repeat(shift, 3)
        all_estimate = vcat(estimate, 0.7 .* estimate, -0.4 .* estimate)
        all_lower = all_estimate .- 0.04
        all_upper = all_estimate .+ 0.06
        all_estimand = repeat(["TE", "NDE", "NIE"]; inner = length(shift))
        all_clamp = repeat([1.2, -0.2, 0.5], 3)
        original_shift = copy(all_shift)
        original_clamp = copy(all_clamp)

        fig, _ = CausalTargeted.plot_mtp_curve(
            all_shift,
            all_estimate,
            all_lower,
            all_upper;
            estimand = all_estimand,
            clamp = all_clamp,
        )
        @test count(item -> item isa CairoMakie.Colorbar, fig.content) == 1

        scratch = CairoMakie.Figure()
        ax = CairoMakie.Axis(scratch[1, 1])
        handles = CausalTargeted.mtp_curve!(
            ax,
            all_shift,
            all_estimate,
            all_lower,
            all_upper;
            estimand = all_estimand,
            clamp = all_clamp,
        )
        @test handles.estimands == [:TE, :NDE, :NIE]
        @test all(series -> series.shift == sort(shift), handles.series)
        @test length(handles.clamp_geometry) == length(unique(shift))
        @test getproperty.(handles.clamp_geometry, :value) == [0.0, 0.5, 1.0]
        @test all(
            rectangle -> rectangle.ymax - rectangle.ymin ≈ 0.055 * handles.geometry.y_span,
            handles.clamp_geometry,
        )
        @test all(
            rectangle -> handles.geometry.y_min - rectangle.ymax ≈
                         0.020 * handles.geometry.y_span,
            handles.clamp_geometry,
        )
        @test all(
            rectangle -> rectangle.xmax - rectangle.xmin ≈ 0.5,
            handles.clamp_geometry,
        )
        @test all_shift == original_shift
        @test all_clamp == original_clamp

        @test_throws ArgumentError CausalTargeted.mtp_curve!(
            ax,
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.1, -0.1],
            [0.1, 0.1];
            estimand = ["TE", "NDE"],
            clamp = [0.1, 0.2],
        )

        _, no_clamp_axis = CausalTargeted.plot_mtp_curve(
            shift,
            estimate,
            lower,
            upper;
            clamp = [missing, NaN, missing],
        )
        @test no_clamp_axis !== nothing
    end

    @testset "native run_lmtp_grid DataFrame interface" begin
        grid = DataFrame(
            delta = shift,
            estimand = fill("TE", 3),
            est = estimate,
            se = fill(0.03, 3),
            lwr = lower,
            upr = upper,
            lwr_sim = lower .- 0.02,
            upr_sim = upper .+ 0.02,
            clamp = [0.2, 0.4, 0.1],
        )
        original = deepcopy(grid)
        fig, _ = CausalTargeted.plot_mtp_curve(grid)
        @test fig isa CairoMakie.Figure
        @test grid == original
        @test count(item -> item isa CairoMakie.Colorbar, fig.content) == 1

        simultaneous, _ = CausalTargeted.plot_mtp_curve(
            grid;
            lower = :lwr_sim,
            upper = "upr_sim",
            ylabel = "Estimated effect (simultaneous 95% band)",
        )
        @test simultaneous isa CairoMakie.Figure

        no_estimand = select(grid, Not(:estimand))
        default_te_fig, _ = CausalTargeted.plot_mtp_curve(no_estimand)
        @test default_te_fig isa CairoMakie.Figure

        invalid = select(grid, Not([:lwr, :upr]))
        message = try
            CausalTargeted.plot_mtp_curve(invalid)
            ""
        catch error
            sprint(showerror, error)
        end
        @test occursin("missing required columns", message)
        @test occursin("lwr", message)
        @test occursin("upr", message)
    end

    @testset "panel_mode and ribbon_alpha" begin
        scratch = CairoMakie.Figure()
        ax = CairoMakie.Axis(scratch[1, 1])
        handles = CausalTargeted.mtp_curve!(
            ax, shift, estimate, lower, upper; panel_mode = true,
        )
        @test handles.series[1].estimate_line.linewidth[] ≈ 0.7 * 120 / 25.4
        marker_size = handles.series[1].points.markersize[]
        @test all(value -> value ≈ 0.8 * 120 / 25.4, Tuple(marker_size))

        @test_throws ArgumentError CausalTargeted.mtp_curve!(
            ax, shift, estimate, lower, upper; ribbon_alpha = -0.1,
        )
        @test_throws ArgumentError CausalTargeted.mtp_curve!(
            ax, shift, estimate, lower, upper; ribbon_alpha = 1.1,
        )
    end

    @testset "exact-size export smoke tests" begin
        mktempdir() do directory
            single, _ = CausalTargeted.plot_mtp_curve(shift, estimate, lower, upper)
            single_png = joinpath(directory, "single.png")
            single_pdf = joinpath(directory, "single.pdf")
            CairoMakie.save(single_png, single; px_per_unit = 320 / 120)
            CairoMakie.save(single_pdf, single; pt_per_unit = 72 / 120)
            @test filesize(single_png) > 0
            @test filesize(single_pdf) > 0
            @test png_dimensions(single_png) == (2720, 1440)
        end
    end
end
