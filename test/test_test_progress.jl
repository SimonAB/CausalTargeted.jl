using Test
isdefined(@__MODULE__, Symbol("@include_test")) || include(joinpath(@__DIR__, "test_progress.jl"))

@testset "per-file test progress retains include semantics" begin
    mktempdir() do directory
        mkpath(joinpath(directory, "nested"))
        write(joinpath(directory, "nested", "success.jl"), "global progress_start_was_flushed = occursin(\"TEST_FILE_START nested/success.jl\", read(progress_log_path, String))\nglobal progress_fixture_value = 41\nprogress_fixture_value + 1\n")
        write(joinpath(directory, "nested", "failure.jl"), "throw(progress_fixture_exception)\n")
        output = joinpath(directory, "progress.log")
        entry = joinpath(directory, "entry.jl")
        write(entry, """
        module TestProgressFixture
            using Main: @include_test
            const progress_log_path = $(repr(output))
            const progress_fixture_exception = ErrorException("deliberate include failure")
            const success_return = @include_test "nested/success.jl"
            const progress_done_was_flushed = occursin("TEST_FILE_DONE nested/success.jl", read(progress_log_path, String))
            const failure_caught = try
                @include_test "nested/failure.jl"
                nothing
            catch exception
                exception
            end
        end
        """)
        open(output, "w") do io
            redirect_stderr(io) do
                Base.include(Main, entry)
            end
        end
        fixture = Base.invokelatest(getproperty, Main, :TestProgressFixture)
        fixture_value(name) = Base.invokelatest(getproperty, fixture, name)
        @test fixture_value(:progress_fixture_value) == 41
        @test fixture_value(:success_return) == 42
        @test fixture_value(:failure_caught) isa LoadError
        @test fixture_value(:failure_caught).error === fixture_value(:progress_fixture_exception)
        @test fixture_value(:progress_start_was_flushed)
        @test fixture_value(:progress_done_was_flushed)
        logs = read(output, String)
        @test occursin("TEST_FILE_START nested/success.jl", logs)
        @test occursin(r"TEST_FILE_DONE nested/success.jl seconds=[0-9.e+-]+", logs)
        @test occursin("TEST_FILE_START nested/failure.jl", logs)
        @test occursin(r"TEST_FILE_FAILED nested/failure.jl seconds=[0-9.e+-]+", logs)
        @test occursin("deliberate include failure", logs)
        @test !occursin("TEST_FILE_DONE nested/failure.jl", logs)
        @test_throws ArgumentError macroexpand(Main, :(@include_test identity("fixture.jl")))
    end
end
