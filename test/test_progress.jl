"""
    @include_test "test_file.jl"

Include a literal path in the caller's module and source context, logging and
flushing per-file progress to stderr. DONE means the include returned, not that
its recorded assertions passed. Exceptions are logged and rethrown unchanged.
"""
macro include_test(path)
    path isa String || throw(ArgumentError("@include_test requires a literal filename"))
    caller_include = esc(:(include($path)))
    quote
        local started = time_ns()
        println(stderr, "TEST_FILE_START ", $path)
        flush(stderr)
        try
            local result = $caller_include
            local seconds = (time_ns() - started) / 1e9
            println(stderr, "TEST_FILE_DONE ", $path, " seconds=", seconds)
            flush(stderr)
            result
        catch exception
            local seconds = (time_ns() - started) / 1e9
            println(stderr, "TEST_FILE_FAILED ", $path, " seconds=", seconds,
                " error=", sprint(showerror, exception))
            flush(stderr)
            rethrow()
        end
    end
end
