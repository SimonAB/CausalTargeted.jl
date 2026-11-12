using Test

@testset "Logistic optional stack stays optional" begin
    code = """
    using CausalTargeted, DataFrames, LinearAlgebra
    @assert Base.get_extension(CausalTargeted, :CausalTargetedRxInferExt) === nothing
    prior = GaussianCoefficientPrior(mean=zeros(2), covariance=Matrix{Float64}(I,2,2))
    data = DataFrame(A=[0.,1.,0.,1.],Y=[0,1,1,0])
    caught = try
        fit_bayesian_logistic(data,:Y,:A,Symbol[];prior)
        nothing
    catch error
        error
    end
    @assert caught isa ArgumentError
    @assert occursin("optional extension",sprint(showerror,caught))
    """
    command = `$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) --threads=auto -e $code`
    @test success(pipeline(command; stdout = devnull))
end
