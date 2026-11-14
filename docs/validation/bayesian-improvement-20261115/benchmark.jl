using CausalTargeted, RxInfer, GraphPPL, DataFrames, LinearAlgebra, Random, Statistics, SHA
BLAS.set_num_threads(1)
source = read(`git show cf28065:src/bayesian_logistic.jl`, String)
first_line = findfirst("function bayesian_policy_contrast(", source).start
baseline = replace(source[first_line:end], "function bayesian_policy_contrast(" => "function _baseline_policy_contrast("; count=1)
Base.include_string(CausalTargeted, baseline, "baseline-cf28065-policy.jl")
data = DataFrame(A=repeat([0,1],25),Y=repeat([0,1,1,0,1],10))
prior=GaussianCoefficientPrior(mean=zeros(2),covariance=2.25Matrix{Float64}(I,2,2))
fit=fit_bayesian_logistic(data,:Y,:A,Symbol[];prior)
policy=smooth_bounded_logit_shift_policy(-1,2)
for k in (1,3,10)
    deltas=collect(range(0.1,0.5;length=k == 1 ? 2 : k))[1:k]
    legacy()=Base.invokelatest(CausalTargeted._baseline_policy_contrast,fit,data,policy;deltas,n_draws=2000,rng=Xoshiro(12))
    current()=bayesian_policy_contrast(fit,data,policy;deltas,n_draws=2000,rng=Xoshiro(12))
    oldtime=@elapsed old=legacy()
    newtime=@elapsed new=current()
    @assert all(getproperty(old,f)==getproperty(new,f) for f in (:mean,:covariance,:lower,:upper,:mcse,:point_estimate))
    times_old=[@elapsed legacy() for _ in 1:30]
    times_new=[@elapsed current() for _ in 1:30]
    bytes_old=@allocated legacy()
    bytes_new=@allocated current()
    println((;k,n_draws=2000,baseline_median_seconds=median(times_old),candidate_median_seconds=median(times_new),
        baseline_allocated_bytes=bytes_old,candidate_allocated_bytes=bytes_new,
        baseline_first_call_seconds=oldtime,candidate_first_call_seconds=newtime,
        baseline_source_sha256=bytes2hex(sha256(source)),identity=true))
end
