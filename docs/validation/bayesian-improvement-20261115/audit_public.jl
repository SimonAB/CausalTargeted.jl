using CSV,TOML,SHA,Statistics,Distributions
length(ARGS)==2 || error("provide decompressed study directory and unpacked generation-source directory")
directory=abspath(ARGS[1])
source_directory=abspath(ARGS[2])
meta=TOML.parsefile(joinpath(directory,"metadata.toml"))
@assert meta["status"]=="complete"
@assert bytes2hex(sha256(read(joinpath(directory,"raw.tsv"))))==meta["raw_sha256"]
for (path,hash) in meta["source_hashes"]
    @assert bytes2hex(sha256(read(joinpath(source_directory,path))))==hash
end
rows=collect(CSV.File(joinpath(directory,"raw.tsv");delim='\t'))
@assert length(rows)==288000
@assert length(unique(r.dataset_id for r in rows))==800
@assert length(unique((r.dataset_id,r.prior_id,r.method) for r in rows))==4800
@assert all(r->r.available && r.reference_available, rows)
groups=Dict{Tuple,Vector{Any}}()
for row in rows
    push!(get!(groups,(row.dataset_id,row.prior_id,row.method,row.draws),Any[]),row)
end
@assert length(groups)==14400
@assert all(g->sort(getproperty.(g,:stream))==collect(1:20),values(groups))
println("AUDIT PASSED: 800 datasets,4800 public fits,288000 rows,14400 complete stream groups; all available; checksum and recorded generation source verified.")
println("elapsed_seconds=",meta["elapsed_seconds"])
for method in ("variational","linear_response"), draws in (500,2000,8000)
    selected=filter(r->r.method==method && r.draws==draws,rows)
    datasets=Dict{Tuple,Vector{Any}}()
    for row in selected
        push!(get!(datasets,(row.cell_id,row.dataset_id),Any[]),row)
    end
    metrics=(:lower_mc_contains_ideal,:upper_mc_contains_ideal,:lower_endpoint_squared_error,:upper_endpoint_squared_error)
    stats=Dict{Symbol,Tuple{Float64,Float64}}()
    for metric in metrics
        bycell=Dict{String,Vector{Float64}}()
        for ((cell,id),block) in datasets
            @assert length(block)==60
            push!(get!(bycell,String(cell),Float64[]),mean(Float64.(getproperty.(block,metric))))
        end
        @assert length(bycell)==8 && all(length(v)==100 for v in values(bycell))
        average=mean(mean(v) for v in values(bycell))
        mcse=sqrt(sum(var(v)/length(v) for v in values(bycell)))/8
        stats[metric]=(average,mcse)
    end
    binomial=Binomial(draws,.025)
    l=quantile(binomial,.025);u=quantile(binomial,.975)+1
    exact=cdf(binomial,u-1)-cdf(binomial,l-1)
    println((;method,draws,lower_mc_inclusion=stats[:lower_mc_contains_ideal],
        upper_mc_inclusion=stats[:upper_mc_contains_ideal],exact_binomial_inclusion=exact,
        lower_endpoint_rmse=sqrt(stats[:lower_endpoint_squared_error][1]),
        upper_endpoint_rmse=sqrt(stats[:upper_endpoint_squared_error][1])))
end
println((maximum_reference_error=maximum(r.reference_error for r in rows),
    maximum_endpoint_refinement=maximum(r.reference_endpoint_change for r in rows)))
