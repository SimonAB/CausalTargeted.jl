using Test, CausalTargeted, StableRNGs

@testset "Explicit inference options survive dispatch" begin
    options = CausalTargeted._lmtp_grid_kwargs((weight_truncation = false, unknown_option = :ignore))
    @test options == (weight_truncation = false,)
    mediation_options = CausalTargeted._mediation_grid_kwargs((weight_truncation = false,), 16)
    @test get(mediation_options, :weight_truncation, nothing) === false
    @test mediation_options.n_mc == 16
end

@testset "Typed discrete execution preserves requested inference and its status" begin
    data, _ = simulate_categorical_treatment_mtp(90; rng = StableRNG(812))
    target = DiscreteInterventionalMean(:A, :Y, [:W], discrete_recode_policy(Dict("2" => "1")))
    for truncated in (false, true)
        result = execute_estimand(target, data; metadata = false, folds = 2,
            weight_truncation = truncated, rng = StableRNG(813))
        @test truncated ? isnan(only(result.se)) : isfinite(only(result.se))
        @test :inference_status in propertynames(result)
        if :inference_status in propertynames(result)
            @test only(result.inference_status) === (truncated ? :truncated_weights : :available)
        end
    end
end

@testset "Small-sample profiles do not select caps by weight variance" begin
    for n in (25, 60, 200), engine in (:lmtp, :mediation)
        @test recommend_run_options(n; engine = engine).cv_trunc === false
    end
end
