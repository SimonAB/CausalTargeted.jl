using Test, CausalTargeted, Statistics

@testset "Normalised NNLS does not lose to its own candidate library" begin
    # A weak mean prediction can acquire a large unconstrained coefficient.
    # Normalising afterwards must not turn an almost-correct candidate into
    # an ensemble with much worse held-out prediction loss.
    predictions = hcat([-2.0, -1.0, 1.0, 2.0], fill(0.01, 4))
    outcome = predictions[:, 1] .+ 0.02
    weights = CausalTargeted._nonneg_ls_weights(predictions, outcome)
    risk = mean(abs2, predictions * weights .- outcome)
    candidate_risk = minimum(mean(abs2, predictions[:, j] .- outcome) for j in axes(predictions, 2))
    @test risk <= candidate_risk + 1e-12
    @test all(weights .>= 0)
    @test sum(weights) ≈ 1

    perfect_predictions = hcat(outcome, zeros(4))
    perfect_weights = CausalTargeted._nonneg_ls_weights(perfect_predictions, outcome)
    @test mean(abs2, perfect_predictions * perfect_weights .- outcome) < 1e-20

    # Retain a useful ensemble when two complementary candidates jointly
    # recover the outcome more accurately than either candidate alone.
    blend_predictions = hcat([-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0])
    blend_outcome = 0.5 .* (blend_predictions[:, 1] .+ blend_predictions[:, 2])
    blend_weights = CausalTargeted._nonneg_ls_weights(blend_predictions, blend_outcome)
    blend_risk = mean(abs2, blend_predictions * blend_weights .- blend_outcome)
    best_candidate_risk = minimum(
        mean(abs2, blend_predictions[:, j] .- blend_outcome)
        for j in axes(blend_predictions, 2)
    )
    @test blend_weights ≈ [0.5, 0.5]
    @test blend_risk < best_candidate_risk
end
