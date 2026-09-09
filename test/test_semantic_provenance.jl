"""Application-layer propagation of CDM semantic provenance."""

@testset "identification certificate carries CDM provenance" begin
    g = DiGraph(2)
    add_edge!(g, 1, 2)
    result = identify(g, TotalEffectQuery(1, 2))
    provenance = CausalDynamics.CDMProvenance(
        graph = "graph-v1", mechanisms = "mechanisms-v1", observation = "assay-v1",
        policy = "policy-v1", exogenous = "noise-v1", intervention = "do-a-v1",
        coupling = "shared-exogenous-draws",
    )
    certificate = identification_certificate(result, :a, :y; provenance = provenance)
    metadata = CausalTargeted.certificate_dict(certificate)

    @test certificate.provenance === provenance
    @test metadata["cdm_coupling"] == "shared-exogenous-draws"
    @test !isempty(metadata["cdm_fingerprint"])

    legacy = IdentificationCertificate(
        :a, :y, result, Symbol.(result.adjustment), Symbol.(result.mediators),
        :graph, nothing,
    )
    @test legacy.provenance === nothing
end
