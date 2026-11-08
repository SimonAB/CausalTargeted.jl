using Test, CausalTargeted, StableRNGs

@testset "Bootstrap cross-fitting groups resampled copies by source identity" begin
    source_rows = [1, 2, 1, 2, 3, 4]
    row_groups = CausalTargeted._bootstrap_fold_groups(source_rows)
    @test row_groups == source_rows
    cluster_groups = CausalTargeted._bootstrap_fold_groups(source_rows;
        cluster_ids = [:a, :a, :b, :b])
    @test cluster_groups == [:a, :a, :a, :a, :b, :b]
    for groups in (row_groups, cluster_groups)
        partitions = crossfit_indices(length(groups), 2, StableRNG(75); cluster = groups)
        for group in unique(groups)
            locations = findall(==(group), groups)
            @test count(partition -> !isempty(intersect(partition, locations)), partitions) == 1
        end
    end
end
