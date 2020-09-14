using Test, RCall

include("base.jl")

@testset "Test subgraphcounts" begin
    n = 50
    g_orig = erdos_renyi(n, 0.05; is_directed=true)
    g_adj = convert(Array{Bool,2}, collect(adjacency_matrix(g_orig)));

    rand_covariate = randn(n) * 0.1
    cat_covariate = rand(1:5, n)

    ## A list of functions which define the model 
    funcs = [
        delta_edge,
        delta_mutual,
        (g, i, j) -> delta_istar(g, i, j, 2),
        (g, i, j) -> delta_ostar(g, i, j, 2),
        delta_m2star,
        delta_ttriple,
        delta_ctriple,
        (g, i, j) -> delta_nodeicov(g, i, j, "cov1"),
        (g, i, j) -> delta_nodeocov(g, i, j, "cov1"),
        (g, i, j) -> delta_nodediff(g, i, j, 2, "cov1")
        ]

    cat_funcs = Function[
        (g, i, j) -> delta_nodeifactor(g, i, j, "cat1"),
        (g, i, j) -> delta_nodeofactor(g, i, j, "cat1"),
        (g, i, j) -> delta_nodemix(g, i, j, "cat1")
        ]

    m = erg_cat(g_adj, funcs, cat_funcs, Dict("cat1" => convert.(UInt32, cat_covariate)); realnodecov=Dict("cov1" => rand_covariate))
    s = subgraphcount(m)
    m2 = erg(g_adj, funcs; realnodecov=Dict("cov1" => rand_covariate))
    s2 = subgraphcount(m2)

    @testset "Test simple subgraphcounts" begin
        @test s[1] == sum(g_adj)              # test edges
        @test s[2] == sum(g_adj .* g_adj') / 2  # test mutual
        @test [sum(g_adj .* cat_covariate' .== x) for x in sort(unique(cat_covariate))] == s[11:15] # test nodeifactor
        @test [sum(g_adj .* cat_covariate .== x) for x in sort(unique(cat_covariate))] == s[16:20] # test nodeofactor
        @test [sum(((g_adj .* (cat_covariate .- 1) * 5.) .+ (g_adj .* (cat_covariate'))) .== x) for x in 1:25] == s[21:end] # test nodemix   
    end
    @testset "Compare erg_cat and erg" begin
        @test s2[1:10] == s[1:10]
    end

    # create variables for R session
    n1 = deepcopy(m.m)
    at_cov = copy(m.realnodecov["cov1"])
    at_cat = convert.(Int64, copy(m.catnodecov["cat1"][1]))
    @rput n1
    @rput at_cov
    @rput at_cat
    R"""
    library(ergm)
    n1a <- as.network(n1)
    set.vertex.attribute(n1a, "cov1", at_cov)
    set.vertex.attribute(n1a, "cat1", at_cat)
    rm1a <- summary(n1a ~ edges + mutual + istar(2) + ostar(2) + m2star + ttriple + ctriple + nodeicov("cov1") + nodeocov("cov1") + absdiff("cov1", pow=2) + nodeifactor("cat1") + nodeofactor("cat1") + nodemix("cat1"))
    nodeifactor_summary <- summary(n1a ~ nodeifactor("cat1"))
    nodeofactor_summary <- summary(n1a ~ nodeofactor("cat1"))
    nodemix_summary <- summary(n1a ~ nodemix("cat1"))
    """

    @rget rm1a
    @rget nodeifactor_summary
    @rget nodeofactor_summary
    @rget nodemix_summary
    @testset "Compare to R's subgraphcounts" begin
        @test isapprox(rm1a[1:10], s[1:10])
        @test nodeifactor_summary == s[12:15]
        @test nodeofactor_summary == s[17:20]
        @test reshape(nodemix_summary, 5, 5) == reshape(s[21:end], 5, 5)'
    end
end

