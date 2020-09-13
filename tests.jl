using Test

include("base.jl")

@testset "categorical vs continuous" begin
    n = 100
    g_orig = erdos_renyi(n, 0.05; is_directed=true)
    g_adj = convert(Array{Bool,2}, collect(adjacency_matrix(g_orig)))

    rand_covariate = randn(n) * 0.1
    cat_covariate = rand(1:5, n)
    funcs = [
        delta_edge, 
        delta_mutual, 
        (g, i, j) -> delta_istar(g, i, j, 2),
        (g, i, j) -> delta_ostar(g, i, j, 2),
        delta_m2star,
        delta_ttriple,
        (g, i, j) -> delta_nodeicov(g, i, j, "cov1"),
        (g, i, j) -> delta_nodeocov(g, i, j, "cov1")
        ]
    m1 = erg(g_adj, funcs; realnodecov=Dict("cov1" => rand_covariate))
    s1 = subgraphcount(m1)
        
    cat_funcs = Function[
        (g, i, j) -> delta_nodeifactor(g, i, j, "cat1"),
        (g, i, j) -> delta_nodeofactor(g, i, j, "cat1"),
        (g, i, j) -> delta_nodemix(g, i, j, "cat1")
        ]
        
    m2 = erg_cat(g_adj, funcs, cat_funcs, Dict("cat1" => convert.(UInt32, cat_covariate)); realnodecov=Dict("cov1" => rand_covariate))
    s2 = subgraphcount(m2)
    @test s2[1:length(funcs)] == s1
    @test s1[1] == sum(m1.m)
end
