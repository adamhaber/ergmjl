# using LightGraphs, MetaGraphs, LinearAlgebra, Distributions
using LinearAlgebra
import LightGraphs:
    AbstractGraph,
    SimpleGraph,
    SimpleDiGraph,
    AbstractEdge,
    SimpleEdge,
    has_edge,
    is_directed,
    edgetype,
    src,
    dst,
    ne,
    nv,
    vertices,
    has_vertex,
    outneighbors,
    inneighbors,
    edges,
    add_edge!,
    rem_edge!,
    erdos_renyi,
    adjacency_matrix,
    indegree,
    outdegree,
    density

#= 
The primary model structure consists of:
1. graph data (currently assumes Boolean true/false; also saves its transpose),
2. List of functions (subgraphs/observables/soft constraints/etc; also saves the length of the list),
3. Real-valued node covariate data, 
4. Int-valued covariate data (for categorical attributes), 
5. An array of edge covariate data (assume Real),
6. Degree data (useful to speed-up certain computations),
7. An array of model parameters associated with the subgraph statistics. =#

struct erg{G <: AbstractArray{Bool},F <: Function} <: AbstractGraph{Int}
    m::G
    fs::Vector{F}
    trans::G
    realnodecov::Union{Dict{String,Array{Float64,1}},Nothing}
    catnodecov::Union{Dict{String,Array{Any,1}},Nothing}
    edgecov::Union{Array{Array{Float64,2}},Nothing}
    indegree::Vector{Int64}
    outdegree::Vector{Int64}
    n_funcs::Int
    kstars_precomputed::Array{Float64,2}
end

# Basic outer constructor when we just have a graph and set of subgraph functions
function erg(
    g::G,
    fs::Vector{F};
    max_star=3,
    realnodecov=nothing,
    catnodecov=nothing,
    edgecov=nothing,
) where {G <: AbstractArray{Bool},F <: Function}
    p = length(fs)
    n = size(g)[1]

    erg(
        g,
        fs,
        collect(transpose(g)),
        realnodecov,
        catnodecov,
        edgecov,
        vec(sum(g, dims=1)),
        vec(sum(g, dims=2)),
        p,
        kstarpre(n, max_star),
    )
end

# Define some needed helper functions
is_directed(g::E) where {E <: erg} = true
edgetype(g::E) where {E <: erg} = SimpleEdge{Int}
ne(g::erg) = sum(g.m)
nv(g::erg) = @inbounds size(g.m)[1]
vertices(g::E) where {E <: erg} = 1:nv(g)
has_vertex(g::E, v) where {E <: erg} = v <= nv(g) && v > 0
indegree(g::E, v::T) where {E <: erg,T <: Int} = @inbounds g.indegree[v]
outdegree(g::E, v::T) where {E <: erg,T <: Int} = @inbounds g.outdegree[v]
density(x) = ne(x) / (nv(x) * (nv(x) - 1))

function has_edge(g::E, s, r) where {E <: erg}
    g.m[s, r]
end

function outneighbors(g::E, node) where {E <: erg}
    @inbounds return (v for v in 1:nv(g) if g.trans[v, node] && node != v)
end

function inneighbors(g::E, node) where {E <: erg}
    @inbounds return (v for v in 1:nv(g) if g.m[v, node] && v != node)
end


# Functions to turn edge on and off - does no checking
function add_edge!(g::E, s, r) where {E <: erg}
    g.m[s, r] = true
    g.trans[r, s] = true
    g.indegree[r] += 1
    g.outdegree[s] += 1
    return nothing
end

function rem_edge!(g::E, s, r) where {E <: erg}
    g.m[s, r] = false
    g.trans[r, s] = false
    g.indegree[r] -= 1
    g.outdegree[s] -= 1
    return nothing
end


# Toggle a single edge in-place
function edge_toggle!(g::E, s, r) where {E <: erg}
    if has_edge(g, s, r)
        rem_edge!(g, s, r)
        return nothing
    else
        add_edge!(g, s, r)
        return nothing
    end
end


# Iterator over edges/arcs
function edges(g::E) where {E <: erg}
    n = nv(g)
    @inbounds return (SimpleEdge(i, j) for j = 1:n for i in 1:n if g.m[i, j] && i != j)
end

# Define iterator over dyads, excluding self-loops
function dyads(g::E) where {E <: AbstractGraph}
    n = nv(g)
    @inbounds return (SimpleEdge(i, j) for j = 1:n for i in 1:n if i != j)
end


# Toggle and edge and get the changes in total number of edges; really basic, but useful.
function delta_edge(g::E, s, r) where {E <: AbstractGraph}
    if !has_edge(g, s, r)
        return 1.0
    else
        return -1.0
    end
end

function count_edges(g::E) where {E <: AbstractGraph}
    sum(g.m)
end

# Toggle edge from s -> r, and get changes in count of reciprocal dyads
function delta_mutual(g::E, s, r) where {E <: AbstractGraph}
    if !g.trans[s, r]
        return 0.0
    elseif !g.m[s, r]
        return 1.0
    else
        return -1.0
    end
end

function count_mutual(g::E) where {E <: AbstractGraph}
    sum(g.m .& g.trans)
end

# delta k-stars using a precalculated table of binomials - MUCH faster
# to maintain compatibility with both erg and simplegraphs, explicitly pass the required degree integer
function delta_istar(g::E, s, r, k) where {E <: AbstractGraph}
    if !has_edge(g, s, r)
        return g.kstars_precomputed[indegree(g, r) + 1, k - 1]
    else
        return -g.kstars_precomputed[indegree(g, r) - 1 + 1, k - 1]
    end
end


function delta_ostar(g::E, s, r, k) where {E <: AbstractGraph}
    if !has_edge(g, s, r)
        return g.kstars_precomputed[outdegree(g, s) + 1, k - 1]
    else
        return -g.kstars_precomputed[outdegree(g, s) - 1 + 1, k - 1]
    end
end


# Change in mixed 2-stars
function delta_m2star(g::G, s, r) where {G <: AbstractGraph}
    if !has_edge(g, s, r) && !has_edge(g, r, s)
        return convert(Float64, indegree(g, s) + outdegree(g, r))
    elseif has_edge(g, s, r) && !has_edge(g, r, s)
        return -convert(Float64, indegree(g, s) + outdegree(g, r))
    elseif !has_edge(g, s, r) && has_edge(g, r, s)
        return convert(Float64, indegree(g, s) + outdegree(g, r) - 2)
    else
        return convert(Float64, -indegree(g, s) - outdegree(g, r) + 2)
    end
end

# Change in transitive triads
# Works very well for arrays, horribly for edgelist
# Note: must use + and * operations to get SIMD - using && is much slower
function delta_ttriple(g::E, s, r) where {E <: AbstractGraph}
    x = 0
    @inbounds for i in vertices(g)
        x +=
            (g.trans[i, r] * g.trans[i, s]) +
            (g.m[i, r] * g.trans[i, s]) +
            (g.m[i, r] * g.m[i, s])
    end
    if !has_edge(g, s, r)
        return convert(Float64, x)
    else
        return -convert(Float64, x)
    end
end


# This works pretty good on arrays and MUCH better on edgelist
# prefer this function
function delta_ttriple2(g::G, s, r) where {G <: AbstractGraph}
    c = 0
    for i in outneighbors(g, r)
        # c += g.trans[i, s]
        c += has_edge(g, s, i)
    end

    for i in inneighbors(g, r)
        # c += g.trans[i, s] + g.m[s, i]
        c += has_edge(g, s, i) + has_edge(g, i, s)
    end

    if !has_edge(g, s, r)
        return convert(Float64, c)
    else
        return -convert(Float64, c)
    end
end


function delta_ctriple(g::E, s, r) where {E <: AbstractGraph}
    x = 0
    # for i in 1:n
    @inbounds for i in vertices(g)
        x += g.trans[i, r] * g.m[i, s]
        # x += has_edge(g, r, i)*has_edge(g, i, s)
    end
    if !has_edge(g, s, r)
        return convert(Float64, x)
    else
        return -convert(Float64, x)
    end
end


# 500 node network benchmarks
# 20 us for simplegraph
# 1.8 us for array
# A combined transitive and cyclic triple function is really no faster
# than separate.
function delta_ttriplectriple(g::E, s, r) where {E <: AbstractGraph}
    a = 0
    b = 0
    @simd for i in vertices(g)
        a +=
            (g.trans[i, r] * g.trans[i, s]) +
            (g.m[i, r] * g.trans[i, s]) +
            (g.m[i, r] * g.m[i, s])
        b += (g.trans[i, r] * g.m[i, s])
    end

    if !has_edge(g, s, r)
        return convert(Float64, a), -convert(Float64, b)
    else
        return -convert(Float64, a), convert(Float64, b)
    end
end


# Effect of covariate on indegrees
function delta_nodeicov(g::G, s, r, cov_name)::Float64 where {G <: AbstractGraph}
    if !has_edge(g, s, r)
        return g.realnodecov[cov_name][r]
    else
        return -g.realnodecov[cov_name][r]
    end
end


# Effect of covariate on outdegrees
function delta_nodeocov(g::G, s, r, cov_name)::Float64 where {G <: AbstractGraph}
    if !has_edge(g, s, r)
        return g.realnodecov[cov_name][s]
    else
        return -g.realnodecov[cov_name][s]
    end
end

# Effect of dyadic distance on covariate p
function delta_nodediff(g::G, s, r, k, cov_name)::Float64 where {G <: AbstractGraph}
    if !has_edge(g, s, r)
        return abs(g.realnodecov[cov_name][s] - g.realnodecov[cov_name][r])^k
    else
        return -abs(g.realnodecov[cov_name][s] - g.realnodecov[cov_name][r])^k
    end
end

# Pre-calculate k-stars
# function kstarpre(g::erg, k)
#     n = nv(g) + 1
#     p = maximum(k)
#     m = zeros(n, p)
#     for j = 1:p
#         for i = 1:n
#             m[i, j] = convert(Float64, binomial(i - 1, j))
#         end
#     end
#     return m
# end

function kstarpre(n::Int, k::Int)
    m = zeros(n, k)
    for j = 1:k
        for i = 1:n
            m[i, j] = convert(Float64, binomial(i - 1, j))
        end
    end
    return m
end


function change_scores(g::E, i, j) where {E <: AbstractGraph}
    x = zeros(g.n_funcs)
    func_list = g.fs
    for k = 1:g.n_funcs
        x[k] = func_list[k](g, i, j)
    end
    return x
end


# Does 500 node graph in 0.18 ms
function subgraphcount(g::E) where {E <: AbstractGraph}
    x = zeros(g.n_funcs)
    g2 = deepcopy(g)

    for j in vertices(g2)
        for i in vertices(g2)
            if i == j
                continue
            elseif !has_edge(g2, i, j)
                continue
            else
                x -= change_scores(g2, i, j)
                edge_toggle!(g2, i, j)
            end
        end
    end
    return x
end


# Generate random graph given a vector of ERGM parameters
# returns total change in graph statistics, number of toggles (and, optionally, the actual graph)
function rgraph(
    theta::Vector{Float64},
    g::E,
    graphstats,
    K::Int64;
    return_graph=true,
) where {E <: AbstractGraph}
    n = nv(g)
    x = copy(graphstats)
    g2 = deepcopy(g)
    toggled = 0

    @inbounds for k = 1:K
        for j = 1:n
            for i = 1:n
                if i == j
                    continue
                else
                    deltastats = change_scores(g2, i, j)
                    if log(rand()) < dot(theta, deltastats)
                        edge_toggle!(g2, i, j)
                        x += deltastats
                        toggled += 1
                    end
                end
            end
        end
    end

    if return_graph == true
        return x, g2, toggled
    else
        return x, toggled
    end
end