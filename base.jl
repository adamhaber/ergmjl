# using LightGraphs, MetaGraphs, LinearAlgebra, Distributions
using Statistics, BenchmarkTools, LinearAlgebra, LightGraphs, FunctionWrappers, LoopVectorization, ThreadPools, FunctionWranglers
using FunctionWrappers: FunctionWrapper
import LinearAlgebra: dot
import LightGraphs: AbstractGraph, SimpleGraph, SimpleDiGraph, AbstractEdge, SimpleEdge, has_edge, is_directed,
edgetype, src, dst, ne, nv, vertices, has_vertex, outneighbors, inneighbors, edges, add_edge!, rem_edge!, erdos_renyi, adjacency_matrix, indegree, outdegree, density

#= 
We use a one-hot representation of categorical features change statistics.
The following is a set of helper functions to efficiently do various
operations on these representations =#

import Base: *

struct SignedOneHotVector <: AbstractVector{Int8}
    ix::UInt32
    of::UInt32
    sign::Int8
end

Base.size(xs::SignedOneHotVector) = (Int64(xs.of),)
Base.getindex(xs::SignedOneHotVector, i::Integer) = (i == xs.ix) * xs.sign

function onehot(i, len, sign=1)
    SignedOneHotVector(i, len, sign)
end

function Base.:-(b::SignedOneHotVector)
    return onehot(b.ix, b.of, -b.sign)
end

function dot(v::Array{Float64,1}, b::SignedOneHotVector)
    return b.sign * v[b.ix]
end

function dot(b::SignedOneHotVector, v::Array{Float64,1})
    return b.sign * v[b.ix]
end

function dot(vs::Array{SignedOneHotVector,1}, b::Array{Float64,1})
    sum = 0
    current_len = 0
    for v in vs
        sum += v.sign * b[v.ix + current_len]
        current_len += v.of
    end
    return sum
end

function dot(b::Array{Float64,1}, vs::Array{SignedOneHotVector,1})
    sum = 0
    current_len = 0
    for v in vs
        sum += v.sign * b[v.ix + current_len]
        current_len += v.of
    end
    return sum
end

function dot(b::Array{Float64,1}, t::Tuple{Array{Float64,1},Array{SignedOneHotVector,1}})
    return dot(b[1:length(t[1])], t[1]) + dot(b[(length(t[1]) + 1):end], t[2])
end

function dot(t::Tuple{Array{Float64,1},Array{SignedOneHotVector,1}}, b::Array{Float64,1})
    return dot(b[1:length(t[1])], t[1]) + dot(b[(length(t[1]) + 1):end], t[2])
end

function Base.:-(b::Array{Float64,1}, t::Tuple{Array{Float64,1},Array{SignedOneHotVector,1}})
    b[1:length(t[1])] -= t[1]
    current_len = length(t[1])
    for v in t[2]
        b[current_len + v.ix] -= v.sign
        current_len += v.of
    end
    b
end

function Base.:+(b::Array{Float64,1}, t::Tuple{Array{Float64,1},Array{SignedOneHotVector,1}})
    b[1:length(t[1])] += t[1]
    current_len = length(t[1])
    for v in t[2]
        b[current_len + v.ix] += v.sign
        current_len += v.of
    end
    b
end

#= 
The primary model structure consists of:
1. graph data (currently assumes Boolean true/false; also saves its transpose),
2. List of functions (subgraphs/observables/soft constraints/etc),
3. Real-valued node covariate data, 
4. Real-valued edge covariate data,
5. In/outdegree data (useful to speed-up certain computations),
6. Number of model parameters,
7. Precomputed quantities for various k-star computations. =#

abstract type AbstractERGM <: AbstractGraph{Int} end

struct erg{G <: AbstractArray{Bool},T <: FunctionWrangler} <: AbstractERGM
    m::G
    fs::T
    trans::G
    realnodecov::Union{Dict{String,Array{Float64,1}},Nothing}
    edgecov::Union{Array{Array{Float64,2}},Nothing}
    indegree::Vector{Int64}
    outdegree::Vector{Int64}
    n_params::Int
    kstars_precomputed::Array{Float64,2}
end

# An variant that supports categorical nodal features as well
struct erg_cat{G <: AbstractArray{Bool},T <: FunctionWrangler} <: AbstractERGM
    m::G
    fs::T
    cat_fs::Vector{Function}
    trans::G
    realnodecov::Union{Dict{String,Array{Float64,1}},Nothing}
    catnodecov::Dict{String,Tuple{Array{UInt32,1},UInt32}}
    edgecov::Union{Array{Array{Float64,2}},Nothing}
    indegree::Vector{Int64}
    outdegree::Vector{Int64}
    n_params::Int
    kstars_precomputed::Array{Float64,2}
end

# constructors
function erg(g::AbstractArray{Bool}, fs::Vector{Function}; max_star=3, realnodecov=nothing, edgecov=nothing)
    p = length(fs)
    n = size(g)[1]
    erg(g, FunctionWrangler(fs), collect(transpose(g)), realnodecov, edgecov, vec(sum(g, dims=1)), vec(sum(g, dims=2)), p, kstarpre(n, max_star))
end

function erg_cat(g::AbstractArray{Bool}, fs::Vector{Function}, fs_cat::Vector{Function}, catnodecov::Dict{String,Array{UInt32,1}}; max_star=3, realnodecov=nothing, edgecov=nothing)
    p = length(fs) 
    n = size(g)[1]

    # hacky way to compute the number of parameters in advance
    tmp = erg_cat(g, FunctionWrangler(fs), fs_cat, collect(transpose(g)), realnodecov, Dict(key => (value, convert(UInt32, length(unique(value)))) for (key, value) in catnodecov), edgecov, vec(sum(g, dims=1)), vec(sum(g, dims=2)), p, kstarpre(n, max_star))
    p_new = p + sum([f(tmp, 1, 1).of for f in fs_cat])

    erg_cat(g, FunctionWrangler(fs), fs_cat, collect(transpose(g)), realnodecov, Dict(key => (value, convert(UInt32, length(unique(value)))) for (key, value) in catnodecov), edgecov, vec(sum(g, dims=1)), vec(sum(g, dims=2)), convert(Int64, p_new), kstarpre(n, max_star))
end


# Define some needed helper functions
# Define some needed helper functions
is_directed(g::AbstractERGM) = true
edgetype(g::AbstractERGM) = SimpleEdge{Int}
ne(g::AbstractERGM) = sum(g.m)
nv(g::AbstractERGM) = @inbounds size(g.m)[1]
vertices(g::AbstractERGM) = 1:nv(g)
has_vertex(g::AbstractERGM, v) = v <= nv(g) && v > 0
indegree(g::AbstractERGM, v::Int) = @inbounds g.indegree[v]
outdegree(g::AbstractERGM, v::Int) = @inbounds g.outdegree[v]
density(x) = ne(x) / (nv(x) * (nv(x) - 1))

@inline function has_edge(g::AbstractERGM, s, r)
    g.m[s, r]
end

function outneighbors(g::AbstractERGM, node)
    @inbounds return (v for v in 1:nv(g) if g.trans[v, node] && node != v)
end

function inneighbors(g::AbstractERGM, node)
    @inbounds return (v for v in 1:nv(g) if g.m[v, node] && v != node)
end


# Functions to turn edge on and off - does no checking
function add_edge!(g::AbstractERGM, s, r)
    g.m[s, r] = true
    g.trans[r, s] = true
    g.indegree[r] += 1
    g.outdegree[s] += 1
    return nothing
end

function rem_edge!(g::AbstractERGM, s, r)
    g.m[s, r] = false
    g.trans[r, s] = false
    g.indegree[r] -= 1
    g.outdegree[s] -= 1
    return nothing
end


# Toggle a single edge in-place
function edge_toggle!(g::AbstractERGM, s, r)
    if has_edge(g, s, r)
        rem_edge!(g, s, r)
        return nothing
    else
        add_edge!(g, s, r)
        return nothing
    end
end


# Iterator over edges/arcs
function edges(g::AbstractERGM)
    n = nv(g)
    @inbounds return (SimpleEdge(i, j) for j = 1:n for i in 1:n if g.m[i, j] && i != j)
end

# Define iterator over dyads, excluding self-loops
function dyads(g::AbstractGraph)
    n = nv(g)
    @inbounds return (SimpleEdge(i, j) for j = 1:n for i in 1:n if i != j)
end


# Toggle and edge and get the changes in total number of edges; really basic, but useful.
@inline function delta_edge(g::AbstractGraph, s, r)
    if !has_edge(g, s, r)
        return 1.0
    else
        return -1.0
    end
end

function count_edges(g::AbstractGraph)
    sum(g.m)
end

# Toggle edge from s -> r, and get changes in count of reciprocal dyads
@inline function delta_mutual(g::AbstractGraph, s, r)
    if !g.trans[s, r]
        return 0.0
    elseif !g.m[s, r]
        return 1.0
    else
        return -1.0
    end
end

function count_mutual(g::AbstractGraph)
    sum(g.m .& g.trans)
end

# delta k-stars using a precalculated table of binomials - MUCH faster
# to maintain compatibility with both erg and simplegraphs, explicitly pass the required degree integer
@inline function delta_istar(g::AbstractGraph, s, r, k)
    if !has_edge(g, s, r)
        return g.kstars_precomputed[indegree(g, r) + 1, k - 1]
    else
        return -g.kstars_precomputed[indegree(g, r) - 1 + 1, k - 1]
    end
end


@inline function delta_ostar(g::AbstractGraph, s, r, k)::Float64
    if !has_edge(g, s, r)
        return g.kstars_precomputed[outdegree(g, s) + 1, k - 1]
    else
        return -g.kstars_precomputed[outdegree(g, s) - 1 + 1, k - 1]
    end
end


# Change in mixed 2-stars
@inline function delta_m2star(g::AbstractGraph, s, r)
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
function delta_ttriple(g::AbstractGraph, s, r)
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
function delta_ttriple2(g::AbstractGraph, s, r)
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


function delta_ctriple(g::AbstractGraph, s, r)
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
function delta_ttriplectriple(g::AbstractGraph, s, r)
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
function delta_nodeicov(g::AbstractGraph, s, r, cov_name)::Float64
    if !has_edge(g, s, r)
        return g.realnodecov[cov_name][r]
    else
        return -g.realnodecov[cov_name][r]
    end
end

# Effect of nodal factor on indegrees
function delta_nodeifactor(g::AbstractGraph, s, r, cov_name)::SignedOneHotVector
    vals, levels = g.catnodecov[cov_name]
    if !has_edge(g, s, r)
        return onehot(vals[r], levels)
    else
        return -onehot(vals[r], levels)
    end
end


# Effect of covariate on outdegrees
function delta_nodeocov(g::AbstractGraph, s, r, cov_name)::Float64
    if !has_edge(g, s, r)
        return g.realnodecov[cov_name][s]
    else
        return -g.realnodecov[cov_name][s]
    end
end

# Effect of nodal factor on outdegrees
function delta_nodeofactor(g::AbstractGraph, s, r, cov_name)::SignedOneHotVector
    vals, levels = g.catnodecov[cov_name]
    if !has_edge(g, s, r)
        return onehot(vals[s], levels)
    else
        return -onehot(vals[s], levels)
    end
end

# Effect of nodal factor on outdegrees
function delta_nodemix(g::AbstractGraph, s, r, cov_name)::SignedOneHotVector
    vals, levels = g.catnodecov[cov_name]
    if !has_edge(g, s, r)
        return onehot((vals[s] - 1) * levels + vals[r], levels^2)
    else
        return -onehot((vals[s] - 1) * levels + vals[r], levels^2)
    end
end

# Effect of dyadic distance on covariate p
function delta_nodediff(g::AbstractGraph, s, r, k, cov_name)::Float64
    if !has_edge(g, s, r)
        return abs(g.realnodecov[cov_name][s] - g.realnodecov[cov_name][r])^k
    else
        return -abs(g.realnodecov[cov_name][s] - g.realnodecov[cov_name][r])^k
    end
end

function kstarpre(n::Int, k::Int)
    m = zeros(n, k)
    for j = 1:k
        for i = 1:n
            m[i, j] = convert(Float64, binomial(i - 1, j))
        end
    end
    return m
end


# function change_scores(g::erg, i, j)
#     x = zeros(g.n_funcs)
#     func_list = g.fs
#     for k = 1:g.n_funcs
#         x[k] = func_list[k](g, i, j)
#     end
#     return x
# end

@inline function change_scores(g::erg, i, j)
    res = zeros(Float64, length(g.fs))
    smap!(res, g.fs, g, i, j)
    return res
end

@inline function change_scores(g::erg_cat, i, j)
    res = zeros(Float64, length(g.fs))
    smap!(res, g.fs, g, i, j)
    return res, [f(g, i, j) for f in g.cat_fs]
end

# Does 500 node graph in 0.18 ms
function subgraphcount(g::AbstractERGM)
    x = zeros(Float64, g.n_params)
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
    g::AbstractGraph,
    graphstats,
    K::Int64;
    return_graph=true,
)
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

function rgraphs(
    theta::Vector{Float64},
    g::AbstractGraph,
    K::Int64,
    N::Int64
)
    n = nv(g)
    g2 = deepcopy(g)
    graphs = Any[]
    for n_samp = 1:N
        @inbounds for k = 1:K
            for j = 1:n
                for i = 1:n
                    if i == j
                        continue
                    else
                        deltastats = change_scores(g2, i, j)
                        if log(rand()) < dot(theta, deltastats)
                            edge_toggle!(g2, i, j)
                        end
                    end
                end
            end
        end
        push!(graphs, deepcopy(g2))
    end
    return graphs
end