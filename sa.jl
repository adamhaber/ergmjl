# The following functions implement the Stochastic Approximation algorithm 
# from the appendix of (Snijders, 2002).

# Compute the quantities in Equation (20)
# This function basically computes subgraphcount for y and y_comp in parallel
# Note that the expression of Pn is slightly changed for numerical stability
function compute_p(g::E, theta::Vector{Float64}) where {E <: AbstractGraph}
    y = deepcopy(g)
    y_comp = deepcopy(g)
    uY = zeros(g.n_funcs)
    uY_comp = zeros(g.n_funcs)

    # y_comp = g's complement
    for j in vertices(y)
        for i in vertices(y)
            edge_toggle!(y_comp, i, j)
        end
    end

    for j in vertices(y)
        for i in vertices(y)
            if i == j
                continue
            else
                if has_edge(y, i, j)
                    uY -= change_scores(y, i, j)
                    edge_toggle!(y, i, j)
                end
                if has_edge(y_comp, i, j)
                    uY_comp -= change_scores(y_comp, i, j)
                    edge_toggle!(y_comp, i, j)
                end
            end
        end
    end
    Pn = 1 / (1 + exp(dot(theta, uY - uY_comp)))    
    return (Pn, uY, uY_comp)
end

# Compute the a single term in the u_bar sum
function compute_u(Ps::Tuple{Float64,Vector{Float64},Vector{Float64}})
    Pn, uY, uY_comp = Ps
    return Pn * uY_comp + (1 - Pn) * uY
end

# Compute the a single term in the D sum
function compute_D(Ps::Tuple{Float64,Vector{Float64},Vector{Float64}})
    Pn, uY, uY_comp = Ps
    return Pn * uY_comp * uY_comp' + (1 - Pn) * uY * uY'
end

# Estimate the D_0 
function phase1(theta::Vector{Float64}, g::E, N::Int64; K::Int64=10, partial_NR::Bool=false) where {E <: AbstractGraph}
    ys = rgraphs(theta, g, K, N)
    ps = map(y -> compute_p(y, theta), ys)
    u_bar = mean(map(compute_u, ps))
    D = mean(map(compute_D, ps)) - u_bar * u_bar'
    if !partial_NR
        return D
    else
        throw("unimplemented")
    end
end

function subphase(theta::Vector{Float64}, D_inv, k::Int64, alpha::Float64, u0::Vector{Float64}, g::E)  where {E <: AbstractGraph}
    N_min = 2^(4(k - 1) / 3) * (7 + g.n_funcs)
    N_max = N_min + 200
    theta_mean = Mean()
    fit!(theta_mean, theta)
    for iter in 1:N_min
        yn = rgraphs(theta, g, 5, 1)[1]
        Pn, uY, uY_comp = compute_p(yn, theta)
        Zn = Pn * uY_comp + (1 - Pn) * uY - u0
        theta -= alpha * D_inv * Zn
        fit!(theta_mean, theta)
    end
    # Add early stopping rule
    for iter in (N_min + 1):N_max
        yn = rgraphs(theta, g, 5, 1)[1]
        Pn, uY, uY_comp = compute_p(yn, theta)
        Zn = Pn * uY_comp + (1 - Pn) * uY - u0
        theta -= alpha * D_inv * Zn
        fit!(theta_mean, theta)
    end    
    return theta
end

# Main parameter estimation phase
function phase2(g::E, D_inv, theta::Vector{Float64}; n_subphases::Int64=4, alpha::Float64=0.1) where {E <: AbstractGraph}
    u0 = subgraphcount(g)
    for k in 1:n_subphases
        theta = subphase(theta, D_inv, k, alpha, u0, g)
        alpha = alpha / 2
    end
    return theta
end

function stochastic_approximation(g::E, theta::Vector{Float64}) where {E <: AbstractGraph}
    D_inv = Diagonal(phase1(theta, g, 7 + 3g.n_funcs))^-1
    theta = phase2(g, D_inv, theta)
    theta_cov = phase1(theta, g, 1000)^-1
    return theta, theta_cov
end