using Distributions, StatsFuns, OnlineStats
using GLM

function pseudo_likelihood(m::erg)
    n = size(m.m, 1)
    X = Array{Float64}(undef, n * (n - 1), m.n_funcs)
    y = Array{Bool,1}(undef, n * (n - 1))
    counter = 1
    for j in 1:n
        for i in 1:n
            i == j && continue
            y[counter] = m.m[i,j]
            if !has_edge(m, i, j)
                add_edge!(m, i, j)
                X[counter,:] = change_scores(m, i, j)
                rem_edge!(m, i, j)
            else
                X[counter,:] = change_scores(m, i, j)
            end
            counter = counter + 1
        end
    end
    return fit(GeneralizedLinearModel, X, y, Binomial(), LogitLink())                                      
end

##########################
# Compute log-likelihood 
# Note - may want to include a rejection step if auxilliary graph is degenerate
##########################
function ll(proposed_graph_parameters, current_parameters, graphstats, auxstats)
    ll = 0.0

    ll += dot(graphstats, proposed_graph_parameters) - dot(graphstats, current_parameters)
    ll += dot(auxstats, current_parameters) - dot(auxstats, proposed_graph_parameters)
    ll +=
        loglikelihood(Normal(0, 1), proposed_graph_parameters) -
        loglikelihood(Normal(0, 1), current_parameters)

    return ll
end


##########################
# Update ERGM parameters 
##########################
function update_ergm(current_parameters, graph, graphstats, K, sigma)

    proposed_ergm = rand(MultivariateNormal(current_parameters, sigma))
    auxstats, _ = rgraph(proposed_ergm, graph, graphstats, K)

    a = log(rand())
    MHratio = ll(proposed_ergm, current_parameters, graphstats, auxstats)

    if a < MHratio
        return proposed_ergm, 1, MHratio
    else
        return current_parameters, 0, MHratio
    end

end

##################################################
# ERGM model using double Metropolis sampler
##################################################
function ergm(graph, startvec, propvec, alphavec, num_iter, num_thin, adapt_iter, K)

    # Create array to hold sampler results
    ergm_parameters = Array{Float64}(undef, length(startvec), num_iter)

    # Record starting values of sampler
    ergm_parameters[:, 1] = startvec

    # Initialize parameters to track
    current_parameters = deepcopy(startvec)
    ergm_count, ergm_ratio = 0, 0.0

    # Track covariance and acceptance rates
    running_ergm_stats = CovMatrix()
    running_acceptance_ergm = Mean()


    fit!(running_ergm_stats, startvec)
    fit!(running_acceptance_ergm, 1)

    graphstats = subgraphcount(graph)
    current_graphstats = copy(graphstats)

    # Sampler
    for mcmciter = 2:num_iter
        for thiniter = 1:num_thin
            if mcmciter <= adapt_iter
                current_parameters, ergm_count, ergm_ratio = update_ergm(
                    current_parameters,
                    graph,
                    current_graphstats,
                    K,
                    alphavec * propvec + I * eps(Float64),
                )
            else
                current_parameters, ergm_count, ergm_ratio = update_ergm(
                    current_parameters,
                    graph,
                    current_graphstats,
                    K,
                    alphavec * (cov(running_ergm_stats) + I * eps(Float64)),
                )
            end

            fit!(running_ergm_stats, current_parameters)
            fit!(running_acceptance_ergm, ergm_count)
        end

        println("Iter: ", mcmciter, "  ERGM: ", round.(current_parameters, digits=2))

        # Record values of sampler
        ergm_parameters[:, mcmciter] = current_parameters
    end
    return ergm_parameters
end
