#load the packages needed
import Pkg
Pkg.add(["RowEchelon", "LinearAlgebra", "JuMP", "Mosek", "MosekTools", "SCS",
    "Plots", "ColorSchemes", "LaTeXStrings", "Dates", "JLD", "TimerOutputs", "Graphs"])

using RowEchelon, LinearAlgebra, JuMP, Mosek, MosekTools, SCS, CSV, DataFrames, Graphs, Plots, ColorSchemes, LaTeXStrings, Dates, JLD, TimerOutputs, Base.Threads, Base

## some helper funcitons
#consensus
function consensus(W, x)
    (a, b) = size(x)
    (N, ~) = size(W)
    if N > 1
        for i in 1:N
            sum_i = 0
            for j in 1:N
                sum_i = sum_i .+ W[i, j] .* x[:, j]
            end
            if i == 1
                y = sum_i
            else
                y = [y sum_i]
            end
        end
    else
        y = x
    end
    return y
end

function e_i(dim::Int, agent, i, N)
    x = zeros(dim, 1)
    x[N*(i-1)+agent] = 1
    return x
end

#generate X0
function generate_X0(dim, N)
    x = e_i(dim, 1, 1, N)
    if N > 1
        for i = 2:N
            v = e_i(dim, i, 1, N)
            x = [x v]
        end
    end
    return x
end

#generate X1 for algorithms using two previous states to generate next state
function generate_X1(X0, G0, mat, stepsize, algorithm)
    if all(stepsize .== stepsize[1])
        X1 = consensus(mat, X0 .- G0 * Diagonal(stepsize))
    else
        X1 = X0 .- G0 * Diagonal(stepsize)
    end
    #X1 = X0 .- G0 * Diagonal(stepsize)
    return X1
end

function generate_XS(dim, N, K)
    XS = Array{Array}(undef, 1)
    vec = e_i(dim, dim - N, 1, N)
    for i = 1:N-1
        v = e_i(dim, dim - N + i, 1, N)
        vec = [vec v]
    end
    XS[1] = vec
    return XS
end

#generate G
function generate_Gradient_matrix(dim, N, K)
    G = Array{Array}(undef, K + 1)
    for i = 1:(K+1)
        v = e_i(dim, N * i + 1, 1, N)
        for j = 2:N
            v = [v e_i(dim, N * i + j, 1, N)]
        end
        G[i] = v
    end
    return G
end

function generate_GavS(dim, N, K)
    GavS = Array{Array}(undef, 1)
    vec = e_i(dim, dim - 2 * N, 1, N)
    for i = 1:N-1
        v = e_i(dim, dim - 2 * N + i, 1, N)
        vec = [vec v]
    end
    GavS[1] = vec
    return GavS
end

function generate_GS(dim, N, K)
    GS = Array{Array}(undef, 1)
    vec = e_i(dim, dim - 3 * N, 1, N)
    for i = 1:N-1
        v = e_i(dim, dim - 3 * N + i, 1, N)
        vec = [vec v]
    end
    GS[1] = vec
    return GS
end

function generate_GavLast(dim, N, K)
    GavLast = Array{Array}(undef, 1)
    vec = e_i(dim, dim - 4 * N, 1, N)
    for i = 1:N-1
        v = e_i(dim, dim - 4 * N + i, 1, N)
        vec = [vec v]
    end
    GavLast[1] = vec
    return GavLast
end
##variables
#scalar variables K = number of iterations N = number of agents
#functions  fav = 1/N*sum_{i=1}^N f_i(xi)
#state variables X0 = [x1_0 x2_0 ... xn_0] ... XK = [x1_K x2_K ... xn_K]  xav_ s = the optimum value of fav
#gradient variables G0 = [g1_0 g2_0 ... gn_0] ... GK = [g1_K g2_K ... gn_K]  GavS = [g1av_s g2av_s ... gnav_s]  GS = [0 ... 0](not used) 
#values of local functions during the iterations F1 = [f1_1 f2_1 ... fn_1] ... FK = [f1_K f2_K ... fn_K]  FavS = [f1av_s f2av_s ... fnav_s] FS = [f1_s f2_s ... fn_s](not used)
# X = [X0 X1 ... XK]  G = [G0 G1 ... GK] GS GavS]    F = [F0 F1 ... FK FS Favs]
##NIDS

##some struct
mutable struct fctParam
    mu::Float64
    L::Float64
    Class::String
end

mutable struct Output
    W::Matrix
    Z::Matrix
    F::Vector
    performance_result::Float64
    performance_criteria::String
    xs_constraint::Int
    x0_constraint::Int
    algorothm_name::String
    stepsize::Array
end

function arg_count(f)
    return [length(m.sig.parameters) - 1 for m in methods(f)]  # Subtract 1 for the function itself
end


function generate_X(K, mat, X0, stepsize, G, algorithm)
    X = Array{Array}(undef, K + 1)
    if K == 0
        X[1] = X0
        return X
    else
        X[1] = X0
        if arg_count(algorithm)[1] == 4
            for i = 1:K
                X_next_state = algorithm(X[i], G[i], mat, stepsize)
                X[i+1] = X_next_state
            end
            return X
        elseif arg_count(algorithm)[1] == 6
            X1 = generate_X1(X0, G[1], mat, stepsize, algorithm)
            X[2] = X1
            for i = 1:K-1
                X[i+2] = algorithm(X[i+1], X[i], G[i+1], G[i], mat, stepsize)
            end
            return X
        elseif arg_count(algorithm)[1] == 7
            X1 = generate_X1(X0, G[1], mat, stepsize, algorithm)
            X[2] = X1
            for i = 1:K-1
                X[i+2] = algorithm(X[i+1], X[i], G[i+1], G[i], mat, stepsize, i)
            end
            return X
        else
            error("generate_X doesn't support algorithm with two previous points")
        end
    end
end

function generate_X_with_combined_method(K, mat, X0, identical_stepsize_L_average, stepsize, G, switch_time)
    X = Array{Array}(undef, K + 1)
    if K == 0
        X[1] = X0
        return X
    else
        X[1] = X0
        if K > switch_time
            for i = 1:switch_time-1
                X[i+1] = GD1(X[i], G[i], mat, stepsize)
            end
            for i = switch_time:K
                X[i+1] = GD1(X[i], G[i], mat, identical_stepsize_L_average)
            end
        else
            for i = 1:K
                X[i+1] = GD1(X[i], G[i], mat, stepsize)
            end
        end
        return X
    end
end

function Model_SDP(fctParams, K, mat, stepsize, algorithm, performance_criteria, xs_constraint, x0_constraint, x_constraint_type, switch_time)

    to = TimerOutput()

    N = size(mat)[1]

    L_sum = fctParams[1].L
    for i = 2:N
        L_sum += fctParams[i].L
    end
    L_av = L_sum / N
    #set up the dimention parameters(here we enable the use of GavLast, so the dimention of P is N(K+6)+1 and F is N(K+4), otherwise it is N(K+5)+1 and N(K+3))
    #P = [X0 | G0 G1 ... GK (GavLast) GS| GavS XS xav_s]  Z = P^T P
    dim_Z = N * (K + 6) + 1
    #F = [F0 | F1 ... FK (FavLast) FS| FavS]
    dim_F = N * (K + 4)

    #identical_stepsize_L_average
    identical_stepsize_L_average = 1.0 / L_av .* ones(N)

    #define the optimal points

    GavS = generate_GavS(dim_Z, N, K)
    GS = generate_GS(dim_Z, N, K)
    GavLast = generate_GavLast(dim_Z, N, K)

    #initial X0 and Gradient

    G = generate_Gradient_matrix(dim_Z, N, K)

    #generate X 
    X0 = generate_X0(dim_Z, N)
    if algorithm == "combined"
        X = generate_X_with_combined_method(K, mat, X0, identical_stepsize_L_average, stepsize, G, switch_time)
    else
        X = generate_X(K, mat, X0, identical_stepsize_L_average, G, algorithm)
    end
    xav_s = e_i(dim_Z, dim_Z, 1, N)
    XS = generate_XS(dim_Z, N, K)
    XavLast = X[K+1][:, 1] ./ N
    if N > 1
        for i = 2:N
            XavLast += X[K+1][:, i] ./ N
        end
    end

    #the case without XS
    X_interp = Array{Array}(undef, N)
    G_interp = Array{Array}(undef, N)
    for i = 1:N
        X_interp[i] = X[1][:, i]
        G_interp[i] = G[1][:, i]
        for j = 2:K+1
            X_interp[i] = [X_interp[i] X[j][:, i]]
            G_interp[i] = [G_interp[i] G[j][:, i]]
        end

        X_interp[i] = [X_interp[i] XavLast]
        G_interp[i] = [G_interp[i] GavLast[1][:, i]]

        X_interp[i] = [X_interp[i] XS[1][:, i]]
        G_interp[i] = [G_interp[i] GS[1][:, i]]

        X_interp[i] = [X_interp[i] xav_s]
        G_interp[i] = [G_interp[i] GavS[1][:, i]]
        if size(G_interp[i]) != size(G_interp[i])
            error("interpolation points error")
        end
    end

    #define the model and choose the optimizer
    model_primal_PEP_with_predefined_stepsize = Model(optimizer_with_attributes(Mosek.Optimizer))
    #high precision
    # set_optimizer_attribute(model_primal_PEP_with_predefined_stepsize, "MSK_DPAR_INTPNT_TOL_PFEAS", 1e-12)
    # set_optimizer_attribute(model_primal_PEP_with_predefined_stepsize, "MSK_DPAR_INTPNT_TOL_DFEAS", 1e-12)
    # set_optimizer_attribute(model_primal_PEP_with_predefined_stepsize, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-12)
    # set_optimizer_attribute(model_primal_PEP_with_predefined_stepsize, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", 1e-12)
    #model_primal_PEP_with_predefined_stepsize = Model(optimizer_with_attributes(SCS.Optimizer))

    #define sdp variable
    @variable(model_primal_PEP_with_predefined_stepsize, Z[1:dim_Z, 1:dim_Z], PSD)
    @variable(model_primal_PEP_with_predefined_stepsize, F[1:dim_F])

    #set up the performance measure
    xav_last = X[K+1][:, 1] .- xav_s
    if N > 1
        for i = 2:N
            xav_last = xav_last + X[K+1][:, i] .- xav_s
        end
    end
    xav_last = xav_last ./ N

    x_last_sum = (X[K+1][:, 1] .- xav_s) * (X[K+1][:, 1] .- xav_s)'
    if N > 1
        for i = 2:N
            x_last_sum = x_last_sum .+ (X[K+1][:, i] .- xav_s) * (X[K+1][:, i] .- xav_s)'
        end
    end

    #define the objective functions
    FavLast = e_i(dim_F, 1, K + 2, N) ./ N
    if N > 1
        for i = 2:N
            FavLast += e_i(dim_F, i, K + 2, N) ./ N
        end
    end

    FKLast = e_i(dim_F, 1, K + 1, N) ./ N
    if N > 1
        for i = 2:N
            FKLast += e_i(dim_F, i, K + 1, N) ./ N
        end
    end

    FavS = e_i(dim_F, 1, K + 4, N) ./ N
    if N > 1
        for i = 2:N
            FavS += e_i(dim_F, i, K + 4, N) ./ N
        end
    end
    objective_expression = AffExpr(0)
    for i = 1:dim_F
        add_to_expression!(objective_expression, FKLast[i] - FavS[i], F[i])
    end

    G_last = G[K+1][:, 1]
    if N >= 2
        for i = 2:N
            G_last += G[K+1][:, i]
        end
    end

    if performance_criteria == "funcValue"
        @objective(model_primal_PEP_with_predefined_stepsize, Max, objective_expression)
    elseif performance_criteria == "state"
        @objective(model_primal_PEP_with_predefined_stepsize, Max, tr(Z * xav_last * xav_last'))
    else
        @objective(model_primal_PEP_with_predefined_stepsize, Max, tr(Z * (G_last * G_last')))
    end
    #set up the iterpolation conditions
    @timeit to "nest 1" begin
        (~, number_of_points) = size(X_interp[1])
        for n = 1:N
            mu = fctParams[n].mu
            L = fctParams[n].L
            class = fctParams[n].Class
            for i = 1:number_of_points
                interpolate_cons = Array{Any}(undef, number_of_points)
                @timeit to "get cons expression" begin
                    @threads for j = 1:number_of_points
                        #the interpolation condition for SmoothStronglyConvex funtion
                        #f_i >= f_j + g_j^T (xi-xj) + 1/2/(1-mu/L) (1/L*||g_i-g_j||^2+mu||xi-xj||^2-2*mu/L(g_j-g_k)^T(xj-xi))
                        if class == "SmoothStronglyConvex"
                            f_i = e_i(dim_F, n, Int(i), N)
                            f_j = e_i(dim_F, n, Int(j), N)
                            g_j = G_interp[n][:, Int(j)]
                            g_i = G_interp[n][:, Int(i)]
                            xi = X_interp[n][:, Int(i)]
                            xj = X_interp[n][:, Int(j)]

                            f_j_i = f_j .- f_i
                            matrix = similar(Z)
                            matrix = ((xi .- xj) * g_j' .+ g_j * (xi .- xj)') / 2 .+ (1 / 2 / (1 - mu / L)) * ((1 / L) .* ((g_i .- g_j) * (g_i .- g_j)') .+ mu .* ((xi .- xj) * (xi .- xj)') .- 2 * mu / L .* ((xj .- xi) * (g_j .- g_i)'+ (g_j .- g_i) * (xj .- xi)')./2)
                            interpolate_cons[j] = @expression(model_primal_PEP_with_predefined_stepsize, dot(f_j_i, F) + sum(Diagonal(Z * matrix)))
                        end
                    end
                    for j = 1:number_of_points
                        @constraint(model_primal_PEP_with_predefined_stepsize, interpolate_cons[j] .<= 0)
                    end
                    interpolate_cons = nothing
                end
            end
        end
    end
    # add initial condition 

    #the optimal condition of fav:the sum of GavS is 0
    gavs_sum = GavS[1][:, 1]
    if N > 1
        for i = 2:N
            gavs_sum = gavs_sum + GavS[1][:, i]
        end
    end
    @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * (gavs_sum * gavs_sum')) .== 0)
    #local optimum all GS = 0
    for i = 1:N
        gs = GS[1][:, i] * GS[1][:, i]'
        @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * gs) .== 0)
    end

    #bounded function heterogeneity(initial gradient constraint)
    # gav0 = G[1][:, 1]
    # if N > 1
    #     for i = 2:N
    #         gav0 = gav0 + G[1][:, i]
    #     end
    # end
    # gav0 = gav0 ./ N

    # G0 = (G[1][:,1] .- gav0)*(G[1][:,1] .- gav0)'
    # if N > 1
    #     for i = 2:N
    #         G0 = G0 + (G[1][:,i] .- gav0)*(G[1][:,i] .- gav0)'
    #     end
    # end
    # @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * G0) .<= N)
    #non-equal start
    # x0av_sum = X[1][:, 1]
    # if N > 1
    #     for i = 2:N
    #         x0av_sum = x0av_sum + X[1][:, i]
    #         #@constraint(model_primal_PEP_with_predefined_stepsize, tr(Z*(((x-xav_s)*(x-xav_s)'))).<=1)
    #     end
    # end
    # x0_av = x0av_sum ./ N
    #@constraint(model_primal_PEP_with_predefined_stepsize, tr(Z*(((x0_av-xav_s)*(x0_av-xav_s)'))).<=1)

    #constri for start value
    #if algorithm == GD1
    if N > 1
        for i = 2:N
            vec = (X[1][:, i] .- X[1][:, 1]) * (X[1][:, i] .- X[1][:, 1])'
            @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * vec) .<= 0)
        end
    end
    #end

    #add constraint that fs = 0 for each i
    # for i = 1:N
    #     @constraint(model_primal_PEP_with_predefined_stepsize, F[(K+2)*N+i]==0)
    # end

    if x_constraint_type == 1
        x0_sum = (X[1][:, 1] .- xav_s) * (X[1][:, 1] .- xav_s)'
        for i = 2:N
            x0_sum = x0_sum + (X[1][:, i] .- xav_s) * (X[1][:, i] .- xav_s)'
        end
        @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * x0_sum) .<= N * x0_constraint)
    else
        for i = 1:N
            x0_i = (X[1][:, i] .- xav_s) * (X[1][:, i] .- xav_s)'
            @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * x0_i) .<= x0_constraint)
        end
    end

    if x_constraint_type == 1
        # ||XS - xav_s||^2 <=1
        xs_sum = (XS[1][:, 1] - xav_s) * (XS[1][:, 1] - xav_s)'
        if N > 1
            for i = 2:N
                xs_sum = xs_sum + (XS[1][:, i] - xav_s) * (XS[1][:, i] - xav_s)'
            end
        end
        @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * xs_sum) .<= xs_constraint * N)
    else
        if N >= 1
            for i = 1:N
                xs_i = (XS[1][:, i] - xav_s) * (XS[1][:, i] - xav_s)'
                @constraint(model_primal_PEP_with_predefined_stepsize, tr(Z * xs_i) .<= xs_constraint)
            end
        end
    end

    #enable silent mode
    set_silent(model_primal_PEP_with_predefined_stepsize)
    #equal start
    #@constraint(model_primal_PEP_with_predefined_stepsize, tr(Z*((X[:,1].-xav_s)*(X[:,1].-xav_s)')).<=1)
    #optimize
    @timeit to "optimization" begin
        @timeit to "optimization cost" optimize!(model_primal_PEP_with_predefined_stepsize)
    end
    # show(to, allocations=true, compact=true)
    # Print violated constraints (if any)
    # for c in all_constraints(model_primal_PEP_with_predefined_stepsize, include_variable_in_set_constraints = true)
    #     constraint_expr = JuMP.constraint_object(c).func   # Get LHS of constraint
    #     constraint_type = JuMP.constraint_object(c).set    # Get constraint type
    #     lhs_value = JuMP.value.(constraint_expr)                 # Evaluate LHS with optimal values

    #     if constraint_type isa MOI.LessThan
    #         rhs_value = constraint_type.upper
    #         if lhs_value > rhs_value  # Violation check
    #             println("Constraint violated: $constraint_expr ≤ $rhs_value (Actual: $lhs_value)")
    #         end
    #     elseif constraint_type isa MOI.GreaterThan
    #         rhs_value = constraint_type.lower
    #         if lhs_value < rhs_value  # Violation check
    #             println("Constraint violated: $constraint_expr ≥ $rhs_value (Actual: $lhs_value)")
    #         end
    #     elseif constraint_type isa MOI.EqualTo
    #         rhs_value = constraint_type.value
    #         if lhs_value ≠ rhs_value  # Violation check
    #             println("Constraint violated: $constraint_expr = $rhs_value (Actual: $lhs_value)")
    #         end
    #     end
    # end

    Z = value.(Z)
    F = value.(F)
    wc_performance = objective_value(model_primal_PEP_with_predefined_stepsize)

    #return the result
    out = Output(mat, Z, F, wc_performance, performance_criteria, xs_constraint, x0_constraint, string(algorithm), stepsize)
    return out
end


#algorithm GD
function GD1(X1, G1, mat, stepsize)
    X2 = consensus(mat, X1 .- G1 * Diagonal(stepsize))
    return X2
end

function run(W, K, fctParams, xs_constraint, x0_constraint, performance_criteria, switch_time)
    #set up agent number and function parameters
    #performance Array
    x_constraint_type = 0
    N = size(W)[1]
    results = Array{Vector}(undef, 2)
    for i in 1:length(results)
        results[i] = []
    end
    Ls = []
    for i in 1:N
        push!(Ls, fctParams[i].L)
    end
    #stepsize 1/L_av
    L_av = sum(Ls) / N

    identical_stepsize_L_average = 1.0 / L_av .* ones(N)

    non_identical_stepsize = []
    for i in 1:N
        push!(non_identical_stepsize, 1 / Ls[i])
    end


    for k = 0:K
        push!(results[1], Model_SDP(fctParams, k, W, identical_stepsize_L_average, GD1, performance_criteria, xs_constraint, x0_constraint, x_constraint_type, switch_time))
        push!(results[2], Model_SDP(fctParams, k, W, non_identical_stepsize, "combined", performance_criteria, xs_constraint, x0_constraint, x_constraint_type, switch_time))
    end
    return results
end



function plot_save_pic(results, fctParams)
    #plot
    num_algorithms = length(results)
    K = length(results[1])
    x = 0:1:K-1
    N = size(results[1][1].W)[1]
    #plot the result

    #set up plot Matrix
    plot_matrix = Array{Vector}(undef, num_algorithms)
    for i = 1:num_algorithms
        plot_matrix[i] = []
    end
    for j = 1:K
        for i = 1:num_algorithms
            push!(plot_matrix[i], results[i][j].performance_result)
        end
    end

    matrix_save = plot_matrix[1]

    for i in 2:num_algorithms
        matrix_save = hcat(matrix_save, plot_matrix[i])
    end

    fig = plot()
    plot!(fig, x, plot_matrix[1], labels=L"GD", linecolor=:blue, line=(2, :solid), marker=:star, xticks=0:5:K)
    plot!(fig, x, plot_matrix[2], labels  = L"Algorithm1", linecolor = :blue, line = (2, :dashdot), marker = :star, xticks=0:5:K)
    plot!(fig, xscale=:log1, yscale=:log10, minorgrid=true)
    plot!(xlabel="Iteration", ylabel=results[1][1].performance_criteria)
    plot!(guidefontsize=10)

    top_dir = mk_output_dir()
    mu = fctParams[1].mu
    Ls = zeros(N, 1)
    for i = 1:N
        Ls[i] = fctParams[i].L
    end

    pic_name = "mu_$(mu)_L"
    for i = 1:N
        pic_name = pic_name * "_$(Ls[i])"
    end
    pic_name = pic_name * ".svg"
    dir_pic = joinpath(top_dir, pic_name)
    savefig(fig, dir_pic)

    #save the data
    #save results
    dir_results = joinpath(top_dir, "results.jld")

    save(dir_results, "results", results)
    
    CSV.write(joinpath(top_dir, "plot_matrix.csv"), DataFrame(matrix_save, :auto))
end

function mk_output_dir()
    timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
    dir_name = joinpath(@__DIR__, "SDP_output", "run_$timestamp")
    @assert !ispath(dir_name) "Somebody else already created the directory"
    mkpath(dir_name)
    return dir_name
end

function generate_random_sparse_graph(n::Int, p::Float64)
    g = erdos_renyi(n, p, is_directed=false)
    while !is_connected(g)
        g = erdos_renyi(n, p, is_directed=false)
    end
    return g
end

function generate_metropolis_mixing_matrix(g::SimpleGraph)
    n = nv(g)
    W = zeros(Float64, n, n)
    
    for i in 1:n
        deg_i = degree(g, i)
        neighbors_i = neighbors(g, i)
        
        for j in neighbors_i
            deg_j = degree(g, j)
            W[i, j] = 1.0 / (1 + max(deg_i, deg_j))
        end

        W[i, i] = 1.0 - sum(W[i, :])  # self-weight to make row sum to 1
    end

    return W
end

