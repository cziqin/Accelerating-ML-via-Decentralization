import Pkg
Pkg.add(["LIBSVMdata", "LinearAlgebra", "Plots", "LaTeXStrings", "TimerOutputs", "CSV", "DataFrames"])

using LinearAlgebra, Plots, LaTeXStrings, LIBSVMdata, Random, TimerOutputs, Base.Threads, Statistics, CSV, DataFrames, Dates

function generate_X1(X0, G0, μs, Ls, storage)
    stepsize = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        stepsize[i] = 1 / Ls[i]
    end
    X1 = X0 .- G0 * diagm(stepsize)
    return X1
end

function switch(storage)
    norms = []
    for i = 1:size(storage)[1]
        push!(norms, norm(gradient(storage[i], A, number_of_agents, y, mu) * ones(number_of_agents)))
    end
    norms = log10.(norms)
    #if size(norms)[1] >= 120 && (norms[end-2] - norms[end])/2 <= (norms[end-12] - norms[end-2])/10
    if size(norms)[1] >= 400 
        return true
    else
        return false
    end
end

function COMBINATION(X2, X1, G2, G1, mat, μs, Ls, storage)
    if switch_sign == 1 && initial_sign == 0
        X3 = GD1_local(X2, G2, mat, μs, Ls, storage)
        global initial_sign = 1
    elseif switch_sign == 1 && initial_sign == 1
        X3 = GD1(X2, X1, G2, G1, mat, μs, Ls, storage)
    else
        X3 = GD1_local(X2, X1, G2, G1, mat, μs, Ls, storage)
    end
    # if abs(norm(gradient(storage[end], A, number_of_agents, y)*ones(number_of_agents))-norm(gradient(storage[end-1], A, number_of_agents, y)*ones(number_of_agents)))<= 10^-10 && initial_sign == 0 #switch criteria
    #     global switch_sign = 1
    # end
    if switch(storage)
        global switch_sign = 1
    end
    return X3
end

function GD1(X1, G1, mat, μs, Ls, storage)
    L_sum = sum(Ls)
    number_of_agents = size(mat)[1]
    stepsize = ones(number_of_agents) .* (1 / (L_sum / number_of_agents))
    out = consensus(mat, X1 .- G1 * diagm(stepsize))
    return out
end

function GD1(X2, X1, G2, G1, mat, μs, Ls, storage)
    L_sum = sum(Ls)
    number_of_agents = size(mat)[1]
    stepsize = ones(number_of_agents) .* (1 / (L_sum / number_of_agents))
    out = consensus(mat, X1 .- G1 * diagm(stepsize))
    return out
end

function GD1_local(X1, G1, mat, μs, Ls, storage)
    L_sum = sum(Ls)
    number_of_agents = size(mat)[1]
    stepsize = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        stepsize[i] = 1 / Ls[i]
    end
    out = consensus(mat, X1 .- G1 * diagm(stepsize))
    return out
end

function GD1_local(X2, X1, G2, G1, mat, μs, Ls, storage)
    L_sum = sum(Ls)
    number_of_agents = size(mat)[1]
    stepsize = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        stepsize[i] = 1 / Ls[i]
    end
    out = consensus(mat, X1 .- G1 * diagm(stepsize))
    return out
end

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

function random_ball(dim, degree)
    y = randn(dim)  # Sample from normal distribution
    y = y / norm(y) # Normalize to unit sphere
    # r = rand()^(1/dim) # Scale by random radius for uniformity
    return y * degree
end


function value(state, A, number_of_agents, y, mu)
    dim = size(A[1])[2]
    F = Vector{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        F_i = 0
        subset_size = size(A[i])[1]
        for j = 1:subset_size
            F_i += 1 / subset_size .* log(1 + exp(-y[i][j] * (A[i][j, :]' * state[:, i]))) + mu / 2 * subset_size .* norm(state[:, i]) #logistic regression 
        end
        #G_i = A[i]'*A[i]*state[:,i] .- A[i]'*y[i] .+ 0 .* state[:,i]
        F[i] = F_i
    end
    return F
end

function gradient(state, A, number_of_agents, y, mu)
    dim = size(A[1])[2]
    G = Array{Float64}(undef, dim, number_of_agents)
    for i = 1:number_of_agents
        G_i = zeros(dim)
        subset_size = size(A[i])[1]
        for j = 1:subset_size
            exp_part = exp(-y[i][j] * (A[i][j, :]' * state[:, i]))/(1 + exp(-y[i][j] * (A[i][j, :]' * state[:, i])))
            isinan = isnan(exp_part)
            if isinan
                exp_part = 1
            end
            G_i += 1 / subset_size .* exp_part / (1 + exp_part) .* (-y[i][j] .* A[i][j, :]) .+ mu / subset_size .* state[:, i]
        end
        #G_i = A[i]'*A[i]*state[:,i] .- A[i]'*y[i] .+ 0 .* state[:,i]
        G[:, i] = G_i
    end
    return G
end

function hessian(state, A, number_of_agents, y, mu)
    dim = size(A[1])[2]
    subset_size = size(A[1])[1]
    H = []
    for i = 1:number_of_agents
        H_i = zeros(dim, dim)
        for j = 1:subset_size
            #H_i = H_i .+ 1/subset_size .* exp(-y[i][j]*(A[i]*state[:,i])[j])/(1+exp(-y[i][j]*(A[i]*state[:,i])[j]))^2 .* (A[i][j,:]*A[i][j,:]') .+ 1/subset_size^2 .* I(dim)
            H_i = H_i .+ 1 / subset_size .* 0.25 .* (y[i][j,:])^2 .* (A[i][j, :] * A[i][j, :]') .+ mu / subset_size .* I(dim)
        end
        #H_i = A[i]'*A[i] .+ 0 .* I(dim)
        push!(H, H_i)
    end
    return H
end

function estimate_Ls(A, number_of_agents, sample_number, y, mu)
    dim = size(A[1])[2]
    Ls = zeros(number_of_agents) .* 1 / size(A[1])[1]
    for i = 1:number_of_agents
        L = 0
        for j = 1:size(A[i])[1]
            if y[i][j] != 0
                L += maximum(eigen(0.25 .* (y[i][j])^2 .* (A[i][j, :] * A[i][j, :]') .+ mu .* I(dim)).values)
            else
                L += maximum(eigen(0.25 .* (A[i][j, :] * A[i][j, :]') .+ mu .* I(dim)).values)
            end
        end
        Ls[i] = L/size(A[i])[1]
    end
    return Ls
end


function estimate_lipschitz(M, labels)
    if ndims(M) == 1
        dim = size(M[1])[2]
        lipschitz_values = [[] for _ in 1:size(M)[1]]
        for i in 1:size(M)[1]
            for j in 1:size(M[i])[1]
                L = 0
                # for j = 1:idx    
                #     L_j = maximum(eigen(exp(-labels[i]*(M[i,:]'*randn(dim)))/(1+exp(-labels[i]*(M[i,:]'*randn(dim))))^2 .* (M[i,:]*M[i,:]')).values)
                #     if L_j > L
                #         L = L_j
                #     end
                # end
                if labels[i][j] != 0
                    L = maximum(eigen(0.25 .* (labels[i][j])^2 .* (M[i][j, :] * M[i][j, :]')).values)
                else
                    L = maximum(eigen(0.25 .* (M[i][j, :] * M[i][j, :]')).values)
                end
                push!(lipschitz_values[i], L)
            end
        end
    else
        dim = size(M)[2]
        lipschitz_values = []
        for j in 1:size(M)[1]
            L = 0
            # for j = 1:idx    
            #     L_j = maximum(eigen(exp(-labels[i]*(M[i,:]'*randn(dim)))/(1+exp(-labels[i]*(M[i,:]'*randn(dim))))^2 .* (M[i,:]*M[i,:]')).values)
            #     if L_j > L
            #         L = L_j
            #     end
            # end
            if labels[j] != 0
                L = maximum(eigen(0.25 .* (labels[j])^2 .* (M[j, :] * M[j, :]')).values)
            else
                L = maximum(eigen(0.25 .* (M[j, :] * M[j, :]')).values)
            end
            push!(lipschitz_values, L)
        end
    end

    return lipschitz_values
end

## test Example usage


number_of_agents = 2
mu = 0.01

A = []
y = []
to = TimerOutput()

#load data
# Get the full path of the current script
script_path = @__FILE__

# Get the directory containing the script
path = dirname(script_path)
for i = 1:number_of_agents
    mat_data = Matrix(CSV.read(joinpath(path, "sorted_dataset", "subset_$i.csv"), DataFrame))
    push!(A, mat_data[:, 1:end-1])
    push!(y, mat_data[:, end])
end


Ls = estimate_Ls(A, number_of_agents, 50, y, mu)
μs = mu.*ones(number_of_agents)
println(Ls)

mat = ones(number_of_agents, number_of_agents) ./ number_of_agents

iteration = 800
run_times = 1
number_of_algorithms = 2
algorithm_list = [GD1, COMBINATION]

dim = size(A[1])[2]


results = []
for _ in 1:run_times
    global switch_sign = 0
    global initial_sign = 0
    global X = [[] for _ in 1:number_of_algorithms]  # A vector of 5 empty integer arrays
    global G = [[] for _ in 1:number_of_algorithms]
    global plot_matrix = zeros(iteration + 1, number_of_algorithms)
    for i = 1:iteration+1
        if i == 1
            vec = random_ball(dim, 1) .* 10^2 #based on l use 10, based on norms using 10, based on labels using 100
            for j = 1:number_of_algorithms
                push!(X[j], vec .* ones(number_of_agents)')
            end
            for j = 1:number_of_algorithms
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))
                #plot_matrix[i, j] = norm(G[j][i]* ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        elseif i == 2
            push!(X[1], GD1(X[1][1], G[1][1], mat, μs, Ls, X[1]))
            push!(G[1], gradient(X[1][2], A, number_of_agents, y, mu))

            push!(X[2], generate_X1(X[2][1], G[2][1], μs, Ls, X[2]))
            push!(G[2], gradient(X[2][2], A, number_of_agents, y, mu))

            # push!(X[3], generate_X1(X[3][1], G[3][1], μs, Ls, X[3]))
            # push!(G[3], gradient(X[3][2], A, number_of_agents, y, mu))

            # push!(X[4], generate_X1(X[4][1], G[4][1], μs, Ls, X[4]))
            # push!(G[4], gradient(X[4][2], A, number_of_agents, y, mu))
            for j = 1:number_of_algorithms
                #plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y, mu) * ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        else
            #@timeit to "algorithm_update" 
            #@timeit to "compute gradient" 
            for j = 1:number_of_algorithms
                push!(X[j], algorithm_list[j](X[j][i-1], X[j][i-2], G[j][i-1], G[j][i-2], mat, μs, Ls, X[j]))
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))

                #plot_matrix[i, j] = norm(G[j][i] * ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        end
    end
    push!(results, plot_matrix)
    println(size(results)[1])
end

means = zeros(iteration+1, number_of_algorithms)
std_errors = zeros(iteration+1, number_of_algorithms)

for i in 1:iteration+1
    # Extract the i-th element from each of the 10 vectors
    for j in 1:number_of_algorithms
        values_at_position = [results[a][i, j] for a in 1:run_times]
        means[i, j] = mean(values_at_position)
        std_errors[i, j] = std(values_at_position) / sqrt(run_times)  # Standard error
    end
end

means
error_bar_index = 1:5:size(means,1)
index = 1:size(means,1)
fig = plot()
plot!(fig, index[error_bar_index], means[:,1][error_bar_index], yscale = :log10, 
yerr=std_errors[error_bar_index],
labels=L"GD", linecolor=:red, line=(2, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,2][error_bar_index], yscale = :log10,
yerr=std_errors[error_bar_index],
labels=L"DGD(Algorithm1)", linecolor=:black, line=(1, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,3][error_bar_index], 
yerr=std_errors[error_bar_index],
labels=L"NIDS-(I+W)/2", linecolor=:blue, line=(2, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,4][error_bar_index], 
yerr=std_errors[error_bar_index],
labels=L"COMBINATION", linecolor=:black, line=(2, :dash),markersize=1, markerstrokecolor=:auto)
plot!(xlabel="Iteration", ylabel="Gradient", yscale=:log10, minorgrid=true)

script_path = @__FILE__

# Get the directory containing the script
script_dir = dirname(script_path)

#savepath = joinpath(script_dir, "examples", "results", "run_$timestamp")
timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
savepath = joinpath(script_dir, "run_$timestamp")
mkdir(savepath)
CSV.write(joinpath(savepath, "means.csv"), DataFrame(means, :auto))
CSV.write(joinpath(savepath, "std_errors.csv"), DataFrame(std_errors, :auto))
write(joinpath(savepath, "information.txt"), "labels")

fig_path = joinpath(savepath, "pic.svg")
savefig(fig, fig_path)

A = []
y = []
to = TimerOutput()

for i = 1:number_of_agents
    mat_data = Matrix(CSV.read(joinpath(path, "sorted_dataset", "norms_$i.csv"), DataFrame))
    push!(A, mat_data[:, 1:end-2])
    push!(y, mat_data[:, end-1])
end

Ls = estimate_Ls(A, number_of_agents, 50, y, mu)
μs = mu.*ones(number_of_agents)
println(Ls)

mat = ones(number_of_agents, number_of_agents) ./ number_of_agents

iteration = 800
run_times = 1
number_of_algorithms = 2
algorithm_list = [GD1, COMBINATION]

dim = size(A[1])[2]


results = []
for _ in 1:run_times
    global switch_sign = 0
    global initial_sign = 0
    global X = [[] for _ in 1:number_of_algorithms]  # A vector of 5 empty integer arrays
    global G = [[] for _ in 1:number_of_algorithms]
    global plot_matrix = zeros(iteration + 1, number_of_algorithms)
    for i = 1:iteration+1
        if i == 1
            vec = random_ball(dim, 1) .* 10^2 #based on l use 10, based on norms using 10, based on labels using 100
            for j = 1:number_of_algorithms
                push!(X[j], vec .* ones(number_of_agents)')
            end
            for j = 1:number_of_algorithms
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))
                #plot_matrix[i, j] = norm(G[j][i]* ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        elseif i == 2
            push!(X[1], GD1(X[1][1], G[1][1], mat, μs, Ls, X[1]))
            push!(G[1], gradient(X[1][2], A, number_of_agents, y, mu))

            push!(X[2], generate_X1(X[2][1], G[2][1], μs, Ls, X[2]))
            push!(G[2], gradient(X[2][2], A, number_of_agents, y, mu))

            # push!(X[3], generate_X1(X[3][1], G[3][1], μs, Ls, X[3]))
            # push!(G[3], gradient(X[3][2], A, number_of_agents, y, mu))

            # push!(X[4], generate_X1(X[4][1], G[4][1], μs, Ls, X[4]))
            # push!(G[4], gradient(X[4][2], A, number_of_agents, y, mu))
            for j = 1:number_of_algorithms
                #plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y, mu) * ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        else
            #@timeit to "algorithm_update" 
            #@timeit to "compute gradient" 
            for j = 1:number_of_algorithms
                push!(X[j], algorithm_list[j](X[j][i-1], X[j][i-2], G[j][i-1], G[j][i-2], mat, μs, Ls, X[j]))
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))

                #plot_matrix[i, j] = norm(G[j][i] * ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        end
    end
    push!(results, plot_matrix)
    println(size(results)[1])
end

means = zeros(iteration+1, number_of_algorithms)
std_errors = zeros(iteration+1, number_of_algorithms)

for i in 1:iteration+1
    # Extract the i-th element from each of the 10 vectors
    for j in 1:number_of_algorithms
        values_at_position = [results[a][i, j] for a in 1:run_times]
        means[i, j] = mean(values_at_position)
        std_errors[i, j] = std(values_at_position) / sqrt(run_times)  # Standard error
    end
end

means
error_bar_index = 1:5:size(means,1)
index = 1:size(means,1)
fig = plot()
plot!(fig, index[error_bar_index], means[:,1][error_bar_index], yscale = :log10, 
yerr=std_errors[error_bar_index],
labels=L"GD", linecolor=:red, line=(2, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,2][error_bar_index], yscale = :log10,
yerr=std_errors[error_bar_index],
labels=L"DGD(Algorithm1)", linecolor=:black, line=(1, :solid),markersize=1, markerstrokecolor=:auto)

script_path = @__FILE__

# Get the directory containing the script
script_dir = dirname(script_path)

#savepath = joinpath(script_dir, "examples", "results", "run_$timestamp")
timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
savepath = joinpath(script_dir, "run_$timestamp")
mkdir(savepath)
CSV.write(joinpath(savepath, "means.csv"), DataFrame(means, :auto))
CSV.write(joinpath(savepath, "std_errors.csv"), DataFrame(std_errors, :auto))
write(joinpath(savepath, "information.txt"), "norms")

fig_path = joinpath(savepath, "pic.svg")
savefig(fig, fig_path)


A = []
y = []
to = TimerOutput()

for i = 1:number_of_agents
    mat_data = Matrix(CSV.read(joinpath(path, "sorted_dataset", "ls_$i.csv"), DataFrame))
    push!(A, mat_data[:, 1:end-2])
    push!(y, mat_data[:, end-1])
end

Ls = estimate_Ls(A, number_of_agents, 50, y, mu)
μs = mu.*ones(number_of_agents)
println(Ls)

mat = ones(number_of_agents, number_of_agents) ./ number_of_agents

iteration = 800
run_times = 1
number_of_algorithms = 2
algorithm_list = [GD1, COMBINATION]

dim = size(A[1])[2]


results = []
for _ in 1:run_times
    global switch_sign = 0
    global initial_sign = 0
    global X = [[] for _ in 1:number_of_algorithms]  # A vector of 5 empty integer arrays
    global G = [[] for _ in 1:number_of_algorithms]
    global plot_matrix = zeros(iteration + 1, number_of_algorithms)
    for i = 1:iteration+1
        if i == 1
            vec = random_ball(dim, 1) .* 10^2 #based on l use 10, based on norms using 10, based on labels using 100
            for j = 1:number_of_algorithms
                push!(X[j], vec .* ones(number_of_agents)')
            end
            for j = 1:number_of_algorithms
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))
                #plot_matrix[i, j] = norm(G[j][i]* ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        elseif i == 2
            push!(X[1], GD1(X[1][1], G[1][1], mat, μs, Ls, X[1]))
            push!(G[1], gradient(X[1][2], A, number_of_agents, y, mu))

            push!(X[2], generate_X1(X[2][1], G[2][1], μs, Ls, X[2]))
            push!(G[2], gradient(X[2][2], A, number_of_agents, y, mu))

            for j = 1:number_of_algorithms
                #plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y, mu) * ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        else
            #@timeit to "algorithm_update" 
            #@timeit to "compute gradient" 
            for j = 1:number_of_algorithms
                push!(X[j], algorithm_list[j](X[j][i-1], X[j][i-2], G[j][i-1], G[j][i-2], mat, μs, Ls, X[j]))
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))

                #plot_matrix[i, j] = norm(G[j][i] * ones(number_of_agents) ./ number_of_agents)
                plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)' * ones(number_of_agents) ./ number_of_agents
            end
        end
    end
    push!(results, plot_matrix)
    println(size(results)[1])
end

means = zeros(iteration+1, number_of_algorithms)
std_errors = zeros(iteration+1, number_of_algorithms)

for i in 1:iteration+1
    # Extract the i-th element from each of the 10 vectors
    for j in 1:number_of_algorithms
        values_at_position = [results[a][i, j] for a in 1:run_times]
        means[i, j] = mean(values_at_position)
        std_errors[i, j] = std(values_at_position) / sqrt(run_times)  # Standard error
    end
end

means
error_bar_index = 1:5:size(means,1)
index = 1:size(means,1)
fig = plot()
plot!(fig, index[error_bar_index], means[:,1][error_bar_index], yscale = :log10, 
yerr=std_errors[error_bar_index],
labels=L"GD", linecolor=:red, line=(2, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,2][error_bar_index], yscale = :log10,
yerr=std_errors[error_bar_index],
labels=L"DGD(Algorithm1)", linecolor=:black, line=(1, :solid),markersize=1, markerstrokecolor=:auto)


script_path = @__FILE__

# Get the directory containing the script
script_dir = dirname(script_path)

#savepath = joinpath(script_dir, "examples", "results", "run_$timestamp")
timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
savepath = joinpath(script_dir, "run_$timestamp")
mkdir(savepath)
CSV.write(joinpath(savepath, "means.csv"), DataFrame(means, :auto))
CSV.write(joinpath(savepath, "std_errors.csv"), DataFrame(std_errors, :auto))
write(joinpath(savepath, "information.txt"), "labels")

fig_path = joinpath(savepath, "pic.svg")
savefig(fig, fig_path)