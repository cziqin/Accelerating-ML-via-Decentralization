import Pkg
Pkg.add(["LIBSVMdata", "LinearAlgebra", "Plots", "LaTeXStrings", "TimerOutputs", "CSV", "DataFrames", "PyCall"])

using LinearAlgebra, Plots, LaTeXStrings, LIBSVMdata, Random, TimerOutputs, Base.Threads, Statistics, CSV, DataFrames, PyCall

# script_path = @__DIR__

# # Get the directory containing the script
# script_dir = dirname(script_path)
# path = joinpath(script_dir, "GD_DGD_Numerical_Results","data.csv")

# julia_df = Matrix(CSV.read(path, DataFrame))



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

function get_norms(M, labels)
    if ndims(M) == 1
        dim = size(M[1])[2]
        norms = [[] for _ in 1:size(M)[1]]
        for i in 1:size(M)[1]
            for j in 1:size(M[i])[1]
                push!(norms[i], M[i][j, :]' * M[i][j, :])
            end
        end
    else
        dim = size(M)[2]
        norms = []
        for j in 1:size(M)[1]
            push!(norms, M[j, :]' * M[j, :])
        end
    end

    return norms
end


##load load_dataset
M, labels = load_dataset("w8a",
    dense=true,
    replace=false,
    verbose=true,
)

all_norms = get_norms(M, labels)
median_norms = median(all_norms)


#load the subset_1 and subset_2
script_path = @__FILE__

# Get the directory containing the script
script_dir = dirname(script_path)
path = script_dir


#get the original dataset
row_index_1 = all_norms .< 10
#selected_rows_1 = randperm(size(M[row_index_1, :], 1))[1:200]
M_1 = M[row_index_1, :]
labels_1 = labels[row_index_1]
matrix_1 = hcat(M_1, labels_1)
selected_rows_1 = randperm(size(matrix_1[labels_1 .< 0, :], 1))[1:200]
matrix_1_saved = matrix_1[selected_rows_1, :]

row_index_2 = all_norms .> 6
#selected_rows_2 = randperm(size(M[row_index_2, :], 1))[1:200]
M_2 = M[row_index_2, :]
labels_2 = labels[row_index_2]
matrix_2 = hcat(M_2, labels_2)
matrix_2_saved = matrix_2[labels_2 .> 0, :][1:200, :]

CSV.write(joinpath(script_dir, "sorted_dataset", "subset_1.csv"), DataFrame(matrix_1_saved, :auto))
CSV.write(joinpath(script_dir, "sorted_dataset", "subset_2.csv"), DataFrame(matrix_2_saved, :auto))


A = []
y = []
number_of_agents = 2
for i = 1:number_of_agents
    mat_data = Matrix(CSV.read(joinpath(path, "sorted_dataset", "subset_$i.csv"), DataFrame))
    push!(A, mat_data[:, 1:end-1])
    push!(y, mat_data[:, end])
end

##sort the data with different criteria

#based on the norms
M = vcat(A[1], A[2])
labels = vcat(y[1], y[2])

norms = get_norms(M, labels)
norms_median = median(norms)
row_index_1 = norms .< norms_median

M_1 = M[row_index_1, :]
labels_1 = labels[row_index_1]
norms_1 = norms[row_index_1]
matrix_norms_1 = hcat(M_1, labels_1, norms_1)

row_index_2 = norms .>= norms_median
M_2 = M[row_index_2, :]
labels_2 = labels[row_index_2]
norms_2 = norms[row_index_2]
matrix_norms_2 = hcat(M_2, labels_2, norms_2)


CSV.write(joinpath(script_dir, "sorted_dataset", "norms_1.csv"), DataFrame(matrix_norms_1, :auto))
CSV.write(joinpath(script_dir, "sorted_dataset", "norms_2.csv"), DataFrame(matrix_norms_2, :auto))


# #based on the ls
M = vcat(A[1], A[2])
labels = vcat(y[1], y[2])
ls = estimate_lipschitz(M, labels)
ls_median = median(ls)
row_index_1 = ls .< ls_median

M_1 = M[row_index_1, :]
labels_1 = labels[row_index_1]
ls_1 = ls[row_index_1]
matrix_ls_1 = hcat(M_1, labels_1, ls_1)

row_index_2 = ls .>= ls_median
M_2 = M[row_index_2, :]
labels_2 = labels[row_index_2]
ls_2 = ls[row_index_2]
matrix_ls_2 = hcat(M_2, labels_2, ls_2)


CSV.write(joinpath(script_dir, "sorted_dataset", "ls_1.csv"), DataFrame(matrix_ls_1, :auto))
CSV.write(joinpath(script_dir, "sorted_dataset", "ls_2.csv"), DataFrame(matrix_ls_2, :auto))


