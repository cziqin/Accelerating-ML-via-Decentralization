path = joinpath(dirname(@__FILE__), "sorted_dataset")
include("sort_data.jl")
include("logistic_regressison.jl")
