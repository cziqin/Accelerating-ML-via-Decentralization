include("PEP_SDP.jl")

#run the switch time experiments
for i in 8:2:12
    #number of iterations performed
    K = 20
    #number of agents 
    N = 2
    #initialize local function prarameters
    fctParams = Vector{fctParam}(undef, N)
    #fctParam(strongly_convex_parameter, Lipschitz_smoothness_parameter, function_class_description)
    fctParams[1] = fctParam(0.1, 1/3, "SmoothStronglyConvex")
    fctParams[2] = fctParam(0.1, 3, "SmoothStronglyConvex")
    plot_save_pic(run(ones(N, N) ./ N, K, fctParams, 1, 100, "funcValue", i), fctParams)
end

#run the average_same L experiments
for i in 0.5:0.1:0.8
    #number of iterations performed
    K = 20
    #number of agents 
    N = 2
    #initialize local function prarameters
    fctParams = Vector{fctParam}(undef, N)
    #fctParam(strongly_convex_parameter, Lipschitz_smoothness_parameter, function_class_description)
    fctParams[1] = fctParam(0.1, i, "SmoothStronglyConvex")
    fctParams[2] = fctParam(0.1, 2-i, "SmoothStronglyConvex")
    plot_save_pic(run(ones(N, N) ./ N, K, fctParams, 1, 100, "funcValue", 10), fctParams)
end

#run the more than 2 agents experiments
for i in 2:2:16
    #number of iterations performed
    K = 10
    #number of agents 
    N = i
    #initialize local function prarameters
    fctParams = Vector{fctParam}(undef, Int(N))
    #fctParam(strongly_convex_parameter, Lipschitz_smoothness_parameter, function_class_description)
    for j in 1:2:i
        fctParams[j] = fctParam(0.1, 1/3, "SmoothStronglyConvex")
    end
    for j in 2:2:i
        fctParams[j] = fctParam(0.1, 3, "SmoothStronglyConvex")
    end
    plot_save_pic(run(ones(N, N) ./ N, K, fctParams, 1, 100, "funcValue", 5), fctParams)
end

