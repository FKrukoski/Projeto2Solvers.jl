module Projeto2Solvers

# standard lib
using LinearAlgebra

# JSO packages
using NLPModels
using SolverTools
using JuMP
using Ipopt

include("uncsolver.jl")
include("newtoncombusca.jl")
include("l_bfgs_rcst.jl")
include("basicos.jl")

end # module
