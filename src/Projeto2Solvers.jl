module Projeto2Solvers

# standard lib
using LinearAlgebra

# JSO packages
using NLPModels
using SolverTools

include("uncsolver.jl")
include("newtoncombusca.jl")
include("l_bfgs_rcst.jl")

end # module
