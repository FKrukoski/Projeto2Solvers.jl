module Projeto2Solvers

# standard lib
using LinearAlgebra
using ForwardDiff

# JSO packages
using NLPModels
using SolverTools
using JuMP
using Ipopt

include("uncsolver.jl")
include("newtoncombusca.jl")
include("l_bfgs_rcst.jl")
include("basicos.jl")
include("StrongWolfe.jl")
include("Newton_rc_bissec.jl")
include("Lbfgs_StrongWolfe.jl")

end # module
