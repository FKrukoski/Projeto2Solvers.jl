export uncsolver

"""
    uncsolver(nlp; options...)

Template for an unconstrained solver using the first order conditions

    ‖∇f(xₖ)‖ ≤ ϵₐ + ϵᵣ‖∇f(x₀)‖,

where ϵₐ is an absolute tolerance and ϵᵣ is a relative tolerance.

Options:
- atol: absolute tolerance for the first order condition (default: 1e-6)
- rtol: relative tolerance for the first order condition (default: 1e-6)
- max_eval: maximum number of [functions] evaluations, use ≤ 0 for unlimited (default: 1000)
- max_iter: maximum number of iterations, use ≤ 0 for unlimited (default: 0)
- max_time: maximum elapsed time in seconds, use ≤ 0 for unlimited (default: 10)

Disclaimers for the developer:
- nlp should be the only mandatory argument
- these five options are the current default for other JSO-compliant solvers
- always return a GenericExecutionStats
"""
function uncsolver(
  nlp::AbstractNLPModel;
  atol::Real = 1e-6,
  rtol::Real = 1e-6,
  max_eval::Int = 1000,
  max_iter::Int = 0,
  max_time::Float64 = 10.0
)

  if !unconstrained(nlp)
    error("Problem is not unconstrained")
  end

  x = copy(nlp.meta.x0)

  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  #= Re: Hessian
  Hessians are expensive, so `hess(nlp, x)` only returns the lower triangle of the symmetric Hessian at x.
  You can use it as a symmetric matrix with

      H(x) = Symmetric(hess(nlp, x), :L)

  However, most symmetric linear solvers only use the lower half.
  Furthermore, for large problems with sparse Hessians, you can obtain the triplet list `(i, j, aᵢⱼ)` using

      rows, cols = hess_structure(nlp)
      vals = hess_coord(nlp, x)
      # or
      hess_coord!(nlp, x, vals)

  Finally, if your method only uses matrix-vector products, you can use

      Hv = hprod(nlp, x, v)
      # or
      hprod!(nlp, x, v, Hv)

  Or potentially, a linear operator that automatically does that (read more on github.com/JuliaSmoothOptimizers/LinearOperators.jl).
  In the example below, the Hessian is not created explicitly (unless the model itself doesn't support it):

      H = hess_op(nlp, x)
      Hv = H * v

  Notice that you can't access elements of the Hessian in this case.
  =#

  fx = f(x)
  ∇fx = ∇f(x)

  ϵ = atol + rtol * norm(∇fx)
  t₀ = time()

  iter = 0
  Δt = time() - t₀
  solved = norm(∇fx) < ϵ # First order stationary
  tired = neval_obj(nlp) ≥ max_eval > 0|| iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations

  # status must be one of a few options found in SolverTools.show_statuses()
  # A good default value is :unknown.
  status = :unknown

  # log_header is up for some rewrite in the future. For now, it simply prints the column names with some spacing
  @info log_header(
    [:iter, :fx, :ngx, :nf, :Δt],
    [Int, Float64, Float64, Int, Float64],
    hdr_override=Dict(:fx => "f(x)", :ngx => "‖∇f(x)‖", :nf => "#f")
  )
  # log_row uses the type information of each value, thus we use `Any` here.
  @info log_row(
    Any[iter, fx, norm(∇fx), neval_obj(nlp), Δt]
  )

  # This template implements a simple steepest descent method without any hopes of working.
  # This is where most of your change will happen
  α = 1.0
  while !(solved || tired)
    x -= α * ∇fx
    α *= 0.99
    fx = f(x)
    ∇fx = ∇f(x)

    iter += 1
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0|| iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations

    @info log_row(
      Any[iter, fx, norm(∇fx), neval_obj(nlp), Δt]
    )
  end

  if solved
    status = :first_order
  elseif tired
    if neval_obj(nlp) ≥ max_eval > 0
      status = :max_eval
    elseif iter ≥ max_iter > 0
      status = :max_iter
    elseif Δt ≥ max_time > 0
      status = :max_time
    end
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution=x,
    objective=f(x),
    dual_feas=norm(∇fx),
    elapsed_time=Δt,
    iter=iter
  )
end