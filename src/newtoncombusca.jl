export newtoncombusca
export tentacholesky

"""
newtoncombusca(nlp, options...)

This package implements the idea of find the minimum τ > 0 such that 
∇²f(xₖ)+τ I is positive definite.

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
  function newtoncombusca(
    nlp::AbstractNLPModel;
    atol::Real = 1e-6,
    rtol::Real = 1e-6,
    max_eval::Int = 10000,
    max_iter::Int = 0,
    max_time::Float64 = 10.0
    )
    
    if !unconstrained(nlp)
      error("Problem is not unconstrained")
    end
    
    x = copy(nlp.meta.x0)
    
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    H(x) = Symmetric(hess(nlp, x), :L)
    fx = f(x)
    ∇fx = ∇f(x)
    
    ϵ = atol + rtol * norm(∇fx)
    t₀ = time()
    
    x⁺ = similar(x) #estava faltando isso!
    
    iter = 0
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0|| 
    iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations
    
    α = 1.0
    η = 1e-2
    Β = 1.0e-3
    
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
    
    while !(solved || tired)
      h = H(x)
      F, status = tentacholesky(h, Β, status)
      if status != :unknown #small_step
        break
      end
      
      d = -(F \ ∇fx)
      
      slope = dot(d, ∇fx)
      
      # Armijo
      α = 1.0
      x⁺ = x + α * d
      f⁺ = f(x⁺)
      while f⁺ ≥ fx + η * α * slope
        α = α / 2
        x⁺ = x + α * d
        f⁺ = f(x⁺)
        if α < 1e-8
          status =:small_step
          break
        end
      end
      if status != :unknown #small_step
        break
      end
      x .= x⁺
      fx = f⁺
      ∇fx = ∇f(x)  
      iter += 1
      
      solved = norm(∇fx) < ϵ # First order stationary
      
      Δt = time() - t₀
      tired = neval_obj(nlp) ≥ max_eval > 0|| 
      iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations
      
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
    objective=fx,
    dual_feas=norm(∇fx),
    elapsed_time=Δt,
    iter=iter
    )
  end
  
  function tentacholesky(h, Β, status)
    F = copy(h)
    fatorou=:inicio
    ρ = minimum(diag(h))       
    if  ρ > 0     
      τ = 0.0
    else
      τ = -ρ + Β
    end  
    k = 0
    while fatorou!=:fatorou
      F = cholesky(h, check = false)
      if !issuccess(F)
        τ = max(2τ, Β)
        h = h+τ*I
        k +=1
        if k > 250
          status=:user 
          break
        end
      else
        fatorou =:fatorou
      end
    end
    return F, status
  end
  
  