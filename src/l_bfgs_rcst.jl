export bfgs_bl
export bfgs_rc
export bfgsH
export Steighaug
export BhaskaraTop

"""
l_bfgs_rcst(nlp; options...) - Ainda não chegamos nesse!

Este método é chamado L-BFGS com região de confiança por Steihaug-Toint.

Tenta-se resolver B_k d = - ∇f(xₖ) usando Gradientes Conjugados. 
Se em algum momento a direção ficar maior que a região de confiança, 
ela é truncada. Bₖ é a aproximação de memória limitada de BFGS. 
Porém, deve-se guardar apenas os últimos p vetores sₖ e yₖ, 
e não criar a matriz Bₖ explicitamente. 

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
  function bfgs_bl(
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
    
    fx = f(x)
    ∇fx = ∇f(x)
    n = length(x)
    Hx = Matrix(1.0I, n, n)
    
    ϵ = atol + rtol * norm(∇fx)
    t₀ = time()
    
    iter = 0
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0 || 
    iter ≥ max_iter > 0 ||
    Δt ≥ max_time > 0 
    # Excess time, iteration, evaluations
    
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
    
    # Aqui começa o show
    
    while !(solved || tired)
      α = 1.0
      η = 1e-2
      
      d = -Hx * ∇fx
      slope = dot(d, ∇fx)
      
      # Armijo
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
      
      s = α * d
      y = ∇f(x⁺) - ∇fx 
      
      
      if dot(s,y) <= 0
        @warn("sᵀy = $(dot(s,y))")
      else
        Hx = bfgsH(Hx, s, y)
      end
      
      x .= x⁺
      fx = f⁺
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
  
  function bfgsH(H, s, y)
    ρ = 1 / dot(s, y)
    H = (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'    
    return H
  end
  




  #REGIÃO DE CONFIANÇA
  function bfgs_rc(
    nlp::AbstractNLPModel;
    atol::Real = 1e-6,
    rtol::Real = 1e-6,
    max_eval::Int = 50000,
    max_iter::Int = 0,
    max_time::Float64 = 10.0
    )
    
    if !unconstrained(nlp)
      error("Problem is not unconstrained")
    end
    
    #Given the starting point
    x = copy(nlp.meta.x0)
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    
    # initial Hessian approximation 
    n = length(x)
    B = Matrix(1.0I, n, n)
    #trust region radius
    Δ = 1.0
    
    fx = f(x)
    ∇fx = ∇f(x)
    #convergence tolerance
    ϵ = atol + rtol * norm(∇fx)
    #parameters η e r
    η = 0.1 #∈ (0,1e-3)
    r = 0.5 #∈ (0,1)
    
    t₀ = time()
    
    iter = 0
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0 || 
                      iter ≥ max_iter > 0 ||
                        Δt ≥ max_time > 0 
    # Excess time, iteration, evaluations
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
    
    # Aqui começa o show
    
    while !(solved || tired)
      #compute sₖ by solving the subproblem
      # (6.27) aqui vai entrar o Steihaug
      s = Steighaug(∇fx, B, Δ) 
      #  @info("x: $x")
      #  @info("s: $s")
      # # @info("x+s: $(x.+s)")
      # @info("y: $(∇f(x.+s)-∇fx)")
      y = ∇f(x.+s) - ∇fx
      ared = fx - f(x.+s)
      pred = -(dot(∇fx, s) + 1/2 * dot(s, B*s)) 
      
      ρ = ared/pred
      @info("ρ: $ρ")

      if ρ > η
        @info("ρ > η: $ρ > $η?")
        # @info("x: $x")
        # @info("s: $s")
        x = x + s
        fx = f(x)
        ∇fx = ∇f(x)
        # @info("x: $x")
      end
      if ρ > 0.75 && norm(s) > 0.8 * Δ
        @info("Aumenta Δ: $Δ")
        Δ = 2*Δ
        if Δ > 10e50
          @error("Δ muito grande")
          status =:user
        end
      end
      if ρ < 0.1
        @info("Reduz Δ: $Δ")
        Δ = 0.5*Δ
        if Δ < 10e-50
          @error("Δ muito pequeno")
          status =:small_step
        end
      end
      
      if status != :unknown #small_step
        break
      end
      yBs = y .-B*s
      # @info("y: $y")
      # @info("B*s: $(B*s)")
      # @info(abs(dot(s,yBs)))
      # @info(r*norm(s,2)*norm(yBs,2))
      if abs(dot(s,yBs)) >= r*norm(s,2)*norm(yBs,2) #6.26
        # @info("aqui tem que ter mágica - LBFGS")
        B = B + (yBs*yBs')/dot(yBs,s)
        @info("novo B: $(B + (yBs*yBs')/dot(yBs,s))")
      end
      
      iter+= 1
      
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
  
  function Steighaug(gx, B, Δ; ϵ = 1.0e-4)  
    m = length(gx)
    zx = zeros(m)
    r = gx
    d = -r
    
    rx = r
    z = zx
    normr = norm(r)
    if normr < ϵ
      return z
    end
    # enquanto estiver dentro da região de confiança
    k=0
    while normr > ϵ || k < 5
      dotdBd = dot(d,B*d)
      if dotdBd ≤ 0 
        # encontrar τ que minimiza o modelo (4.5) e satisfaz ||pk|| = Δk
        m1, m2 = BhaskaraTop(z, d, Δ)
        # @info("m2 :$m2")
        return z + m2*d
      end
      
      dotrr = dot(r,r)
      α = dotrr/dotdBd
      zx = z + α*d
      if norm(zx) ≥ Δ
        # encontrar τ >=0 tal que pk e satisfaz ||pk|| = Δk
        m1, m2 = BhaskaraTop(z, d, Δ)    
        # @info("m1 :$m1")
        return z + m1*d
      end
      
      rx = r + α*B*d
      if norm(rx) < ϵ
        return zx
      end
      β = dot(rx,rx)/dotrr
      d = -rx+ β*d
      r = rx 
      z = zx 
      k+=1   
    end  
  end
  
  function BhaskaraTop(z, d, Δ)
    a=dot(d,d)
    b=2dot(z,d)
    c=dot(z,z) - Δ^2
    Delta = b^2-4*a*c
    if Delta < 0 
      @warn("Δ<0")
    else
      t1 = (-b+sqrt(Delta))/2a
      t2 = (-b-sqrt(Delta))/2a
      m1 = max(t1,t2)  
      m2 = min(t1,t2) 
      return m1, m2
    end                 
  end