export bfgs_bl
export bfgs_rc
export bfgsH

"""
Modelos básicos...
bfgs_bl (busca linear)
bfgs_rc (região de confiança - Steihaug Toint)

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
    
    #trust region radius
    Δ = 1.0

    
    fx = f(x)
    ∇fx = ∇f(x)

    #convergence tolerance
    ϵ = atol + rtol * norm(∇fx)
    #parameters η e r
    η = 0.01 #∈ (0,1e-3)
    r = 0.001 #∈ (0,1)
    
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
    #determinar o primero B
    # B⁰ₖ = δₖI, wher δₖ = dot(y,y)/dot(s,y) Nocedal pg178 (eq.7.20)
    B = Matrix(1.0I, n, n)
    s = Steighaug(∇fx, B, Δ) 
    y = ∇f(x.+s) - ∇fx
    γ = dot(y,y)/dot(s,y)
    if γ >= 1.0
      B = Matrix(γ*I, n, n)
    end
    
    while !(solved || tired)
      #compute sₖ by solving the subproblem
      # (6.27) aqui vai entrar o Steihaug
      s = Steighaug(∇fx, B, Δ) 
      y = ∇f(x.+s) - ∇fx
      ared = fx - f(x.+s)
      pred = -(dot(∇fx, s) + 1/2 * dot(s, B*s)) 
      ρ = ared/pred
      if ρ < η
        Δ = Δ/2
        if Δ < 10e-50
          @error("Δ muito pequeno")
          status =:small_step
        end
      else 
        x = x + s
        fx = f(x)
        ∇fx = ∇f(x)
        if ρ > 0.75 && norm(s) > 0.8 * Δ
          Δ = 2*Δ
          if Δ > 10e50
            @error("Δ muito grande")
            status =:user
          end
        end
      end
      
      if status != :unknown #small_step
        break
      end
      yBs = y .-B*s
      syBs = dot(s, yBs)
      if abs(syBs) > r*norm(s,2)*norm(yBs,2) #era maior ou igual, deixei maior
        # em alguns casos, y - B*s pode ser zero... e nesse caso a atualização 
        #de B será uma divisão por zero.
        
        #why?!?!
        #6.26 - evita atualizações se o denominador é pequeno
        #Simmetric-rank 1 method: prevent breakdown; better approximations 
        # to the true Hessian matrix;
        B = B + (yBs*yBs')/syBs
      end
      
      iter+= 1
      
      Δt = time() - t₀
      solved = norm(∇fx) < ϵ # First order stationary
      tired = neval_obj(nlp) ≥ max_eval > 0|| iter ≥ max_iter > 0 || 
                          Δt ≥ max_time > 0 # Excess time, iteration, evaluations
      
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
    while normr > ϵ || k < m # CG vai em no máximo m-variáveis direções
      dotdBd = dot(d,B*d)
      @info("dotdBd: $dotdBd")
      if dotdBd ≤ 0 
        # para o método se a direção dⱼ é direção de curvatura não positiva 
        # encontrar τ >=0 tal que pk e satisfaz ||pk|| = Δk
        # ou seja, a interseção da direção com a região de confiança
        m1, m2 = BhaskaraTop(z, d, Δ)
        return z + m2*d
      end
      
      dotrr = dot(r,r)
      α = dotrr/dotdBd
      @info("α: $α")
      zx = z + α*d
      if norm(zx) ≥ Δ
        #para se zⱼ₊₁ viola os limites da região de confiança    
        # encontrar τ >=0 tal que pk e satisfaz ||pk|| = Δk
        # ou seja, a interseção da direção com a região de confiança
        m1, m2 = BhaskaraTop(z, d, Δ)    
        return z + m1*d
      end
      
      rx = r + α*B*d #Conjugate Gradient
      #se α = 0 => rx = r , β = 1
      if norm(α) < ϵ^2 
        #@error("α menor que zero") 
        return zeros(m)
      end
      
      if norm(rx) < ϵ
        return zx
      end
      β = dot(rx,rx)/dotrr
      d = -rx+ β*d
      r = rx 
      normr = norm(r)
      z = zx 
      k+=1
    end  
    if k >= m
      #@error ("não encontrou a borda, direção multipla da própria direção")
      return zeros(m)
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