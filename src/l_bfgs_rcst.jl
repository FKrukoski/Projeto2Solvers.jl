export l_bfgs_rcst
export Steighaug
export BhaskaraTop
export LimitedMemory

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
  
  function Steighaug(gx, B, Δ)  
    # ϵ = 1e-4
    ϵ = min(0.5,(norm(gx))^0.5)*norm(gx) # tolerância sugerida Algor 7.1 
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
      if dotdBd ≤ 0 
        # pára o método se a direção dⱼ é direção de curvatura não positiva 
        # encontrar τ >=0 tal que pk e satisfaz ||pk|| = Δk
        # ou seja, a interseção da direção com a região de confiança
        m1, m2 = BhaskaraTop(z, d, Δ)
        return z + m1*d
      end
      
      dotrr = dot(r,r)
      α = dotrr/dotdBd
      zx = z + α*d
      if norm(zx) ≥ Δ
        #para se zⱼ₊₁ viola os limites da região de confiança    
        # encontrar τ >=0 tal que pk e satisfaz ||pk|| = Δk
        # ou seja, a interseção da direção com a região de confiança
        m1, m2 = BhaskaraTop(z, d, Δ)    
        return z + m1*d
      end
      
      rx = r + α*B*d #Conjugate Gradient
      #se α = 0 => rx = r , β = 1 (logo abaixo)

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

  # L-BGFS-RC-Steighaug-Toint
  function l_bfgs_rcst(
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
    
    #Given the starting point
    x = copy(nlp.meta.x0)
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)

    fx = f(x)
    ∇fx = ∇f(x)
    # initial Hessian approximation 
    n = length(x)

    #trust region radius
    Δ = 1.0
    
    B = Matrix(1.0*I, n, n)
    s = Steighaug(∇fx, B, Δ) 
    y = ∇f(x.+s) - ∇fx
    γ = dot(y,y)/dot(s,y) #ver Nocedal p. 178
    B = Matrix(γ*I, n, n)
    
    # Criar matrizes para armazenar s e y
    if n < 5
      m_vetores = n
    else
      m_vetores = 5
    end

    S = zeros(n, m_vetores)
    Y = zeros(n, m_vetores)
  
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
      s = Steighaug(∇fx, B, Δ) 
      y = ∇f(x.+s) - ∇fx
      if norm(y) < 10e-18
        status=:small_step
      end
      ared = fx - f(x.+s)
      pred = -(dot(∇fx, s) + 1/2 * dot(s, B*s)) 
    
      ρ = ared/pred
      if ρ < η
        Δ = Δ/2
        if Δ < 10e-50
          status =:user
        end
      else 
        x = x + s
        fx = f(x)
        ∇fx = ∇f(x)
        if ρ > 0.75 && norm(s) > 0.8 * Δ
          Δ = 2*Δ
          if Δ > 10e50 #evita que o raio aumente muito
            status =:not_desc
          end
        end
      end
      
      if status != :unknown #small_step
        break
      end
      if iter < m_vetores
        yBs = y .-B*s
        if abs(dot(s,yBs)) > r*norm(s,2)*norm(yBs,2) 
          #6.26 - evita atualizações se o denominador é pequeno
          B = B + (yBs*yBs')/dot(yBs,s)
        end
        S[:,iter+1] = s
        Y[:,iter+1] = y
      else
        yBs = y .-B*s
        if abs(dot(s,yBs)) > r*norm(s,2)*norm(yBs,2) #evita situações em que são ambas zero
          #6.26 - evita atualizações se o denominador é pequeno
          #aqui tem que ter mágica - LBFGS
          δₖ = dot(y,y)/dot(s, y)
          B = LimitedMemory(δₖ, S, Y, iter, n, m_vetores)
        end
        
        cpatu = mod(iter, m_vetores) + 1 
        S[:,cpatu] = s
        Y[:,cpatu] = y
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

  function LimitedMemory(δₖ, S, Y, iter, n, m)
    Dₖ = zeros(m,m)
    Lₖ = zeros(m,m)
    
    B1 = δₖ.*I(n)
    inicio = mod(iter, m) + 1
    Sₖ= [S[:, inicio:m] S[:, 1:inicio-1]]
    Yₖ = [Y[:, inicio:m] Y[:, 1:inicio-1]]
    # Sₖ= [S[:, inicio-1:-1:1] S[:, m:-1:inicio] ] #sey_inverso
    # Yₖ = [Y[:, inicio-1:-1:1] Y[:, m:-1:inicio] ]

    δS = δₖ.*Sₖ
    B2 = [δS Yₖ]
    for i=1:m, j=1:m
      if i>j
        Lₖ[i,j] = dot(S[:,i], Y[:,j]) 
      end
      if i==j
        Dₖ[i,j] = dot(S[:,i], Y[:,j])
      end
    end
    # B3 = inv([δS'*Sₖ Lₖ; Lₖ' -Dₖ])

    B4 = [δS';Yₖ']

    return B1 - B2*(([δS'*Sₖ Lₖ; Lₖ' -Dₖ])\B4) 
    
  end

  function Unrolling(B, S, Y, iter, n, m)
    #Unrolling the BFGS formula - Nocedal p.184
    inicio = mod(iter, m) + 1
    Sₖ= [S[:, inicio:m] S[:, 1:inicio-1]]
    Yₖ = [Y[:, inicio:m] Y[:, 1:inicio-1]]
    # Sₖ= [S[:, inicio-1:-1:1] S[:, m:-1:inicio] ] #sey_inverso
    # Yₖ = [Y[:, inicio-1:-1:1] Y[:, m:-1:inicio] ]
    a = zeros(n,m)
    b = zeros(n,m)
    for i = 1:m
      b[:,i] = Y[:,i]/sqrt(dot(Y[:,i], S[:,i]))
      a[:,i] = B*S[:,i] + sum(dot(b[:,j],S[:,i])*b[:,j] -
                                    dot(a[:,j],S[:,i])*a[:,j] 
                                    for j=1:m)
    a[:,i] = a[:,i]/sqrt(dot(S[:,i], a[:,i]))
    end
    return B + sum(b[:,i]*b[:,i]' - a[:,i]*a[:,i]' for i=1:m)
  end
