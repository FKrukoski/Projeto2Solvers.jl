export gradiente
export newton
export newton_rc

function gradiente(
  nlp :: AbstractNLPModel;
  max_time = 10.0,
  max_iter = 10_000,
  η₁ = 1e-2,
  atol = 1e-6,
  rtol = 1e-6,
)

t₀ = time()
Δt = time() - t₀
iter = 0

x = copy(nlp.meta.x0)
f(x) = obj(nlp, x)
∇f(x) = grad(nlp, x)
fx = f(x)
gx = ∇f(x)

x⁺ = similar(x)

ϵ = atol + rtol * norm(gx)

status = :unknown

# Cond. de parada
resolvido = norm(gx) < ϵ # true ou false
cansado = Δt > max_time || iter > max_iter
while !(resolvido || cansado) # Não tiver satisfeito as condições de parada
  # Calculo a direção e passo e xₖ₊₁
  d = -gx
  slope = dot(gx, d)
  
  α = 1.0
  x⁺ = x + α * d
  f⁺ = f(x⁺)
  while f⁺ ≥ fx + η₁ * α * slope
      α = α / 2
      x⁺ = x + α * d
      f⁺ = f(x⁺)
      if α < 1e-8
          status = :small_step
          break
      end
  end
  if status != :unknown
      break
  end
  x .= x⁺
  fx = f⁺
  gx = ∇f(x)
  
  resolvido = norm(gx) < ϵ
  Δt = time() - t₀
  iter += 1
  cansado = Δt > max_time || iter > max_iter
end


if resolvido
  status = :first_order
elseif cansado
  if Δt > max_time
      status = :max_time
  elseif iter > max_iter
      status = :max_iter
  end
end

return GenericExecutionStats(status, nlp, objective=fx,
  solution=x, dual_feas=norm(gx), iter=iter, elapsed_time=Δt)
end

##

function newton(
  nlp :: AbstractNLPModel;
  max_time = 10.0,
  max_iter = 10_000,
  η₁ = 1e-2,
  atol = 1e-6,
  rtol = 1e-6,
)

t₀ = time()
Δt = time() - t₀
iter = 0

x = copy(nlp.meta.x0)
f(x) = obj(nlp, x)
∇f(x) = grad(nlp, x)
H(x) = Symmetric(hess(nlp, x), :L)
fx = f(x)
gx = ∇f(x)

x⁺ = similar(x)

ϵ = atol + rtol * norm(gx)

status = :unknown

# Cond. de parada
resolvido = norm(gx) < ϵ # true ou false
cansado = Δt > max_time || iter > max_iter
while !(resolvido || cansado) # Não tiver satisfeito as condições de parada
  # Calculo a direção e passo e xₖ₊₁
  Hx = H(x)
  F = cholesky(Hx, check=false)
  if !issuccess(F)
      status = :not_desc
      break
  end
  d = -(F \ gx)
  slope = dot(gx, d)
  
  α = 1.0
  x⁺ = x + α * d
  f⁺ = f(x⁺)
  while f⁺ ≥ fx + η₁ * α * slope
      α = α / 2
      x⁺ = x + α * d
      f⁺ = f(x⁺)
      if α < 1e-8
          status = :small_step
          break
      end
  end
  if status != :unknown
      break
  end
  x .= x⁺
  fx = f⁺
  gx = ∇f(x)
  
  resolvido = norm(gx) < ϵ
  Δt = time() - t₀
  iter += 1
  cansado = Δt > max_time || iter > max_iter
end


if resolvido
  status = :first_order
elseif cansado
  if Δt > max_time
      status = :max_time
  elseif iter > max_iter
      status = :max_iter
  end
end

return GenericExecutionStats(status, nlp, objective=fx,
  solution=x, dual_feas=norm(gx), iter=iter, elapsed_time=Δt)
end


##

function newton_rc(
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
  H(x) = Symmetric(hess(nlp, x), :L)

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
  η₁ = 1e-2
  η₂ = 0.75
  Δ = 1.0
  iter = 0
  
  while !(solved || tired)
    fx = f(x)
    gx = ∇f(x)
    Hx = H(x)
    model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    @variable(model, d[1:2])
    @objective(model, Min, fx + dot(d, gx) + dot(d, Hx * d) / 2)
    @NLconstraint(model, d[1]^2 + d[2]^2 ≤ Δ^2)
    optimize!(model)
    d = value.(d)
    
    Ared = f(x) - f(x + d)
    Pred = f(x) - (fx + dot(d, gx) + dot(d, Hx * d) / 2)
    ρ = Ared / Pred
    if ρ < η₁
        Δ = Δ / 2
    elseif ρ < η₂
        x = x + d
    else
        x = x + d
        Δ = 2Δ
    end
    
    iter += 1
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0|| #Evaluations
            iter ≥ max_iter > 0 || #iterations
            Δt ≥ max_time > 0 # Excess time

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