using LinearAlgebra, ForwardDiff, NLPModels

export Bisseccao, Newton_rc_bissec

function Bisseccao(g, a, b, max_bissec; λ = 0)  
    ϵ = 1e-4
    status= :resolvido
    iter = 0
    while g(λ) > ϵ
        if g(a) * g(b) < 0
            while abs(b-a) > ϵ
                λ = (b + a) / 2
                if g(λ) * g(a) < 0
                    b = λ
                else g(λ) * g(a) < 0
                    a = λ 
                end
            end
        end 
        if g(a) * g(b) > 0
            if b != 10001
                b = b + 50
            elseif a < 8000
                a = a + 40
            else
                status= :bisseccao_falhou
                break
            end
        end

        if iter > max_bissec
            println("Maximum bissection.")
            break
        end
        iter +=1
    end
    return λ, status
end

function Newton_rc_bissec(
    nlp::AbstractNLPModel;
    η1::Float64 = 1e-2,
    η2::Float64 = 0.75,
    Δ::Float64 = 5.0,
    max_bissec = 1000,
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
    n = nlp.meta.nvar
    
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    H(x) = hess(nlp, x)
    
    fx = f(x)
    ∇fx = ∇f(x)
    Hx = Matrix(H(x))
    g(λ) = norm(inv(Hx + λ * I) * ∇f(x)) - Δ 
    d = - Hx \ ∇fx
    
    ϵ = atol + rtol * norm(∇fx)
    t₀ = time()
    
    iter = 0
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0|| iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations
    
    status = :unknown
    
    @info log_header(
    [:iter, :fx, :ngx, :nf, :Δt],
    [Int, Float64, Float64, Int, Float64],
    hdr_override=Dict(:fx => "f(x)", :ngx => "‖∇f(x)‖", :nf => "#f")
    )
    
    @info log_row(
    Any[iter, fx, norm(∇fx), neval_obj(nlp), Δt]
    )
    
    while !(solved || tired)
        if norm(d) < Δ
            d = - Hx \ ∇f(x)
        else
            λ, stat = Bisseccao(g, -Δ, Δ, max_bissec)
            if stat == :bisseccao_falhou
                @warn("Bissection fail")
                break
            end
            d = - (Hx + λ * I) \ ∇f(x)
        end

        Ared = f(x) - f(x + d)
        Pred = f(x) - (f(x) + dot(d, ∇f(x)) + dot(d, Hx * d) /  2)
        ρ = Ared / Pred
        if ρ < η1
            Δ = Δ / 2
        elseif ρ < η2
            x = x + d
        else
            x = x + d
            Δ = 2Δ
        end   
        
        fx = f(x)
        ∇fx = ∇f(x)
        Hx = Matrix(H(x))
        
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