using LinearAlgebra, ForwardDiff, Random, NLPModels

export Zoom, StrongWolfe_LineSearch



function Zoom(nlp::AbstractNLPModel, x, r, alo, ahi, η1::Float64, η2::Float64; max_iter = 20)

    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    
    iter = 0
    α = 0
    while true 
        a = (alo + ahi) / 2

        if f(x + a * r) > f(x) + η1 * a * dot(∇f(x), r) || f(x + a * r) ≥ f(x + alo * r)
            ahi = a 
        else
            if abs(dot(∇f(x + a * r), r)) ≤ - η2 * dot(∇f(x), r)
                α = a
                break
            end
            if dot(∇f(x + a * r), r) * (ahi - alo) ≥ 0
                ahi = alo
            end
            alo = a
        end
        
        if iter == max_iter
            α = a
            break
        end

        iter +=1
    end
    
    return α, iter 
end

function StrongWolfe_LineSearch(nlp::AbstractNLPModel, x, r, a1::Float64, η1::Float64, η2::Float64, p::Float64; max_iter = 100)

    amax = 10 * a1
    
    a0 = 0
    iter = 0
    
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)

    f_old = 0
    a = 0
    
    while true
        if f(x + a1 * r) > f(x) + η1 * a1 * dot(∇f(x),r) || (iter > 0 && f(x + a1 * r) > f_old)
            a, iter = Zoom(nlp, x, r, a0, a1, η1, η2);
            break
        end
        
        if abs(dot(∇f(x + a1 * r), r)) ≤ - η2 * dot(∇f(x), r)
            a = a1
            break
        end

        if dot(∇f(x + a1 * r), r) ≥ 0
        a, iter = Zoom(nlp, x, r, a1, a0, η1, η2);
            break
        end
        
        iter += 1 
        a0 = a1
        a1 = p * a0 + (1 - p) * amax
        # a1 += p * (amax - a1)
        f_old = f(x + a0 * r)

        if iter == max_iter
            # println("LineaSearch Max iter")
            a = a1
            break
        end
    
    end
    return a, iter
end
