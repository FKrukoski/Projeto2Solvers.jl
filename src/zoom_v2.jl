using LinearAlgebra, ForwardDiff, Random

export zoom, LinearSearch

f(x) = (x[1] - 1)^2 + 4 * (x[2] - 2)^2
∇f(x) = ForwardDiff.gradient(f, x)
H(x) = ForwardDiff.hessian(f, x)

x = rand(2)
s = rand(2)
y = rand(2)

d = -H(x) * ∇f(x)
ϕ(α) = f(x + α * d)
ϕd(α) = ForwardDiff.derivative(ϕ, α)


function zoom(f, x, alo, ahi; η1 = 1e-4, η2 = 0.5, max_iter = 100)

    ∇f(x) = ForwardDiff.gradient(f, x)
    H(x) = ForwardDiff.hessian(f, x)
    d = -H(x) * ∇f(x)
    ϕ(α) = f(x + α * d)
    ϕd(α) = ForwardDiff.derivative(ϕ, α)
    
    # η1 = 1e-4
    # η2 = 0.5
    # max_iter = 100
    
    iter = 0
    α = 0
    while true 
        a = (alo + ahi) / 2

        if ϕ(a) > ϕ(0) + η1 * a * ϕd(0) || ϕ(a) ≥ ϕ(alo)
            ahi = a 
            # println("Dentro do primeio if. a = $a e ahi = $ahi")
        else
            if abs(ϕd(a)) ≤ - η2 * ϕd(0)
                α = a
                # println("Dentro do segundo if. a = $a e α = $α.")
                break
            end
            if ϕd(a) * (ahi - alo) ≥ 0
                ahi = alo
                # println("Dentro do terceiro if. ahi = $ahi e alo = $alo.")
            end
            alo = a
            # println("Fora dos if. a = $a, alo = $alo, ahi = $ahi e α = $α.")
        end
        
        if iter == max_iter
            α = a
            break
        end

        iter +=1
    end
    
    return α, iter 
end
# zoom(f, x, -10, 10)

function LinearSearch(f, x, a1; η1 = 1e-4, η2 = 0.5, ρ = 0.9, max_iter = 100)

    # η1 = 1e-4
    # η2 = 0.5
    # ρ = 0.8
    # max_iter = 100
    amax = 10 * a1
    
    a0 = 0
    i = 1
    
    ∇f(x) = ForwardDiff.gradient(f, x)
    H(x) = ForwardDiff.hessian(f, x)
    d = -H(x) * ∇f(x)
    ϕ(α) = f(x + α * d)
    ϕd(α) = ForwardDiff.derivative(ϕ, α)
    
    ϕ_old = 0
    a = 0
    
    while true 
        if ϕ(a1) > ϕ(0) + η1 * a1 * ϕd(0) || ( i > 1 && ϕ(a1) > ϕ_old)
            a, iter = zoom(f,x, a0, a1);
            # println("Estou no primeiro if. a = $a")
            break
        end
        
        if abs(ϕd(a1)) ≤ - η2 * ϕd(0)
            a = a1
            # println("Estou no segundo if. a = $a")
            break
        end

        if ϕd(a1) ≥ 0
            a, iter = zoom(f, x, a1, a0)
            # println("Estou no terceiro if. a = $a")
            break
        end
        
        if i == max_iter
            # println("Iterações máximas em Linear Search.")
            a = a1
            break
        end
        
        i+=1
        a0 = a1
        a1 = ρ * a0 + (1 - ρ) * amax
        ϕ_old = ϕ(a0)

    end
    # println("Sai do loop. a = $a, a1 = $a1, a0 = $a0")
    return a, i
end
# LinearSearch(f, rand(2), 1)







    



 