using LinearAlgebra, Random, ForwardDiff
Random.seed!(0)

include("zoom_v2.jl")

function L_BFGS(f, x, m; a1 = 0.5)
    
    ∇f(x) = ForwardDiff.gradient(f, x)
    k = 0
    n = length(x)
    max_iter = 1000
    
    r = ∇f(x)
    H0 = Matrix(I, n, n)# Defino uma H0
    y = zeros(n ,m)
    s = zeros(n, m)
    while norm(∇f(x)) > 1e-5      
        α, iteracao = LinearSearch(f,x,a1)  #Calcular um passo com α de Wolfe
        
        xt = x - α * r #Atualiza o passo xₖ₊₁ = xₖ - αₖ r     
        
        #calcular sₖ e yₖ
        s[:, k%m+1] = xt - x   
        y[:, k%m+1] = ∇f(xt) - ∇f(x)
        
        if k > m
            q = ∇f(x)
            a = zeros(m)
            for i = ((k - 1) % m + 1):-1:((k - m) % m + 1)
                ρ = 1 / dot(y[:,i],s[:,i])
                a[i] = ρ * dot(s[:,i], q)
                q -= a[i] * y[:,i]
            end
            r = H0 * q
            for i = ((k - m) % m + 1):((k - 1) % m + 1)
                ρ = 1 / dot(y[:,i],s[:,i])
                β = ρ * dot(y[:,i],r)
                r += s[:,i] * (a[i] - β) # retorna a direção r = Hₖ∇f
            end
        end
        
        if k == max_iter
            @warn("Máximo de iterações")
            break
        end
        k += 1
        x = xt
    end
    return x, k
end

#f(x) = log(exp(-x[1]) + exp(-x[2]) + exp(x[1] + x[2]))
f(x) = (x[1] - 1)^2 + 4 * (x[2] - 2)^2
∇f(x) = ForwardDiff.gradient(f, x)

x, k = L_BFGS(f, rand(2), 5)
println("x = $x e k = $k")

println(∇f(x))

println("###################################")

#= A Hessiana começa como identidade, dai depois que as iterações passarem m você começa a estocar o primeiro =#