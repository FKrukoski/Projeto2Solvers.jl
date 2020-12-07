using LinearAlgebra, ForwardDiff

function NewtonProjeto(f, x)
    ∇f(x) = ForwardDiff.gradient(f, x)
    H(x) = ForwardDiff.hessian(f, x)
    m, n = size(H(x))
    η = 1e-2
    Β = 1.0e-3
    num_backtrack = 0
    iter = 0
    while norm(∇f(x)) > 1e-4
        h = H(x)

        if minimum(diag(h)) > 0     
            ρ = 0.0
        else
            ρ = -minimum(diag(h)) + Β
        end       

        k = 0  
        while   issuccess(cholesky(h + ρ*I, check=false)) == false
                ρ = max(2ρ, Β)
                k = k+1
        end    
        
        F = cholesky(h + ρ*I)
        J = F.L  
        y = - J \ ∇f(x)
        M = J'
        d = M \ y
        
        # Armijo
        α = 1.0
        while f(x + α * d) ≥ f(x) + η * α * dot(d, ∇f(x))
            α = α / 2
            num_backtrack = num_backtrack + 1
            if α < 1e-8
                error("Erro no backtracking")
            end
        end
        x = x + α * d
        
        iter += 1
        if iter > 10000
            error("Nao converge")
        end
    end
    
    return x, iter, num_backtrack
end