@testset "Unconstrained" begin
  @testset "Basic" begin
    # These basic tests are simple ways of checking that your solver isn't breaking.
    nlp = ADNLPModel(
      # x -> (x[1] - 1)^2 + (x[2] - 2)^2 / 4,
      # zeros(2)
      x -> (x[1] - 1.0)^2 + 100 *(x[2] - x[1]^2)^2,
    [-0.1; 1.0]
    )
    # output = with_logger(NullLogger()) do
    #   uncsolver(nlp)
    output = newtoncombusca(nlp)    
    print(output)
    @test isapprox(output.solution, [1.0; 1.0], rtol=1e-2)
    @test output.objective < 1e-3
    @test output.dual_feas < 1e-3
    @test output.status == :first_order
  
    # output = with_logger(NullLogger()) do
    #   gradiente(nlp)    
    # end
    # #@test isapprox(output.solution, [1.0; 2.0], rtol=1e-4)
    # @test output.objective < 1e-4
    # @test output.dual_feas < 1e-4
    # @test output.status == :first_order

  end

  @testset "Failures" begin
    # This checks that your solver is failing accordingly
    nlp = ADNLPModel(
      x -> (3- x[1])^2 + 10 * (x[1] - x[2]^2)^2,
    [-1.2; 1.0]
    )
    output = with_logger(NullLogger()) do
      newtoncombusca(nlp, max_eval = 1)
    end
    @test output.status == :max_eval
    output = with_logger(NullLogger()) do
      newtoncombusca(nlp, max_iter = 1)
    end
    @test output.status == :max_iter
    output = with_logger(NullLogger()) do
      newtoncombusca(nlp, max_time = 1e-8)
    end
    @test output.status == :max_time
  end
end