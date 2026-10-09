# Usage:
#   julia bench_gd.jl plain 1000   # your original loops, plain Julia CPU
#   julia bench_gd.jl cpu   1000   # same math compiled by Reactant, CPU
#   julia bench_gd.jl gpu   1000   # same math compiled by Reactant, GPU
using Random
using Reactant

mode = length(ARGS) >= 1 ? ARGS[1] : "plain"
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
lr = 0.05
epochs = 2000

Random.seed!(0)
xs = collect(range(0.0, 5.0; length=N))
ys = 2.0 .* xs .+ 5.0 .+ 0.5 .* randn(N)
println("mode = ", mode, ", N = ", N)

function linreg(x, y; learning_rate=0.01, epochs=1000)
    m = 0.0
    b = 0.0
    n = length(y)
    for epoch in 1:epochs
        m_gradient = 0.0
        b_gradient = 0.0
        for i in 1:n
            y_pred = m * x[i] + b
            error = y_pred - y[i]
            m_gradient += (2 / n) * x[i] * error
            b_gradient += (2 / n) * error
        end
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient
    end
    return m, b
end

if mode == "plain"
    linreg(xs, ys; learning_rate=lr, epochs=epochs)   # warmup
    times = [@elapsed(linreg(xs, ys; learning_rate=lr, epochs=epochs)) for _ in 1:3]
    println("result (m, b) = ", linreg(xs, ys; learning_rate=lr, epochs=epochs))
    println("run times (s): ", round.(times; digits=4))
    println("best: ", round(minimum(times); digits=4), " s")
else
    Reactant.set_default_backend(mode)   # "cpu" or "gpu"; before any arrays

    function linreg_reactant(x, y, lr)
        n = length(y)
        m = zero(lr)      # starts as traced numbers, not a plain array
        b = zero(lr)
        @trace for _ in 1:2000          # must match `epochs` above
            err = m .* x .+ b .- y
            mg = (2 / n) * sum(x .* err)
            bg = (2 / n) * sum(err)
            m = m - lr * mg
            b = b - lr * bg
        end
        return m, b
    end

    x_r = Reactant.to_rarray(xs)
    y_r = Reactant.to_rarray(ys)
    lr_r = Reactant.ConcreteRNumber(lr)

    # pull both results back to Julia floats (also forces the device to finish)
    fetch(r) = (Float64(r[1]), Float64(r[2]))

    t_compile = @elapsed begin
        global compiled = @compile linreg_reactant(x_r, y_r, lr_r)
    end
    println("compile time: ", round(t_compile; digits=2), " s")

    fetch(compiled(x_r, y_r, lr_r))     # warmup
    times = [@elapsed(fetch(compiled(x_r, y_r, lr_r))) for _ in 1:5]
    println("result (m, b) = ", fetch(compiled(x_r, y_r, lr_r)))
    println("run times (s): ", round.(times; digits=4))
    println("best: ", round(minimum(times); digits=4), " s")
end