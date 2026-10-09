# Usage:  julia bench_nuts.jl cpu 1000
#         julia bench_nuts.jl gpu 100000
using Reactant
using Reactant: ProbProg
using Statistics
using Random

backend = length(ARGS) >= 1 ? ARGS[1] : "cpu"
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000

# Must happen BEFORE any Reactant arrays/RNG are created
Reactant.set_default_backend(backend)
println("backend = ", backend, ", N = ", N)

# Synthetic data (true slope 2, intercept 5, noise sd 0.5); fixed seed
Random.seed!(0)
xs = collect(range(0.0, 5.0; length=N))
ys = 2.0 .* xs .+ 5.0 .+ 0.5 .* randn(N)

function model(rng, xs)
    _, slope = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope)
    _, intercept = ProbProg.sample(rng, ProbProg.Normal(0.0, 10.0, (1,)); symbol=:intercept)
    _, ys = ProbProg.sample(
        rng,
        ProbProg.Normal(slope .* xs .+ intercept, 1.0, (length(xs),));
        symbol=:ys,
    )
    return ys
end

obs = ProbProg.Constraint(:ys => ys)
obs_tensor = ProbProg.flatten_constraint(obs)
constrained_addresses = ProbProg.extract_addresses(obs)
selection = ProbProg.select(ProbProg.Address(:slope), ProbProg.Address(:intercept))

function infer(rng, xs, obs_tensor, step_size, inverse_mass_matrix)
    trace, = ProbProg.generate(rng, obs_tensor, model, xs; constrained_addresses)
    trace, = ProbProg.mcmc(
        rng, trace, model, xs;
        selection, algorithm=:NUTS,
        step_size, inverse_mass_matrix,
        num_warmup=200, num_samples=500,
    )
    return trace
end

rng = Reactant.ReactantRNG()
step_size = Reactant.ConcreteRNumber(0.1)
inverse_mass_matrix = Reactant.ConcreteRArray([1.0 0.0; 0.0 1.0])

# 1) compile time, measured separately
t_compile = @elapsed begin
    global compiled_infer = @compile optimize=:probprog infer(
        rng, xs, obs_tensor, step_size, inverse_mass_matrix,
    )
end
println("compile time: ", round(t_compile; digits=2), " s")

# 2) warmup call (not timed)
Array(compiled_infer(rng, xs, obs_tensor, step_size, inverse_mass_matrix))

# 3) timed runs; Array(...) forces the device to finish before the clock stops
reps = 5
times = [@elapsed(Array(compiled_infer(rng, xs, obs_tensor, step_size, inverse_mass_matrix)))
         for _ in 1:reps]
println("run times (s): ", round.(times; digits=4))
println("best: ", round(minimum(times); digits=4), " s   median: ", round(median(times); digits=4), " s")

# 4) sanity check
samples = Array(compiled_infer(rng, xs, obs_tensor, step_size, inverse_mass_matrix))
println("slope mean = ", mean(samples[:, 1]), "  intercept mean = ", mean(samples[:, 2]))