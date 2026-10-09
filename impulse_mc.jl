using Reactant
using Reactant: ProbProg
using Statistics

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [7.1, 8.9, 11.2, 13.0, 15.1]

function model(rng, xs)
    #vairables + line equation
    _, slope = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope)
    _, intercept = ProbProg.sample(rng, ProbProg.Normal(0.0, 10.0, (1,)); symbol=:intercept)
    _, ys = ProbProg.sample(
        rng,
        ProbProg.Normal(slope .* xs .+ intercept, 1.0, (length(xs),));
        symbol=:ys,
    )
    return ys
end

#conditioning on the data
obs = ProbProg.Constraint(:ys => ys) #based on given data
obs_tensor = ProbProg.flatten_constraint(obs) 
constrained_addresses = ProbProg.extract_addresses(obs) #list of fixed sites

# which sites nuts should infer 
selection = ProbProg.select(ProbProg.Address(:slope), ProbProg.Address(:intercept))

# training the model using NUTS
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

#ready to run for themachine
compiled_infer = @compile optimize=:probprog infer(
    rng, xs, obs_tensor, step_size, inverse_mass_matrix,
)

#has all the explored pairs
trace = compiled_infer(rng, xs, obs_tensor, step_size, inverse_mass_matrix)
samples = Array(trace)  # 500x2: col 1 = slope, col 2 = intercept

println("mean slope     = ", mean(samples[:, 1]))
println("mean intercept = ", mean(samples[:, 2]))