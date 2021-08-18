#=
Author: Stephan Scholz
Year: 2021

This file contains a machine learning approach (SciML) for the van der Pol oscillator
=#

# ODE function of van der Pol oscillator
function vdp(dx, x, p, t)

    μ = p[1]

    dx[1] = μ * (x[1] - 1/3 * x[1]^3 - x[2] ) 
    dx[2] = 1/μ * x[1]
end

tspan = (0.0, 100.0) # time span
x0    = [0.5, 0.5]   # initial values
p     = [1.0]        # parameter: μ
tsteps= 0.1          # saved time steps


using DifferentialEquations
prob = ODEProblem( vdp, x0, tspan, p )      # Build ODEProblem
sol = solve(prob, Tsit5(), saveat = tsteps) # Solve ODEProblem

# using Plots
# plot(sol, title="Solution of the van der Pol oscillator")


# MACHINE LEARNING
function loss(p)
    sol_pred = solve(prob, Tsit5(), p=p, saveat = tsteps)
    loss = sum(abs2, sol - sol_pred)
    return loss, sol_pred
end

pred_param = [] # Stores all found parameters

callback = function (p, l, pred)
    display(l)
    append!(pred_param, p)
   
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

using DiffEqFlux

p_ml = [0.1] # Initial parameter
loss(p_ml)   # Test whether loss function works

# Run ML training
result_ode = DiffEqFlux.sciml_train(loss, p_ml, ADAM(0.1), cb = callback, maxiters=100)

# Build and plot Cost function
x_min = 0.1
x_max = 2
x_step = 0.01
x_range = range(x_min,x_max,step=x_step)
Nlength = length(x_range)

loss_results = zeros(Nlength)
idx = 1;
for i in x_range
   loss_results[idx],_ = loss([i])
   idx += 1;
end

# Cost function 
plot(x_range, loss_results, label="loss", title="Cost function", xaxis=("Parameter μ"))


# Save Machine Learning results as gif
@gif for pp in pred_param
    sol_remade = solve(remake(prob, p=[pp]), Tsit5(), saveat=tsteps)
    
    # Phase portrait
    pp_text = string( "μ = ", string(round(pp; digits=3)) )
    plot(sol_remade[1,:], sol_remade[2,:], annotations=(0.1, 0, Plots.text(pp_text, :left)), title="Phase portrait of van der Pol oscillator", label=false)
    
    # Oscillation
    # plot(sol_remade, title="Machine Learning of van der Pol oscillator", label=["x1" "x2"], legend=:bottomright)
    
end

