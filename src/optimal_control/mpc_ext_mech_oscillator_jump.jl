#=
    Model Predictive Control (feedback) approach for a mechanical oscillator with two masses, springs and attenuators 

    Inspired by: https://github.com/rdeits/DynamicWalking2018.jl

    Author: Stephan Scholz
    Year: 2021
=#

using JuMP, Ipopt

my_optimizer = with_optimizer(Ipopt.Optimizer, print_level = 0)


const d₁ = 0.5; # Damping
const d₂ = 0.5;
const c₁ = 0.1; # Spring constant
const c₂ = 0.1;
const m₁ = 1.0; # Mass
const m₂ = 2.0;


# System matrix
const A = [0 0 1 0; 0 0 0 1; (-1/m₁)*(c₁ + c₂) c₂/m₁ (-1/m₁)*(d₁ + d₂) d₂/m₁; c₂/m₂ -c₂/m₂ d₂/m₂ -d₂/m₂]

const B = [0 0; 0 0; 1/m₁ 0; 0 1/m₂]  # Input matrix
const C = [1 0 0 0; 0 1 0 0]          # Output matrix

mech_osc(x,u) = A * x + B * u   # System dynamics


function mpc_emo(x_init)

    model = Model(my_optimizer)
    
    ΔT = 0.1            #  Sampling time
    n_horizon = 20;     # MPC horizon

    Nx = size(A)[1]
    Nu = size(B)[2]

    # Input constraints
    umin = -100.0
    umax = 100.0 

    ref = [2.0, 4.0, 0.0, 0.0] # Reference 

    @variables model begin
        x[1:Nx, 1:n_horizon]    # States
        e[1:Nx, 1:n_horizon]    # Errors = Reference - States
        umin <= u[1:Nu, 1:n_horizon] <= umax  # Input constraints
    end


    @constraint(model, x[1:Nx, 1] .== x_init )                          # Initial values
    @constraint(model, [k=1:n_horizon], e[:, k] .== ref[:] - x[:, k])   # Error = Reference - States
    @constraint(model, [k=1:n_horizon-1], x[:, k+1] .== x[:, k] + ΔT * mech_osc(x[:,k], u[:,k])) # System dynamics

    J = @NLexpression(model, 100*sum(e[i,end]^2 for i in 1:Nx) + sum(u[j,end]^2 for j in 1:Nu) )
    
    @NLobjective( model, Min, J)

    optimize!(model)
    return JuMP.value.(x), JuMP.value.(u), JuMP.value.(e)
end

xpos = rand(4)          # Recent states
x_hist = zeros(4,0)     # History of states


for i in 1:50
    
    # Run the MPC control optimization
    x_plan, u_plan, e_plan = mpc_emo(xpos)
    
    # Save states
    x_hist = hcat(x_hist, x_plan[:,1])

    # Apply the planned input signals and simulate one step in time
    ΔT = 0.1
    xpos = xpos + ΔT * mech_osc(xpos, u_plan[:,1])
end

using Plots
plot(x_hist', labels=["x1" "x2" "x3" "x4"], title="Model Predictive Control of a mechanical oscillator")