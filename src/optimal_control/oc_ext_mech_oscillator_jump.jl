#=
    Optimal Control (feed-forward) approach for a mechanical oscillator with two masses, springs and attenuators 

    Author: Stephan Scholz
    Year: 2021
=#


using JuMP,Ipopt, LinearAlgebra

#model initialization
mod = Model( optimizer_with_attributes( Ipopt.Optimizer , "max_iter" => 1000))

const d₁ = 0.5; # Damping
const d₂ = 0.5;
const c₁ = 0.1; # Spring constant
const c₂ = 0.1;
const m₁ = 1.0; # Mass
const m₂ = 2.0;

# System matrix
A = [0 0 1 0; 0 0 0 1; (-1/m₁)*(c₁ + c₂) c₂/m₁ (-1/m₁)*(d₁ + d₂) d₂/m₁; c₂/m₂ -c₂/m₂ d₂/m₂ -d₂/m₂]

B = [0 0; 0 0; 1/m₁ 0; 0 1/m₂]  # Input matrix
C = [1 0 0 0; 0 1 0 0]          # Output matrix

Nx = size(A)[1]
Nu = size(B)[2]

# Input constraints
umin = -10.0;
umax = 10.0;

K = 200; # Number of time steps
x0 = zeros(4); # Initial conditions; also possible: x0 = 3*rand(4); 

∆T = 0.1
Tf = K * ∆T; # Final time

Q = 100 * Diagonal(ones(Nx))    # Weighing matrix for states / error: e' Q e or x' Q x
R = Diagonal(ones(Nu))          # Weighing matrix for inputs: u' R u

ref = [2.0, 4.0, 0.0, 0.0]      # Reference

@variable(mod, umin <= u[1:2, 1:K-1] <= umax) # Control signals

@variable(mod, x[1:Nx, 1:K])            # States x
@constraint(mod, x[1:Nx, 1] .== x0 )    # Initial values

@variable(mod, e[1:Nx, 1:K] )           # Error between reference and states
@constraint(mod,  e[1:Nx, 1:K] .== ref[1:Nx] - x[1:Nx, 1:K])

for k = 1: K-1
    @constraint(mod, x[:, k+1] .== x[:, k] + ∆T *(A * x[:,k] + B * u[:,k] )) # Discrete-time system dynamics 
end
    
J = @NLexpression(mod, 0.5 * ∆T * sum( Q[1,1]*e[1,k]^2 + Q[2,2]*e[2,k]^2 + Q[3,3]*e[3,k]^2 + Q[4,4]*e[4,k]^2 + R[1,1]*u[1,i]^2 + R[2,2]*u[2,i] for k=1:K, i=1:K-1 ))
@NLobjective( mod, Min, J)
optimize!(mod)

# Optimization results
x_sol = JuMP.value.(x)
u_sol = JuMP.value.(u)

using Plots
tspan = 0 : ∆T : (K-1)*∆T
plot(tspan, x_sol', label=["x1" "x2" "x3" "x4"], title="States") 
plot(tspan[1:end-1], u_sol', label=["u1" "u2"], title="Control signals") 


# Rebuild solution as ODE with feed-forward input: u = optimization result 

# Definition of ODE
function mech_oscillator(dx,x,p,t)
    n = floor(Int64,(t/Tf)*K) + 1
    u = u_sol2[:,n] # [u1; u2]
    dx .= A*x + B*u # Right-hand side of ODE
end


using DifferentialEquations

u_sol2 = hcat(u_sol, zeros(2,1)) # Input signal found by optimizer
trange = (0, (K-1)*∆T)

prob = ODEProblem(mech_oscillator, x0, trange)
sol  = DifferentialEquations.solve(prob, Euler(), dt = ∆T)
plot(sol, labels=["x1" "x2" "x3" "x4"], title="Optimal Control of a mechanical oscillator")