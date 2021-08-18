#=
Author: Stephan Scholz
Year: 2021

This file contains a Simulation of the Heat equation with a proportional controller
Boundary conditions: 
- left side: heat induction (proportional control)
- right side: heat transfer and heat radiation (nonlinear)
Dimension: 1D
Spatial Approximation: Finite Differences
Temporal Approximation: Forward-Euler method
=#

using LinearAlgebra, Plots, SparseArrays, DifferentialEquations

# Define the constants for the PDE
const λ = 45.0          # Conductivity
const ρ = 7800.0;       # Density
const cap = 480.0;      # Specific heat capacitivity
const α = λ/(cap * ρ)   # Diffusivity
const L = 0.1 # 10*10^(-3)    # Length of rod


#Boundary conditions
const h = 10        # Heat transfer coefficient
const ϵ = 0.6;      # Emissivity
const sb = 5.67*10^(-8) # Stefan-Boltzmann constant
const k = ϵ * sb # Radiation coefficient

# Discretization  

const N = 101;      # Number of elements
const Δx = L/(N-1)  # Spatial sampling
const Tf = 3000.0 # 15.0;
const dt = 10^(-2) # 1.0# 10^(-4); # Time step width
# o--o--o--o--o--o



#Laplace operator
const M = spdiagm(-1 => ones(N-1), 0 => -2*ones(N), 1 => ones(N-1)) 
M[1,2] = 2
M[end,end-1] = 2

const Mx = zeros(N)


# x'(t) = A x(t) + B u(t) + E w(t)
const A = α/(Δx^2) * M
const b = 1
const B = spzeros(N); # Input vector
const B[1] = 2 * (α/(λ*Δx)) * b
const E = spzeros(N); # Disturbance vector
const E[end] = 2 * (α/(λ*Δx)) * b



const q = α*dt/(Δx^2);
if q > 0.5
  error("Numerical stability is NOT guaranteed! STOP program!")
end


# Temperatures
const θ₀   = 300.0  # Initial temperature
const θamb = 298.0  # Ambient temperature
const θref = 400.0  # Reference temperature

# Flux towards environment
ϕout(θ) = -h * (θ - θamb)  - k*(θ^4 - θamb^4) 

### Controller ###
const Kp = 10^3 # Proportional gain
const yref = 400.0 # Reference temperature

u_in(err) = Kp * err # input signals



global u_hist
u_hist = zeros(1,0) # History of input signals

function heat_eq!(dx,x,p,t)
    global u_hist
  
    err = yref - x[end]
    u = u_in(err)
    
    u_hist = hcat(u_hist, u)

    dx .= A * x + B * u + E * ϕout( x[end] )    # Integration of temperatures
   
  end


x₀ = θ₀ * ones(N)
tspan = (0.0, Tf)

prob = ODEProblem( heat_eq!, x₀, tspan ) # ODE Problem
sol = solve(prob,Euler(),dt=dt ,progress=true, saveat=1.0, save_start=true);

# 1-dimensional grid
xspan = 0 : Δx : L

using Plots
# 2D Heatmap plot: xaxis = time ; yaxis = position
heatmap(sol.t, xspan, sol[1:end,1:end], xaxis="Time [s]", yaxis="Position [m]", title="Evolution of temperature")

# Temperature at left and right side
# plot(sol.t,[sol[1,:], sol[end-1,:]], label=["Left" "Right"], title="Temperature at the left/right end", legend=:bottomright)