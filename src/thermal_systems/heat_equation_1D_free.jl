#=
Author: Stephan Scholz
Year: 2021

This file contains a Simulation of the Heat equation
Boundary conditions: heat transfer and heat radiation (nonlinear)
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
const L = 0.1           # Length of rod


#Boundary conditions
const h = 10            # Conduction coefficient
const ϵ = 0.6;          # Emissivity
const sb = 5.67*10^(-8) # Stefan-Boltzmann constant
const k = ϵ * sb        # Radiation coefficient

# Discretization  
const dx = L/100;                     # Spatial sampling: o--o--o--o--o--o
const N = round(Int, (L / dx + 1));   # Number of elements


const Tf = 300.0    # Final time
const dt = 10^(-2); # Time step width
const t_len = round(Int, (Tf/dt + 1))


#Laplace operator
const M = spdiagm(-1 => ones(N-1), 0 => -2*ones(N), 1 => ones(N-1))
M[1,2] = 2
M[end,end-1] = 2

const Mx = zeros(N)
const C = spzeros(N,2) # Boundary Conditions: Output / Outflow

C[1,1] = 1	  # Left side
C[end,2] = 1  # Right side

const q = α*dt/(dx^2);
if q > 0.5
  error("Numerical stability is NOT guaranteed! STOP program!")
end

# Temperatures
const ϑ₀   = 400.0; # Initial temperature
const ϑamb = 273.0; # Ambient temperature

# Define initial temperature
function init_temperature(ϑoffset, scale, dx, num_points)
	a = scale
	b = ϑoffset
	
	L = (num_points-1)*dx
	x = 0 : dx : L
	
	ϑinit = a*sin.(x*2*pi/L) .+ b
	
	return ϑinit
end


# Define the discretized PDE as an ODE function
function heat_eq(dθ,θ,p,t)
 
  λ = p[1]
  ρ = p[2]   
  cap = p[3]
 
  mul!(Mx,M,θ[1:end]) # means: Mx = M * Θ[1:end]

  # Now: zero Neumann boundary conditions
  bc = zeros(2)

  # Later: Natural boundary condition: heat conduction + radiation
  bc[1] =  -1*(h*(θ[1] - ϑamb) + k*(θ[1]^4 - ϑamb^4)); 
  bc[2] =  -1*(h*(θ[end] - ϑamb) + k*(θ[end]^4 - ϑamb^4)); 	
  
  flux_out =(2*dx/λ)*C*bc                     # Temperature input and output
  
  dθ .= λ/(ρ*cap) * 1/(dx^2) * (Mx + flux_out)   # Integration of temperatures

end


θinit = init_temperature(ϑ₀, 30.0, dx, N)

tspan = (0.0, Tf) # Time span
param = [λ, ρ, cap] # Parameter that shall be learned

prob = ODEProblem( heat_eq, θinit,tspan, param )
sol = solve(prob,Euler(),dt=dt,progress=true,save_everystep=true,save_start=true) # use 'save_everystep' only for small systems with few timesteps 

xspan = 0 : dx : L

# Temperature at the beginning (t=0) vs at the end (t=Tf)
# plot(xspan, [sol[:,1], sol[:,end]], label=["t=0" "t=Tf"], xlabel="Position x", ylabel="Temperature")

# Temperature at left and right side
# plot(sol.t,[sol[1,:], sol[end,:]], label=["Left" "Right"], xlabel="Time t", ylabel="Temperature", legend=:bottomright) 


# 2D Heatmap plot: xaxis = time ; yaxis = position
# heatmap(sol.t, xspan, sol[:,:], xlabel="Time t", ylabel="Position x")
