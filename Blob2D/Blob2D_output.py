# Blob2D_output.py
# Blob2D using DG advection / CG elliptic solve
# Based on Firedrake example:
# https://www.firedrakeproject.org/demos/DG_advection.py.html

import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from irksome import Dt, TimeStepper, GaussLegendre
import time

# Mesh
mesh_res = 32
mesh = SquareMesh(mesh_res, mesh_res, 1.0, quadrilateral=True)

# Function spaces
# NB: Canonical name for DG elements on a quadrilateral is DQ
V_w = FunctionSpace(mesh, "DQ", 4, variant="spectral")  # for w (vorticity)
V_n = FunctionSpace(mesh, "DQ", 4, variant="spectral")  # for n (electron density)
V = V_w * V_n
V_phi = FunctionSpace(mesh, "CG", 4)  # for phi (electrostatic potential) - to be obtained separately
V_driftvel = VectorFunctionSpace(mesh, "CG", 4)  # for driftvel (drift velocity) - to be obtained separately

# Time parameters
T = 5.0 # end time
time_res = 100
skip = 5
t = Constant(0.0) # current time
dt = Constant(T / time_res) # time-step size

# Time-stepper parameters (Irksome):
# Butcher tableau for the implicit midpoint rule (2nd-order accurate):
# u_{n+1} = u_n + h f( (u_n + u_{n+1}) / 2, t_n + h / 2 )
butcher_tableau = GaussLegendre(1) 

# Model parameters
L_par = 10.0
height = 0.5
blob_width = 0.05  # width

# Coordinates
x, y = SpatialCoordinate(mesh)

# Functions
solution = Function(V)
w, n = split(solution)
# w, n, p_e, p_h = split(solution)

# Test functions
v_w, v_n = TestFunctions(V)

# Initial conditions
solution.sub(0).interpolate(0.0)
solution.sub(1).interpolate(1 + height * exp(-((x - 0.5)**2 + (y - 0.5)**2) / (blob_width**2)))

# Save initial conditions for verification
# VTKFile("Blob2D_output_initial.pvd").write(solution.sub(0), solution.sub(1))

# Electrostatic potential (to be solved first)
phi = TrialFunction(V_phi)
v_phi = TestFunction(V_phi)
phi_s = Function(V_phi)

# Electrostatic-potential weak form 
L_phi = inner(grad(phi), grad(v_phi)) * dx
R_phi = - w * v_phi * dx

# Boundary condition for phi
bc_phi = DirichletBC(V_phi, 0, 'on_boundary')

# Solver parameters for phi
linparams_phi = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

# Drift velocity (to be solved second)
driftvel = Function(V_driftvel)
norm = FacetNormal(mesh)
driftvel_n = 0.5 * (dot(driftvel, norm) + abs(dot(driftvel, norm)))

# Main problem
F = (
    Dt(w) * v_w * dx
    - w * dot(driftvel, grad(v_w)) * dx
    # - div(w * driftvel) * v_w * dx
    + Constant(20.0 / 9.0) * grad(n)[1] * v_w * dx
    + Dt(n) * v_n * dx
    - n * dot(driftvel, grad(v_n)) * dx
    # - div(n * driftvel) * v_n * dx
    - phi_s * n * (v_w + v_n) / L_par * dx
    + driftvel_n('-') * ( w('-') - w('+') ) * v_w('-') * dS
    + driftvel_n('+') * ( w('+') - w('-') ) * v_w('+') * dS
    + driftvel_n('-') * ( n('-') - n('+') ) * v_n('-') * dS
    + driftvel_n('+') * ( n('+') - n('-') ) * v_n('+') * dS
)

# Time-stepper parameters, from Cahn-Hilliard example:
# https://www.firedrakeproject.org/Irksome/demos/demo_cahnhilliard.py.html
# https://petsc.org/release/manualpages/SNES/
solver_params = {
    'snes_monitor': None,
    'snes_max_it': 100,
    'snes_linesearch_type': 'l2',
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
stepper = TimeStepper(F, butcher_tableau, t, dt, solution, solver_parameters=solver_params)

# Main loop
output_file = VTKFile("Blob2D_output.pvd")
start_time = time.time()
cnt = 0

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))
    
    # Solve for electrostatic potential
    solve(L_phi == R_phi, phi_s, solver_parameters=linparams_phi, bcs=bc_phi)
    
    # Compute drift velocity
    driftvel.interpolate(as_vector([grad(phi_s)[1], -grad(phi_s)[0]]))
    driftvel_n = 0.5 * (dot(driftvel, norm) + abs(dot(driftvel, norm)))
    
    # Output results every 5 steps
    if(cnt % skip == 0):
        print("Saving output...\n")
        w_s, n_s = solution.subfunctions
        output_file.write(w_s, n_s, phi_s)
        
    # Advance solution in time
    cnt += 1
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(f"Current time: {float(t)}, Time step: {float(dt)}")

end_time = time.time()
print(f"Simulation complete. Total wall time: {end_time - start_time:.2f} seconds")

# Save output at end time
VTKFile("Blob2D_output_final.pvd").write(solution.sub(0), solution.sub(1), phi_s)
