# Blob2D-Te-Ti
# Solves coupled drift-reduced plasma equations using DG for advection and CG for elliptic solve

import os
from firedrake import *
from irksome import Dt, TimeStepper, GaussLegendre
import time

# ======================
# SIMULATION PARAMETERS
# ======================

# Disable OpenMP threading for better performance with MPI
os.environ["OMP_NUM_THREADS"] = "1"

# Physical parameters
END_TIME = 10.0
TIME_STEPS = 100
BLOB_HEIGHT = 0.5
BLOB_WIDTH = 0.05
PARALLEL_CONNECTION_LENGTH = 10.0

# Numerical parameters
MESH_RESOLUTION = 64
OUTPUT_INTERVAL = 1

# =================
# MESH SETUP
# =================

mesh = SquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, 1.0, quadrilateral=True)
x, y = SpatialCoordinate(mesh)

# ======================
# FUNCTION SPACES
# ======================

# NB: Canonical name for DG elements on a quadrilateral is DQ
V_w = FunctionSpace(mesh, "DQ", 1) # Discontinuous Galerkin spaces for vorticity
V_n = FunctionSpace(mesh, "DQ", 1) # Discontinuous Galerkin spaces for electron density
V_p_e = FunctionSpace(mesh, "DQ", 1) # Discontinuous Galerkin spaces for electron pressure
V_p_h = FunctionSpace(mesh, "DQ", 1) # Discontinuous Galerkin spaces for ion pressure
V_phi = FunctionSpace(mesh, "CG", 1) # Continuous Galerkin space for electrostatic potential

# Mixed function space for coupled system
V = V_w * V_n * V_p_e * V_p_h * V_phi

# ======================
# TIME DISCRETISATION
# ======================

V_t = FunctionSpace(mesh, "R", 0)

t = Function(V_t) # current time
dt = Function(V_t) # time space

t.assign(0.0)
dt.assign(END_TIME / TIME_STEPS)

# Time integrator (Irksome):
# Implicit midpoint rule (2nd-order accurate):
# u_{n+1} = u_n + h f( (u_n + u_{n+1}) / 2, t_n + h / 2 )
butcher_tableau = GaussLegendre(1) 

# ======================
# FIELD VARIABLES
# ======================

# Functions
solution = Function(V)
w, n, p_e, p_h, phi = solution.subfunctions # concrete
w_s, n_s, p_e_s, p_h_s, phi_s = split(solution) # symbolic representations for weak forms

w.rename("vorticity")
n.rename("density")
p_e.rename("electron_pressure")
p_h.rename("ion_pressure")
phi.rename("potential")

# Test functions
v_w, v_n, v_p_e, v_p_h, v_phi = TestFunctions(V)

# ======================
# INITIAL CONDITIONS
# ======================

# Zero initial vorticity
w.interpolate(0.0)

# Initial Gaussian blob density profile
n0 = 1 + BLOB_HEIGHT * exp(-((x - 0.5)**2 + (y - 0.5)**2) / (BLOB_WIDTH**2))
n.interpolate(n0)

# Additional initial conditions
T_e = 1
p_e.interpolate(n0 * T_e)
p_h.interpolate(0.001 * n0 * T_e)

# ======================
# BOUNDARY CONDITIONS
# ======================

# Zero potential
bc_phi = DirichletBC(V.sub(4), 0, 'on_boundary')

# ======================
# PHYSICAL QUANTITIES
# ======================

# Normal vector for facet integrals
normal = FacetNormal(mesh)

# ExB drift velocity
driftvel = as_vector([phi_s.dx(1), -phi_s.dx(0)])

# Upwind flux term (for DG advection)
driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))

# ======================
# WEAK FORMULATION
# ======================

def advection_term(w, v_w, driftvel, driftvel_n):
    """Discontinuous Galerkin advection term with upwinding."""
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

alpha = Constant(10.0)
gamma_e = Constant(2.0)

F = (
    # Vorticity equation
    Dt(w_s) * v_w * dx
    + advection_term(w_s, v_w, driftvel, driftvel_n)
    - Constant(1.0 / 2.25) * (p_e_s + p_h_s).dx(1) * v_w * dx
    - phi_s * n_s * v_w / PARALLEL_CONNECTION_LENGTH * dx
    
    # Density equation
    + Dt(n_s) * v_n * dx
    + advection_term(n_s, v_n, driftvel, driftvel_n)
    - phi_s * n_s * v_n / PARALLEL_CONNECTION_LENGTH * dx
    
    # Electron pressure equation
    + Dt(p_e_s) * v_p_e * dx
    + advection_term(p_e_s, v_p_e, driftvel, driftvel_n)
    + gamma_e * sqrt(T_e) * p_e_s * v_p_e / PARALLEL_CONNECTION_LENGTH * dx
    
    # Ion pressure equation
    + Dt(p_h_s) * v_p_h * dx
    + advection_term(p_h_s, v_p_h, driftvel, driftvel_n)
    
    # Poisson equation for potential (interior penalty formulation)
    + w_s * v_phi * dx
    + inner(grad(phi_s + p_h_s), grad(v_phi)) * dx
    # DG terms for p_h+
    - dot(avg(grad(v_phi)), jump(p_h_s, normal)) * dS  # Consistency term
    - dot(jump(v_phi, normal), avg(grad(p_h_s))) * dS  # Symmetry term
    + (alpha/h_avg) * dot(jump(p_h_s, normal), jump(v_phi, normal)) * dS  # Penalty term
)

# ======================
# SOLVER CONFIGURATION
# ======================

# https://petsc.org/release/manualpages/SNES/
# solver_params = {
#     'snes_monitor': None, # Print SNES convergence
#     'snes_max_it': 100, # Maximum nonlinear iterations
#     'snes_linesearch_type': 'l2', # Line search algorithm
#     'mat_type': 'aij', # Matrix type
#     'ksp_type': 'preonly', # Only use preconditioner
#     'pc_type': 'lu', # LU factorization
#     'pc_factor_mat_solver_type': 'mumps', # Parallel solver
# }
solver_params = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_max_it': 100,
    'snes_rtol': 1e-8,
    'snes_linesearch_type': 'bt', # backtracking line search
    # Matrix and preconditioner
    'mat_type': 'aij',
    'ksp_type': 'fgmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_schur_precondition': 'selfp',
    # Define block structure
    'pc_fieldsplit_0_fields': '0,1,2,3',  # w, n, p_e, p_h
    'pc_fieldsplit_1_fields': '4',        # phi
    # Field 0 (w, n): use additive Schwarz + GMRES
    'fieldsplit_0_ksp_type': 'gmres',
    'fieldsplit_0_ksp_rtol': 1e-6,
    'fieldsplit_0_pc_type': 'bjacobi',
    'fieldsplit_0_sub_pc_type': 'ilu',
    # Field 1 (phi): use multigrid or LU
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'hypre',
}

# Create time stepper
stepper = TimeStepper(F, butcher_tableau, t, dt, solution, solver_parameters=solver_params, bcs=bc_phi)

# ======================
# MAIN SIMULATION LOOP
# ======================

output_file = VTKFile("./Blob2D-Te-Ti/Blob2D-Te-Ti_output.pvd")
start_time = time.time()
step_counter = 0

while step_counter < TIME_STEPS:
    # Save output every OUTPUT_INTERVAL steps
    if(step_counter % OUTPUT_INTERVAL == 0):
        print(f"Saving output at t = {float(t)}\n")
        w, n, p_e, p_h, phi = solution.subfunctions
        output_file.write(w, n, p_e, p_h, phi)
        
    # Advance solution in time
    step_counter += 1
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(f"Current time: {float(t)}/{END_TIME}, Time step: {float(dt)}")

end_time = time.time()
print(f"Simulation complete. Total wall time: {end_time - start_time} seconds")
