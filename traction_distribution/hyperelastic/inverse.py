import os
import time
from typing import List

# Set JAX device (before importing jax)
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # or 'gpu'

import jax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem.mma import optimize
from jax_fem import logger
from scipy.optimize import minimize, Bounds

import numpy as onp
import matplotlib.pyplot as plt

import logging
logger.setLevel(logging.DEBUG)

# Check JAX backend and devices
print('JAX backend:', jax.default_backend())
print('Devices:', jax.devices())

# Save setup
file_dir = 'data/inverse'
os.makedirs(file_dir, exist_ok=True)
file_name = 'load1_noise-0-inverse'

# Load data (measured displacement)
sol_measured = onp.loadtxt('load1_noise-0.txt') # (number of nodes, 3) in 3D

# Material Properties 
mu = 3. # MPa
lmbda = 148. # MPa

# Mesh info
ele_type = 'TET10'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
Lx, Ly, Lz = 1., 1., 0.05 # domain
Nx, Ny, Nz = 60, 60, 2 # number of elements in x-dir, y-dir
dim = 3
# Meshes
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz,
                            domain_x=Lx, domain_y=Ly, domain_z=Lz,
                            data_dir=file_dir, ele_type=ele_type)
# Input mesh info into Mesh class
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# # Helper function for normalizing parameters
# def normalize(val, vmin, vmax): # original params -> normalized params
#     return (val - vmin) / (vmax - vmin)

# def unnormalize(val, vmin, vmax): # normalized params -> original params
#     return val * (vmax - vmin) + vmin

# Traction Distribution
def traction_true(point):
    return np.exp(-(np.power(point[0] - Lx/2., 2)) / (2.*(Lx/5.)**2))

# Weak forms
class HyperElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
            
    # Tensor
    def get_tensor_map(self):
        def psi(F):
            kappa = lmbda + (2./3.) * mu
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x, load_value):
            return np.array([0, load_value, 0])
        return [surface_map]

    def set_params(self, params):
        self.internal_vars_surfaces = []
        # Interpolate params (nodal) to quadrature points for each defined boundary surface
        for i in range(len(self.boundary_inds_list)):
            var_index = 0 # We assume single variable system (u) for the mesh
            
            # Get connectivity and shape functions for this boundary
            # cells_face: (num_selected_faces, num_nodes_per_elem_face)
            cells_face = self.cells_list_face_list[i][var_index]
            
            # Fetch nodal values of parameters for the faces
            # point_vals: (num_selected_faces, num_nodes_per_elem_face)
            point_vals = params[cells_face] 
            
            # shape_vals: (num_selected_faces, num_face_quads, num_nodes_per_elem_face)
            shape_vals = self.selected_face_shape_vals[i] 
            
            # Interpolate to quadrature points: sum_k (N_k * rho_k)
            # vals_at_quads: (num_selected_faces, num_face_quads)
            vals_at_quads = np.sum(point_vals[:, None, :] * shape_vals, axis=-1)
            
            # Append as the argument lists for the surface map functions
            self.internal_vars_surfaces.append([vals_at_quads])

# Boundary Locations
def bottom(point):
    return np.isclose(point[1], np.min(mesh.points, axis=0)[1], atol=1e-5) # True for y = 0

def top(point):
    return np.isclose(point[1], np.max(mesh.points, axis=0)[1], atol=1e-5) # True for y = Ly

# Dirichlet boundary values
# in the direction normal to the boundary("1": tension, "-1": compression)
def zero_dirichlet_val(point):
    return 0.

# Dirichlet boundary info
# [plane, direction, displacement]
# number of elements in plane, direction, displacement should match
dirichlet_bc_info = [[bottom]*3,
                     [0, 1, 2], 
                     [zero_dirichlet_val]*3]

# Neumann boundary locations
# "get_surface_maps" performs surface integral to get the traction
location_fns = [top]

# Instance of the problem
problem = HyperElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           location_fns=location_fns)

# AD wrapper : critical step that makes the problem solver differentiable
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

# TV Loss
def TV_reg(u, alpha=1, epsilon = 1e-6):
    """
    Args:
        epsilon: small value to avoid division by zero
    """
    # np.roll: shifts the entire image one pixel to the right
    u_shift_x = np.roll(u, shift=1, axis=1)
    u_shift_y = np.roll(u, shift=1, axis=0)

    # Zero out the boundary (to prevent wrap-around effect of np.roll)
    u_shift_x = u_shift_x.at[:, 0].set(0.)
    u_shift_y = u_shift_y.at[0, :].set(0.)

    grad_x = u - u_shift_x
    grad_y = u - u_shift_y

    return 0.5 * alpha * np.sum(np.sqrt(grad_x**2 + grad_y**2 + epsilon))

# Objective Function
# TODO: try TV regularization
def J_total(params): # J(u(theta), theta)
    # Solve w/ params
    sol_list = fwd_pred(params)
    # Data term
    u_difference = sol_measured - sol_list[0]
    # Regularization term
    alpha = 1. #1e-9
    TV_reg_term = TV_reg(sol_list[0], alpha=alpha)
    # Objective function
    J = 0.5 * np.linalg.norm(u_difference)**2 + 0.5 * TV_reg_term
    return J

# Gradient of J
def J_grad(rho):
    dJ = jax.grad(J_total)(rho) # (...)
    dJ = dJ.reshape(-1)
    return dJ

# Solution(u) from forward problem
outputs = []
params = []
def output_sol(intermediate_result):
    # Solve
    sol_list = fwd_pred(intermediate_result.x)
    # Save vtu
    vtu_name = f'{file_name}/sol_{output_sol.counter:03d}.vtu'
    save_sol(problem.fes[0], 
             np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
             os.path.join(file_dir, vtu_name), 
             point_infos=[('traction', intermediate_result.x)])
    print(f"Iteration:{output_sol.counter}")
    print(f"Obj:{intermediate_result.fun}")
    outputs.append(intermediate_result.fun)
    params.append(intermediate_result.x)
    output_sol.counter += 1
    return 
output_sol.counter = 1

# Initial guess
rho_ini = 0.01 * np.ones(problem.fes[0].num_total_nodes) # (num_nodes,)
sol_list = fwd_pred(rho_ini)
# Initial solution
start_time = time.time() # Start timing
# params = params + [rho_ini]
# save_sol(problem.fes[0], 
#          np.ones((len(sol_measured), 4)),
#          os.path.join(file_dir, f'{file_name}/sol_000.vtu'),
#          point_infos=[('traction', rho_ini)])


# Optimization setup
numConstraints = 1
optimizationParams = {'maxiter':100, 'disp':True} # 'ftol':1e-4

# Optimize
results = minimize(J_total, jac=J_grad, 
                   x0=rho_ini, 
                   options=optimizationParams, 
                   method='L-BFGS-B', callback=output_sol)
end_time = time.time() # End timing
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)


# Print log and save to text file
log_info = f"""Total optimization runtime: {hours}h {minutes}m {seconds}s
Total Iteration: {output_sol.counter}
Final Objective Value: {results.fun}
"""

print(log_info)

# Save results to text file
results_file_path = os.path.join(file_dir, file_name, 'optimization_results.txt')
with open(results_file_path, 'w') as f:
    f.write(log_info)
    f.write(f"\nOptimization Success: {results.success}\n")
    f.write(f"Optimization Message: {results.message}\n")
    f.write(f"Number of Function Evaluations: {results.nfev}\n")
    f.write(f"Number of Gradient Evaluations: {results.njev}\n")

print(f"Results saved to: {results_file_path}")

# Plot the optimization results
obj = onp.array(outputs)
plt.figure(1, figsize=(10, 8))
plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
# plt.title(rf'L2 Regularization with $\lambda = 1e-9$', fontsize=20)

# Save
plt.savefig(os.path.join(file_dir, file_name, 'optimization_graph.png'), dpi=300, bbox_inches='tight')