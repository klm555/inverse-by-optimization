import os
import time
from typing import List
import itertools

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
file_name = 'load2_noise-0_reg_all-vertical-three-pts'

# Load data (measured displacement)
sol_measured = onp.loadtxt('load2_noise-0.txt') # (number of nodes, 3) in 3D

# Material Properties 
E = 1.0e3 # MPa

# Mesh info
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
Lx, Ly, Lz = 1., 1., 0.05 # domain
Nx, Ny, Nz = 40, 40, 2 # number of elements in x-dir, y-dir
dim = 3
# Meshes
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz,
                            domain_x=Lx, domain_y=Ly, domain_z=Lz,
                            data_dir=file_dir, ele_type=ele_type)
# Input mesh info into Mesh class
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Target points and their indices for calculating data mismatch
# Front surface (z = Lz)
target_z_mask = onp.isclose(mesh.points[:, 2], Lz, atol=1e-5)
target_y_mask = onp.isclose(mesh.points[:, 1], 0.5 * Ly, atol=1e-5)
target_x_mask = (mesh.points[:, 0] >= Lx/4.) & (mesh.points[:, 0] <= 3.*Lx/4.)

target_mask = target_x_mask & target_y_mask & target_z_mask
target_idx = onp.where(target_mask)[0]

# # Helper function for normalizing parameters
def normalize(val, vmin, vmax): # original params -> normalized params
    return (val - vmin) / (vmax - vmin)

def unnormalize(val, vmin, vmax): # normalized params -> original params
    return val * (vmax - vmin) + vmin

# Weak forms
class LinearElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
            
    # Tensor
    def get_tensor_map(self):
        def stress(u_grad): # stress tensor
            nu = 0.33
            mu = E / (2.*(1+nu))
            lmbda = E * nu / ((1+nu)*(1-2*nu))
            # strain-displacement relation
            epsilon = 0.5 * (u_grad + u_grad.T) # u_grad = 3x3
            # stress-strain relation
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

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
problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           location_fns=location_fns)

# AD wrapper : critical step that makes the problem solver differentiable
solver_opts = {'petsc_solver': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'pc_factor_mat_solver_type': 'mumps'}}

fwd_pred = ad_wrapper(problem, solver_options=solver_opts, adjoint_solver_options=solver_opts)

# Extract mesh edges for TV regularization
def get_mesh_edges(cells):
    """
    Extract unique edges(a pair of nodes) from mesh cells.
    """
    cells = onp.array(cells) # Ensure numpy array
    num_nodes_per_cell = cells.shape[1]
    
    # Generate all possible node pairs within a single element
    pairs = list(itertools.combinations(range(num_nodes_per_cell), 2))
    
    edges_list = []
    for i, j in pairs:
        edges_list.append(cells[:, [i, j]])
        
    # Combine all edges into one array
    all_edges = onp.vstack(edges_list)
    
    # 정렬 후 중복 제거 (방향성 무시: 1-2와 2-1은 같음)
    all_edges.sort(axis=1)
    unique_edges = onp.unique(all_edges, axis=0)
    
    return unique_edges

mesh_edges = get_mesh_edges(problem.fes[0].cells)
# JAX 장치로 이동 (최적화 루프 내에서 사용하기 위함)
mesh_edges_jax = jax.device_put(mesh_edges)

# TV Loss
def TV_reg(rho, edges, alpha=1., epsilon = 1e-6):
    """
    Args:
        edges: (num_edges, 2) a pair of node indices representing an edge
        epsilon: small value to avoid division by zero
    """
    # Fetch parameter values at the edge nodes
    rho_i = rho[edges[:, 0]]
    rho_j = rho[edges[:, 1]]

    # Difference between two adjacent nodes
    diff = rho_i - rho_j
    return alpha * np.sum(np.sqrt(diff**2 + epsilon))

# Objective Function
# TODO: try TV regularization
def J_total(params): # J(u(theta), theta)
    # params = unnormalize(params, vmin=0., vmax=1.0)
    # Solve w/ params
    sol_list = fwd_pred(params)
    # Data term
    u_difference = sol_measured[target_idx, 0] - sol_list[0][target_idx, 0]
    # Regularization term
    alpha = 1e-5 #1e-9
    TV_reg_term = TV_reg(params, mesh_edges_jax, alpha=alpha)
    # Objective function
    J = 1e6 * 0.5 * np.linalg.norm(u_difference)**2 + 0.5 * TV_reg_term
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
    
    # Create (n, 3) shape array with zeros in the first and last columns
    is_top = np.isclose(mesh.points[:, 1], np.max(mesh.points[:, 1]), atol=1e-5)
    is_top = is_top[:, None] # Reshape to (N, 1) for broadcasting
    
    zeros = np.zeros_like(intermediate_result.x)
    traction = np.stack([zeros, intermediate_result.x, zeros], axis=1) * is_top
    
    save_sol(problem.fes[0], 
             np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
             os.path.join(file_dir, vtu_name), 
             point_infos=[('traction', traction)])
    print(f"Iteration:{output_sol.counter}")
    print(f"Obj:{intermediate_result.fun}")
    outputs.append(intermediate_result.fun)
    params.append(intermediate_result.x)
    output_sol.counter += 1
    return 
output_sol.counter = 1

# Initial guess
rho_ini = 0.01 * np.ones(problem.fes[0].num_total_nodes) # (num_nodes,)
# rho_ini_norm = normalize(rho_ini, vmin=0., vmax=1.0)
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
optimizationParams = {'maxiter':200, 'disp':True} # 'ftol':1e-4

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