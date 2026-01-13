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
file_name = 'ellipse_hole-extended_domain'

# Load data (measured displacement)
sol_measured = onp.loadtxt('ellipse_hole-extended_domain.txt') # (number of nodes, 3) in 3D

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

# Helper function for normalizing parameters
def normalize(val, vmin, vmax): # original params -> normalized params
    return (val - vmin) / (vmax - vmin)

def unnormalize(val, vmin, vmax): # normalized params -> original params
    return val * (vmax - vmin) + vmin

# Weak forms
class HyperElasticity(Problem):
    def custom_init(self): # TODO: integrate geometry with this?
            self.fes[0].flex_inds = np.arange(len(self.fes[0].cells)) # "flex_inds" : idx of the inner domain cells
            
    # Tensor
    def get_tensor_map(self):
        def stress(u_grad, theta): # stress tensor
            Emax = 2.35e3
            Emin = 1.0e-3
            nu = 0.33
            # penal = 3.
            # E = Emin + (Emax - Emin)*theta[0]**penal
            E = Emin + (Emax - Emin)*theta[0]
            mu = E / (2.*(1+nu))
            lmbda = E * nu / ((1+nu)*(1-2*nu))
            # strain-displacement relation
            epsilon = 0.5 * (u_grad + u_grad.T) # u_grad = 3x3
            # stress-strain relation
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    # def get_surface_maps(self):
    #     def surface_map(u, x): # traction
    #         return np.array([-15., 0., 0.]) # tr_x = -15 ((-): tension, (+): compression)
    #     return [surface_map]

    def set_params(self, params): # params = [x, y, z, r, h]
        # x = normalize(45., x_bound[0], x_bound[1])
        # y = normalize(50., y_bound[0], y_bound[1])    
        
        # Inner domain indices
        inner_domain = Geometry(params[0], params[1], length=params[2], 
                                length2=params[3], angle=params[4], 
                                cells=self.fes[0].cells, 
                                points=self.fes[0].points)
        full_params = inner_domain.ellipse()
        # The line below is only needed in case of 3D problems!
        full_params = np.expand_dims(full_params, axis=1) # (n, 1)
        # Match "full_params" to the number of quadrature points
        thetas = np.repeat(full_params[:, None, :], self.fes[0].num_quads, axis=1) # (n, num_quads, 1)
        # Geometry class doesn't use 'flex_inds', and directly assigns 'theta' values to the cells
        self.params = params
        self.full_params = full_params
        self.internal_vars = [thetas]

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

# Objective Function
# TODO: try TV regularization
def J_total(params): # J(u(theta), theta)
    # Solve w/ params
    sol_list = fwd_pred(params)
    # Data term
    u_difference = sol_measured - sol_list[0]
    # Regularization term
    lambda_reg = 0 #1e-9
    u_grad = problem.fes[0].sol_to_grad(sol_list[0])
    l2_reg_term = lambda_reg * np.linalg.norm(u_grad)**2 # l2 regularization
    # Objective function
    J = 0.5 * np.linalg.norm(u_difference)**2 + 0.5 * l2_reg_term
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
             cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"Iteration:{output_sol.counter}")
    print(f"Obj:{intermediate_result.fun}")
    outputs.append(intermediate_result.fun)
    params.append(intermediate_result.x)
    output_sol.counter += 1
    return 
output_sol.counter = 1

# Set the max/min for the design variables
# TODO: make bound function to refactor the lines below  
bound_min, bound_max = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
bound_diff = bound_max - bound_min
bound_sum = bound_max + bound_min
x_bound = (bound_min[0], bound_max[0])
y_bound = (bound_min[1], bound_max[1])
l_bound = (min(Lx/Nx, Ly/Ny), (min(bound_diff[0], bound_diff[1]) / 2) * 0.99) # 1% less than the boundary
l2_bound = (min(Lx/Nx, Ly/Ny), (min(bound_diff[0], bound_diff[1]) / 2) * 0.99)
angle_bound = (0, np.pi/2 * 0.99)

# Initial guess
rho_ini = np.array([bound_sum[0]/2, bound_sum[1]/2, l_bound[1]/2, l2_bound[1]/2, angle_bound[1]/2]) # it doesn't work with 0.0
# rho_ini = np.array([bound_sum[0]/2, bound_sum[1]/2, 0.1, 0.1, 1]) # it doesn't work with 0.0
x_ini_normalized = normalize(rho_ini[0], x_bound[0], x_bound[1])
y_ini_normalized = normalize(rho_ini[1], y_bound[0], y_bound[1])
l_ini_normalized = normalize(rho_ini[2], l_bound[0], l_bound[1])
l2_ini_normalized = normalize(rho_ini[3], l2_bound[0], l2_bound[1])
angle_ini_normalized = normalize(rho_ini[4], angle_bound[0], angle_bound[1])
rho_ini_normalized = np.array([x_ini_normalized, y_ini_normalized, 
                               l_ini_normalized, l2_ini_normalized, 
                               angle_ini_normalized])

# Initial solution
start_time = time.time() # Start timing
params = params + [rho_ini]
save_sol(problem.fes[0], 
         np.hstack(np.ones((len(sol_measured), 4))),
         os.path.join(file_dir, f'{file_name}/sol_000.vtu'),
         cell_infos=[('theta', np.ones(problem.fes[0].num_cells))])

# Sharpness for sigmoid
k1 = 500.

# Exact solution
mid_point = (np.max(mesh.points, axis=0) + np.min(mesh.points, axis=0)) / 2 # mid_point = (40, 40)
cen_exact = mid_point + np.array([5, 10])
rho_exact = np.array([cen_exact[0], cen_exact[1], 5.0, 2.0, np.pi/3]) # [x, y, l, l2, angle]
x_exact_normalized = normalize(rho_exact[0], x_bound[0], x_bound[1])
y_exact_normalized = normalize(rho_exact[1], y_bound[0], y_bound[1])
l_exact_normalized = normalize(rho_exact[2], l_bound[0], l_bound[1])
l2_exact_normalized = normalize(rho_exact[3], l2_bound[0], l2_bound[1])
angle_exact_normalized = normalize(rho_exact[4], angle_bound[0], angle_bound[1])
rho_exact_normalized = np.array([x_exact_normalized, y_exact_normalized, l_exact_normalized, l2_exact_normalized, angle_exact_normalized])
exact_obj = J_total(rho_exact_normalized)

# Optimization setup
numConstraints = 1
optimizationParams = {'maxiter':100, 'disp':True} # 'ftol':1e-4

# Optimize
results = minimize(J_total, jac=J_grad, 
                   x0=rho_ini_normalized, 
                   bounds=[(1e-5, 1)] * len(rho_ini), # normalized
                   options=optimizationParams, 
                   method='L-BFGS-B', callback=output_sol)
end_time = time.time() # End timing
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Unnormalize the final parameters
x_unnormalized = unnormalize(results.x[0], x_bound[0], x_bound[1])
y_unnormalized = unnormalize(results.x[1], y_bound[0], y_bound[1])
l_unnormalized = unnormalize(results.x[2], l_bound[0], l_bound[1])
l2_unnormalized = unnormalize(results.x[3], l2_bound[0], l2_bound[1])
angle_unnormalized = unnormalize(results.x[4], angle_bound[0], angle_bound[1])
final_param_unnormalized = np.array([x_unnormalized, y_unnormalized, l_unnormalized, l2_unnormalized, angle_unnormalized])

# Print log and save to text file
log_info = f"""Total optimization runtime: {hours}h {minutes}m {seconds}s
Total Iteration: {output_sol.counter}
Sharpness: {k1}
Final Parameters: {final_param_unnormalized}
Exact Parameters: {rho_exact}
Final Objective Value: {results.fun}
Exact Objective Value: {exact_obj}
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
plt.axhline(y=exact_obj, color='r', linestyle='--', label='J = %f' % exact_obj)
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
# plt.title(rf'L2 Regularization with $\lambda = 1e-9$', fontsize=20)

# Save
plt.savefig(os.path.join(file_dir, file_name, 'optimization_graph.png'), dpi=300, bbox_inches='tight')