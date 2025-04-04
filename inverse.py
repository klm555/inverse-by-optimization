import os
import time
from typing import List

import jax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
from jax_fem.mma import optimize
from jax_fem import logger
from scipy.optimize import minimize, Bounds

import numpy as onp
import matplotlib.pyplot as plt
import meshio

import logging
logger.setLevel(logging.DEBUG)

file_name = 'u_dogbone_0.01_any'

# Load the measured displacement data
sol_measured = onp.loadtxt('u_dogbone_0.01_any.txt') # (number of nodes, 3) for 3D case

# Create inner domain (differentiable)
class Geometry:
    def __init__(self, cen_x, cen_y, cen_z=None, length=None, height=None, 
                 cells=None, points=None):
        """
        Create a geometry instance that handles both 2D and 3D cases.
        
        Args:
            cen_x: Center x coordinate
            cen_y: Center y coordinate
            cen_z: Center z coordinate
            length: Radius of the circle or side length of the square
            height: Height of the cylinder or the cube (3D only)
            cells: Mesh cells
            points: Mesh points
            param_mins: Dict of minimum values for min-max scaling
            param_maxes: Dict of maximum values for min-max scaling
        """
        # Map normalized parameters back to original values
        def unnormalize(val, vmin, vmax):
            return val * (vmax - vmin) + vmin
        
        bound_min, bound_max = np.min(points, axis=0), np.max(points, axis=0)
        bound_diff = bound_max - bound_min
        r_max = (min(bound_diff[0], bound_diff[1]) / 2) * 0.99 # 1% less than the boundary
        h_max = (bound_diff[2] / 2) * 0.99
        
        # Unnormalize the parameters
        self.cen_x = unnormalize(cen_x, bound_min[0], bound_max[0])
        self.cen_y = unnormalize(cen_y, bound_min[1], bound_max[1])
        if cen_z is None:
            self.cen_z = None
        else:
            self.cen_z = unnormalize(cen_z, bound_min[2], bound_max[2])
        if length is None:
            self.length = None
        else:
            self.length = unnormalize(length, 0, r_max)
        if height is None:
            self.height = None
        else:
            self.height = unnormalize(height, 0, h_max)

        # Store the cells and points
        self.cells = cells
        self.points = points

    def circle(self) -> np.ndarray:
        """
        Get indices of cells inside the circle (2D) or cylinder (3D).
        """
        is_2d = self.cen_z == None
        
        domain_squared = (self.points[:,0] - self.cen_x)**2 + (self.points[:,1] - self.cen_y)**2
        r_squared = self.length ** 2
        k = 1.0  # Controls transition sharpness

        # Find the indices of the points in the inner domain
        if is_2d: # 2D
            # When r_squared < r_param: sigmoid ≈ 1 (inside)
            # When r_squared > r_param: sigmoid ≈ 0 (outside)
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared))
            
        else: # 3D  
            z_squared = (self.points[:, 2] - self.cen_z)**2
            h_squared = self.height ** 2
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared))
            z_indicators = jax.nn.sigmoid(k * (z_squared - h_squared))
            # np.round() makes indifferentiable
            
            # point_indicators = 1 only if both are 1
            point_indicators = np.maximum(point_indicators, z_indicators)

        # Find the point_indicators by the indices stored in the cells
        cell_indicators = point_indicators[self.cells] # Assign point indicators by cell indices
        # cell_indicators = 1 only if all points are 1
        cell_indicators = np.prod(cell_indicators, axis=1)
        return cell_indicators # array of shape (n,)
    
    def circle_relu(self) -> np.ndarray:
        """
        Get indices of cells inside the circle (2D) or cylinder (3D).
        """
        is_2d = self.cen_z == None
        
        domain_squared = (self.points[:,0] - self.cen_x)**2 + (self.points[:,1] - self.cen_y)**2
        r_squared = self.length ** 2
        k = 50.0  # Controls transition sharpness

        # Find the indices of the points in the inner domain
        if is_2d: # 2D
            # When r_squared < r_param: sigmoid ≈ 1 (inside)
            # When r_squared > r_param: sigmoid ≈ 0 (outside)
            point_indicators = jax.nn.relu(domain_squared - r_squared)
            
        else: # 3D  
            z_squared = (self.points[:, 2] - self.cen_z)**2
            h_squared = self.height ** 2
            point_indicators = jax.nn.relu(domain_squared - r_squared)
            z_indicators = jax.nn.relu(z_squared - h_squared)
            # np.round() makes indifferentiable
            
            # point_indicators = 1 only if both are 1
            point_indicators = np.maximum(point_indicators, z_indicators)

        # Find the point_indicators by the indices stored in the cells
        cell_indicators = point_indicators[self.cells] # Assign point indicators by cell indices
        # cell_indicators = 1 only if all points are 1
        cell_indicators = np.prod(cell_indicators, axis=1)
        cell_indicators = jax.nn.sigmoid(cell_indicators)
        return cell_indicators # array of shape (n,)

# Weak forms
class LinearElasticity(Problem):
    def custom_init(self): # TODO: integrate geometry with this?
            # self.fes[0].flex_inds : indices of the inner domain cells
            # Set up 'self.fes[0].flex_inds' to create the inner domain
            self.fes[0].flex_inds = np.arange(len(self.fes[0].cells))
            
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

    def set_params(self, params): # params = [x, y, z, r, h]
        # Geometry class doesn't use 'flex_inds', but directly assigns 'theta' values to the cells
        
        # Normalize r & h
        def normalize(val, vmin, vmax):
            return (val - vmin) / (vmax - vmin)
        
        bound_min, bound_max = np.min(self.fes[0].points, axis=0), np.max(self.fes[0].points, axis=0)
        bound_diff = bound_max - bound_min
        r_max = (min(bound_diff[0], bound_diff[1]) / 2) * 0.99
        h_max = (bound_diff[2] / 2) * 0.99
        length = normalize(4.0, 0, r_max)
        height = normalize(0.6, 0, h_max)
        
        inner_domain = Geometry(params[0], params[1], params[2], length, height, 
                                self.fes[0].cells, self.fes[0].points)
        full_params = inner_domain.circle()
        full_params = np.expand_dims(full_params, axis=1)
        thetas = np.repeat(full_params[:, None, :], self.fes[0].num_quads, axis=1)
        self.params = params
        self.full_params = full_params
        self.internal_vars = [thetas]
        
    # Traction
    def get_surface_maps(self):
        def surface_map(u, x):
            # Traction components in each direction
            return np.array([-15., 0., 0.])
        return [surface_map]
    
# Mesh info
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
dim = 3

# Create meshes using meshio : 3rd party library
msh_file = 'Dogbone_0.01.msh'
meshio_mesh = meshio.read(msh_file)
# Input mesh info into Mesh class
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Boundary Locations
def left(point):
    return np.isclose(point[0], np.min(mesh.points, axis=0)[0], atol=1e-5) # True for x = 0

def right(point):
    return np.isclose(point[0], np.max(mesh.points, axis=0)[0], atol=1e-5) # True for x = Lx

# Dirichlet boundary values
def zero_dirichlet_val(point):
    return 0.

# Dirichlet boundary info
# [plane, direction, displacement]
# number of elements in plane, direction, displacement should match
dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]

# Neumann boundary locations
location_fns = [right]

# Create an instance of the problem.
problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           location_fns=location_fns)

##################################################################
# Apply the automatic differentiation wrapper.
# This is a critical step that makes the problem solver differentiable.
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

# Objective Function
def J_total(params):
    # J(u(theta), theta)   
    # Set the parameters into problem
    sol_list = fwd_pred(params)
    u_difference = sol_measured - sol_list[0]
    lambda_reg = 0 #1e-9
    u_grad = problem.fes[0].sol_to_grad(sol_list[0])
    # l2_reg_term = lambda_reg * np.linalg.norm(u_grad)**2 # l2 regularization
    TV_reg_term = lambda_reg * np.linalg.norm(u_grad)**2 # TV regularization
    J: float = 0.5 * np.linalg.norm(u_difference)**2 + 0.5 * TV_reg_term
    return J

def J_grad(rho):
    J: float # ()
    dJ: np.ndarray # (...)
    dJ = jax.grad(J_total)(rho)
    # output_sol(rho, J)
    dJ = dJ.reshape(-1)
    return dJ

# Output solution(displacement u) from solving the forward problem
outputs = []
params = []
def output_sol(intermediate_result):
    # Store the solution to local file (vtu).
    sol_list = fwd_pred(intermediate_result.x)
    vtu_path = f'data/inverse/{file_name}/sol_{output_sol.counter:03d}.vtu'
    save_sol(problem.fes[0], np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
             vtu_path, cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"Iteration:{output_sol.counter}")
    print(f"Obj:{intermediate_result.fun}")
    outputs.append(intermediate_result.fun)
    params.append(intermediate_result.x)
    output_sol.counter += 1
    return 
output_sol.counter = 1

# constraint function = 0 (will be not considered in this problem.)
def consHandle(rho, epoch):
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    def constraint_fn(rho):
        # g = np.mean(rho)/vf - 1.
        g = np.array(0.)
        return g
    c, gradc = jax.value_and_grad(constraint_fn)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc

# Normalize r & h
def normalize(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin)

# Set the max/min for the design variables
bound_min, bound_max = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
bound_diff = bound_max - bound_min
bound_sum = bound_max + bound_min
x_bound = (bound_min[0], bound_max[0])
y_bound = (bound_min[1], bound_max[1])
z_bound = (bound_min[2], bound_max[2])
r_bound = (0, (min(bound_diff[0], bound_diff[1]) / 2) * 0.99) # 1% less than the boundary
h_bound = (0, (bound_diff[2] / 2) * 0.99)
rho_ini = np.array([bound_sum[0] / 4, bound_sum[1] / 2, bound_sum[2] / 2])
rho_ini_normalized = np.array([normalize(rho_ini[0], bound_min[0], bound_max[0]),
                               normalize(rho_ini[1], bound_min[1], bound_max[1]),
                               normalize(rho_ini[2], bound_min[2], bound_max[2])])

# Optimization problem setting
numConstraints = 1
optimizationParams = {'maxiter':100, 'disp':True} # 'ftol':1e-4
start_time = time.time() # Start timing

# Minimize the objective function
params = params + [rho_ini]
save_sol(problem.fes[0], np.hstack(np.ones((len(sol_measured), 4))), 
         f'data/inverse/{file_name}/sol_000.vtu', 
         cell_infos=[('theta', np.ones(problem.fes[0].num_cells))])
minimize(J_total, jac=J_grad, x0=rho_ini_normalized, 
        #  bounds=[(0, 1)] * len(rho_ini),
         bounds=[(0.2, 0.6), (0, 1), (0, 1)],
         options=optimizationParams, method='L-BFGS-B', callback=output_sol)
end_time = time.time() # End timing
elapsed_time = end_time - start_time

# Convert to hours, minutes, seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Total optimization runtime: {hours}h {minutes}m {seconds}s")
print("Total Iteration:", output_sol.counter)
print(f"As a reminder, L2 norm of the displacement differences = {J_total(rho_ini)} for full material")

# Plot the optimization results.
obj = onp.array(outputs)
plt.figure(1, figsize=(10, 8))
plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.axhline(y=obj[-1], color='r', linestyle='--', label='J = %f' % obj[-1])
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
# plt.title(rf'L2 Regularization with $\lambda = 1e-9$', fontsize=20)
# plt.title(rf'Initial Guess with $\theta = {vf}$', fontsize=20)

save_dir = 'data/inverse/figures'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
plt.savefig('data/inverse/figures/%s_obj.png' %file_name, dpi=300, bbox_inches='tight')
# plt.show()