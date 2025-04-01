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

# Customize gradients 
# (Straight-Through Estimator is a custom rule for how gradients flow.)
@jax.custom_vjp # Vector-Jacobian Product
def where_ste(cond):
    """
    Straight-Through step function.
    Forward pass: 0/1 threshold
    Backward pass: d( step_st(x) )/dx = 1 (i.e. pass-through)
    """
    return np.where(cond >= 0.0, 0.0, 1.0)

def where_ste_fwd(cond):
    return np.where(cond >= 0.0, 0.0, 1.0), cond
    # (output, residual saved for backward pass)

def where_ste_bwd(res, g):
    (x,) = res, # 1-element tuple
    return (g,) # gradient

# Register the forward & backward functions to "where_ste"
where_ste.defvjp(where_ste_fwd, where_ste_bwd)

# Create inner domain (differentiable)
class Geometry:
    def __init__(self, cen_x, cen_y, cen_z=None, length=None, height=None, 
                 cells=None, points=None, param_mins=None, param_maxes=None):
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
        # If no min/max provided, default to pass-through (i.e. range=1).
        if param_mins is None:
            param_mins = {'cen_x': 0, 'cen_y': 0, 'cen_z': 0,
                          'length': 0, 'height': 0}
        if param_maxes is None:
            param_maxes = {'cen_x': 1, 'cen_y': 1, 'cen_z': 1,
                           'length': 1, 'height': 1}

        # Map normalized parameters back to original values
        def unnormalize(val, vmin, vmax):
            return val * (vmax - vmin) + vmin

        self.cen_x = unnormalize(cen_x, param_mins['cen_x'], param_maxes['cen_x'])
        self.cen_y = unnormalize(cen_y, param_mins['cen_y'], param_maxes['cen_y'])
       
        if cen_z is None:
            self.cen_z = None
        else:
            self.cen_z = unnormalize(cen_z, param_mins['cen_z'], param_maxes['cen_z'])
        
        if length is None:
            self.length = None
        else:
            self.length = unnormalize(length, param_mins['length'], param_maxes['length'])
        
        if height is None:
            self.height = None
        else:
            self.height = unnormalize(height, param_mins['height'], param_maxes['height'])
        
        self.cells = cells
        self.points = points
        
        # self.cen_x: float = cen_x
        # self.cen_y: float = cen_y
        # self.cen_z: float = cen_z
        # self.length: float = length
        # self.height: float = height
        # self.cells: np.ndarray = cells
        # self.points: np.ndarray = points
        
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
            # point_indicators = np.round(point_indicators)
            
        else: # 3D  
            z_squared = (self.points[:, 2] - self.cen_z)**2
            h_squared = self.height ** 2
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared))
            z_indicators = jax.nn.sigmoid(k * (z_squared - h_squared))
            # np.round() makes indifferentiable
            # point_indicators, z_indicators = np.round(point_indicators), np.round(z_indicators)
            
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
        k = 100.0  # Controls transition sharpness

        # Find the indices of the points in the inner domain
        if is_2d: # 2D
            # When r_squared < r_param: sigmoid ≈ 1 (inside)
            # When r_squared > r_param: sigmoid ≈ 0 (outside)
            point_indicators = jax.nn.relu(domain_squared - r_squared)
            # point_indicators = np.round(point_indicators)
            
        else: # 3D  
            z_squared = (self.points[:, 2] - self.cen_z)**2
            h_squared = self.height ** 2
            point_indicators = jax.nn.relu(domain_squared - r_squared)
            z_indicators = jax.nn.relu(z_squared - h_squared)
            # np.round() makes indifferentiable
            # point_indicators, z_indicators = np.round(point_indicators), np.round(z_indicators)
            
            # point_indicators = 1 only if both are 1
            point_indicators = np.maximum(point_indicators, z_indicators)

        # Find the point_indicators by the indices stored in the cells
        cell_indicators = point_indicators[self.cells] # Assign point indicators by cell indices
        # cell_indicators = 1 only if all points are 1
        cell_indicators = np.prod(cell_indicators, axis=1)
        cell_indicators = jax.nn.sigmoid(cell_indicators)
        return cell_indicators # array of shape (n,)

    # Straight-Through Estimator
    def circle_st(self) -> np.ndarray:
        """
        Get indices of cells inside the circle (2D) or cylinder (3D).
        """
        is_2d = self.cen_z == None
        
        domain_squared = (self.points[:,0] - self.cen_x)**2 + (self.points[:,1] - self.cen_y)**2
        r_squared = self.length ** 2

        if is_2d:
            # For each mesh point, 1 if inside circle, else 0 (straight-through in backward pass)
            point_indicators = where_ste(domain_squared - r_squared)
            print(point_indicators)
        else:
            # 3D logic: must be inside the disk in XY and also within [cen_z +/- height]
            z_squared = (self.points[:,2] - self.cen_z) ** 2
            h_squared = self.height ** 2

            cond1 = domain_squared - r_squared
            cond2 = z_squared - h_squared
            cond_both = cond1 & cond2
            point_indicator = where_ste(cond_both)
            # z_indicators = where_ste(z_squared - h_squared)
            # 1 only if inside the circle in XY *and* within height range in Z            
            # point_indicators = point_indicators + z_indicators
            # point_indicators = np.maximum(point_indicators, 1.0)

        # "Cell" indicator = product of that cell's corner indicators
        cell_indicators = point_indicators[self.cells]
        cell_indicators = np.prod(cell_indicators, axis=1)
        return cell_indicators
    
    def rectangle(self) -> np.ndarray:
        # Set the edges of the square
        left, right = self.center[0], self.center[0] + self.length
        bottom, top = self.center[1], self.center[1] + self.length
        z_bot, z_top = self.center[2] - self.height, self.center[2] + self.height
        # Find the indices of the points in the inner domain
        domain_points = np.where((self.points[:,0] >= left) & (self.points[:,0] <= right) &
                                 (self.points[:,1] >= bottom) & (self.points[:,1] <= top) &
                                 (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Find the indices of the cells in the inner domain
        domain_cells = np.all(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]

        return flex_inds # array of shape (n,)


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
            # E = Emax + (Emin - Emax)*theta[0]
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
        bound_min, bound_max = np.min(self.fes[0].points, axis=0), np.max(self.fes[0].points, axis=0)
        bound_diff = bound_max - bound_min
        bound_sum = bound_max + bound_min
        inner_domain = Geometry(params[0], params[1], params[2], params[3], params[4], 
                                self.fes[0].cells, self.fes[0].points,
                                param_mins={'cen_x': bound_min[0], 'cen_y': bound_min[1], 'cen_z': bound_min[2],
                                            'length': 0., 'height': 0.},
                                param_maxes={'cen_x': bound_max[0], 'cen_y': bound_max[1], 'cen_z': bound_max[2],
                                             'length': min(bound_diff[0], bound_diff[1]) / 2, 'height': bound_diff[2] / 2})

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
# This means on the 'top' side, we will perform the surface integral to get 
# the tractions with the function 'get_surface_maps' defined in the class 'LinearElasticity'.
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
             vtu_path, cell_infos=[('theta', problem.full_params[:, 0])],
             ) # point_infos=[('center', intermediate_result.x[])]
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


# Set the max/min for the design variables
bound_min, bound_max = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
bound_diff = bound_max - bound_min
bound_sum = bound_max + bound_min
r_bound = (0, (min(bound_diff[0], bound_diff[1]) / 2) * 0.99) # 1% less than the boundary
h_bound = (0, (bound_diff[2] / 2) * 0.99)
x_bound = (bound_min[0], bound_max[0])
y_bound = (bound_min[1], bound_max[1])
z_bound = (bound_min[2], bound_max[2])
rho_ini = np.array([bound_sum[0] / 2, bound_sum[1] / 2, bound_sum[2] / 2, 0., 0.])
rho_ini_normalized = np.array([(rho_ini[0] - bound_min[0]) / bound_diff[0],
                               (rho_ini[1] - bound_min[1]) / bound_diff[1],
                               (rho_ini[2] - bound_min[2]) / bound_diff[2],
                               rho_ini[3] / r_bound[1],
                               rho_ini[4] / h_bound[1]])

# Optimization problem setting
numConstraints = 1
optimizationParams = {'maxiter':100, 'disp':True}
start_time = time.time() # Start timing

# Minimize the objective function
params = params + [rho_ini]
save_sol(problem.fes[0], np.hstack(np.zeros((len(sol_measured), 4))), 
         f'data/inverse/{file_name}/sol_000.vtu', 
         cell_infos=[('theta', np.zeros(problem.fes[0].num_cells))])
minimize(J_total, jac=J_grad, x0=rho_ini_normalized, 
         bounds=[(0, 1)]*5,
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
plt.title(rf'L2 Regularization with $\lambda = 1e-9$', fontsize=20)
# plt.title(rf'Initial Guess with $\theta = {vf}$', fontsize=20)

save_dir = 'data/inverse/figures'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
plt.savefig('data/inverse/figures/%s_obj.png' %file_name, dpi=300, bbox_inches='tight')


# # Plot the parameter changes
# fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True) # share the same x-axis

# # Plot parameter x (index 0)
# axs[0].plot(onp.arange(len(params)), params[:, 0], label='x')
# axs[0].axhline(y=bound_sum[0] / 2, color='r', linestyle='--', label='Center x')
# axs[0].set_ylabel('Parameter Value')
# axs[0].legend()

# # Plot parameter y (index 1)
# axs[1].plot(onp.arange(len(params)), params[:, 1], label='y')
# axs[1].axhline(y=bound_sum[1] / 2, color='r', linestyle='--', label='Center y')
# axs[1].set_ylabel('Parameter Value')
# axs[1].legend()

# # Plot parameter z (index 2)
# axs[2].plot(onp.arange(len(params)), params[:, 2], label='z')
# axs[2].axhline(y=bound_sum[2] / 2, color='r', linestyle='--', label='Center z')
# axs[2].set_ylabel('Parameter Value')
# axs[2].legend()

# # Plot parameter r (index 3)
# axs[3].plot(onp.arange(len(params)), params[:, 3], label='r')
# axs[3].axhline(y=4.0, color='r', linestyle='--', label='Radius r')
# axs[3].set_ylabel('Parameter Value')
# axs[3].legend()

# # Plot parameter h (index 4)
# axs[4].plot(onp.arange(len(params)), params[:, 4], label='h')
# axs[4].axhline(y=0.6, color='r', linestyle='--', label='Height h')
# axs[4].set_xlabel('Iteration')
# axs[4].set_ylabel('Parameter Value')
# axs[4].legend()

# # Main title for entire figure
# fig.suptitle('Parameter Changes', fontsize=20)

# os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
# plt.savefig('data/inverse/figures/%s_params.png' %file_name, dpi=300, bbox_inches='tight')
plt.show()