# TODO
# check how it stops : f-based, g-based, x-based 
# check whether the gradient sign changes

# do many cases() -> average

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
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
from jax_fem.mma import optimize
from jax_fem import logger
from scipy.optimize import minimize, Bounds

import numpy as onp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio

import logging
logger.setLevel(logging.DEBUG)

# Check JAX backend and devices
print('JAX backend:', jax.default_backend())
print('Devices:', jax.devices())

# Save setup
file_dir = '../data/local_min_test/3var-sigmoid1-dogbone_3'
os.makedirs(file_dir, exist_ok=True)
file_name = '3var-sigmoid0.005_-dogbone_3'

# Load data (measured displacement)
sol_measured = onp.loadtxt('../dogbone_3.txt') # (number of nodes, 3) in 3D

# Mesh info
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
dim = 3
# Meshes
msh_file = '../Dogbone_0.05.msh'
meshio_mesh = meshio.read(msh_file) # meshio : 3rd party library
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Helper function for normalizing parameters
def normalize(val, vmin, vmax): # original params -> normalized params
    return (val - vmin) / (vmax - vmin)

def unnormalize(val, vmin, vmax): # normalized params -> original params
    return val * (vmax - vmin) + vmin

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
        """
        bound_min, bound_max = np.min(points, axis=0), np.max(points, axis=0)
        bound_min = bound_min.at[1].set(785.321) # since y coords of dogbone shape is varying,
        bound_max = bound_max.at[1].set(798.321) # but l-bfgs-b cannot set varying bounds, set them to be always inside the domain
        bound_diff = bound_max - bound_min
        r_max = min(bound_diff[0], bound_diff[1]) * 0.5 * 0.99 # 1% less than the boundary
        h_max = bound_diff[2] * 0.5 * 0.99
        
        # Unnormalize the parameters
        self.cen_x = unnormalize(cen_x, bound_min[0], bound_max[0])
        self.cen_y = unnormalize(cen_y, bound_min[1], bound_max[1])
        self.cen_z = None if cen_z is None else unnormalize(cen_z, bound_min[2], bound_max[2])
        self.length = None if length is None else unnormalize(length, 0, r_max)
        self.height = None if height is None else unnormalize(height, 0, h_max)

        # Store data
        self.cells = cells
        self.points = points

    def circle(self) -> np.ndarray:
        # Height
        z_bot, z_top = self.cen_z - self.height, self.cen_z + self.height
        # Indices of the points in the inner domain (x**2 + y**2 <= r**2) & (abs(z) <= h)            
        domain_points = np.where(((self.points[:,0] - self.cen_x)**2 +
                                  (self.points[:,1] - self.cen_y)**2 <= self.length ** 2) &
                                 (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Indices of the cells
        domain_cells = np.any(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]    
        return flex_inds # array of shape (n,)

    def circle_sigmoid(self) -> np.ndarray:  
        # Point indices
        is_2d = self.cen_z == None # True if "self.cen_z" is None     
        # Sharpness for sigmoid
        k = 1.0
        # Squared distances
        domain_squared = (self.points[:,0] - self.cen_x)**2 + (self.points[:,1] - self.cen_y)**2
        r_squared = self.length ** 2
        z_squared = None if is_2d is None else (self.points[:,2] - self.cen_z)**2
        h_squared = None if is_2d is None else self.height ** 2

        if is_2d: # 2D
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared)) # (+): outside, (-): inside  
        else: # 3D  
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared))
            z_indicators = jax.nn.sigmoid(k * (z_squared - h_squared))            
            # Choose maximum sigmoid value
            point_indicators = np.maximum(point_indicators, z_indicators)

        # Cell indices cell (cell_indicators = [point1_indicators, point2_indicators, point3_indicators, point4_indicators])
        cell_indicators = point_indicators[self.cells] # Assign point indicators to cell indicators
        cell_indicators = np.prod(cell_indicators, axis=1) # "1" if all points are 1
        return cell_indicators # array (n,)

# Weak forms
class LinearElasticity(Problem):
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
        # Geometry class doesn't use 'flex_inds', and directly assigns 'theta' values to the cells
        bound_min, bound_max = np.min(self.fes[0].points, axis=0), np.max(self.fes[0].points, axis=0)
        bound_min = bound_min.at[1].set(785.321) # since y coords of dogbone shape is varying,
        bound_max = bound_max.at[1].set(798.321) # but l-bfgs-b cannot set varying bounds, set them to be always inside the domain
        bound_diff = bound_max - bound_min
        r_max = (min(bound_diff[0], bound_diff[1]) / 2) # 1% less than the boundary
        h_max = (bound_diff[2] / 2)
        # y = normalize((bound_min[1] + bound_max[1]) / 2, bound_min[1], bound_max[1])
        # z = normalize((bound_min[2] + bound_max[2]) / 2, bound_min[2], bound_max[2])
        length = normalize(4.0, 0, r_max)
        height = normalize(0.6, 0, h_max)
        
        # Inner domain indices
        inner_domain = Geometry(params[0], params[1], params[2], 
                                length, height, 
                                self.fes[0].cells, 
                                self.fes[0].points)
        full_params = inner_domain.circle_sigmoid()
        full_params = np.expand_dims(full_params, axis=1)

        # Match "full_params" to the number of quadrature points
        thetas = np.repeat(full_params[:, None, :], self.fes[0].num_quads, axis=1)

        self.params = params
        self.full_params = full_params
        self.internal_vars = [thetas]

# Boundary Locations
def left(point):
    return np.isclose(point[0], np.min(mesh.points, axis=0)[0], atol=1e-5) # True for x = 0

def right(point):
    return np.isclose(point[0], np.max(mesh.points, axis=0)[0], atol=1e-5) # True for x = Lx

# Dirichlet boundary values
# in the direction normal to the boundary("1": tension, "-1": compression)
def zero_dirichlet_val(point):
    return 0.

def one_dirichlet_val(point):
    return 1.

def minus_one_dirichlet_val(point):
    return -1.

# Dirichlet boundary info
# [plane, direction, displacement]
# number of elements in plane, direction, displacement should match
dirichlet_bc_info = [[left]*3 + [right]*3, 
                     [0, 1, 2]*2, 
                     [minus_one_dirichlet_val] + [zero_dirichlet_val]*2 + [one_dirichlet_val] + [zero_dirichlet_val]*2]

# Neumann boundary locations
location_fns = [right]

# Instance of the problem
problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info)
                        #    location_fns=location_fns)

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
    # Save vtu
    vtu_name = "sol_%03d.vtu" % count
    save_sol(problem.fes[0], sol_list[0], os.path.join(file_dir, vtu_name), 
             cell_infos=[('theta', problem.full_params[:, 0].reshape(-1))])
    return J

# Gradient of J
def J_grad(rho):
    dJ = jax.grad(J_total)(rho) # (...)
    dJ = dJ.reshape(-1)
    return dJ

# Set the max/min for the design variables
bound_min, bound_max = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
bound_min = bound_min.at[1].set(785.321) # since y coords of dogbone shape is varying,
bound_max = bound_max.at[1].set(798.321) # but l-bfgs-b cannot set varying bounds, set them to be always inside the domain
x_points = np.linspace(bound_min[0], bound_max[0], 12)
y_points = np.linspace(bound_min[1], bound_max[1], 6)
z_points = np.linspace(bound_min[2], bound_max[2], 3)

outputs = []
count = 1
start_time = time.time()
for x in x_points:
    for y in y_points:
        for z in z_points:
            # Initial guess
            x_normalized = normalize(x, bound_min[0], bound_max[0])
            y_normalized = normalize(y, bound_min[1], bound_max[1])
            z_normalized = normalize(z, bound_min[2], bound_max[2])
            rho_normalized = np.array([x_normalized, y_normalized, z_normalized])
            # Objective function
            output = J_total(rho_normalized)
            outputs.append(output)
            print("Iteration : %d" % count)
            count += 1
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Total running time: {hours}h {minutes}m {seconds}s")

# Plot the optimization results
x_center = (bound_min[0] + bound_max[0]) / 2
y_center = (bound_min[1] + bound_max[1]) / 2
z_center = (bound_min[2] + bound_max[2]) / 2
obj_0 = J_total(np.array([normalize(x_center, bound_min[0], bound_max[0]),
                normalize(y_center, bound_min[1], bound_max[1]),
                normalize(z_center, bound_min[2], bound_max[2])]))
obj = onp.array(outputs)
# Reshape 'obj' into a grid nx x ny
obj_grid = obj.reshape(len(x_points), len(y_points), len(z_points))

# Mesh for plotting
X, Y, Z = onp.meshgrid(x_points, y_points, z_points)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((16.5, 1.9, 0.32))
scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=obj_grid, cmap='viridis', s=20, alpha=0.5)
ax.set_xlabel(r"x-coordinate")
ax.set_ylabel(r"y-coordinate")
ax.set_zlabel(r"z-coordinate")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title('Objective function over x, y', fontsize=16)
# ax.tick_params(labelsize=20)
fig.colorbar(scatter, label='Objective value', shrink=0.5)

# Save
plt.savefig(os.path.join(file_dir, '%s.png' %file_name), dpi=300, bbox_inches='tight')

end_time = time.time()
print('Total running time : %f secs' % (end_time - start_time))