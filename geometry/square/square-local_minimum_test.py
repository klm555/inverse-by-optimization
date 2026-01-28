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
file_dir = 'data/local_min_test/ellipse_hole-extended_domain-shape-TV'
os.makedirs(file_dir, exist_ok=True)
file_name = 'ellipse_hole-extended_domain-shape-TV'

# Load data (measured displacement)
sol_measured = onp.loadtxt('ellipse_hole-extended_domain.txt') # (number of nodes, 3) in 3D

# Mesh info
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
Lx, Ly = 80., 80. # domain
Nx, Ny = 160, 160 # number of elements in x-dir, y-dir
dim = 2
# Meshes
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
# Input mesh info into Mesh class
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Helper function for normalizing parameters
def normalize(val, vmin, vmax): # original params -> normalized params
    return (val - vmin) / (vmax - vmin)

def unnormalize(val, vmin, vmax): # normalized params -> original params
    return val * (vmax - vmin) + vmin

class Geometry:
    def __init__(self, cen_x, cen_y, cen_z=None, length=None, height=None, 
                 length2=None, angle=None, cells=None, points=None):
        """
        Create a geometry instance that handles both 2D and 3D cases.
        
        Args:
            cen_x: Center x coordinate
            cen_y: Center y coordinate
            cen_z: Center z coordinate
            length: Radius of the circle or side length of the square
            height: Height of the cylinder or the cube (3D only)
            length2: Length in the minor axis direction (for ellipses)
            angle: Rotation angle (for ellipses)
            cells: Mesh cells
            points: Mesh points
        """       
        # Unnormalize the parameters
        self.cen_x = unnormalize(cen_x, x_bound[0], x_bound[1])
        self.cen_y = unnormalize(cen_y, y_bound[0], y_bound[1])
        # self.cen_z = None if cen_z is None else unnormalize(cen_z, bound_min[2], bound_max[2])
        self.length = None if length is None else unnormalize(length, l_bound[0], l_bound[1])
        # self.height = None if height is None else unnormalize(height, 0, h_max)
        self.length2 = None if length2 is None else unnormalize(length2, l2_bound[0], l2_bound[1])
        self.angle = None if angle is None else unnormalize(angle, angle_bound[0], angle_bound[1])

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
        """
        Get indices of cells inside the circle (2D) or cylinder (3D).
        """    
        # Point indices
        is_2d = self.cen_z == None # True if "self.cen_z" is None
        
        # Squared distances
        domain_squared = (self.points[:,0] - self.cen_x)**2 + (self.points[:,1] - self.cen_y)**2
        r_squared = self.length ** 2
        z_squared = None if is_2d is None else (self.points[:,2] - self.cen_z)**2
        h_squared = None if is_2d is None else self.height ** 2

        if is_2d: # 2D
            point_indicators = jax.nn.sigmoid(k1 * (domain_squared - r_squared)) # (+): outside, (-): inside  
        else: # 3D  
            point_indicators = jax.nn.sigmoid(k1 * (domain_squared - r_squared))
            z_indicators = jax.nn.sigmoid(k2 * (z_squared - h_squared))            
            # Choose maximum sigmoid value
            point_indicators = np.maximum(point_indicators, z_indicators)

        # Cell indices
        # cell_indicators = [point1_indicators, point2_indicators, point3_indicators, point4_indicators]
        cell_indicators = point_indicators[self.cells] # Assign point indicators to cell indicators
        cell_indicators = np.prod(cell_indicators, axis=1) # "1" if all points are 1
        return cell_indicators # array (n,)
    
    def ellipse(self) -> np.ndarray:   
        """
        Get indices of cells inside the ellipse (only 2D available).
        """   
        point_indicators = jax.nn.sigmoid(k1 * # (+): outside, (-): inside  
                                          (((self.points[:,0] - self.cen_x) * np.cos(self.angle) + 
                                            (self.points[:,1] - self.cen_y) * np.sin(self.angle))**2 
                                            / self.length**2 +
                                           ((self.points[:,0] - self.cen_x) * np.sin(self.angle) - 
                                            (self.points[:,1] - self.cen_y) * np.cos(self.angle))**2 
                                            / self.length2**2 - 1))        

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
        x = normalize(45., x_bound[0], x_bound[1])
        y = normalize(50., y_bound[0], y_bound[1])
        angle = normalize(np.pi/3, angle_bound[0], angle_bound[1])

        # Inner domain indices
        inner_domain = Geometry(x, y, length=params[0], 
                                length2=params[1], angle=angle, 
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
dirichlet_bc_info = [[left]*2 + [right]*2, 
                     [0, 1]*2, 
                     [minus_one_dirichlet_val] + [zero_dirichlet_val] + [one_dirichlet_val] + [zero_dirichlet_val]]

# Neumann boundary locations
location_fns = [right]

# Instance of the problem
problem = LinearElasticity(mesh,
                           vec=2,
                           dim=2,
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
    lambda_reg = 10. #1e-9
    u_grad = problem.fes[0].sol_to_grad(sol_list[0])
    l2_reg_term = lambda_reg * np.linalg.norm(u_grad)**2 # l2 regularization
    # Objective function
    J = 0.5 * np.linalg.norm(u_difference)**2 + 0.5 * l2_reg_term
    # Save vtu
    vtu_name = "sol_%03d.vtu" % count
    save_sol(problem.fes[0], sol_list[0], os.path.join(file_dir, vtu_name), 
             cell_infos=[('theta', problem.full_params[:, 0].reshape(-1))],
             point_infos=[('sol_difference',0.5*u_difference)])
    return J

# Gradient of J
def J_grad(rho):
    dJ = jax.grad(J_total)(rho) # (...)
    dJ = dJ.reshape(-1)
    return dJ

# Sharpness for sigmoid
k1 = 500.0

# Set the max/min for the design variables
bound_min, bound_max = np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)
bound_diff = bound_max - bound_min
bound_sum = bound_max + bound_min
x_bound = (bound_min[0], bound_max[0])
y_bound = (bound_min[1], bound_max[1])
l_bound = (1e-5, (min(bound_diff[0], bound_diff[1]) / 2) * 0.99) # 1% less than the boundary
l2_bound = (1e-5, (min(bound_diff[0], bound_diff[1]) / 2) * 0.99)
angle_bound = (1e-5, np.pi/2 * 0.99)

l_points = np.linspace(l_bound[0], l_bound[1], 25)
l2_points = np.linspace(l2_bound[0], l2_bound[1], 25)

outputs = []
count = 1
start_time = time.time()
for l in l_points:
    for l2 in l2_points:
        # Initial guess
        l_normalized = normalize(l, l_bound[0], l_bound[1])
        l2_normalized = normalize(l2, l2_bound[0], l2_bound[1])
        rho_normalized = np.array([l_normalized, l2_normalized])
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

# Exact solution
mid_point = (np.max(mesh.points, axis=0) + np.min(mesh.points, axis=0)) / 2 # mid_point = (20, 20)
cen_exact = mid_point + np.array([5, 10])
rho_exact = np.array([5.0, 2.0]) # [x, y, l, l2, angle]
l_exact_normalized = normalize(rho_exact[0], l_bound[0], l_bound[1])
l2_exact_normalized = normalize(rho_exact[1], l2_bound[0], l2_bound[1])
rho_exact_normalized = np.array([l_exact_normalized, l2_exact_normalized])
exact_obj = J_total(rho_exact_normalized)
obj = onp.array(outputs)
# Reshape 'obj' into a grid nx x ny
obj_grid = obj.reshape(len(l_points), len(l2_points))

# Mesh for plotting
X, Y = onp.meshgrid(l_points, l2_points, indexing='ij')

fig, ax = plt.subplots(figsize=(10, 10))
contour_plot = ax.contourf(X, Y, obj_grid, levels=100, cmap='viridis')
ax.set_xlabel(r"radius in major axis", fontsize=14)
ax.set_ylabel(r"radius in minor axis", fontsize=14)
ax.set_title('Objective function', fontsize=20)
# ax.tick_params(labelsize=20)
ax.scatter(X, Y, c='red', marker='.')
fig.colorbar(contour_plot, label='Objective value')
ax.set_aspect('equal')

# Save
plt.savefig(os.path.join(file_dir, '%s.png' %file_name), dpi=300, bbox_inches='tight')

end_time = time.time()
print('Total running time : %f secs' % (end_time - start_time))