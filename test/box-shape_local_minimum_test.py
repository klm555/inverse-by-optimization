# TODO
# check how it stops : f-based, g-based, x-based 
# try make J to be very large 10**8
# check whether the gradient sign changes
# show more evidence if you convince it is local minimum!

# need to be symmetric(same dirichlet in opposite)
# do many cases() -> average
# box shaped geometry with symmmetric meshes
# can be the problem of the mesh

import os
import time
from typing import List

import jax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem.mma import optimize
from jax_fem import logger
from scipy.optimize import minimize, Bounds

import numpy as onp
import matplotlib.pyplot as plt
import meshio

import logging
logger.setLevel(logging.DEBUG)

# Save setup
file_dir = '../data/local_min_test/box-shape_two_nonzero_dirichlet'
os.makedirs(file_dir, exist_ok=True)
file_name = 'box-shape_two_nonzero_dirichlet'

# Load data (measured displacement)
sol_measured = onp.loadtxt('../box-shape_two_nonzero_dirichlet.txt') # (number of nodes, 3) in 3D

# Mesh info
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 25, 5, 5
dim = 3
# Meshes
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
                            data_dir=file_dir,
                            ele_type=ele_type)
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
        bound_diff = bound_max - bound_min
        r_max = min(bound_diff[0], bound_diff[1]) * 0.5 # 1% less than the boundary
        h_max = bound_diff[2] * 0.5
        
        # Unnormalize the parameters
        self.cen_x = unnormalize(cen_x, bound_min[0], bound_max[0])
        self.cen_y = unnormalize(cen_y, bound_min[1], bound_max[1])
        self.cen_z = None if cen_z is None else unnormalize(cen_z, bound_min[2], bound_max[2])
        self.length = None if length is None else unnormalize(length, 0, r_max)
        self.height = None if height is None else unnormalize(height, 0, h_max)

        # Store data
        self.cells = cells
        self.points = points

    def rectangle(self) -> np.ndarray:
        # Edges of the square
        left, right = self.cen_x - self.length, self.cen_x + self.length
        bottom, top = self.cen_y - self.length, self.cen_y + self.length
        z_bot, z_top = self.cen_z - self.height, self.cen_z + self.height
        # Indices of the points in the inner domain
        domain_points = np.where((self.points[:,0] >= left) & (self.points[:,0] <= right) &
                                 (self.points[:,1] >= bottom) & (self.points[:,1] <= top) &
                                 (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Indices of the cells
        domain_cells = np.any(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]
        return flex_inds # array of shape (n,)
    
    def circle(self) -> np.ndarray:
        # Height
        z_bot, z_top = self.cen_z - self.height, self.cen_z + self.height
        # Indices of the points in the inner domain (x**2 + y**2 <= r**2) & (abs(z) <= h)            
        domain_points = np.where(((self.points[:,0] - self.cen_x)**2 +
                                  (self.points[:,1] - self.cen_y)**2 <= self.length ** 2) &
                                 (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Indices of the cells
        domain_cells = np.all(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]    
        return flex_inds # array of shape (n,)

    def circle_sigmoid(self) -> np.ndarray:       
        # Sharpness for sigmoid
        k = 1.0
        # Squared distances
        domain_squared = (self.points[:,0] - self.cen_x)**2 + (self.points[:,1] - self.cen_y)**2
        r_squared = self.length ** 2
        z_squared = None if is_2d is None else (self.points[:,2] - self.cen_z)**2
        h_squared = None if is_2d is None else self.height ** 2

        # Point indices
        is_2d = self.cen_z == None

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
        # (temp) Fix parameters except "x" coord
        bound_min, bound_max = np.min(self.fes[0].points, axis=0), np.max(self.fes[0].points, axis=0)
        bound_diff = bound_max - bound_min
        r_max = (min(bound_diff[0], bound_diff[1]) / 2)
        h_max = (bound_diff[2] / 2)
        y = normalize((bound_min[1] + bound_max[1]) / 2, bound_min[1], bound_max[1])
        z = normalize((bound_min[2] + bound_max[2]) / 2, bound_min[2], bound_max[2])
        length = normalize(2/5/2, 0, r_max)
        height = normalize(2/5/2, 0, h_max)

        # Inner domain indices
        inner_domain = Geometry(params[0], y, z, 
                                length, height, 
                                self.fes[0].cells, 
                                self.fes[0].points)
        flex_inds = inner_domain.rectangle()
        
        # Assign density(0/1) to entire domain
        full_params = np.ones((problem.num_cells, 1)) # (num_cells, 1)
        full_params = full_params.at[flex_inds].set(0.) # assign "0" to the cells with "flex_inds"
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
x_points = np.linspace(bound_min[0], bound_max[0], 200)

outputs = []
count = 1
start_time = time.time()
for x in x_points:
    # Initial guess
    rho = np.array([x])
    rho_normalized = normalize(rho, bound_min[0], bound_max[0])
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
print(f"Total running runtime: {hours}h {minutes}m {seconds}s")

# Plot the optimization results
x_center = np.array([(bound_min[0] + bound_max[0]) / 2])
obj_0 = J_total(normalize(x_center, bound_min[0], bound_max[0]))
obj = onp.array(outputs)

plt.figure(1, figsize=(10, 8))
plt.plot(x_points, obj, linestyle='-', linewidth=2, color='black')
plt.scatter(x_points, obj, color='black', s=10)
plt.axhline(y=obj_0, color='r', linestyle='--', label='J = %f' % obj_0)
plt.axvline(x=x_center, color='r', linestyle='--', label='x = %f' % x_center[0])
plt.xlabel(r"x-coordinate", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(labelsize=20)
plt.title('Objective function w.r.t x-coordinate', fontsize=20)

# Save
plt.savefig(os.path.join(file_dir, '%s.png' %file_name), dpi=300, bbox_inches='tight')

end_time = time.time()
print('Total running time : %f secs' % (end_time - start_time))