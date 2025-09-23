import os
from typing import List
import json

# Set JAX device (before importing jax)
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # or 'gpu'

import jax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
from jax_fem import logger
import numpy as onp

import logging
logger.setLevel(logging.DEBUG)

# Check JAX backend and devices
print('JAX backend:', jax.default_backend())
print('Devices:', jax.devices())

# Save setup
file_dir = 'data/forward'
os.makedirs(file_dir, exist_ok=True)
file_name = 'eval_solutions-no_hole-unit_domain'

# Elastic modulus
E_inner = 1.0e-3 # Inner domain (soft material)
E_outer = 1.0e3 # Outer domain (hard material)

# Mesh info
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
Lx, Ly = 2., 2. # domain
Nx, Ny = 200, 200 # number of elements in x-dir, y-dir
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

# Inner domain
class Geometry:
    def __init__(self, center, length, height=None, length2=None, angle=None, cells=None, points=None):
        self.center: List[float]  = center
        self.length: float = length
        self.height: float = None if height is None else height
        self.length2: float = None if length2 is None else length2
        self.angle: float = None if angle is None else angle
        self.cells: np.ndarray = cells
        self.points: np.ndarray = points
        
    def circle(self) -> np.ndarray:
        # Check dimensions
        is_2d = self.height == None

        if is_2d: # 2D
            domain_points = np.where((self.points[:,0] - self.center[0])**2 +
                                     (self.points[:,1] - self.center[1])**2 <= self.length ** 2)[0]
        
        else: # 3D
            # Height
            z_bot, z_top = self.center[2] - self.height, self.center[2] + self.height
            # Indices of the points in the inner domain (x**2 + y**2 <= r**2) & (abs(z) <= h)            
            domain_points = np.where(((self.points[:,0] - self.center[0])**2 +
                                      (self.points[:,1] - self.center[1])**2 <= self.length ** 2) &
                                     (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Indices of the cells
        domain_cells = np.any(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]
        return flex_inds # array of shape (n,)
    
    def ellipse(self) -> np.ndarray: # only for 2D
        # Indices of the points in the inner domain (x**2 + y**2 <= r**2) & (abs(z) <= h)            
        domain_points = np.where(((self.points[:,0] - self.center[0]) * np.cos(self.angle) + 
                                  (self.points[:,1] - self.center[1]) * np.sin(self.angle))**2 / self.length**2 +
                                 ((self.points[:,0] - self.center[0]) * np.sin(self.angle) - 
                                  (self.points[:,1] - self.center[1]) * np.cos(self.angle))**2 / self.length2**2
                                  <= 1)[0]
        # Indices of the cells
        domain_cells = np.any(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]
        return flex_inds # array of shape (n,)
    
    def rectangle(self) -> np.ndarray:
        # Edges of the square
        left, right = self.center[0] - self.length, self.center[0] + self.length
        bottom, top = self.center[1] - self.length, self.center[1] + self.length
        z_bot, z_top = self.center[2] - self.height, self.center[2] + self.height
        # Indices of the points in the inner domain
        domain_points = np.where((self.points[:,0] >= left) & (self.points[:,0] <= right) &
                                 (self.points[:,1] >= bottom) & (self.points[:,1] <= top) &
                                 (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Indices of the cells
        domain_cells = np.any(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]
        return flex_inds # array of shape (n,)

# Weak forms
class LinearElasticity(Problem):
    def custom_init(self): # TODO: integrate geometry with this?
            self.fes[0].flex_inds = np.arange(len(self.fes[0].cells)) # "flex_inds" : idx of the inner domain cells

    def get_tensor_map(self):
        def stress(u_grad, E): # stress tensor
            nu = 0.33
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

    def set_params(self, params):
        # Override base class method.
        # full_params = np.ones((self.num_cells, params.shape[1]))
        # full_params = full_params.at[0:10].set(1e-5)
        # thetas = np.repeat(full_params, self.fes[0].num_quads, axis=1)
        # self.full_params = full_params
        self.internal_vars = [params]

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

def custom_dirichlet_val(point):
    return 0.05

def minus_custom_dirichlet_val(point):
    return -0.05

# Dirichlet boundary info
# [plane, direction, displacement]
# number of elements in plane, direction, displacement should match
dirichlet_bc_info = [[left]*2 + [right]*2, 
                     [0, 1]*2, 
                     [minus_custom_dirichlet_val] + [zero_dirichlet_val] + [custom_dirichlet_val] + [zero_dirichlet_val]]

# Neumann boundary locations
# "get_surface_maps" performs surface integral to get the traction
location_fns = [right]

# Instance of the problem
problem = LinearElasticity(mesh,
                           vec=2,
                           dim=2,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,)
                        #    location_fns=location_fns)

# TODO: I didn't integrate the Geometry class into the problem class, 
# for more flexibility during the testing phase.

# Inner domain parameters
mid_point = (np.max(mesh.points, axis=0) + np.min(mesh.points, axis=0)) / 2 # mid_point = (20, 20)
center_inner = mid_point + np.array([0.125, 0.125]) # default : 5, 5
length_inner = 0.0 # side length of the square / radius of the circle / length in major axis of the ellipse
length2_inner = 0.0 # default : 5, 2, pi/3
angle_inner = np.pi / 3 # radian

# Inner domain indices
inner_domain = Geometry(center_inner, 
                        length_inner, 
                        length2=length2_inner,
                        angle=angle_inner,
                        cells=problem.fes[0].cells, 
                        points=problem.fes[0].points)
flex_inds = inner_domain.ellipse()

# Assign elastic modulus to entire domain
full_params = E_outer * np.ones((problem.num_cells, 1)) # (num_cells, 1)
full_params = full_params.at[flex_inds].set(E_inner) # assign "E_inner" to the cells with "flex_inds"
elastic_modulus = full_params[:,0] # (num_cells,)

# Match "full_params" to the number of quadrature points
E = np.repeat(full_params, problem.fes[0].num_quads, axis=1)

# Assign elastic modulus into "Problem"
problem.set_params(E)

# Solve
sol_list = solver(problem, solver_options={'umfpack_solver': {}})

# Save
vtu_name = 'vtk/%s.vtu' %file_name
save_sol(problem.fes[0], 
         np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
         os.path.join(file_dir, vtu_name),
         cell_infos=[('elastic_modulus', elastic_modulus)]) 
# 2nd arg makes the solution 3D, which enables warping in Paraview

# Save the result in JSON file
with open('%s.json' %file_name, 'w') as f:
    json.dump(sol_list[0].tolist(), f, indent=4)