import os
from typing import List

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
import meshio

import logging
logger.setLevel(logging.DEBUG)

# Check JAX backend and devices
print('JAX backend:', jax.default_backend())
print('Devices:', jax.devices())

# Save setup
file_dir = 'data/forward'
os.makedirs(file_dir, exist_ok=True)
file_name = 'dogbone_4'

# Elastic modulus
E_inner = 1.0e-3 # Inner domain (soft material)
E_outer = 2.35e3 # Outer domain (hard material)

# Mesh info
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type) # convert 'QUAD4' to 'quad' in meshio
dim = 3
# Meshes
msh_file = 'Dogbone_0.05.msh'
meshio_mesh = meshio.read(msh_file) # meshio : 3rd party library
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

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

# Dirichlet boundary info
# [plane, direction, displacement]
# number of elements in plane, direction, displacement should match
dirichlet_bc_info = [[left]*3 + [right]*3, 
                     [0, 1, 2]*2, 
                     [minus_one_dirichlet_val] + [zero_dirichlet_val]*2 + [one_dirichlet_val] + [zero_dirichlet_val]*2]

# Neumann boundary locations
# "get_surface_maps" performs surface integral to get the traction
location_fns = [right]

# Instance of the problem
problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,)
                        #    location_fns=location_fns)

# TODO: I didn't integrate the Geometry class into the problem class, 
# for more flexibility during the testing phase.

# Inner domain parameters
center_inner = (np.max(mesh.points, axis=0) + np.min(mesh.points, axis=0)) / 2
center_inner_origin = (np.max(mesh.points, axis=0) + np.min(mesh.points, axis=0)) / 2
center_inner = center_inner_origin + np.array([42.0, -1.8, 0.0])
length_inner = 2.5 # side length of the square / radius of the circle
height_inner = 1.0

# Inner domain
class Geometry:
    def __init__(self, center, length, height, cells, points):
        self.center: List[float]  = center
        self.length: float = length
        self.height: float = height
        self.cells: np.ndarray = cells
        self.points: np.ndarray = points
        
    def circle(self) -> np.ndarray:
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

# Inner domain indices
inner_domain = Geometry(center_inner, 
                        length_inner, 
                        height_inner, 
                        problem.fes[0].cells, 
                        problem.fes[0].points)
flex_inds = inner_domain.circle()

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
         sol_list[0],
         os.path.join(file_dir, vtu_name),
         cell_infos=[('elastic_modulus', elastic_modulus)]) 
# 2nd arg makes the solution 3D, which enables warping in Paraview
onp.savetxt('%s.txt' %file_name, sol_list[0])