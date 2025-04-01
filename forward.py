import os
from typing import List

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

file_name = 'test'

# Inner domain (soft material)
E_inner = 1.0e-2

# Outer domain (hard material)
E_outer = 2.35e3

# Weak forms
class LinearElasticity(Problem):
    def custom_init(self): # TODO: integrate geometry with this?
            # self.fes[0].flex_inds : indices of the inner domain cells
            # Set up 'self.fes[0].flex_inds' to create the inner domain
            self.fes[0].flex_inds = np.arange(len(self.fes[0].cells))

    # Tensor
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

    def set_params(self, params):
        # Override base class method.
        # full_params = np.ones((self.num_cells, params.shape[1]))
        # full_params = full_params.at[0:10].set(1e-5)
        # thetas = np.repeat(full_params, self.fes[0].num_quads, axis=1)
        # self.full_params = full_params
        self.internal_vars = [params]
        
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
# TODO: I didn't integrate the Geometry class into the problem class, 
# for more flexibility during the testing phase.

# Inner domain info
center_inner = (np.max(mesh.points, axis=0) + np.min(mesh.points, axis=0)) / 2
length_inner = 4. # side length of the square / radius of the circle
height_inner = 0.6

# Create inner domain
class Geometry:
    def __init__(self, center, length, height, cells, points):
        self.center: List[float]  = center
        self.length: float = length
        self.height: float = height
        self.cells: np.ndarray = cells
        self.points: np.ndarray = points

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
        
    def circle(self) -> np.ndarray:
        # Set the edges of the square
        z_bot, z_top = self.center[2] - self.height, self.center[2] + self.height
        # Find the indices of the points in the inner domain(x**2 + y**2 <= r**2) & (abs(z) <= h)            
        domain_points = np.where(((self.points[:,0] - self.center[0])**2 +
                                  (self.points[:,1] - self.center[1])**2 <= self.length ** 2) &
                                 (self.points[:,2] >= z_bot) & (self.points[:,2] <= z_top))[0]
        # Find the indices of the cells in the inner domain
        domain_cells = np.all(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]
        
        return flex_inds # array of shape (n,)

# Create an array of size (num_cells, 1)
# TODO: possible to change '1' to params.shape[1] ?
full_params = E_outer * np.ones((problem.num_cells, 1))

# Find the indices of the inner domain
inner_domain = Geometry(center_inner, length_inner, height_inner, problem.fes[0].cells, problem.fes[0].points)
flex_inds = inner_domain.circle()

# Assign the elastic modulus "E_in" to the cells with flex_inds
full_params = full_params.at[flex_inds].set(E_inner)

# Repeat the full_params to match the number of quadrature points
E = np.repeat(full_params, problem.fes[0].num_quads, axis=1)
# thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)

# Set the parameters into problem
problem.set_params(E)

##################################################################
# Solve the defined problem.
sol_list = solver(problem, solver_options={'umfpack_solver': {}})
# solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

# (num_cells,)
elastic_modulus = full_params[:,0]

# Store the solution to local file.
# vtk_path = os.path.join(os.path.dirname(__file__), 'data', 'forward/vtk/%s.vtu' %file_name)
save_sol(problem.fes[0], np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1))))
         , 'data/forward/vtk/%s.vtu' %file_name
         , cell_infos=[('elastic_modulus', elastic_modulus)]) 
# second argument makes the solution 3D, which makes it able to show the warping in Paraview
onp.savetxt('%s.txt' %file_name, sol_list[0]) # displacements (1681, 2)

# # Postprocess for stress evaluations
# # (num_cells, num_quads, vec, dim)
# u_grad = problem.fes[0].sol_to_grad(sol_list[0])
# epsilon = 0.5 * (u_grad + u_grad.transpose(0,1,3,2))
# # (num_cells, bnum_quads, 1, 1) * (num_cells, num_quads, vec, dim)
# # -> (num_cells, num_quads, vec, dim)
# E = E
# nu = 0.3
# mu = E / (2.*(1+nu))
# lmbda = E * nu / ((1+nu)*(1-2*nu))
# sigma = lmbda * np.trace(epsilon) * np.eye(problem.dim) + 2*mu*epsilon
# # (num_cells, num_quads)
# cells_JxW = problem.JxW[:,0,:]
# # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1) ->
# # (num_cells, vec, dim) / (num_cells, 1, 1)
# #  --> (num_cells, vec, dim)
# sigma_average = np.sum(sigma * cells_JxW[:,:,None,None], axis=1) / np.sum(cells_JxW, axis=1)[:,None,None]

# # Von Mises stress
# # (num_cells, dim, dim)
# s_dev = (sigma_average - 1/problem.dim * np.trace(sigma_average, axis1=1, axis2=2)[:,None,None]
#                                        * np.eye(problem.dim)[None,:,:])
# # (num_cells,)
# vm_stress = np.sqrt(3./2.*np.sum(s_dev*s_dev, axis=(1,2)))
# elastic_modulus = full_params[:,0]