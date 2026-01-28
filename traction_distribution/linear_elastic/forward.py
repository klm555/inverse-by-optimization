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
from jax_fem.generate_mesh import rectangle_mesh, box_mesh_gmsh, get_meshio_cell_type, Mesh
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
file_name = 'load2_noise-0'

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

# Traction Distribution
def traction_true(point):
    # return 1e-2 * np.exp(-(np.power(point[0] - Lx/2., 2)) / (2.*(Lx/5.)**2)) # Load 1
    return np.where(point[0] >= Lx/2., 1e-2, 0.0) # Load 2

class LinearElasticity(Problem):
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
        def surface_map(u, x):
            return np.array([0, traction_true(x), 0])
        return [surface_map]

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

# Solve
sol_list = solver(problem, solver_options={'petsc_solver': {}})

# Traction data for saving
def compute_traction(point):
    is_top = top(point)

    traction_vector = np.array([0., traction_true(point), 0.])
    
    # Return only where is_top is True, else 0.
    return traction_vector * is_top # vector*True = vector / vector*False = 0

compute_traction_vmap = jax.vmap(compute_traction)
traction_data = compute_traction_vmap(mesh.points)


# Save
vtu_name = 'vtk/%s.vtu' %file_name
save_sol(problem.fes[0], 
         np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
         os.path.join(file_dir, vtu_name),
         point_infos=[('traction', traction_data)]) 
# 2nd arg makes the solution 3D, which enables warping in Paraview
onp.savetxt('%s.txt' %file_name, sol_list[0])

# Save the result in JSON file
with open('%s.json' %file_name, 'w') as f:
    json.dump(sol_list[0].tolist(), f, indent=4)