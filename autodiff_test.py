import os
import time
from typing import List

import jax
import jax.numpy as np
from scipy.optimize import minimize, Bounds
import numpy as onp

# Applied approaches : Sigmoid / ReLU / Straight-Through Estimator / Gumbel Softmax
# Toy example : 2D square mesh
points = np.array([[0, 0], [1, 0], [2, 0], [3, 0],
                   [0, 1], [1, 1], [2, 1], [3, 1],
                   [0, 2], [1, 2], [2, 2], [3, 2],
                   [0, 3], [1, 3], [2, 3], [3, 3]])

cells = np.array([[0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3],
                 [4, 8, 9, 5], [5, 9, 10, 6], [6, 10, 11, 7],
                 [8, 12, 13, 9], [9, 13, 14, 10], [10, 14, 15, 11]])

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

# "Differentiable" function to create a domain
class Geometry:
    def __init__(self, cen_x, cen_y, cen_z=None, length=None, height=None, cells=None, points=None):
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
        self.cen_x: float = cen_x
        self.cen_y: float = cen_y
        self.cen_z: float = cen_z
        self.length: float = length
        self.height: float = height
        self.cells: np.ndarray = cells
        self.points: np.ndarray = points
    
    # Gumbel softmax (Gumbel trick)
    def gumbel_softmax(self, logits, temperature=1.0, eps=1e-20, separate=False):
        # Create uniform random values
        uniform_rand = jax.random.uniform(jax.random.PRNGKey(0), shape=self.points.shape)
        
        # Convert uniform random values to Gumbel noise (-log(-log(uniform_rand)))
        gumbel_noise = -np.log(-np.log(uniform_rand + eps) + eps) # eps to avoid log(0)
        
        # Add Gumbel noise to logits(inputs)
        perturbed_logits = logits + gumbel_noise

        # Softmax
        khot_list = []
        onehot_approx = np.zeros_like(perturbed_logits, dtype=np.float32)

        jax.nn.softmax(perturbed_logits / temperature)

        if separate:
            return khot_list
        else:
            return jax.lax.reduce_sum(khot_list, 0)

    def circle(self) -> np.ndarray:
        """
        Get indices of cells inside the circle (2D) or cylinder (3D).
        """
        is_2d = self.cen_z == None

        # Find the indices of the points in the inner domain
        if is_2d: # 2D
            # Find the indices of the points in the inner domain(x**2 + y**2 <= r**2) & (abs(z) <= h)            
            domain_points = np.where(self.length ** 2 - (self.points[:,0] - self.cen_x)**2 -
                                     (self.points[:,1] - self.cen_y)**2)[0]
            
        else: # 3D
            # Set the edges of the square
            z_bot, z_top = self.cen_z - self.height, self.cen_z + self.height
            # Find the indices of the points in the inner domain(x**2 + y**2 <= r**2) & (abs(z) <= h)            
            domain_points = np.where((self.length ** 2 - (self.points[:,0] - self.cen_x)**2 -
                                     (self.points[:,1] - self.cen_y)**2) &
                                     (self.points[:,2] - z_bot) & (z_top - self.points[:,2]))[0]

        # Find the indices of the cells in the inner domain
        domain_cells = np.all(np.isin(self.cells, domain_points), axis=1)
        flex_inds = np.where(domain_cells)[0]
        return flex_inds # array of shape (n,)
    
    def circle_sigmoid(self) -> np.ndarray:
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
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared))
            # point_indicators = np.round(point_indicators)
            # print(point_indicators)
            
        else: # 3D  
            z_squared = (self.points[:, 2] - self.cen_z)**2
            h_squared = self.height ** 2
            point_indicators = jax.nn.sigmoid(k * (domain_squared - r_squared))
            z_indicators = jax.nn.sigmoid(k * (z_squared - h_squared))
            # np.round() makes indifferentiable
            # point_indicators, z_indicators = np.round(point_indicators), np.round(z_indicators)
            
            # point_indicators = 1 only if both are 1
            point_indicators = point_indicators + z_indicators
            point_indicators = np.max(point_indicators, 1.0)

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
        k = 500  # Controls transition sharpness

        # Find the indices of the points in the inner domain
        if is_2d: # 2D
            # When r_squared < r_param: sigmoid ≈ 1 (inside)
            # When r_squared > r_param: sigmoid ≈ 0 (outside)
            point_indicators = jax.nn.relu6(k * (domain_squared - r_squared))
            # point_indicators = np.round(point_indicators)
            
        else: # 3D  
            z_squared = (self.points[:, 2] - self.cen_z)**2
            h_squared = self.height ** 2
            point_indicators = jax.nn.relu6(k * (domain_squared - r_squared))
            z_indicators = jax.nn.relu6(k * (z_squared - h_squared))
            # np.round() makes indifferentiable
            # point_indicators, z_indicators = np.round(point_indicators), np.round(z_indicators)
            
            # point_indicators = 1 only if both are 1
            point_indicators = np.max(point_indicators, z_indicators)

        # Find the point_indicators by the indices stored in the cells
        cell_indicators = point_indicators[self.cells] # Assign point indicators by cell indices
        # cell_indicators = 1 only if all points are 1
        cell_indicators = np.prod(cell_indicators, axis=1)
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
            point_indicators = where_ste(r_squared - domain_squared)
            print(point_indicators)
        else:
            # 3D logic: must be inside the disk in XY and also within [cen_z +/- height]
            z_squared = (self.points[:,2] - self.cen_z) ** 2
            h_squared = self.height ** 2

            radial_indicator = where_ste(r_squared - domain_squared)
            z_indicator = where_ste(h_squared - z_squared)
            # 1 only if inside the circle in XY *and* within height range in Z
            point_indicators = radial_indicator * z_indicator

        # "Cell" indicator = product of that cell's corner indicators
        cell_indicators = point_indicators[self.cells]
        cell_indicators = np.min(cell_indicators, axis=1)

        return cell_indicators
        

def set_params(params): # params = [x, y, z, r, h]
    # Geometry class doesn't use 'flex_inds', but directly assigns 'theta' values to the cells
    inner_domain = Geometry(params[0], params[1], length=params[2], cells=cells, points=points)
    full_params = inner_domain.circle_relu()
    # full_params = np.zeros((len(cells), 1)) # 'circle' function only
    # full_params = full_params.at[flex_inds].set(1) # 'circle' function only
    full_params = np.expand_dims(full_params, axis=1)
    thetas = np.repeat(full_params[:, None, :], 2, axis=1)
    internal_vars = [thetas]
    print(full_params)
    return np.max(thetas) # Just for testing
    

def main(rho):
    sol = set_params(rho)
    sol_grad = jax.grad(set_params)(rho)
    # print(sol)
    print('grad=', sol_grad)

if __name__ == "__main__":
    rho = [1., 0., 1.]
    main(rho)