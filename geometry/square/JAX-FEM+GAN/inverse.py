import os
import time
import jax
import jax.numpy as np
import numpy as onp
import optax
from flax import linen as nn
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
from jax_fem.utils import save_sol
from jax_fem import logger
import logging

# ---------------------------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------------------------
logger.setLevel(logging.WARNING) # Reduce log noise
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # Or 'gpu' if available

file_dir = 'data/inverse'
os.makedirs(file_dir, exist_ok=True)
file_name = '2D-ellipse2-refine_10'

# Load Measurement Data (Ground Truth)
sol_measured = onp.loadtxt('../ellipse_hole-extended_domain2.txt')

# Mesh Constants (Match with forward.py)
Lx, Ly = 80., 80.
Nx, Ny = 160, 160
dim = 2
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)

# Create Mesh
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Material Constants (Match with inverse.py)
E_max = 1.0e3
E_min = 1.0e-3
nu = 0.33

# ---------------------------------------------------------------------------
# 2. Generator Network (CNN-based)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Generates a 160x160 density field from a latent vector z.
    """
    @nn.compact
    def __call__(self, z):
        # # z shape: (batch, latent_dim)
        # # 1. Project and Reshape
        # x = nn.Dense(features=10 * 10 * 256)(z)
        # x = nn.relu(x)
        # x = x.reshape((z.shape[0], 10, 10, 256))
        
        # # 2. Upsampling (Deconvolution)
        # # 10x10 -> 20x20
        # x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        # # [수정] BatchNorm -> LayerNorm (상태 관리 필요 없음, 배치 사이즈 1에서도 동작)
        # x = nn.LayerNorm()(x) 
        # x = nn.relu(x)
        
        # # 20x20 -> 40x40
        # x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        # # [수정] BatchNorm -> LayerNorm
        # x = nn.LayerNorm()(x)
        # x = nn.relu(x)
        
        # # 40x40 -> 80x80
        # x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        # # [수정] BatchNorm -> LayerNorm
        # x = nn.LayerNorm()(x)
        # x = nn.relu(x)
        
        # # 80x80 -> 160x160
        # x = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        
        # # 3. Output Activation (Sigmoid for 0~1 density)
        # density_map = nn.sigmoid(x) 
        
        # return density_map.reshape((z.shape[0], -1, 1))
        
        x = nn.Dense(256)(z)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(512)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(784)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(160*160)(x)
        x = nn.tanh(x)
        return x.reshape((z.shape[0], -1, 1))

# ---------------------------------------------------------------------------
# 3. Differentiable Physics (FEM Solver)
# ---------------------------------------------------------------------------
class DifferentiablePhysics(Problem):
    def get_tensor_map(self):
        def stress(u_grad, theta):
            # SIMP Interpolation
            # theta: density (0~1)
            # Penalization factor p=3 for better binary separation
            p = 3.0
            E = E_min + (E_max - E_min) * (theta[0] ** p)
            # E = E_min + (E_max - E_min) * theta[0]
            
            mu = E / (2. * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            return sigma
        return stress

    def set_params(self, params):
        # params: density field (num_cells, 1)
        # Map density to quadrature points
        self.internal_vars = [np.repeat(params[:, None, :], self.fes[0].num_quads, axis=1)]

# Boundary Conditions (From inverse.py)
def left(point): return np.isclose(point[0], 0., atol=1e-5)
def right(point): return np.isclose(point[0], Lx, atol=1e-5)
def zero_bc(p): return 0.
def one_bc(p): return 0.05
def minus_one_bc(p): return -0.05

dirichlet_bc_info = [[left]*2 + [right]*2, 
                     [0, 1]*2, 
                     [minus_one_bc, zero_bc, one_bc, zero_bc]]

# Initialize Problem
problem = DifferentiablePhysics(mesh, vec=2, dim=2, ele_type=ele_type, 
                                dirichlet_bc_info=dirichlet_bc_info)

# Create Differentiable Function (Auto-Diff Wrapper)
# Input: Material Density -> Output: Displacement Field
fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

# ---------------------------------------------------------------------------
# 4. Refinement Step (Material Optimization)
# ---------------------------------------------------------------------------
def Discriminator(initial_density, u_measured, steps=20, lr=0.5):
    """
    Optimizes the material density directly to match measured displacement.
    """
    density = initial_density # Start from Generator's output

    # Total Variation Regularization to encourage smoothness
    def tv_reg(rho, alpha=0.01):
        img = rho.reshape((Nx, Ny))
        dx = img - np.roll(img, 1, axis=0)
        dy = img - np.roll(img, 1, axis=1)
        return alpha * np.sum(np.sqrt(dx**2 + dy**2 + 1e-6))

    def loss_fn(rho):
        # rho shape: (num_cells, 1)
        sol_list = fwd_pred(rho)
        u_pred = sol_list[0]
        
        # Data Mismatch
        mse_loss = 0.5 * np.sqrt(np.sum((u_pred - u_measured)**2))
        # Regularization
        reg_loss = tv_reg(rho)
        
        return mse_loss # + reg_loss

    # Optimization Loop (Adam)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(density)

    for i in range(steps):
        # Gradient Function (w.r.t. rho) 
        loss_val, grads = jax.value_and_grad(loss_fn)(density)
        updates, opt_state = optimizer.update(grads, opt_state, density)
        density = optax.apply_updates(density, updates)
        # Project to [0, 1] valid range
        density = np.clip(density, 0., 1.)
        
    return density, loss_val

# ---------------------------------------------------------------------------
# 5. Generator Training Step
# ---------------------------------------------------------------------------
@jax.jit
def train_generator(g_params, opt_state, z, target_density):
    """
    Trains Generator to mimic the refined (optimized) density.
    """
    def g_loss_fn(params):
        generated = generator.apply(params, z)[0] # (num_cells, 1)
        # Supervised Loss: Generator Output <-> Refined Target
        return 0.5 * np.sqrt(np.sum((generated - target_density)**2))
    
    loss, grads = jax.value_and_grad(g_loss_fn)(g_params)
    updates, opt_state = optimizer.update(grads, opt_state, g_params)
    new_g_params = optax.apply_updates(g_params, updates)
    
    return new_g_params, opt_state, loss

# ---------------------------------------------------------------------------
# 6. Main Execution Loop
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize Generator
    key = jax.random.PRNGKey(42)
    latent_dim = 64
    generator = Generator()
    
    key, subkey = jax.random.split(key)
    dummy_z = jax.random.normal(subkey, (1, latent_dim))
    g_params = generator.init(subkey, dummy_z)
    
    # Initialize Optimizer (Adam)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(g_params)
    
    print(">>> Starting Training: Refinement-based Generative Inverse Design <<<")
    
    num_epochs = 500
    refinement_steps = 10  # How many FEM steps to refine the guess
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 1. Sample Noise
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (1, latent_dim))
        
        # 2. Generator Forward (Initial Guess)
        initial_density = generator.apply(g_params, z)[0]
        
        # 3. Refinement Step (Physics-based Optimization)
        # "Discriminator" phase: Optimize material using Solver + L2 Loss
        refined_density, d_loss = Discriminator(initial_density, sol_measured, 
                                          steps=refinement_steps, lr=0.5)
        
        # 4. Generator Update Step (Learning)
        # "Generator" phase: Learn to output the refined density directly
        g_params, opt_state, g_loss = train_generator(g_params, opt_state, z, refined_density)
        
        elapsed = time.time() - start_time
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | G_Loss: {g_loss:.6f} | D_Loss: {d_loss:.6f} | Time: {elapsed:.2f}s")
            
            # Save Visualization
            # Save Refined Density (Target)
            save_sol(problem.fes[0], 
                     np.zeros((len(problem.fes[0].points), 3)), # Dummy displacement
                     os.path.join(file_dir, f'{file_name}_refined_{epoch:03d}.vtu'), 
                     cell_infos=[('density', refined_density[:, 0])])
            
            # Save Generator Output (Prediction)
            gen_out = generator.apply(g_params, z)[0]
            save_sol(problem.fes[0], 
                     np.zeros((len(problem.fes[0].points), 3)), # Dummy displacement
                     os.path.join(file_dir, f'{file_name}_gen_{epoch:03d}.vtu'), 
                     cell_infos=[('density', gen_out[:, 0])])