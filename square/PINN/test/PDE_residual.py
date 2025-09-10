import jax
import jax.numpy as np




def residual(u, x, y):
    def stress(x, y): # tensor
        dim = 2
        E = 2.35e3
        nu = 0.33
        mu = E / (2. * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
        def u_vec(coords): # make (n, 2)
            return u(coords[0], coords[1])
        u_jac = jax.jacrev(u_vec) # (2, 2) Jacobian matrix
        u_jac_val = u_jac((np.array([x, y])))
        epsilon = 0.5 * (u_jac_val + u_jac_val.T) # (2, 2)
        # Stress-Strain relationship
        sigma = lmbda * np.trace(epsilon) * np.eye(dim) + 2 * mu * epsilon
        return sigma

    # σ = [[σ11, σ12], [σ21, σ22]]
    jacobian_wrt_x = jax.jacobian(stress, argnums=0)(x, y) # [[∂σ11/∂x, ∂σ12/∂x], [∂σ21/∂x, ∂σ22/∂x]]
    jacobian_wrt_y = jax.jacobian(stress, argnums=1)(x, y) # [[∂σ11/∂y, ∂σ12/∂y], [∂σ21/∂y, ∂σ22/∂y]]
    dsigma11_dx = jacobian_wrt_x[0, 0] # ∂σ11 / ∂x
    dsigma12_dy = jacobian_wrt_y[0, 1] # ∂σ12 / ∂y
    dsigma21_dx = jacobian_wrt_x[1, 0] # ∂σ21 / ∂x
    dsigma22_dy = jacobian_wrt_y[1, 1] # ∂σ22 / ∂y
    lhs1 = dsigma11_dx + dsigma12_dy
    rhs1 = 0.
    lhs2 = dsigma21_dx + dsigma22_dy
    rhs2 = 0.
    return lhs1 - rhs1, lhs2 - rhs2