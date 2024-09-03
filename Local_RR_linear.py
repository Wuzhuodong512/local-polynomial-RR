import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.scipy.stats import multivariate_normal
from jax import grad

def data_generate(n, key):
    mean1 = jnp.array([0, 0, 0])
    cov1 = jnp.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
    mean2 = jnp.array([2, 2, 2])
    cov2 = jnp.array([[1,-0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 1]])
    Z1 = random.multivariate_normal(key, mean1, cov1, (n,))
    Z2 = random.multivariate_normal(key, mean2, cov2, (n,))
    
    indicator = np.random.binomial(1, 0.5, n)
    indicator = indicator[:, None]
    Z = indicator * Z1 + (1 - indicator) * Z2
    
    return jnp.array(Z)


def p_0(a, x):
    mean1 = jnp.array([0, 0, 0])
    cov1 = jnp.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
    mean2 = jnp.array([2, 2, 2])
    cov2 = jnp.array([[1,-0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 1]])
    p1 = multivariate_normal.pdf(jnp.concatenate([jnp.array([a]), x]), mean=mean1, cov = cov1)
    p2 = multivariate_normal.pdf(jnp.concatenate([jnp.array([a]), x]), mean=mean2, cov = cov2)
    p = 0.5 * p1 + 0.5 * p2
    return p


def alpha_0(a, x):
    dp_da = grad(p_0, argnums=0)(a, x)
    return -dp_da / p_0(a, x)

@jit
def k(z_1, z_2, h):
    return jnp.exp(-jnp.sum((z_1 - z_2)**2) / h)

@jit
def partial_dk(z_1, z_2, h):
    def k_a(a):
        z_1_new = jnp.concatenate((jnp.array([a]), z_1[1:]))
        return k(z_1_new, z_2, h)
    return grad(k_a)(z_1[0])

@jit
def U2K(z, z_0, h):
    U = jnp.concatenate([jnp.array([1.0]), z-z_0])
    matrix = k(z, z_0, h) * jnp.outer(U, U)
    return matrix

@jit
def m_UK (z, z_0, h):
    U = jnp.concatenate([jnp.array([1.0]), z-z_0])
    mUK = partial_dk(z, z_0, h) * U
    mUK = mUK.at[1].add(k(z, z_0, h))
    return mUK

@jit
def RR_local_linear(z_0, Z, h):
    P_nU2K = jnp.mean(vmap(U2K, (0, None, None))(Z, z_0, h), axis=0)
    P_nmUK = jnp.mean(vmap(m_UK, (0, None, None))(Z, z_0, h), axis=0)
    beta = jnp.linalg.solve(P_nU2K, P_nmUK)
    return beta[0]
    

n = 2000
h = 1

m = 20
key = random.PRNGKey(0)
keys = random.split(key, m)
MSE = np.zeros(m)

for i in range(m):
    Z = data_generate(n, keys[i])
    A, X = Z[:, 0], Z[:, 1:]
    alpha_true = vmap(alpha_0, (0, 0))(A, X)
    alpha_local_linear = vmap(RR_local_linear, (0, None, None))(Z, Z, h)
    MSE[i] = jnp.mean((alpha_local_linear - alpha_true)**2)

mean_MSE = np.mean(MSE)
print(mean_MSE)
