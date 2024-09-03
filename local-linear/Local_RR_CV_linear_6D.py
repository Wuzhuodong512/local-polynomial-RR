import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.scipy.stats import multivariate_normal
from jax import grad
from sklearn.model_selection import KFold


def data_generate(n, key):
    mean1 = jnp.array([0, 0, 0, 0, 0, 0])
    cov1 = jnp.array([[1, 0.5, 0, 0.2, 0, 0], [0.5, 1, 0.5, 0, 0, 0], [0, 0.5, 1, 0.5, 0 ,0], [0.2, 0, 0.5, 1, 0, 0], [0, 0, 0, 0, 1, 0.5], [0, 0, 0, 0, 0.5, 1]])
    mean2 = jnp.array([2, 2, 2, 2, 2, 2])
    cov2 = jnp.array([[1,-0.5, 0.2, 0, 0, 0], [-0.5, 1, -0.5, 0, 0, 0], [0.2, -0.5, 1, -0.5, 0, 0], [0, 0, -0.5, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    Z1 = random.multivariate_normal(key, mean1, cov1, (n,))
    Z2 = random.multivariate_normal(key, mean2, cov2, (n,))
    
    indicator = np.random.binomial(1, 0.5, n)
    indicator = indicator[:, None]
    Z = indicator * Z1 + (1 - indicator) * Z2
    
    return jnp.array(Z)


def p_0(a, x):
    mean1 = jnp.array([0, 0, 0, 0, 0, 0])
    cov1 = jnp.array([[1, 0.5, 0, 0.2, 0, 0], [0.5, 1, 0.5, 0, 0, 0], [0, 0.5, 1, 0.5, 0 ,0], [0.2, 0, 0.5, 1, 0, 0], [0, 0, 0, 0, 1, 0.5], [0, 0, 0, 0, 0.5, 1]])
    mean2 = jnp.array([2, 2, 2, 2, 2, 2])
    cov2 = jnp.array([[1,-0.5, 0.2, 0, 0, 0], [-0.5, 1, -0.5, 0, 0, 0], [0.2, -0.5, 1, -0.5, 0, 0], [0, 0, -0.5, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
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

def RR_a(a, z_0, Z, h):
    z_0_new = jnp.concatenate((jnp.array([a]), z_0[1:]))
    return RR_local_linear(z_0_new, Z, h)


def partial_alpha(z_0, Z, h):
    return grad(RR_a)(z_0[0], z_0, Z, h)

@jit
def loss_one(z_0, Z, h):
    alpha = RR_local_linear(z_0, Z, h)
    m_alpha = partial_alpha(z_0, Z, h)
    l = jnp.square(alpha) - 2 * m_alpha
    return l


def loss(Z_train, Z_val, h):
    l = jnp.sum(vmap(loss_one, (0, None, None))(Z_val, Z_train, h))
    return l
        
        
def CV_loss(h, Z, split_index):
    l = 0
    for train_index, val_index in split_index:
        Z_train, Z_val = Z[train_index], Z[val_index]
        l += loss(Z_train, Z_val, h)
    return l


def CV_bandwidth(Z, bandwidths, k_folds=10):
    kf = KFold(n_splits=k_folds)
    split_index = list(kf.split(Z))
    cv_scores = vmap(CV_loss, (0, None, None))(bandwidths, Z, split_index)
    best_h = bandwidths[jnp.argmin(cv_scores)]
    return best_h



n = 2000
m = 10
key = random.PRNGKey(1)
keys = random.split(key, m)
MSE = np.zeros(m)
bandwidth = np.linspace(0.5, 5, 10)
h = np.zeros(m)

for i in range(m):
    Z = data_generate(n, keys[i])
    A, X = Z[:, 0], Z[:, 1:]
    h[i] = CV_bandwidth(Z, bandwidth)
    
    alpha_true = vmap(alpha_0, (0, 0))(A, X)
    alpha_LocalP = vmap(RR_local_linear, (0, None, None))(Z, Z, h[i])
    MSE[i] = jnp.mean((alpha_LocalP - alpha_true)**2)

mean_MSE = np.mean(MSE)
print("MSE:",mean_MSE)
print("selected bandwidth:",h)
