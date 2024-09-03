import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.scipy.stats import multivariate_normal
from jax import grad
from functools import partial
from sklearn.model_selection import KFold


def data_generate(n, key):
    mean1 = jnp.array([0, 0])
    cov1 = jnp.array([[1,0.5], [0.5, 1]])
    mean2 = jnp.array([2, 2])
    cov2 = jnp.array([[1,-0.5], [-0.5, 1]])
    Z1 = random.multivariate_normal(key, mean1, cov1, (n,))
    Z2 = random.multivariate_normal(key, mean2, cov2, (n,))
    
    indicator = np.random.binomial(1, 0.5, n)
    indicator = indicator[:, None]
    Z = indicator * Z1 + (1 - indicator) * Z2
    
    return jnp.array(Z)


def p_0(a, x):
    mean1 = jnp.array([0, 0])
    cov1 = jnp.array([[1,0.5], [0.5, 1]])
    mean2 = jnp.array([2, 2])
    cov2 = jnp.array([[1,-0.5], [-0.5, 1]])
    p1 = multivariate_normal.pdf(jnp.array([a, x]), mean=mean1, cov = cov1)
    p2 = multivariate_normal.pdf(jnp.array([a, x]), mean=mean2, cov = cov2)
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
def RR_regular_x(z_0, Z, h):
    a_0 = z_0[0]
    x_0 = z_0[1]
    A = Z[:, 0] - a_0
    X = Z[:, 1] - x_0
    K = vmap(partial(k, z_2=z_0, h=h))(Z)
    partial_K = vmap(partial(partial_dk, z_2=z_0, h=h))(Z)
    sum_K = jnp.sum(K)
    sum_K_A = jnp.sum(K * A)
    sum_K_A2 = jnp.sum(K * A**2)
    sum_K_X = jnp.sum(K * X)
    sum_K_X2 = jnp.sum(K * X**2)
    sum_K_AX = jnp.sum(K * A * X)
    d_matrix = jnp.array([[sum_K, sum_K_A, sum_K_X], [sum_K_A, sum_K_A2, sum_K_AX], [sum_K_X, sum_K_AX, sum_K_X2]])
    b = jnp.array([jnp.sum(partial_K), jnp.sum(K)+jnp.sum(partial_K*A), jnp.sum(partial_K*X)])
    beta = jnp.linalg.solve(d_matrix, b)
    return beta[0]

def RR_a(a, z_0, Z, h):
    z_0_new = jnp.concatenate((jnp.array([a]), z_0[1:]))
    return RR_regular_x(z_0_new, Z, h)


def partial_alpha(z_0, Z, h):
    return grad(RR_a)(z_0[0], z_0, Z, h)

@jit
def loss_one(z_0, Z, h):
    alpha = RR_regular_x(z_0, Z, h)
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
m = 100
key = random.PRNGKey(0)
keys = random.split(key, m)
MSE = np.zeros(m)

for i in range(m):
    Z = data_generate(n, keys[i])
    A, X = Z[:, 0], Z[:, 1]
    h = CV_bandwidth(Z, jnp.linspace(0.1, 2, 20))
    
    alpha_true = vmap(alpha_0, (0, 0))(A, X)
    alpha_LocalP = vmap(RR_regular_x, (0, None, None))(Z, Z, h)
    MSE[i] = jnp.mean((alpha_LocalP - alpha_true)**2)

mean_MSE = np.mean(MSE)
print(mean_MSE)
