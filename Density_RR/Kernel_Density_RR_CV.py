import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.scipy.stats import multivariate_normal
from jax import grad
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
    return jnp.exp(-jnp.sum((z_1 - z_2)**2) / h) / h


@jit
def density(z, Z, h):  ## multiplied by a constant
    p = jnp.sum(vmap(k, (0, None, None))(Z, z, h))
    return p


@jit
def RR_kd(z, Z, h):
    def density_a(a):
        z_new = jnp.concatenate((jnp.array([a]), z[1:]))
        return density(z_new, Z, h)
    alpha = -grad(density_a)(z[0]) / density(z, Z, h)
    return alpha


@jit
def score(Z_train, Z_val, h):
    likelihood = vmap(density, (0, None, None))(Z_val, Z_train, h)
    s = jnp.sum(jnp.log(likelihood))
    return s


def CV_score(h, Z, split_index):
    s = 0
    for train_index, val_index in split_index:
        Z_train, Z_val = Z[train_index], Z[val_index]
        s += score(Z_train, Z_val, h)
    return s


def CV_bandwidth(Z, bandwidths, k_folds=10):
    kf = KFold(n_splits=k_folds)
    split_index = list(kf.split(Z))
    cv_scores = vmap(CV_score, (0, None, None))(bandwidths, Z, split_index)
    best_h = bandwidths[jnp.argmax(cv_scores)]
    return best_h


n = 2000
m = 100
key = random.PRNGKey(0)
keys = random.split(key, m)
MSE = np.zeros(m)


for i in range(m):
    Z = data_generate(n, keys[i])
    A, X = Z[:, 0], Z[:, 1]
    h = CV_bandwidth(Z, jnp.linspace(0.1, 0.8, 30))
    
    alpha_true = vmap(alpha_0, (0, 0))(A, X)
    alpha_kd = vmap(RR_kd, (0, None, None))(Z, Z, h)
    MSE[i] = jnp.mean((alpha_kd - alpha_true)**2)

mean_MSE = np.mean(MSE)
print(mean_MSE)
