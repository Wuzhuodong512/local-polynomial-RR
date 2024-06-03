import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.scipy.stats import multivariate_normal
from jax import grad
from functools import partial

@jit
def p_0(a, x):
    mean = jnp.zeros(2)
    cov = jnp.array([[1,0.5], [0.5, 1]])
    return multivariate_normal.pdf(jnp.array([a, x]), mean=mean, cov = cov)

@jit
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
def RR_special(z_0, Z, h):
    a_0 = z_0[0]
    A = Z[:, 0] - a_0
    K = vmap(partial(k, z_2=z_0, h=h))(Z)
    sum_K = jnp.sum(K)
    sum_K_A = jnp.sum(K * A)
    sum_K_A2 = jnp.sum(K * A**2)
    d_matrix = jnp.array([[sum_K, sum_K_A], [sum_K_A, sum_K_A2]])
    b = jnp.array([0, jnp.sum(K)])
    beta = jnp.linalg.solve(d_matrix, b)
    return beta[0]

@jit
def RR_regular(z_0, Z, h):
    a_0 = z_0[0]
    A = Z[:, 0] - a_0
    K = vmap(partial(k, z_2=z_0, h=h))(Z)
    partial_K = vmap(partial(partial_dk, z_2=z_0, h=h))(Z)
    sum_K = jnp.sum(K)
    sum_K_A = jnp.sum(K * A)
    sum_K_A2 = jnp.sum(K * A**2)
    d_matrix = jnp.array([[sum_K, sum_K_A], [sum_K_A, sum_K_A2]])
    b = jnp.array([jnp.sum(partial_K), jnp.sum(K)+jnp.sum(partial_K*A)])
    beta = jnp.linalg.solve(d_matrix, b)
    return beta[0]

@jit
def RR_special_x(z_0, Z, h):
    a_0 = z_0[0]
    x_0 = z_0[1]
    A = Z[:, 0] - a_0
    X = Z[:, 1] - x_0
    K = vmap(partial(k, z_2=z_0, h=h))(Z)
    sum_K = jnp.sum(K)
    sum_K_A = jnp.sum(K * A)
    sum_K_A2 = jnp.sum(K * A**2)
    sum_K_X = jnp.sum(K * X)
    sum_K_X2 = jnp.sum(K * X**2)
    sum_K_AX = jnp.sum(K * A * X)
    d_matrix = jnp.array([[sum_K, sum_K_A, sum_K_X], [sum_K_A, sum_K_A2, sum_K_AX], [sum_K_X, sum_K_AX, sum_K_X2]])
    b = jnp.array([0, jnp.sum(K), 0])
    beta = jnp.linalg.solve(d_matrix, b)
    return beta[0]

@jit
def RR_normal_x(z_0, Z, h):
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


n = 10000
key = random.PRNGKey(0)
mean = jnp.zeros(2)
cov = jnp.array([[1,0.5], [0.5, 1]])
Z = random.multivariate_normal(key, mean, cov, (n,))
A, X = Z[:, 0], Z[:, 1]

h = 0.5
alpha_true = vmap(alpha_0, (0, 0))(A, X)
alpha_special = vmap(RR_special, (0, None, None))(Z, Z, h)
alpha_special_x = vmap(RR_special_x, (0, None, None))(Z, Z, h)
alpha_regular = vmap(RR_regular, (0, None, None))(Z, Z, h)
alpha_regular_x = vmap(RR_regular_x, (0, None, None))(Z, Z, h)

MSE_special = jnp.mean((alpha_special - alpha_true)**2)
MSE_regular = jnp.mean((alpha_regular - alpha_true)**2)
MSE_special_x = jnp.mean((alpha_special_x - alpha_true)**2)
MSE_regular_x = jnp.mean((alpha_regular_x - alpha_true)**2)
MSE_diff = jnp.mean((alpha_special - alpha_normal)**2)
MSE_diff_x = jnp.mean((alpha_special_x - alpha_normal_x)**2)

print("MSE_special", MSE_special)
print("MSE_regular", MSE_regular)
print("MSE_special_x", MSE_special_x)
print("MSE_regular_x", MSE_regular_x)
print("MSE_diff", MSE_diff)
print("MSE_diff_x", MSE_diff_x)
print("var_alpha" , jnp.var(alpha_true))
