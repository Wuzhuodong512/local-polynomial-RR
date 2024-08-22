import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax import jit, grad
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

# Define the functions
@jit
def p_0(a, x):
    mean1 = jnp.array([0, 0])
    cov1 = jnp.array([[1,0.5], [0.5, 1]])
    mean2 = jnp.array([2, 2])
    cov2 = jnp.array([[1,-0.5], [-0.5, 1]])
    p1 = multivariate_normal.pdf(jnp.array([a, x]), mean=mean1, cov = cov1)
    p2 = multivariate_normal.pdf(jnp.array([a, x]), mean=mean2, cov = cov2)
    p = 0.5 * p1 + 0.5 * p2
    return p

@jit
def alpha_0(a, x):
    dp_da = grad(p_0, argnums=0)(a, x)
    return -dp_da / p_0(a, x)

# Create a grid of values for a and x
a_values = np.linspace(-2, 4, 100)
x_values = np.linspace(-2, 4, 100)
a_grid, x_grid = np.meshgrid(a_values, x_values)

# Compute alpha_0 for each pair of a and x values
alpha_values = np.array([[alpha_0(a, x) for a in a_values] for x in x_values])

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(a_grid, x_grid, alpha_values, cmap='viridis')

ax.set_xlabel('a')
ax.set_ylabel('x')
ax.set_zlabel('alpha_0(a, x)')
ax.set_title('3D plot of alpha_0(a, x)')

plt.show()