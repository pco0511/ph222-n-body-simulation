import tqdm
import time

import jax
import jax.numpy as jnp
import numpy as np
from hamiltonian import Eular, RK45, ForestRuth
import matplotlib.pyplot as plt

# example of coupled oscillator

N = 4
# m = np.array([1, 1, 1, 1])
# k = np.array([2, 1, 1, 1, 2])

# def kinetic(p):
#     return np.sum(p**2 / (2 * m))

# def velocity(p):
#     return p / m

# def potential(q):
#     return (1/2) * k[0] * q[0] ** 2 + np.sum((1/2) * k[1:N] * (q[:-1] - q[1:]) ** 2) + (1/2) * k[N] * q[N-1] ** 2

# def force(q):
#     return np.array([
#         -k[0] * q[0] - k[1] * (q[0] - q[1]),
#         k[1] * (q[0] - q[1]) - k[2] * (q[1] - q[2]),
#         k[2] * (q[1] - q[2]) - k[3] * (q[2] - q[3]),
#         k[3] * (q[2] - q[3]) - k[4] * q[3]
#     ])

m = jnp.array([1, 1, 1, 1])
k = jnp.array([2, 1, 1, 1, 2])
def kinetic(p):
    return jnp.sum(p ** 2 / 2 * m)

def potential(q):
    interaction = np.sum((1/2) * k[1:N] * (q[:-1] - q[1:]) ** 2)
    boundary = (1/2) * k[0] * q[0] ** 2 + (1/2) * k[N] * q[N-1] ** 2
    return interaction + boundary

q0 = jnp.array([0.1, 0.2, 0, -0.2])
p0 = jnp.array([0.05, 0.07, -0.01, -0.03])


integrator = ForestRuth(
    kinetic,
    potential,
    0,
    q0,
    p0,
    4,
    0.0001
)
# integrator._velocity = velocity
# integrator._force = force

n_steps = 400000


t = np.empty((n_steps, ))
q = np.empty((n_steps, N))
p = np.empty((n_steps, N))
energy = np.empty((n_steps, ))


for i in tqdm.trange(n_steps):
    integrator.step()
    t[i] = integrator.t
    p[i,:] = np.array(integrator.p)
    q[i,:] = np.array(integrator.q)
    energy[i] = integrator.energy

print(energy[0])
print(max(energy-energy[0]))
plt.figure(figsize=(8, 5))
plt.plot(t, energy, label="energy")
# for i in range(N):
#     plt.plot(t, q[:,i], label=f"coordinate[{i}]")
plt.legend()
plt.tight_layout()
plt.show()