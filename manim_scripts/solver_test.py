import tqdm
import time

import jax
import jax.numpy as jnp
import numpy as np
from hamiltonian_solver import Eular, RK23, RK45, ForestRuth
import matplotlib.pyplot as plt

# example of coupled oscillator

N = 4
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
    0.001
)

n_steps = 10_000_000

t = np.empty((n_steps, ))
q = np.empty((n_steps, N))
p = np.empty((n_steps, N))
energy = np.empty((n_steps, ))

# for i in tqdm.trange(n_steps):
#     integrator.step()
#     t[i] = integrator.t
#     p[i,:] = np.array(integrator.p)
#     q[i,:] = np.array(integrator.q)
#     energy[i] = integrator.energy

start = time.time()
qs, ps, ts = integrator.solve(n_steps)
t = np.array(ts)
q = np.array(qs)
p = np.array(ps)
energy = np.array(jax.vmap(kinetic)(ps) + jax.vmap(potential)(qs))
end = time.time()
print(f"elapsed: {end - start} seconds")
print(energy[0])
print(max(energy-energy[0]))
plt.figure(figsize=(8, 5))
plt.plot(t, energy, label="energy")
# for i in range(N):
#     plt.plot(t, q[:,i], label=f"coordinate[{i}]")
plt.legend()
plt.tight_layout()
plt.show()