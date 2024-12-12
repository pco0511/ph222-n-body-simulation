import tqdm
import time

import jax
import jax.numpy as jnp
import numpy as np
from hamiltonian_solver import Eular, RK23, RK45, ForestRuth
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 0.1
m1 = 10
m2 = 1
m = jnp.array([m1, m2])
q0 = jnp.array([-0.25, 0, 2.5, 0])
P = m1 * m2 * jnp.sqrt(G / ((m1 + m2) * 2.75))
p0 = jnp.array([0, -P, 0, P])

def kinetic(p):
    p2 = p ** 2
    return jnp.sum((p2[0::2] + p2[1::2]) / (2 * m))

def potential(q):
    x = q[0::2]
    y = q[1::2]
    epsilon = 1e-12
    
    delta_xij2 = (x[jnp.newaxis, :] - x[:, jnp.newaxis]) ** 2
    delta_yij2 = (y[jnp.newaxis, :] - y[:, jnp.newaxis]) ** 2
    delta_rij2 = delta_xij2 + delta_yij2 + epsilon
    
    mimj = m[jnp.newaxis, :] * m[:, jnp.newaxis]
    return -G * jnp.sum(jnp.triu(mimj / jnp.sqrt(delta_rij2), k=1))


integrator = RK45(
    kinetic,
    potential,
    0,
    q0,
    p0,
    4,
    1e-2
)

mult = 80
n_steps = 1000 * 20 * mult

# print(q0)
# print(p0)
# print(kinetic(p0))
# print(potential(q0))

start = time.time()
qs, ps, ts = integrator.solve(n_steps)
end = time.time()


t = np.array(ts[0::mult])
q = np.array(qs[0::mult])
x1 = q[:, 0]
y1 = q[:, 1]
x2 = q[:, 2]
y2 = q[:, 3]
p = np.array(ps[0::mult])
px1 = p[:, 0]
py1 = p[:, 1]
px2 = p[:, 2]
py2 = p[:, 3]
energy = np.array(jax.vmap(kinetic)(ps) + jax.vmap(potential)(qs))
print(x1[0], y1[0], x2[0], y2[0])
print(f"elapsed: {end - start} seconds")
print(energy[0])
print(max(energy-energy[0]))




plt.figure(figsize=(8, 5))
plt.plot(ts, energy, label="energy")
# plt.plot(ts, qs[:, 0], label=f"x1")
plt.legend()
plt.tight_layout()
plt.show()






# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(-4, 4)  # 데이터 값의 범위에 맞게 설정
# ax.set_ylim(-4, 4)

# # 점 생성
# point1, = ax.plot([], [], 'ro', label="Point 1")  # 빨간 점
# point2, = ax.plot([], [], 'bo', label="Point 2")  # 파란 점
# ax.legend()

# def update(frame):
#     point1.set_data(x1[frame], y1[frame])
#     point2.set_data(x2[frame], y2[frame])
#     return point1, point2


# fps = 60
# interval = 1000 / fps  # 60fps에 맞는 간격(ms 단위)
# ani = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=True)

# # 애니메이션 실행
# plt.show()