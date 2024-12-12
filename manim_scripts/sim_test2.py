import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt

import hamiltonian_solver

# 시뮬레이션 파라미터
num_particles = 1000                # 입자 수
box_size = jnp.array([10.0, 10.0])  # 상자 크기 (가로, 세로)
dt = 0.01                           # 시간 간격
num_steps = 100000                  # 시뮬레이션 스텝 수
record_interval = 5                # 시각화를 위해 기록할 간격




# 시뮬레이션 실행
trajectory = run_simulation(num_particles, box_size, dt, num_steps, record_interval)

# 시각화
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, box_size[0])
ax.set_ylim(0, box_size[1])
ax.set_title('2D Free Particle Simulation in a Box')

for pos in trajectory:
    ax.clear()
    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    ax.set_title('2D Free Particle Simulation in a Box')
    ax.scatter(pos[:,0], pos[:,1], s=1)
    plt.pause(dt * record_interval)

plt.show()
