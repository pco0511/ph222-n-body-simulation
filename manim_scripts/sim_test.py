import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt

# 시뮬레이션 파라미터
num_particles = 1000                # 입자 수
box_size = jnp.array([10.0, 10.0])  # 상자 크기 (가로, 세로)
dt = 0.01                           # 시간 간격
num_steps = 100000                  # 시뮬레이션 스텝 수
record_interval = 5                # 시각화를 위해 기록할 간격


# 초기 위치와 속도 랜덤 설정
def initialize(num_particles, box_size, key):
    positions = box_size * jax.random.uniform(key, (num_particles, 2))
    velocities = jax.random.uniform(key, (num_particles, 2)) * 2 - 1  # 속도 범위 [-1, 1]
    return positions, velocities

# 위치 업데이트 함수
@jit
def update_positions(positions, velocities, dt):
    return positions + velocities * dt

# 속도 업데이트 함수 (경계 충돌 처리)
@jit
def update_velocities(positions, velocities, box_size):
    # 좌표별로 벽과 충돌하는지 확인
    collided_lower = positions < 0.0
    collided_upper = positions > box_size

    # 속도 반전
    velocities = jnp.where(collided_lower | collided_upper, -velocities, velocities)

    # 위치 보정 (벽을 넘어가지 않도록)
    positions = jnp.where(collided_lower, 0.0, positions)
    positions = jnp.where(collided_upper, box_size, positions)

    return velocities, positions

# 시뮬레이션 스텝 함수
@jit
def simulation_step(state, _):
    positions, velocities, box_size, dt = state
    positions = update_positions(positions, velocities, dt)
    velocities, positions = update_velocities(positions, velocities, box_size)
    return (positions, velocities, box_size, dt), positions  # 반환값: (새 상태, 기록할 값)

# 메인 시뮬레이션 루프 (jax.lax.scan 사용)
@partial(jit, static_argnums=1)  # num_steps를 static_argnums로 지정
def run_simulation_scan(initial_state, num_steps):
    # jax.lax.scan은 (state, carry) -> (new_state, output)을 반복
    final_state, trajectory = jax.lax.scan(simulation_step, initial_state, None, length=num_steps)
    return trajectory  # 모든 스텝의 positions 반환

# 시뮬레이션 실행
def run_simulation(num_particles, box_size, dt, num_steps, record_interval):
    key = jax.random.PRNGKey(0)
    key, subkey1 = jax.random.split(key)
    positions, velocities = initialize(num_particles, box_size, subkey1)
    initial_state = (positions, velocities, box_size, dt)
    
    # 시뮬레이션 수행
    trajectory = run_simulation_scan(initial_state, num_steps)
    
    # 시각화를 위해 일정 간격으로 데이터 추출
    trajectory = trajectory[::record_interval]
    
    return trajectory

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
