import numpy as np
import matplotlib.pyplot as plt

# 전기장 함수 예제 (여기서는 간단한 전기장 함수 사용)
def electric_field(x, y):
    # 예: 원형 전기장 예제
    Ex = -y / (x**2 + y**2 + 1e-9)
    Ey = x / (x**2 + y**2 + 1e-9)
    return Ex, Ey

# 전기력선 시작점 생성 (원형 배치)
def generate_starting_points(num_points=20, radius=1.5):
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    for theta in angle:
        x_start = radius * np.cos(theta)
        y_start = radius * np.sin(theta)
        points.append((x_start, y_start))
    return points

# 그리드 설정
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 전기장 계산
Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
for i in range(len(x)):
    for j in range(len(y)):
        Ex[i, j], Ey[i, j] = electric_field(X[i, j], Y[i, j])

# 전기력선 시작점 생성
start_points = generate_starting_points()

# 전기력선 그리기
fig, ax = plt.subplots(figsize=(6, 6))
ax.streamplot(X, Y, Ex, Ey, color="blue", linewidth=0.8, start_points=start_points)

plt.title("Electric Field Lines from Given Field")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
