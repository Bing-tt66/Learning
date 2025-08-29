import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.linalg import eigh

# 物理参数
M = 0.1
t1 = 1.0
t2 = 0.2
phi_val = 0.2 * np.pi
a = 1.0

# 晶格矢量
a1 = np.array([0, a])
a2 = np.array([-np.sqrt(3) / 2 * a, -a / 2])
a3 = np.array([np.sqrt(3) / 2 * a, -a / 2])

# 倒格子矢量
b1 = a2 - a3
b2 = a3 - a1
b3 = a1 - a2

# 泡利矩阵
sigma0 = np.eye(2)
sigma1 = np.array([[0, 1], [1, 0]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma3 = np.array([[1, 0], [0, -1]])


def hamiltonian(kx, ky):
    k = np.array([kx, ky])

    # 计算哈密顿量各项
    epsilon = 2 * t2 * np.cos(phi_val) * (np.cos(np.dot(k, b1)) +
                                          np.cos(np.dot(k, b2)) +
                                          np.cos(np.dot(k, b3)))

    dx = t1 * (np.cos(np.dot(k, a1)) +
               np.cos(np.dot(k, a2)) +
               np.cos(np.dot(k, a3)))

    dy = t1 * (np.sin(np.dot(k, a1)) +
               np.sin(np.dot(k, a2)) +
               np.sin(np.dot(k, a3)))

    dz = M - 2 * t2 * np.sin(phi_val) * (np.sin(np.dot(k, b1)) +
                                         np.sin(np.dot(k, b2)) +
                                         np.sin(np.dot(k, b3)))

    # 构建哈密顿量
    H = epsilon * sigma0 + dx * sigma1 + dy * sigma2 + dz * sigma3
    return H


# 设置k空间网格
num = 100
A1, B1 = -np.pi, np.pi  # kx范围
A2, B2 = -np.pi, np.pi  # ky范围

deltakx = (B1 - A1) / num
deltaky = (B2 - A2) / num

kx_points = np.linspace(A1, B1 - deltakx, num)
ky_points = np.linspace(A2, B2 - deltaky, num)
KX, KY = np.meshgrid(kx_points, ky_points)

# 计算本征值和本征态
eigenvalues = np.zeros((num, num, 2))
eigenvectors = np.zeros((num, num, 2, 2), dtype=complex)

for i in range(num):
    for j in range(num):
        H = hamiltonian(kx_points[i], ky_points[j])
        eigvals, eigvecs = eigh(H)
        eigenvalues[i, j] = eigvals
        eigenvectors[i, j] = eigvecs.T  # 转置以便按行访问本征矢

# 选择能带计算 (0或1)
band = 0

# 提取选定能带的波函数
vec = eigenvectors[:, :, band, :]

# 计算略偏离kx的波函数
vec_delta_kx = np.zeros((num, num, 2), dtype=complex)
for i in range(num):
    for j in range(num):
        kx_shifted = kx_points[i] + deltakx
        # 处理周期性边界条件
        if kx_shifted > B1:
            kx_shifted = A1 + (kx_shifted - B1)
        H_shifted = hamiltonian(kx_shifted, ky_points[j])
        eigvals, eigvecs = eigh(H_shifted)
        vec_delta_kx[i, j] = eigvecs[:, band]  # 选择能带

# 计算略偏离ky的波函数
vec_delta_ky = np.zeros((num, num, 2), dtype=complex)
for i in range(num):
    for j in range(num):
        ky_shifted = ky_points[j] + deltaky
        # 处理周期性边界条件
        if ky_shifted > B2:
            ky_shifted = A2 + (ky_shifted - B2)
        H_shifted = hamiltonian(kx_points[i], ky_shifted)
        eigvals, eigvecs = eigh(H_shifted)
        vec_delta_ky[i, j] = eigvecs[:, band]  # 选择能带

# 计算略偏离kx和ky的波函数
vec_delta_kx_ky = np.zeros((num, num, 2), dtype=complex)
for i in range(num):
    for j in range(num):
        kx_shifted = kx_points[i] + deltakx
        ky_shifted = ky_points[j] + deltaky
        # 处理周期性边界条件
        if kx_shifted > B1:
            kx_shifted = A1 + (kx_shifted - B1)
        if ky_shifted > B2:
            ky_shifted = A2 + (ky_shifted - B2)
        H_shifted = hamiltonian(kx_shifted, ky_shifted)
        eigvals, eigvecs = eigh(H_shifted)
        vec_delta_kx_ky[i, j] = eigvecs[:, band]  # 选择能带

# Wilson loop计算
line1 = np.zeros((num, num), dtype=complex)
line2 = np.zeros((num, num), dtype=complex)
line3 = np.zeros((num, num), dtype=complex)
line4 = np.zeros((num, num), dtype=complex)

for i in range(num):
    for j in range(num):
        line1[i, j] = np.dot(np.conj(vec[i, j]), vec_delta_kx[i, j])
        line2[i, j] = np.dot(np.conj(vec_delta_kx[i, j]), vec_delta_kx_ky[i, j])
        line3[i, j] = np.dot(np.conj(vec_delta_kx_ky[i, j]), vec_delta_ky[i, j])
        line4[i, j] = np.dot(np.conj(vec_delta_ky[i, j]), vec[i, j])

# 贝里曲率计算
berry_curvature = np.zeros((num, num))
for i in range(num):
    for j in range(num):
        product = line1[i, j] * line2[i, j] * line3[i, j] * line4[i, j]
        berry_curvature[i, j] = -np.angle(product)

# 陈数计算
chern_number = np.sum(berry_curvature) / (2 * np.pi)
print(f"Chern number = {chern_number}")

# 准备数据用于绘图
kx_grid, ky_grid = np.meshgrid(kx_points, ky_points)
berry_data = np.zeros((num * num, 3))
idx = 0
for i in range(num):
    for j in range(num):
        berry_data[idx] = [kx_points[i], ky_points[j], berry_curvature[i, j]]
        idx += 1

# 创建3D图
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制贝里曲率曲面
surf = ax.plot_trisurf(berry_data[:, 0], berry_data[:, 1], berry_data[:, 2],
                       cmap=cm.coolwarm, linewidth=0, alpha=0.8)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Berry Curvature')

# 设置标签和标题
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('Berry Curvature')
ax.set_title('3D Berry Curvature of Haldane Model', fontsize=16)

# 调整视角
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# 创建等高线图
plt.figure(figsize=(10, 8))
contour = plt.contourf(kx_grid, ky_grid, berry_curvature, 50, cmap='rainbow')
plt.colorbar(contour, label='Berry Curvature')
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('Contour Plot of Berry Curvature')
plt.show()

# 创建密度图
plt.figure(figsize=(10, 8))
im = plt.imshow(berry_curvature, extent=[A1, B1, A2, B2],
                origin='lower', cmap='rainbow', aspect='auto')
plt.colorbar(im, label='Berry Curvature')
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('Density Plot of Berry Curvature')
plt.show()