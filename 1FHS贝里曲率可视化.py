import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)


def hamiltonian(kx, ky, m=0.1):
    # 三个最近邻矢量
    d1 = np.array([0, a_0])
    d2 = np.array([a_0 * np.sqrt(3) / 2, -a_0 / 2])
    d3 = np.array([-a_0 * np.sqrt(3) / 2, -a_0 / 2])

    # 计算跃迁项
    h = 0
    for d in [d1, d2, d3]:
        h += np.exp(1j * (kx * d[0] + ky * d[1]))

    H = np.array([
        [m, V_ppi_0 * h],
        [V_ppi_0 * np.conj(h), -m]
    ])
    return H


def calculate(n=100):
    delta = 2 * np.pi / a_0 / n
    chern_number = 0
    berry_curvature = np.zeros((n, n, 2), dtype=complex)

    # 预先计算k点网格
    k_points = np.linspace(-np.pi / a_0, np.pi / a_0, n)

    for band in range(2):
        for i, kx in enumerate(k_points):
            for j, ky in enumerate(k_points):
                H = hamiltonian(kx, ky)
                eigenvalue, eigenvector = np.linalg.eig(H)
                vector = eigenvector[:, np.argsort(np.real(eigenvalue))[band]]  # 价带波函数

                H_delta_kx = hamiltonian(kx + delta, ky)
                eigenvalue, eigenvector = np.linalg.eig(H_delta_kx)
                vector_delta_kx = eigenvector[:, np.argsort(np.real(eigenvalue))[band]]

                H_delta_ky = hamiltonian(kx, ky + delta)
                eigenvalue, eigenvector = np.linalg.eig(H_delta_ky)
                vector_delta_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[band]]

                H_delta_kx_ky = hamiltonian(kx + delta, ky + delta)
                eigenvalue, eigenvector = np.linalg.eig(H_delta_kx_ky)
                vector_delta_kx_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[band]]

                ux = np.vdot(vector, vector_delta_kx)
                ux = ux / abs(ux)

                uy = np.vdot(vector, vector_delta_ky)
                uy = uy / abs(uy)

                ux_y = np.vdot(vector_delta_ky, vector_delta_kx_ky)
                ux_y = ux_y / abs(ux_y)

                uy_x = np.vdot(vector_delta_kx, vector_delta_kx_ky)
                uy_x = uy_x / abs(uy_x)

                F12 = np.log(ux * uy_x * (1 / ux_y) * (1 / uy))

                berry_curvature[i, j, band] = F12 / (delta * delta * 1j)
                chern_number += F12

    chern_number = chern_number / (2 * np.pi * 1j)
    return chern_number, berry_curvature


# 计算贝里曲率
n = 100  # 降低精度以加快计算速度，可根据需要调整
chern_number, berry_curvature = calculate(n)
print('Chern number = %.6f' % np.real(chern_number))

# 创建k空间网格
kx = np.linspace(-np.pi / a_0, np.pi / a_0, n)
ky = np.linspace(-np.pi / a_0, np.pi / a_0, n)
KX, KY = np.meshgrid(kx, ky)

# 提取能带的贝里曲率（取实部）
berry_curvature_valence = np.real(berry_curvature[:, :, 0])  # 价带
berry_curvature_conduction = np.real(berry_curvature[:, :, 1])  # 导带

# 创建图形
fig = plt.figure(figsize=(18, 12))

# 1. 3D 价带贝里曲率
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(KX, KY, berry_curvature_valence, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True, alpha=0.8, rcount=n, ccount=n)
ax1.set_xlabel('kx (1/nm)')
ax1.set_ylabel('ky (1/nm)')
ax1.set_zlabel('Berry Curvature')
ax1.set_title('Valence Band Berry Curvature (3D)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Berry Curvature')

# 2. 3D 导带贝里曲率
ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(KX, KY, berry_curvature_conduction, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True, alpha=0.8, rcount=n, ccount=n)
ax2.set_xlabel('kx (1/nm)')
ax2.set_ylabel('ky (1/nm)')
ax2.set_zlabel('Berry Curvature')
ax2.set_title('Conduction Band Berry Curvature (3D)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Berry Curvature')

# 3. 2D 价带等高线图
ax3 = fig.add_subplot(223)
contour1 = ax3.contourf(KX, KY, berry_curvature_valence, 50, cmap='RdBu_r')
ax3.set_xlabel('kx (1/nm)')
ax3.set_ylabel('ky (1/nm)')
ax3.set_title('Valence Band Berry Curvature (2D)')
fig.colorbar(contour1, ax=ax3, label='Berry Curvature')

# 4. 2D 导带等高线图
ax4 = fig.add_subplot(224)
contour2 = ax4.contourf(KX, KY, berry_curvature_conduction, 50, cmap='RdBu_r')
ax4.set_xlabel('kx (1/nm)')
ax4.set_ylabel('ky (1/nm)')
ax4.set_title('Conduction Band Berry Curvature (2D)')
fig.colorbar(contour2, ax=ax4, label='Berry Curvature')

plt.tight_layout()
plt.show()