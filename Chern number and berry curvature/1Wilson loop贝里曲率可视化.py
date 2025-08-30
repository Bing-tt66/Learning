import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)

def hamiltonian(kx, ky, m=0.1):
    # 三个最近邻矢量
    d1 = np.array([0, a_0])
    d2 = np.array([a_0 * np.sqrt(3) / 2,-a_0 / 2])
    d3 = np.array([-a_0 * np.sqrt(3) / 2,-a_0 / 2])

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
    chern_number = np.zeros(2)
    berry_curvature = np.zeros((n, n, 2),dtype=complex)
    delta = 2 * np.pi/ a_0 / n
    kx = np.linspace(-np.pi / a_0, np.pi / a_0, n)
    ky = np.linspace(-np.pi / a_0, np.pi / a_0, n)
    vector = np.zeros((n, n, 2, 2),dtype=complex)
    for band in range(2):
        for i in range(n):
            for j in range(n):
                H = hamiltonian(kx[i], ky[j], m=0.1)
                eigenvalue, eigenvector = np.linalg.eigh(H)
                vector[i, j,:,band] = eigenvector[:, np.argsort(np.real(eigenvalue))[band]]

    for band in range(2):
        total_flux = 0
        for i in range(n):
            for j in range(n):
                i_next = (i + 1) % n
                j_next = (j + 1) % n
                u00 = vector[i, j, :,band]
                u10 = vector[i_next, j,:, band]
                u11 = vector[i_next, j_next,:, band]
                u01 = vector[i, j_next, :,band]

                l1 = np.vdot(u00, u10)
                l2 = np.vdot(u10, u11)
                l3 = np.vdot(u11, u01)
                l4 = np.vdot(u01, u00)

                product =l1 * l2 * l3 * l4
                flux = np.angle(product)
                total_flux += flux
                berry_curvature[i, j, band] = flux / (delta * delta)
        chern_number[band] = total_flux / (2 * np.pi)
    return chern_number,berry_curvature

n = 100
chern_number,berry_curvature = calculate(n)
print(f"价带陈数: {chern_number[0]:.11f}")
print(f"导带陈数: {chern_number[1]:.11f}")

# 创建k空间网格
kx = np.linspace(-np.pi / a_0, np.pi / a_0, n)
ky = np.linspace(-np.pi / a_0, np.pi / a_0, n)
KX, KY = np.meshgrid(kx, ky)

# 提取能带的贝里曲率（取实部）
berry_curvature_valence = np.real(berry_curvature[:, :, 0])  # 价带
berry_curvature_conduction = np.real(berry_curvature[:, :, 1])  # 导带

# 创建图形 - 使用GridSpec调整子图大小
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

# 1. 3D 价带贝里曲率
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
surf1 = ax1.plot_surface(KX, KY, berry_curvature_valence, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True, alpha=0.8, rcount=n, ccount=n)
ax1.set_xlabel('kx (1/nm)')
ax1.set_ylabel('ky (1/nm)')
ax1.set_zlabel('Berry Curvature')
ax1.set_title('Valence Band Berry Curvature (3D)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Berry Curvature')

# 2. 3D 导带贝里曲率
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
surf2 = ax2.plot_surface(KX, KY, berry_curvature_conduction, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True, alpha=0.8, rcount=n, ccount=n)
ax2.set_xlabel('kx (1/nm)')
ax2.set_ylabel('ky (1/nm)')
ax2.set_zlabel('Berry Curvature')
ax2.set_title('Conduction Band Berry Curvature (3D)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Berry Curvature')

# 3. 2D 价带等高线图
ax3 = fig.add_subplot(gs[1, 0])
contour1 = ax3.contourf(KX, KY, berry_curvature_valence, 50, cmap='RdBu_r')
ax3.set_xlabel('kx (1/nm)')
ax3.set_ylabel('ky (1/nm)')
ax3.set_title('Valence Band Berry Curvature (2D)')
fig.colorbar(contour1, ax=ax3, label='Berry Curvature')

# 4. 2D 导带等高线图
ax4 = fig.add_subplot(gs[1, 1])
contour2 = ax4.contourf(KX, KY, berry_curvature_conduction, 50, cmap='RdBu_r')
ax4.set_xlabel('kx (1/nm)')
ax4.set_ylabel('ky (1/nm)')
ax4.set_title('Conduction Band Berry Curvature (2D)')
fig.colorbar(contour2, ax=ax4, label='Berry Curvature')

plt.tight_layout()
plt.show()