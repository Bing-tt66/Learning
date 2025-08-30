import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 最近邻跃迁能 (eV)
t2 = 0.1  # 次近邻跃迁强度 (eV)
phi = np.pi / 2  # Haldane相位

def haldane_hamiltonian(kx, ky, m=0.1):
    # 最近邻矢量 (从A到B)
    d1 = a_0 * np.array([0, 1])
    d2 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    d3 = a_0 * np.array([-np.sqrt(3) / 2, -1 / 2])

    # 计算最近邻跃迁项
    h = 0
    for d in [d1, d2, d3]:
        h += np.exp(1j * (kx * d[0] + ky * d[1]))

    # 次近邻矢量 (从A到A或B到B)
    # 这些是连接同一子晶格上原子的矢量
    a1 = d2 - d3
    a2 = d3 - d1
    a3 = d1 - d2

    # 计算次近邻跃迁项
    # 对于A子晶格，相位为+φ
    h2_A = 0
    for a in [a1, a2, a3]:
        h2_A += np.exp(1j * (kx * a[0] + ky * a[1]) + 1j * phi)
        h2_A += np.exp(1j * (kx * (-a[0]) + ky * (-a[1])) - 1j * phi)

    # 对于B子晶格，相位为-φ
    h2_B = 0
    for a in [a1, a2, a3]:
        h2_B += np.exp(1j * (kx * a[0] + ky * a[1]) - 1j * phi)
        h2_B += np.exp(1j * (kx * (-a[0]) + ky * (-a[1])) + 1j * phi)

    H = np.array([
        [m + t2 * h2_A, V_ppi_0 * h],
        [V_ppi_0 * np.conj(h), -m + t2 * h2_B]
    ])
    return H

def calculate_chern_number(n=100, m=0.1):
    # k点网格 - 使用更大的布里渊区范围以包含狄拉克点
    kx = np.linspace(-4 * np.pi / (3 * a_0), 4 * np.pi / (3 * a_0), n)
    ky = np.linspace(-4 * np.pi / (3 * a_0), 4 * np.pi / (3 * a_0), n)
    dk = kx[1] - kx[0]

    # 存储所有k点的波函数
    eigenvectors = np.zeros((n, n, 2, 2), dtype=complex)

    # 计算所有k点的本征态
    for i in range(n):
        for j in range(n):
            H = haldane_hamiltonian(kx[i], ky[j], m)
            eigvals, eigvecs = np.linalg.eigh(H)
            # 按能量排序
            idx = np.argsort(eigvals)
            eigenvectors[i, j, :, 0] = eigvecs[:, idx[0]]  # 价带
            eigenvectors[i, j, :, 1] = eigvecs[:, idx[1]]  # 导带

    # 计算每个能带的陈数
    chern_numbers = np.zeros(2)
    berry_curvature = np.zeros((n, n, 2))

    for band in range(2):
        total_flux = 0.0
        for i in range(n):
            for j in range(n):
                # 周期性边界条件
                i_next = (i + 1) % n
                j_next = (j + 1) % n

                # 获取四个点的波函数
                u00 = eigenvectors[i, j, :, band]
                u10 = eigenvectors[i_next, j, :, band]
                u01 = eigenvectors[i, j_next, :, band]
                u11 = eigenvectors[i_next, j_next, :, band]

                # 计算U(1)联络
                U1 = np.vdot(u00, u10)
                U2 = np.vdot(u10, u11)
                U3 = np.vdot(u11, u01)
                U4 = np.vdot(u01, u00)

                # 计算小方格的通量
                product = U1 * U2 * U3 * U4
                # 避免数值误差导致的问题
                if np.abs(product) < 1e-10:
                    flux = 0
                else:
                    flux = np.angle(product)
                total_flux += flux
                berry_curvature[i, j, band] = flux

        chern_numbers[band] = total_flux / (2 * np.pi)

    return chern_numbers, berry_curvature, kx, ky

# 计算陈数和贝利曲率
n = 100
chern_numbers, berry_curvature, kx, ky = calculate_chern_number(n)
print(f"价带陈数: {chern_numbers[0]:.6f}")
print(f"导带陈数: {chern_numbers[1]:.6f}")

# 创建k点网格
KX, KY = np.meshgrid(kx, ky, indexing='ij')

# 使用coolwarm颜色映射的可视化
fig = plt.figure(figsize=(16, 12))

# 价带贝利曲率 - 等高线图
ax1 = fig.add_subplot(221)
# 使用coolwarm颜色映射，调整透明度
contour1 = ax1.contourf(KX, KY, berry_curvature[:, :, 0], 100, cmap=cm.coolwarm, alpha=0.8)
ax1.set_title(f'Valence Band Berry Curvature (Chern number = {chern_numbers[0]:.3f})', fontsize=12)
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
fig.colorbar(contour1, ax=ax1)

# 导带贝利曲率 - 等高线图
ax2 = fig.add_subplot(222)
contour2 = ax2.contourf(KX, KY, berry_curvature[:, :, 1], 100, cmap=cm.coolwarm, alpha=0.8)
ax2.set_title(f'Conduction Band Berry Curvature (Chern number = {chern_numbers[1]:.3f})', fontsize=12)
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
fig.colorbar(contour2, ax=ax2)

# 价带贝利曲率 - 3D曲面图
ax3 = fig.add_subplot(223, projection='3d')
# 使用更密集的网格采样以提高可视化效果，使用coolwarm颜色映射
surf3 = ax3.plot_surface(KX[::2, ::2], KY[::2, ::2], berry_curvature[::2, ::2, 0],
                        cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.7)
ax3.set_title('Valence Band Berry Curvature (3D View)', fontsize=12)
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.set_zlabel('Berry Curvature')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

# 导带贝利曲率 - 3D曲面图
ax4 = fig.add_subplot(224, projection='3d')
surf4 = ax4.plot_surface(KX[::2, ::2], KY[::2, ::2], berry_curvature[::2, ::2, 1],
                        cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.7)
ax4.set_title('Conduction Band Berry Curvature (3D View)', fontsize=12)
ax4.set_xlabel('kx')
ax4.set_ylabel('ky')
ax4.set_zlabel('Berry Curvature')
fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

# 绘制贝利曲率的绝对值对数图，使用coolwarm颜色映射
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))

# 价带贝利曲率绝对值对数图
log_berry_valence = np.log10(np.abs(berry_curvature[:, :, 0]) + 1e-10)
im1 = ax5.imshow(log_berry_valence.T, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
                origin='lower', cmap=cm.coolwarm, aspect='auto', alpha=0.8)
ax5.set_title('Valence Band Log10(|Berry Curvature|)', fontsize=12)
ax5.set_xlabel('kx')
ax5.set_ylabel('ky')
fig2.colorbar(im1, ax=ax5)

# 导带贝利曲率绝对值对数图
log_berry_conduction = np.log10(np.abs(berry_curvature[:, :, 1]) + 1e-10)
im2 = ax6.imshow(log_berry_conduction.T, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
                origin='lower', cmap=cm.coolwarm, aspect='auto', alpha=0.8)
ax6.set_title('Conduction Band Log10(|Berry Curvature|)', fontsize=12)
ax6.set_xlabel('kx')
ax6.set_ylabel('ky')
fig2.colorbar(im2, ax=ax6)

plt.tight_layout()
plt.show()