import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)
a1 = np.array([np.sqrt(3) * a / 2, a / 2])
a2 = np.array([np.sqrt(3) * a / 2, -a / 2])

# 倒空间基矢
b1 = 2 * np.pi * (np.array([a2[1], -a2[0]])) / (np.cross(a1, a2))
b2 = 2 * np.pi * (np.array([-a1[1], a1[0]])) / (np.cross(a1, a2))

# 高对称点（倒空间坐标）
Gamma = np.array([0, 0])  # Γ点
K = (b1 + 2 * b2) / 3  # K点
M = (b2 + b1) / 2  # M点
K_prime = (2 * b1 + b2) / 3  # K'点

# 设置k空间网格
n_k = 100  # 增加点数使曲面更平滑
kx = np.linspace(-np.pi / a, np.pi / a, n_k)
ky = np.linspace(-np.pi / a, np.pi / a, n_k)
KX, KY = np.meshgrid(kx, ky)

energies_3d = np.zeros((2, n_k, n_k))


# 紧束缚模型哈密顿量
def graphene_hamiltonian(k):
    kx, ky = k
    # 三个最近邻矢量
    d1 = np.array([0, a_0])
    d2 = np.array([a_0 * np.sqrt(3) / 2, -a_0  / 2])
    d3 = np.array([-a_0 * np.sqrt(3) / 2, -a_0  / 2])

    # 计算跃迁项
    h = 0
    for d in [d1, d2, d3]:
        h += np.exp(1j * (kx * d[0] + ky * d[1]))

    H = np.array([
        [0, V_ppi_0 * h],
        [V_ppi_0 * np.conj(h), 0]
    ])
    return H


# 计算每个(kx,ky)点的能量
for i in range(n_k):
    for j in range(n_k):
        k = np.array([KX[i, j], KY[i, j]])
        H = graphene_hamiltonian(k)
        eigenvalues = np.linalg.eigvalsh(H)
        energies_3d[:, i, j] = eigenvalues

# 创建自定义颜色映射
colors_positive = [(0, '#2A00FF'), (0.5, '#00A0FF'), (1, '#00F9FF')]  # 蓝-青
colors_negative = [(0, '#FF2A00'), (0.5, '#FFA000'), (1, '#FFF800')]  # 红-黄
cmap_pos = LinearSegmentedColormap.from_list('positive_band', colors_positive)
cmap_neg = LinearSegmentedColormap.from_list('negative_band', colors_negative)

# 计算布里渊区六边形边界
r = 4 * np.pi / (3 * a)  # 布里渊区顶点半径
angles = np.linspace(0, 2 * np.pi, 7)  # 7个点形成闭合六边形
hexagon_x = r * np.cos(angles + np.pi / 6)  # 旋转30度对齐
hexagon_y = r * np.sin(angles + np.pi / 6)

# 高对称点路径（Γ→K→M→Γ→K'→M→Γ）
path_points = [
    Gamma, K, M, Gamma, K_prime, M, Gamma
]
path_kx = [p[0] for p in path_points]
path_ky = [p[1] for p in path_points]

# 计算路径上的能量
path_energies = []
for k in path_points:
    H = graphene_hamiltonian(k)
    eigenvalues = np.linalg.eigvalsh(H)
    path_energies.append(eigenvalues)

# 创建3D图
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

# 绘制能带曲面 (使用不同的颜色映射)
for band in range(2):
    # 选择颜色映射：价带用红-黄，导带用蓝-青
    cmap = cmap_neg if band == 0 else cmap_pos

    # 归一化颜色范围
    vmin, vmax = np.min(energies_3d[band]), np.max(energies_3d[band])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 绘制曲面
    surf = ax.plot_surface(
        KX, KY, energies_3d[band],
        cmap=cmap,
        norm=norm,
        edgecolor='none',
        alpha=0.9,
        rstride=2,
        cstride=2,
        antialiased=True,
        zorder=band
    )

# 绘制布里渊区边界 (在能量最小值平面)
min_energy = np.min(energies_3d) - 0.5
ax.plot(hexagon_x, hexagon_y, min_energy,
        color='black', linewidth=2.5, linestyle='-', zorder=10)
ax.text(hexagon_x[0], hexagon_y[0], min_energy, 'BZ',
        color='black', fontsize=12, ha='right', va='top')

# 绘制高对称点路径
path_energies = np.array(path_energies)
ax.plot(path_kx, path_ky, path_energies[:, 0],
        'r-', linewidth=3, zorder=15, label='Valence Band Path')
ax.plot(path_kx, path_ky, path_energies[:, 1],
        'b-', linewidth=3, zorder=15, label='Conduction Band Path')

# 标注高对称点
sym_points = {
    'Γ': Gamma,
    'K': K,
    "K'": K_prime,
    'M': M
}
for label, point in sym_points.items():
    H = graphene_hamiltonian(point)
    energy = np.linalg.eigvalsh(H)[0]  # 取价带能量标注

    # 绘制点
    ax.scatter([point[0]], [point[1]], [energy],
               color='purple', s=100, depthshade=False, zorder=20)

    # 添加标签 (带背景)
    ax.text(point[0], point[1], energy + 0.3, label,
            fontsize=14, weight='bold', color='white',
            bbox=dict(facecolor='purple', alpha=0.8, boxstyle='round,pad=0.2'),
            zorder=25, ha='center', va='center')

# 添加图例和标签
ax.set_xlabel('$k_x$ (nm$^{-1}$)', fontsize=14, labelpad=15)
ax.set_ylabel('$k_y$ (nm$^{-1}$)', fontsize=14, labelpad=15)
ax.set_zlabel('Energy (eV)', fontsize=14, labelpad=15)
ax.set_title('3D Band Structure of Monolayer Graphene', fontsize=18, pad=20)

# 设置视角和范围
ax.view_init(elev=28, azim=-45)
ax.set_xlim([-np.pi / a, np.pi / a])
ax.set_ylim([-np.pi / a, np.pi / a])
ax.set_zlim([min_energy, np.max(energies_3d) + 0.5])

# 添加网格和色标
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle=':', alpha=0.6)

# 添加自定义图例
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='r', lw=3, label='Valence Band Path'),
    Line2D([0], [0], color='b', lw=3, label='Conduction Band Path'),
    Line2D([0], [0], color='black', lw=2, label='Brillouin Zone'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
           markersize=10, label='High Symmetry Points')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)
#plt.savefig('graphene_3d_band_structure.png', dpi=300)
plt.show()