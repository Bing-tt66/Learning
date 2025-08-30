import numpy as np
import matplotlib.pyplot as plt

# 参数设置
a = np.sqrt(2) + 1  # 大正方形边长
b = 1  # 小正方形边长
c = (a - b) / np.sqrt(2)  # 对角线键长
t = 1  # 小正方形内跃迁
t_prime = 1  # 中心原子到小正方形原子的跃迁
pi = np.pi


def hamiltonian(kx, ky):
    k_vec = np.array([kx, ky])

    # 最近邻矢量
    d12 = np.array([(a - b) / 2, (a - b) / 2])
    d13 = np.array([- (a - b) / 2, (a - b) / 2])
    d14 = np.array([- (a - b) / 2, - (a - b) / 2])
    d15 = np.array([(a - b) / 2, - (a - b) / 2])
    d23 = np.array([b, 0])
    d25 = np.array([0, b])
    d34 = np.array([0, b])
    d45 = np.array([- b, 0])

    # 跃迁项
    h12 = np.exp(1j * np.dot(d12, k_vec))
    h13 = np.exp(1j * np.dot(d13, k_vec))
    h14 = np.exp(1j * np.dot(d14, k_vec))
    h15 = np.exp(1j * np.dot(d15, k_vec))
    h23 = np.exp(1j * np.dot(d23, k_vec))
    h25 = np.exp(1j * np.dot(d25, k_vec))
    h34 = np.exp(1j * np.dot(d34, k_vec))
    h45 = np.exp(1j * np.dot(d45, k_vec))

    # Hamiltonian matrix
    H = np.array([
        [0, t_prime * h12, t_prime * h13, t_prime * h14, t_prime * h15],
        [t_prime * np.conj(h12), 0, t * h23, 0, t * h25],
        [t_prime * np.conj(h13), t * np.conj(h23), 0, t * h34, 0],
        [t_prime * np.conj(h14), 0, t * np.conj(h34), 0, t * h45],
        [t_prime * np.conj(h15), t * np.conj(h25), 0, t * np.conj(h45), 0]
    ])

    return H

def calculate(n=100):
    chern_number = np.zeros(5)
    berry_curvature = np.zeros((n, n, 5), dtype=complex)
    delta = 4 * pi / n
    kx = np.linspace(-pi, pi, n)
    ky = np.linspace(-pi, pi, n)
    vector = np.zeros((n, n, 5, 5), dtype=complex)

    for band in range(5):
        for i in range(n):
            for j in range(n):
                H = hamiltonian(kx[i], ky[j])
                eigenvalue, eigenvector = np.linalg.eigh(H)
                vector[i, j, :, band] = eigenvector[:, np.argsort(np.real(eigenvalue))[band]]

    for band in range(5):
        total_flux = 0
        for i in range(n):
            for j in range(n):
                i_next = (i + 1) % n
                j_next = (j + 1) % n
                u00 = vector[i, j, :, band]
                u10 = vector[i_next, j, :, band]
                u11 = vector[i_next, j_next, :, band]
                u01 = vector[i, j_next, :, band]

                l1 = np.vdot(u00, u10)
                l2 = np.vdot(u10, u11)
                l3 = np.vdot(u11, u01)
                l4 = np.vdot(u01, u00)

                product = l1 * l2 * l3 * l4
                flux = np.angle(product)
                if i == (n-1 or 0) and j != (n-1 or 0) :
                    flux /= 2
                elif i != (n-1 or 0) and j == (n-1 or 0) :
                    flux /= 2
                elif i == (n-1 or 0) and j == (n-1 or 0) :
                    flux /= 4
                else:
                    flux = flux
                total_flux += flux
                berry_curvature[i, j, band] = flux / (delta * delta)
        chern_number[band] = total_flux / (2 * pi)
    return chern_number, berry_curvature, kx, ky

# 计算贝利曲率和陈数
n = 100
chern_number, berry_curvature, kx, ky = calculate(n)
print("Chern numbers:", chern_number)

# 创建kx和ky的网格
KX, KY = np.meshgrid(kx, ky, indexing='ij')

fig = plt.figure(figsize=(20, 12))
axes_2d = []
axes_3d = []

# 创建所有子图
for band in range(5):
    ax_2d = fig.add_subplot(2, 5, band + 1)
    axes_2d.append(ax_2d)

    ax_3d = fig.add_subplot(2, 5, band + 6, projection='3d')
    axes_3d.append(ax_3d)

for band in range(5):
    bc_real = np.real(berry_curvature[:, :, band])

    # --- 二维热图 ---
    im = axes_2d[band].pcolormesh(KX, KY, bc_real.T, cmap='RdBu_r', shading='auto')
    axes_2d[band].set_aspect('equal')
    axes_2d[band].set_title(f'Band {band + 1}, Chern={chern_number[band]:.2f}', fontsize=10)
    fig.colorbar(im, ax=axes_2d[band], shrink=0.6, pad=0.03, aspect=20)  # 紧凑colorbar

    # --- 三维曲面 ---
    surf = axes_3d[band].plot_surface(KX, KY, bc_real, cmap='RdBu_r',
                                      edgecolor='none', alpha=0.9, rstride=2, cstride=2)
    axes_3d[band].set_title(f'Band {band + 1}', fontsize=10)
    fig.colorbar(surf, ax=axes_3d[band], shrink=0.6, pad=0.1, aspect=20)  # 紧凑colorbar

plt.subplots_adjust(
    left=0.05,
    right=0.98,
    bottom=0.05,
    top=0.95,
    wspace=0.02,
    hspace=0.1 )

# plt.savefig('optimized_layout.png', dpi=300, bbox_inches='tight')
plt.show()