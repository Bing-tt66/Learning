import numpy as np

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
    kx = np.linspace(-2 * np.pi / (3 * a_0), 2 * np.pi / (3 * a_0), n)
    ky = np.linspace(-2 * np.pi / (3 * a_0), 2 * np.pi / (3 * a_0), n)
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

    return chern_numbers, berry_curvature


# 计算陈数
n = 100
chern_numbers, berry_curvature = calculate_chern_number(n)
print(f"价带陈数: {chern_numbers[0]:.6f}")
print(f"导带陈数: {chern_numbers[1]:.6f}")
