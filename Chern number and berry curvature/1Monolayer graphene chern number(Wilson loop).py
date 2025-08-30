import numpy as np
from scipy.linalg import eig

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)

# 倒空间基矢
a1 = np.array([np.sqrt(3) * a / 2, a / 2])
a2 = np.array([np.sqrt(3) * a / 2, -a / 2])
b1 = 2 * np.pi * (np.array([a2[1], -a2[0]])) / (np.cross(a1, a2))
b2 = 2 * np.pi * (np.array([-a1[1], a1[0]])) / (np.cross(a1, a2))


# 紧束缚模型哈密顿量
def graphene_hamiltonian(k,m):
    kx, ky = k
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


# 计算陈数的函数(Wilson loop法)
def compute_chern_number(n_k):
    # 生成k空间网格 (使用倒空间基矢坐标)
    k1 = np.linspace(-np.pi / a_0, np.pi / a_0, n_k, endpoint=False)
    k2 = np.linspace(-np.pi / a_0, np.pi / a_0, n_k, endpoint=False)
    K1, K2 = np.meshgrid(k1, k2, indexing='ij')

    # 存储所有k点的波函数 (价带)
    u_valence = np.zeros((n_k, n_k, 2), dtype=complex)

    # 计算每个k点的波函数
    for i in range(n_k):
        for j in range(n_k):
            # 将倒空间坐标转换为笛卡尔坐标
            k_cart = K1[i, j] * b1  + K2[i, j] * b2
            H = graphene_hamiltonian(k_cart,0.1)
            # 对角化哈密顿量
            eigvals, eigvecs = eig(H)
            # 获取价带（能量较低的能带）
            idx = np.argsort(eigvals.real)[0]
            u_valence[i, j] = eigvecs[:, idx]

    # 初始化陈数
    chern_number = 0.0

    # 遍历所有小格子
    for i in range(n_k):
        for j in range(n_k):
            # 获取小格子的四个顶点 (使用周期性边界条件)
            i_next = (i + 1) % n_k
            j_next = (j + 1) % n_k

            # 获取四个顶点的波函数
            u00 = u_valence[i, j]
            u10 = u_valence[i_next, j]
            u11 = u_valence[i_next, j_next]
            u01 = u_valence[i, j_next]

            # 计算四个重叠 (内积)
            s1 = np.vdot(u00, u10)  # <u(k_i,j)|u(k_i+1,j)>
            s2 = np.vdot(u10, u11)  # <u(k_i+1,j)|u(k_i+1,j+1)>
            s3 = np.vdot(u11, u01)  # <u(k_i+1,j+1)|u(k_i,j+1)>
            s4 = np.vdot(u01, u00)  # <u(k_i,j+1)|u(k_i,j)>

            # 计算小格子的通量 (贝里相位)
            # 注意: 由于是单带系统，不需要行列式
            product = s1 * s2 * s3 * s4
            flux = np.angle(product)  # 取辐角

            # 累加到总陈数
            chern_number += flux

    # 最终陈数
    chern_number /= (2 * np.pi)

    return chern_number


# 计算陈数
n_k = 100  # 网格点数
chern_num = compute_chern_number(n_k)
print(f"计算得到单层石墨烯的陈数: {chern_num:.22f}")