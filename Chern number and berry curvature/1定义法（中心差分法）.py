import numpy as np

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)

def hamiltonian(kx, ky,m=0.1):
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
        [V_ppi_0 * np.conj(h),-m]
    ])
    return H

# 中心差分格式
def calculate(n=100):
    delta = 1e-9
    chern_number = 0
    for kx in np.arange(-np.pi, np.pi, 2 * np.pi / n):
        for ky in np.arange(-np.pi, np.pi, 2 * np.pi / n):
            H0 = hamiltonian(kx, ky)
            eigenvalue0, eigenvector0 = np.linalg.eig(H0)
            vector0 = eigenvector0[:, np.argsort(np.real(eigenvalue0))[0]]

            H1 = hamiltonian(kx - delta, ky)
            eigenvalue1, eigenvector1 = np.linalg.eig(H1)
            vector1 = eigenvector1[:, np.argsort(np.real(eigenvalue1))[0]]
            a1 = np.vdot(vector0,vector1)

            H2 = hamiltonian(kx + delta, ky)
            eigenvalue2, eigenvector2 = np.linalg.eig(H2)
            vector2 = eigenvector2[:, np.argsort(np.real(eigenvalue2))[0]]
            a2 = np.vdot(vector0,vector2)

            H3 = hamiltonian(kx, ky - delta)
            eigenvalue3, eigenvector3 = np.linalg.eig(H3)
            vector3 = eigenvector3[:, np.argsort(np.real(eigenvalue3))[0]]
            a3 = np.vdot(vector0,vector3)

            H4 = hamiltonian(kx, ky + delta)
            eigenvalue4, eigenvector4 = np.linalg.eig(H4)
            vector4 = eigenvector4[:, np.argsort(np.real(eigenvalue4))[0]]
            a4 = np.vdot(vector0,vector4)

            F = (a2 - a1)/(2 * delta)-(a4 - a3)/(2 * delta)

            chern_number += F * (2 * np.pi / n) ** 2

    chern_number = chern_number / (2 * np.pi * 1j)
    return chern_number

chern_number = calculate(n=200)
print(chern_number)
