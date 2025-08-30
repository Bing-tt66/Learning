import numpy as np

# 物理参数
a_0 = 0.142 # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)


def graphene_hamiltonian(kx, ky, m=0):
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


def main():
    n = 200
    delta = 2 * np.pi/ a_0 / n
    chern_number = 0  # 陈数初始化
    for kx in np.linspace(-np.pi / a_0, np.pi / a_0, n):
        for ky in np.linspace(-np.pi / a_0, np.pi / a_0, n):
            H = graphene_hamiltonian(kx, ky)
            eigenvalue, eigenvector = np.linalg.eig(H)
            vector = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 价带波函数

            H_delta_kx = graphene_hamiltonian(kx + delta, ky)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_kx)
            vector_delta_kx = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离kx的波函数

            H_delta_ky = graphene_hamiltonian(kx, ky + delta)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_ky)
            vector_delta_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离ky的波函数

            H_delta_kx_ky = graphene_hamiltonian(kx + delta, ky + delta)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_kx_ky)
            vector_delta_kx_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离kx和ky的波函数

            ux = np.dot(np.conj(vector), vector_delta_kx)
            ux = ux / abs(ux)

            uy = np.dot(np.conj(vector), vector_delta_ky)
            uy = uy / abs(uy)

            ux_y = np.dot(np.conj(vector_delta_ky), vector_delta_kx_ky)
            ux_y = ux_y / abs(ux_y)

            uy_x = np.dot(np.conj(vector_delta_kx), vector_delta_kx_ky)
            uy_x = uy_x / abs(uy_x)

            F12 = np.log(ux * uy_x * (1 / ux_y) * (1 / uy))
            # print(F12)

            # 陈数(chern number)
            chern_number += F12
    chern_number = chern_number / (2 * np.pi * 1j)
    print('Chern number = %.11f' % np.real(chern_number))


if __name__ == '__main__':
    main()