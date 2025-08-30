import numpy as np

# 物理参数
a_0 = 0.142  # C-C键长 (nm)
a = np.sqrt(3) * a_0  # 晶格常数 (nm)
V_ppi_0 = -2.7  # 跃迁能 (eV)

def hamiltonian(kx, ky,m=0):
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

def calculate(n=100):
    delta = 1e-9
    chern_number = 0 # 陈数初始化
    for kx in np.arange(-np.pi, np.pi, 2 * np.pi / n):
        for ky in np.arange(-np.pi, np.pi, 2 * np.pi / n):
            H = hamiltonian(kx, ky)
            eigenvalue, eigenvector = np.linalg.eig(H)
            vector = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 价带波函数
            # print(np.argsort(np.real(eigenvalue))[0])  # 排序索引（从小到大）
            # print(eigenvalue)  # 排序前的本征值
            # print(np.sort(np.real(eigenvalue)))  # 排序后的本征值（从小到大）

            H_delta_kx = hamiltonian(kx + delta, ky)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_kx)
            vector_delta_kx = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离kx的波函数

            H_delta_ky = hamiltonian(kx, ky + delta)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_ky)
            vector_delta_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离ky的波函数

            H_delta_kx_ky = hamiltonian(kx + delta, ky + delta)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_kx_ky)
            vector_delta_kx_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离kx和ky的波函数

            # 价带的波函数的贝里联络(berry connection) # 求导后内积
            A_x = np.dot(vector.transpose().conj(), (vector_delta_kx - vector) / delta)  # 贝里联络Ax（x分量）
            A_y = np.dot(vector.transpose().conj(), (vector_delta_ky - vector) / delta)  # 贝里联络Ay（y分量）

            A_x_delta_ky = np.dot(vector_delta_ky.transpose().conj(),
                                  (vector_delta_kx_ky - vector_delta_ky) / delta)  # 略偏离ky的贝里联络Ax
            A_y_delta_kx = np.dot(vector_delta_kx.transpose().conj(),
                                  (vector_delta_kx_ky - vector_delta_kx) / delta)  # 略偏离kx的贝里联络Ay

            # 贝里曲率(berry curvature)
            F = (A_y_delta_kx - A_y) / delta - (A_x_delta_ky - A_x) / delta

            # 陈数(chern number)
            chern_number = chern_number + F * (2 * np.pi / n) ** 2
    chern_number = chern_number / (2 * np.pi * 1j)
    return chern_number
chern_number = calculate()
print('Chern number = %.11f'%np.real(chern_number))