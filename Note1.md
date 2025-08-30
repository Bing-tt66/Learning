# 1.晶体结构绘制
## 1.1单层石墨烯
### 1.绘图代码
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
a=1.42#碳碳键长  
sqrt3=np.sqrt(3)  
  
#定义基矢  
a1=a*np.array([3/2,sqrt3/2])  
a2=a*np.array([3/2,-sqrt3/2])  
  
#两类原子坐标  
atom_A=a*np.array([0,0])  
atom_B=a*np.array([1,0])  
  
N=10  
positions_A=[]  
positions_B=[]  
for n in range(-N,N+1):  
    for m in range(-N,N+1):  
        R=n*a1+m*a2#晶格矢量  
        pos_A=R+atom_A  
        positions_A.append(pos_A)  
        pos_B=R+atom_B  
        positions_B.append(pos_B)  
  
positions_A=np.array(positions_A)  
positions_B=np.array(positions_B)  
# print(positions_A,sep='\t\t')  
# print(positions_B,sep='\t\t')  
  
  
#绘图  
plt.figure(figsize=(10,10))  
plt.axis('equal')  
plt.title('Single Layer Graphene Lattice')  
plt.xlabel('X(Å)')  
plt.ylabel('Y(Å)')  
  
#绘制原子  
plt.scatter(positions_A[:,0],positions_A[:,1],s=30,c='red',label='A sublattice')  
plt.scatter(positions_B[:,0],positions_B[:,1],s=30,c='blue',label='B sublattice')  
  
#绘制连线（键）  
bond_length = a * 1.1  # 键长容差范围  
  
# 预计算所有原子位置  
all_positions = np.vstack([positions_A, positions_B])  
num_A = len(positions_A)  
  
# 使用集合记录已绘制的键，避免重复  
drawn_bonds = set()  
  
# 只连接A子晶格和B子晶格之间的键  
for i in range(num_A):  
    pos_A = positions_A[i]  
    # 查找该A原子的最近邻B原子  
    for j in range(len(positions_B)):  
        pos_B = positions_B[j]  
  
        # 计算距离  
        dx = pos_A[0] - pos_B[0]  
        dy = pos_A[1] - pos_B[1]  
        distance = np.sqrt(dx ** 2 + dy ** 2)  
  
        # 如果距离在键长范围内且未绘制过  
        if distance < bond_length:  
            # 使用原子索引对作为键的唯一标识，确保不重复  
            bond_key = tuple(sorted([i, j + num_A]))  
            if bond_key not in drawn_bonds:  
                drawn_bonds.add(bond_key)  
                plt.plot([pos_A[0], pos_B[0]], [pos_A[1], pos_B[1]],  
                         'black', linewidth=1, alpha=0.8)  
  
#绘制图例  
plt.legend(loc='best')  
  
plt.grid(True,alpha=0.3)  
plt.tight_layout()  
plt.savefig('Single Layer Graphene Lattice')  
plt.show()
```
### 2.绘制结果：

![[Single Layer Graphene Lattice.png|图1：单层石墨烯]] 

## 1.2双层扭角石墨烯
### 1.绘图代码：
```python
from shapely.geometry import Point, Polygon  
import numpy as np   
import copy  
import os  
import matplotlib.pyplot as plt   
from math import *  
  
#with open('in.dat', 'r') as file:  
#    m = int(file.readline())  # 读取第一行并转为整数  
#    n = int(file.readline())  # 读取第二行并转为整数  
#print(m,n)  
m = 3  
n = 1  
  
x_arange = np.arange(-10,15.1)  
y_arange = np.arange(-10,15.1)  
coordinates = []  
for x in x_arange:  
    for y in y_arange:  
        coordinates.append([0+x*3/2+y*3/2,0+x*np.sqrt(3)/2-y*np.sqrt(3)/2])  
        coordinates.append([1+x*3/2+y*3/2,0+x*np.sqrt(3)/2-y*np.sqrt(3)/2])  
#计算图层1的范围  
x_range1 = max(np.array(coordinates)[:,0])-min(np.array(coordinates)[:,0])  
y_range1 = max(np.array(coordinates)[:,1])-min(np.array(coordinates)[:,1])  
  
#深度复制第一层到第二层  
theta = np.arccos((m**2 + n**2 + 4*m*n)/(2*(m**2 + n**2 + m*n)))  
      
#theta = round(np.degrees(theta_0), 2)*np.pi/180  
print(round(np.degrees(theta), 2))  
rotation_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])  
coordinates2 = copy.deepcopy(coordinates)  
  
#通过旋转矩阵修改第二层  
for i in range(len(coordinates)):  
    coordinates2[i] = np.dot(rotation_matrix,coordinates[i])  
  
#计算第二层范围  
x_range2 = max(np.array(coordinates2)[:,0])-min(np.array(coordinates2)[:,0])  
y_range2 = max(np.array(coordinates2)[:,1])-min(np.array(coordinates2)[:,1])  
  
#计算总范围  
x_range = max([x_range1,x_range2])  
y_range = max([y_range1,y_range2])  
  
# 计算四个点的坐标来构成边界  
  
boundary_points = []  
boundary_points.append([0,0])  
boundary_points.append([0+(m+n)*3/2,0+(m-n)*np.sqrt(3)/2])  
boundary_points.append([0+(m+n*2)*3/2,0+m*3*np.sqrt(3)/2])  
boundary_points.append([0+n*3/2,0+(m*2+n)*np.sqrt(3)/2])  
      
# 定义边界形成多边形  
polygon = Polygon(np.array(boundary_points))  
# 设定一个阈值范围，允许顶点在这个范围内被视为多边形内部  
polygon_with_buffer = polygon.buffer(0.001)  # 缓冲区大小为 0.01  
x_polygon, y_polygon = zip(*boundary_points)  
#两点间间距  
def distance(x_i,y_i,x_j,y_j):  
    d = np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)  
    return d  
  
#相邻点连线图层一（fig1）  
def plot_dots_1(ax, coordinates):  
    for i in range(len(coordinates)):  
        for j in range(len(coordinates)):  
            x_i = coordinates[i][0]  
            y_i = coordinates[i][1]  
            x_j = coordinates[j][0]  
            y_j = coordinates[j][1]  
            d = distance(x_i,y_i,x_j,y_j)  
            if d < 1.1:  
                ax.plot([coordinates[i][0],coordinates[j][0]],[coordinates[i][1],coordinates[j][1]],'-k',linewidth=0.2)  
    for i in range(len(coordinates)):  
        ax.plot(coordinates[i][0],coordinates[i][1],'ro',markersize=0.9)  
  
  
  
#相邻点连线图层二（fig1）  
def plot_dots_2(ax, coordinates2):  
    for i in range(len(coordinates2)):  
        for j in range(len(coordinates2)):  
            x_i = coordinates2[i][0]  
            y_i = coordinates2[i][1]  
            x_j = coordinates2[j][0]  
            y_j = coordinates2[j][1]  
            d = distance(x_i,y_i,x_j,y_j)  
            if d < 1.1:  
                ax.plot([coordinates2[i][0],coordinates2[j][0]],[coordinates2[i][1],coordinates2[j][1]],'--k',linewidth=0.2)  
    for i in range(len(coordinates2)):  
        ax.plot(coordinates2[i][0],coordinates2[i][1],'bo',markersize=0.9)  
  
#标记原点  
def plot_dots_0(ax, coordinates):  
    for i in range(len(coordinates)):  
        ax.plot(coordinates[i][0],coordinates[i][1],'ko',markersize=0.9)  
  
#创建图板调整比例（长宽）  
  
fig, ax = plt.subplots(figsize=(9*x_range/y_range,9))  
ax.plot(x_polygon + (x_polygon[0],), y_polygon + (y_polygon[0],), 'r-', linewidth=1)  # 多边形边界  
plot_dots_1(ax, coordinates)  
plot_dots_2(ax, coordinates2)   
plot_dots_0(ax, [[0,0]])#标记原点[0,0]  
plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95)#调整边距  
plt.axis('off')#关闭坐标轴显示  
#plt.savefig('TBG.png',dpi=300)  
plt.show()
```
### 2.绘制结果：

![[TBG.png|图2：双层扭角石墨烯]] 

## 1.3特殊结构晶体
### 1.绘图代码：
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
# 设置参数  
a = 2  # 大正方形边长  
b = 1  # 小正方形边长  
N = 5  # 晶胞数量（每边）  
c = (a-b)/np.sqrt(2) # 对角线键长  
a1 = a*np.array([1,0]) # 晶格基矢  
a2 = a*np.array([0,1])  
  
# 生成大正方形顶点坐标  
big_square_points = []  
for m in range(-N, N + 1):  
    for n in range(-N, N + 1):  
        r = m * a1 + n * a2  
        big_square_points.append([0+r[0],0+r[1]])  
  
# 生成小正方形顶点坐标  
small_square_points = []  
for m in range(-N, N):  
    for n in range(-N, N):  
        # 计算晶胞中心  
        center = m * a1 + n * a2 + [a / 2, a / 2]  
        # 生成小正方形的四个顶点  
        small_square_points.append([center[0] - b / 2, center[1] - b / 2])  # 左下  
        small_square_points.append([center[0] + b / 2, center[1] - b / 2])  # 右下  
        small_square_points.append([center[0] + b / 2, center[1] + b / 2])  # 右上  
        small_square_points.append([center[0] - b / 2, center[1] + b / 2])  # 左上  
all_points = big_square_points + small_square_points  
  
# 创建图形  
fig, ax = plt.subplots(figsize=(10, 10))  
  
# 绘制所有点  
for point in all_points:  
    ax.plot(point[0], point[1], 'ko', markersize=4)  
  
# 连接小正方形的边  
for i in range(0, len(small_square_points), 4):  
    # 获取当前小正方形的四个点  
    p1 = small_square_points[i]  # 左下  
    p2 = small_square_points[i + 1]  # 右下  
    p3 = small_square_points[i + 2]  # 右上  
    p4 = small_square_points[i + 3]  # 左上  
  
    # 连接小正方形的四条边  
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1)  
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-', linewidth=1)  
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'k-', linewidth=1)  
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], 'k-', linewidth=1)  
  
# 连接小正方形顶点到大正方形顶点  
for i in range(len(small_square_points)):  
    # 当前小正方形顶点  
    sx, sy = small_square_points[i]  
    # 找到最近的大正方形顶点（四个方向）  
    for bx, by in big_square_points:  
        # 计算距离  
        distance = np.sqrt((sx - bx) ** 2 + (sy - by) ** 2)  
        # 检查是否在四个对角线方向上  
        if abs(distance - np.sqrt((a / 2 - b / 2) ** 2 + (a / 2 - b / 2) ** 2)) < 0.01:  
            # 绘制连接线  
            ax.plot([sx, bx], [sy, by], 'k-', linewidth=1)  
  
# 标记原点  
ax.plot(0, 0, 'ro', markersize=6)  
# 设置图形属性  
plt.axis('equal')  
plt.axis('off')  
plt.title('Special Monolayer Lattice',size=16)  
plt.tight_layout()  
#plt.savefig('Special Monolayer Lattice.png')  
plt.show()
```
### 2.绘制结果：

![[Figure_1.png|图3：特殊结构晶体]]

# 2.能带结构
## 2.1单层石墨烯
### 理论推导：
哈密顿量：
$$
\hat{H}_{\alpha\beta}(\mathbf{k})=\epsilon_{\alpha}\delta_{\alpha\beta}+\sum_{\mathbf{d}}{}t_{\alpha\beta}(\mathbf{d})\exp(i\mathbf{k}\cdot\mathbf{d})

$$
- $t_{\alpha\beta}$为跃迁积分

最近邻矢量：
$$
\begin{aligned}
\mathbf{d_1}&=a(1,0)\\
\mathbf{d_2}&=a(-\frac{1}{2},\frac{\sqrt{3}}{2})\\
\mathbf{d_3}&=a(-\frac{1}{2},-\frac{\sqrt{3}}{2})
\end{aligned}
$$
跃迁相位：
$$
f(\mathbf{k})=exp(\mathbf{k}\cdot\mathbf{d_1})+exp(\mathbf{k}\cdot\mathbf{d_2})+exp(\mathbf{k}\cdot\mathbf{d_3})
$$
哈密顿矩阵：
$$
\hat{H}(\mathbf{k})=\begin{pmatrix}
 0 & -t*f(\mathbf{k})\\
 -t*f^*(\mathbf{k}) & 0
\end{pmatrix}
$$
本征值方程：
$$
\hat{H}(\mathbf{k})\psi(\mathbf{k})=E(\mathbf{k})\psi(\mathbf{k})
$$
解得：
$$
E=\pm t\cdot f(\mathbf{k})
$$
### 代码实现：
2D:
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)a1=np.array([np.sqrt(3)*a/2,a/2])  
a2=np.array([np.sqrt(3)*a/2,-a/2])  
  
# 倒空间基矢  
b1 = 2 * np.pi * (np.array([a2[1], -a2[0]])) / (np.cross(a1, a2))  
b2 = 2 * np.pi * (np.array([-a1[1], a1[0]])) / (np.cross(a1, a2))  
  
# 高对称点（倒空间坐标）  
Gamma = np.array([0, 0])  # Γ点  
K = ( b1 + 2*b2) / 3  # K点  
M =(b2+b1)/ 2  # M点  
K_prime = ( 2 *b1 +  b2) / 3  # K'点  
  
# 创建k空间路径：K → Γ → M → K'  
num_segments = 30  
path_K_Gamma = np.linspace(K, Gamma, num_segments)  
path_Gamma_M = np.linspace(Gamma, M, num_segments)  
path_M_Kprime = np.linspace(M, K_prime, num_segments)  
k_path = np.concatenate([path_K_Gamma, path_Gamma_M, path_M_Kprime])  
  
# 紧束缚模型哈密顿量  
def graphene_hamiltonian(k):  
    kx, ky = k  
    # 三个最近邻矢量  
    d1 = np.array([a_0,0])  
    d2 = np.array([-a_0/2,a_0*np.sqrt(3)/2])  
    d3 = np.array([-a_0/2,-a_0*np.sqrt(3)/2])  
  
    # 计算跃迁项  
    h = 0  
    for d in [d1, d2, d3]:  
        h += np.exp(1j * (kx * d[0] + ky * d[1]))  
  
    H = np.array([  
        [0, V_ppi_0 * h],  
        [V_ppi_0 * np.conj(h), 0]  
    ])  
    return H  
  
  
# 计算能带  
energies = []  
for k in k_path:  
    H = graphene_hamiltonian(k)  
    eigvals = np.linalg.eigvalsh(H)  # 计算本征值  
    energies.append(eigvals)  
  
energies = np.array(energies)  
  
# 绘制能带图  
plt.figure(figsize=(10, 6))  
  
# 绘制两条能带  
plt.plot(energies[:, 0], 'b-', linewidth=1.5)  
plt.plot(energies[:, 1], 'r-', linewidth=1.5)  
  
# 设置高对称点标记  
ticks_positions = [0, num_segments, 2 * num_segments, 3 * num_segments]  
plt.xticks(ticks_positions, ['K', 'Γ', 'M', "K'"])  
  
# 标签和标题  
plt.ylabel('Energy (eV)', fontsize=12)  
plt.title('Graphene Band Structure: K → Γ → M → K\'', fontsize=14)  
plt.grid(alpha=0.3)  
plt.ylim(-9, 9)  # 设置能量范围  
  
# 图例  
plt.legend(['Valence Band', 'Conduction Band'], loc='upper right')  
plt.tight_layout()  
plt.show()
```
3D:
```python
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.colors as mcolors  
from matplotlib.colors import LinearSegmentedColormap  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)a1 = np.array([np.sqrt(3) * a / 2, a / 2])  
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
    d1 = np.array([a_0, 0])  
    d2 = np.array([-a_0 / 2, a_0 * np.sqrt(3) / 2])  
    d3 = np.array([-a_0 / 2, -a_0 * np.sqrt(3) / 2])  
  
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
```
### 结果：

![[Monolayer Graphere (literature reproduction).png|图4：单层石墨烯能带结构（文献复现）]]
![[graphene_3d_band_structure.png|图5：单层石墨烯3D图]]
## 2.2双层扭角石墨烯

### 理论推导：**（没细推）

### 代码实现：
```python
from shapely.geometry import Point, Polygon  
import numpy as np   
import copy  
import matplotlib.pyplot as plt   
from math import *  
from scipy.linalg import eigh  
from datetime import datetime  
  
  
#常量  
a_0 = 0.142  #单位是nm  
a = 0.246  
d_0 = 0.335  
V_ppi_0 = -2.7#单位是eV  
V_ppx_0 = 0.48  
delta = 0.1848*a  
m = 4   #角度相关量  
n = 3  
dmax = 1  
theta = np.arccos((m**2 + n**2 + 4*m*n)/(2*(m**2 + n**2 + m*n)))  
L = a*(m-n)/(2*np.sin(theta/2))  
#超晶格越大，x_arrange和y_arrange越大  
x_arange = np.arange(-30,80.1)  
y_arange = np.arange(-30,80.1)  
#第一层  
coordinates = []  
for x in x_arange:  
    for y in y_arange:  
        coordinates.append([0+x*3/2+y*3/2,0+x*np.sqrt(3)/2-y*np.sqrt(3)/2])  
        coordinates.append([1+x*3/2+y*3/2,0+x*np.sqrt(3)/2-y*np.sqrt(3)/2])  
  
#旋转矩阵  
rotation_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])  
#第二层  
coordinates2 = copy.deepcopy(coordinates)#深度复制修改不影响第一层  
coordinates = np.array(coordinates)  # 转为 NumPy 数组  
  
for i in range(len(coordinates)):  
    coordinates2[i] = np.dot(rotation_matrix,coordinates[i])  
  
# 定义超晶格边界（显式转换为浮点数以避免精度问题）  
boundary_points = [  
    [0.0, 0.0],  
    [(m + n) * 3 / 2, (m - n) * np.sqrt(3) / 2],  
    [(m + n * 2) * 3 / 2, m * 3 * np.sqrt(3) / 2],  
    [n * 3 / 2, (m * 2 + n) * np.sqrt(3) / 2]  
]  
  
# 创建多边形  
polygon = Polygon(boundary_points)  
  
# 将边界点转换为set以便快速查找（使用四舍五入避免浮点精度问题）  
boundary_set = {tuple(np.round(point, 8)) for point in boundary_points}  
  
# 遍历所有坐标，检查哪些点在超晶格内部（严格内部，不包括边界）  
inside_points1 = []  
inside_points2 = []  
inside_points1.append(np.array([0,0]))  
inside_points2.append(np.array([0,0]))  
  
for coord in coordinates:  
    # 将坐标四舍五入到8位小数进行比较  
    rounded_coord = tuple(np.round(coord, 8))  
    point = Point(coord)  
    if polygon.contains(point) and rounded_coord not in boundary_set:  
        inside_points1.append(coord)  
  
for coord in coordinates2:  
    rounded_coord = tuple(np.round(coord, 8))  
    point = Point(coord)  
    if polygon.contains(point) and rounded_coord not in boundary_set:  
        inside_points2.append(coord)  
  
print(len(inside_points1))  
print(len(inside_points2))  
print(4*np.pi/3/L)  
if len(inside_points1) != len(inside_points2):  
    print("Error: points dismatch")  
#print(inside_points1)  
#print(inside_points2)  
# 绘制多边形边界  
x_polygon, y_polygon = zip(*boundary_points)  
  
def distance(x_i,y_i,x_j,y_j):  
    d = np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)  
    return d  
#能量函数  
def V_ppi(d):  
    V_ppi = V_ppi_0*np.exp((a_0-d)/delta)  
    return V_ppi  
  
def V_ppx(d):  
    V_ppx = V_ppx_0*np.exp((d_0-d)/delta)  
    return V_ppx  
  
#层内跃迁  
def t_11(inside_points, kx, ky):  
    N = len(inside_points)  
    H_ii = np.zeros((N, N), dtype=complex)  # Initialize the Hamiltonian matrix  
  
    for i1 in range(N):  
        for i2 in range(N):  
            if i1 == i2:  
                H_ii[i1, i2] = 0  
            else:  
                x_i1, y_i1 = inside_points[i1]  
                x_i2, y_i2 = inside_points[i2]  
                x_1 = x_i2 - x_i1  
                y_1 = y_i2 - y_i1  
                x_2 = x_1 + ((m + n) * 3 / 2)  
                y_2 = y_1 + ((m - n) * np.sqrt(3) / 2)  
                x_3 = x_1 - ((m + n) * 3 / 2)  
                y_3 = y_1 - ((m - n) * np.sqrt(3) / 2)  
                x_4 = x_1 + (n * 3 / 2)  
                y_4 = y_1 + ((m * 2 + n) * np.sqrt(3) / 2)  
                x_5 = x_1 - (n * 3 / 2)  
                y_5 = y_1 - ((m * 2 + n) * np.sqrt(3) / 2)  
                x_6 = x_1 + ((m + n) * 3 / 2) + (n * 3 / 2)  
                y_6 = y_1 + ((m - n) * np.sqrt(3) / 2) + ((m * 2 + n) * np.sqrt(3) / 2)  
                x_7 = x_1 + ((m + n) * 3 / 2) - (n * 3 / 2)  
                y_7 = y_1 + ((m - n) * np.sqrt(3) / 2) - ((m * 2 + n) * np.sqrt(3) / 2)  
                x_8 = x_1 - ((m + n) * 3 / 2) - (n * 3 / 2)  
                y_8 = y_1 - ((m - n) * np.sqrt(3) / 2) - ((m * 2 + n) * np.sqrt(3) / 2)  
                x_9 = x_1 - ((m + n) * 3 / 2) + (n * 3 / 2)  
                y_9 = y_1 - ((m - n) * np.sqrt(3) / 2) + ((m * 2 + n) * np.sqrt(3) / 2)  
                points = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5],  
                                   [x_6, y_6], [x_7, y_7], [x_8, y_8], [x_9, y_9]])  
  
                condition = np.sum(points**2, axis=1) <= dmax**2 +0.01  
                filtered_points = points[condition]  
  
                distance = a_0 * np.linalg.norm(filtered_points, axis=1)  
                Hii = []  
                for i in range(len(filtered_points)):  
                    Hi = 0  
                    d = distance[i]  
                    x, y = filtered_points[i]  
                    Hi = -np.exp(1j * (kx * x + ky * y)*a_0) * V_ppi(d)  
                    Hii.append(Hi)  
                H_ii[i1, i2] = sum(Hii)  
  
    return H_ii  
  
#层与层之间跃迁  
def t_12(inside_points_1,inside_points_2,kx,ky):  
    H_ij = np.zeros((len(inside_points_1),len(inside_points_2)),dtype=np.complex128)  
    for i1 in range(len(inside_points_1)):  
        for i2 in range(len(inside_points_2)):  
            x_i1, y_i1 = inside_points_1[i1]  
            x_i2, y_i2 = inside_points_2[i2]  
            x_1 = x_i2 - x_i1  
            y_1 = y_i2 - y_i1  
            x_2 = x_1 + ((m + n) * 3 / 2)  
            y_2 = y_1 + ((m - n) * np.sqrt(3) / 2)  
            x_3 = x_1 - ((m + n) * 3 / 2)  
            y_3 = y_1 - ((m - n) * np.sqrt(3) / 2)  
            x_4 = x_1 + (n * 3 / 2)  
            y_4 = y_1 + ((m * 2 + n) * np.sqrt(3) / 2)  
            x_5 = x_1 - (n * 3 / 2)  
            y_5 = y_1 - ((m * 2 + n) * np.sqrt(3) / 2)  
            x_6 = x_1 + ((m + n) * 3 / 2) + (n * 3 / 2)  
            y_6 = y_1 + ((m - n) * np.sqrt(3) / 2) + ((m * 2 + n) * np.sqrt(3) / 2)  
            x_7 = x_1 + ((m + n) * 3 / 2) - (n * 3 / 2)  
            y_7 = y_1 + ((m - n) * np.sqrt(3) / 2) - ((m * 2 + n) * np.sqrt(3) / 2)  
            x_8 = x_1 - ((m + n) * 3 / 2) - (n * 3 / 2)  
            y_8 = y_1 - ((m - n) * np.sqrt(3) / 2) - ((m * 2 + n) * np.sqrt(3) / 2)  
            x_9 = x_1 - ((m + n) * 3 / 2) + (n * 3 / 2)  
            y_9 = y_1 - ((m - n) * np.sqrt(3) / 2) + ((m * 2 + n) * np.sqrt(3) / 2)  
            points = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5],  
                                   [x_6, y_6], [x_7, y_7], [x_8, y_8], [x_9, y_9]])  
  
            condition = np.sum(d_0**2 + points**2 * a_0**2, axis=1) <= dmax**2 * a_0**2 +0.01   
filtered_points = points[condition]  
            distance = np.sqrt((a_0 * np.linalg.norm(filtered_points, axis=1))**2 + d_0**2)   
  
            Hij = []  
            for i in range(len(filtered_points)):  
                Hj = 0  
                d = distance[i]  
                x, y = filtered_points[i]  
                term1 = 1 - (d_0/d)**2  
                term2 = (d_0/d)**2  
                Hj = -np.exp(1j * (kx * x + ky * y)*a_0) * (V_ppi(d)*term1+ V_ppx(d)*term2)  
                Hij.append(Hj)  
            H_ij[i1,i2] = sum(Hij)  
             
  
    return H_ij  
  
def H_Matrix(inside_points_1,inside_points_2,kx,ky):  
    H_11 = t_11(inside_points_1,kx,ky)  
    H_12 = t_12(inside_points_1,inside_points_2,kx,ky)  
    H_21 = t_12(inside_points_2,inside_points_1,kx,ky)  
    H_22 = t_11(inside_points_2,kx,ky)  
    H_matrix = np.block([[H_11,H_12],[H_21,H_22]])  
      
    return H_matrix  
  
  
lenth = 2*np.pi/3/a_0/(m**2 + m*n + n**2)  
Gamma = np.array([0,0])  
K1 = lenth*np.array([n,np.sqrt(3)/3 * (2*m+n)])  
M = lenth*np.array([(m+2*n)/2,m*np.sqrt(3)/2])  
K2 = lenth*np.array([(m+n),(m-n)*np.sqrt(3)/3])  
  
num_points = 10  
path_K1_Gamma = np.linspace(K1,Gamma,num_points)  
path_Gamma_M = np.linspace(Gamma,M,num_points)  
path_M_K2 = np.linspace(M,K2,int(num_points/2))  
  
k_path = np.concatenate([path_K1_Gamma, path_Gamma_M, path_M_K2], axis=0)  
print(k_path)  
  
num_bands_to_plot = 14  
  
bands = np.zeros((len(k_path), num_bands_to_plot))    
  
for i in range(len(k_path)):  
    kx_val, ky_val = k_path[i]  
    t0 = datetime.now()  
    H = H_Matrix(inside_points1, inside_points2, kx_val, ky_val)  
    t1 = datetime.now()  
    print("matrix calculation time:",i, t1-t0)  
  
    if i == 0:  
        print("output matrix",kx_val,ky_val)  
        np.savetxt('matrix.txt', H, fmt='%.6e', delimiter='\t')  
  
    eigenvalues = np.linalg.eigvalsh(H)  # 已排序  
    t0 = datetime.now()  
    print("Hamiltonian diagonalization time:",i, t0-t1)  
    mid = len(eigenvalues)//2  
  
    # 例如取中间 14 条  
    # 这里 [mid-7 : mid+7] => 取 band indices from mid-7 to mid+6    num_bands2 = int(num_bands_to_plot/2)  
    bands[i,:] = eigenvalues[mid-num_bands2 : mid+num_bands2]  
  
num_total = len(k_path)  # 3 * num_points  
x_axis = np.arange(num_total)  
#print(x_axis)  
  
plt.figure(figsize=(8,6))  
for band_idx in range(num_bands_to_plot):  
    plt.plot(x_axis, bands[:, band_idx], linewidth=1.0)  
plt.xticks(  
    ticks=[0, num_points, int(2*num_points), int(2*num_points+num_points/2)],  
    labels=["K₁", "Γ", "M", "K₂"]  
)  
plt.ylabel("Energy (eV)")  
plt.title("Band Structure along K1->Gamma->M->K2")  
plt.grid(True, alpha=0.3)  
plt.show()  
  
  
  
'''  
# 生成 ky 路径（kx固定为0）  
num_points = 500  # 增加点数使曲线更平滑  
ky_path = np.linspace(-4*np.pi/3/L - 0.1, 4*np.pi/3/L + 0.1, num_points)# 注意根据输出的2np.pi/L的值来调整ky_path  
kx_fixed = 0  # 固定 kx=0  
# 存储能带数据（仅存储中间4条）  
bands = np.zeros((num_points, 14))  # 第36-39条能带（索引35-38）  
  
for i, ky_val in enumerate(ky_path):  
    # 计算哈密顿量（根据你的体系，可能需要使用 H_Matrix 而非 t_11）  
    H = H_Matrix(inside_points1,inside_points2,kx_fixed, ky_val)    # H = H_Matrix(inside_points1, inside_points2, kx_fixed, ky_val)  # 双层哈密顿量  
    # 计算本征值（已排序）  
    eigenvalues = np.linalg.eigvalsh(H)    # 提取中间4条能带（假设体系总自由度为74，索引36-39对应第37-40条能带）  
    middle_index = len(eigenvalues) // 2    bands[i, :] = eigenvalues[middle_index-7 : middle_index+7]  # 取中间4条  
  
# 绘制能带图  
plt.figure(figsize=(10, 6))  
for band_idx in range(14):  
    plt.plot(ky_path, bands[:, band_idx], linewidth=1.5)  
# 设置坐标轴和标题  
plt.xlim(-4*np.pi/3/L - 0.1, 4*np.pi/3/L - 0.1)# 根据ky_path来调整  
plt.xlabel("$k_y$ (沿 $k_x=0$ 路径)")  
plt.ylabel("Energy (eV)")  
plt.title("Band Structure (Middle 4 Bands)")  
plt.show()  
  
# 标记高对称点（根据你的布里渊区定义添加）  
# plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)  
# plt.axvline(x=np.pi/L, color='gray', linestyle='--', linewidth=0.8)  
  
plt.grid(True, alpha=0.3)  
plt.tight_layout()  
plt.savefig('figure.png', dpi=300, bbox_inches='tight')  
'''
```
### 结果：  
![[Band structure along K1 to Gamma to M to K2（双层）.png|图7：双层扭角石墨烯]]
## 2.3正方形晶格结构
### 理论推导
#### 1.超胞模型
##### 哈密顿矩阵
$$
\hat{H}=\begin{pmatrix}
0&t*(e^{ik_xa}+e^{-ik_xa})&t*(e^{ik_y}+e^{-ik_ya})&0\\
t*(e^{ik_xa}+e^{-ik_xa})&0&0&t*(e^{ik_y}+e^{-ik_ya})\\
t*(e^{ik_y}+e^{-ik_ya})&0&0&t*(e^{ik_xa}+e^{-ik_xa})\\
0&t*(e^{ik_y}+e^{-ik_ya})&t*(e^{ik_xa}+e^{-ik_xa})&0
\end{pmatrix}
$$
#### 2.单胞模型
$$
\begin{align}
E(\mathbf{k}) &= -t \left[ e^{ik_x a} + e^{-ik_x a} + e^{ik_y a} + e^{-ik_y a} \right]\\
&=-t \left[ 2\cos(k_x a) + 2\cos(k_y a) \right]
\end{align}
$$
### 代码
#### 超胞：
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
a = 1.0  # Atomic spacing  
t = 1.0  # Hopping integral  
  
# 修正基矢：二维正方形晶格 (2a x 2a 超胞)  
a1 = np.array([2 * a, 0])  # 基矢1  
a2 = np.array([0, 2 * a])  # 基矢2  
  
# 倒空间基矢  
b1 = 2 * np.pi * np.array([a2[1], -a2[0]]) / (np.cross(a1, a2))  
b2 = 2 * np.pi * np.array([-a1[1], a1[0]]) / (np.cross(a1, a2))  
# 高对称点（使用正方形晶格标准点）  
Gamma = np.array([0, 0])  # Γ点  
X = np.array([b1[0] / 2, 0])  # X点 (π/(2a), 0)M = np.array([b1[0] / 2, b2[1] / 2])  # M点 (π/(2a), π/(2a))  
def H_k(kx, ky):  
    # 考虑超胞边界的相位因子 (周期2a)  
    h_AB = t * ( np.exp(1j * kx * a) + np.exp(-1j * kx * a))  
    h_AC = t * ( np.exp(1j * ky * a) + np.exp(-1j * ky * a))  
    h_BD = t * ( np.exp(1j * ky * a) + np.exp(-1j * ky * a))  
    h_CD = t * ( np.exp(1j * kx * a) + np.exp(-1j * kx * a))  
  
    return np.array([  
        [0, h_AB, h_AC, 0],  
        [np.conj(h_AB), 0, 0, h_BD],  
        [np.conj(h_AC), 0, 0, h_CD],  
        [0, np.conj(h_BD), np.conj(h_CD), 0]  
    ])  
  
  
# 高对称路径：Γ → X → M → Γ  
n_points = 50  
path_Gamma_X = np.linspace(Gamma, X, n_points)  
path_X_M = np.linspace(X, M, n_points)  
path_M_Gamma = np.linspace(M, Gamma, n_points)  
k_path = np.concatenate([path_Gamma_X, path_X_M, path_M_Gamma])  
  
# 计算能带  
eigenvalues = []  
for k in k_path:  
    kx, ky = k  
    H = H_k(kx, ky)  
    evals = np.linalg.eigvalsh(H)  # 确保使用厄米矩阵对角化  
    eigenvalues.append(evals)  
  
eigenvalues = np.array(eigenvalues).real  # 取实部 (忽略数值误差)  
  
# 绘制能带  
plt.figure(figsize=(10, 6))  
for i in range(4):  
    plt.plot(eigenvalues[:, i], linewidth=1.5)  
  
# 标记高对称点  
ticks_positions = [0, n_points, 2 * n_points, 3 * n_points]  
plt.xticks(ticks_positions, ['Γ', 'X', 'M', 'Γ'])  
plt.axvline(x=n_points, color='k', linestyle='--', alpha=0.3)  
plt.axvline(x=2 * n_points, color='k', linestyle='--', alpha=0.3)  
  
plt.xlabel('k-path')  
plt.ylabel('Energy (t)')  
plt.title('Square Lattice Band Structure (4-atom unit cell)')  
plt.grid(alpha=0.3)  
plt.show()
```
#### 单胞：
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
a = 1.0  # Atomic spacing  
t = 1.0  # Hopping integral  
  
# 单原子原胞基矢  
a1 = np.array([a, 0])  
a2 = np.array([0, a])  
  
# 倒空间基矢  
b1 = 2 * np.pi * np.array([1, 0]) / a  
b2 = 2 * np.pi * np.array([0, 1]) / a  
  
# 高对称点  
Gamma = np.array([0, 0])  
X = np.array([np.pi/a, 0])        # X点 (π/a, 0)M = np.array([np.pi/a, np.pi/a])  # M点 (π/a, π/a)  
def H_k(kx, ky):  
    """单原子原胞哈密顿量"""  
    return -2 * t * (np.cos(kx * a) + np.cos(ky * a))  
  
# 高对称路径：Γ → X → M → Γ  
n_points = 100  
path_Gamma_X = np.linspace(Gamma, X, n_points)  
path_X_M = np.linspace(X, M, n_points)  
path_M_Gamma = np.linspace(M, Gamma, n_points)  
k_path = np.concatenate([path_Gamma_X, path_X_M, path_M_Gamma])  
  
# 计算能带  
energies = []  
for k in k_path:  
    kx, ky = k  
    energies.append(H_k(kx, ky))  
  
# 绘制能带  
plt.figure(figsize=(10, 6))  
plt.plot(energies, linewidth=2.0, color='b')  
  
# 标记高对称点  
ticks_positions = [0, n_points, 2*n_points, 3*n_points]  
plt.xticks(ticks_positions, ['Γ', 'X', 'M', 'Γ'])  
plt.axvline(x=n_points, color='k', linestyle='--', alpha=0.3)  
plt.axvline(x=2*n_points, color='k', linestyle='--', alpha=0.3)  
  
plt.xlabel('k-path')  
plt.ylabel('Energy (t)')  
plt.title('Square Lattice Band Structure (Single Atom Unit Cell)')  
plt.grid(alpha=0.3)  
plt.ylim(-5, 5)  # 能带范围 -4t 到 4tplt.show()
```
### 结果
![[Square Lattice Band Structure(4-atom unit cell).png]]
![[Square Lattice Band Structure (Single Atom Unit Cell).png]]
## 2.4特殊晶体

### 理论推导：

我们研究一种特殊单层晶格结构，包含五种原子位点：
- **A1**：位于大正方形中心
- **A2**：小正方形左下顶点
- **A3**：小正方形右下顶点
- **A4**：小正方形右上顶点
- **A5**：小正方形左上顶点

定义几何参数：（这种情况下两个跃迁积分相等）
$$
\begin{align*}
a &= \sqrt{2} + 1  \quad &\text{（大正方形边长）} \\
b &= 1  \quad &\text{（小正方形边长）} \\
c &= \frac{a - b}{\sqrt{2}}  \quad &\text{（对角线键长）}
\end{align*}
$$

实空间基矢
$$
\vec{a_1} = a\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
\vec{a_2} = a\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

倒易空间基矢
$$
\vec{b_1} = \frac{2\pi}{a}\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
\vec{b_2} = \frac{2\pi}{a}\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

最近邻矢量

| 跃迁对   | 矢量表示                                                                 |
| ----- | -------------------------------------------------------------------- |
| A₁→A₂ | $\vec{d_{12}} = \frac{a-b}{2}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$   |
| A₁→A₃ | $\vec{d_{13}} = \frac{a-b}{2}\begin{bmatrix} -1 \\ 1 \end{bmatrix}$  |
| A₁→A₄ | $\vec{d_{14}} = \frac{a-b}{2}\begin{bmatrix} -1 \\ -1 \end{bmatrix}$ |
| A₁→A₅ | $\vec{d_{15}} = \frac{a-b}{2}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$  |
| A₂→A₃ | $\vec{d_{23}} = b\begin{bmatrix} 1 \\ 0 \end{bmatrix}$               |
| A₂→A₅ | $\vec{d_{25}} = b\begin{bmatrix} 0 \\ 1 \end{bmatrix}$               |
| A₃→A₄ | $\vec{d_{34}} = b\begin{bmatrix} 0 \\ 1 \end{bmatrix}$               |
| A₄→A₅ | $\vec{d_{45}} = b\begin{bmatrix} -1 \\ 0 \end{bmatrix}$              |
 跃迁积分参数
- $t$：小正方形顶点原子间跃迁积分
- $t'$：中心原子与顶点原子间跃迁积分

k空间相位因子
对于波矢 $\vec{k} = (k_x, k_y)$，各跃迁相位因子为：
$$
\begin{align*}
\phi_{12}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{12}}\right) \\
\phi_{13}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{13}}\right) \\
\phi_{14}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{14}}\right) \\
\phi_{15}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{15}}\right) \\
\phi_{23}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{23}}\right) \\
\phi_{25}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{25}}\right) \\
\phi_{34}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{34}}\right) \\
\phi_{45}(\vec{k}) &= \exp\left(i\vec{k} \cdot \vec{d_{45}}\right)
\end{align*}
$$

哈密顿矩阵
构建5×5紧束缚哈密顿量：
$$
H(\vec{k}) = 
\begin{pmatrix}
0 & t'\phi_{12} & t'\phi_{13} & t'\phi_{14} & t'\phi_{15} \\
t'\phi_{12}^* & 0 & t\phi_{23} & 0 & t\phi_{25} \\
t'\phi_{13}^* & t\phi_{23}^* & 0 & t\phi_{34} & 0 \\
t'\phi_{14}^* & 0 & t\phi_{34}^* & 0 & t\phi_{45} \\
t'\phi_{15}^* & t\phi_{25}^* & 0 & t\phi_{45}^* & 0
\end{pmatrix}
$$

能带计算与布里渊区路径

高对称点
$$
\Gamma = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad
X = \begin{bmatrix} \pi/a \\ 0 \end{bmatrix}, \quad
M = \begin{bmatrix} \pi/a \\ \pi/a \end{bmatrix}
$$

k空间路径
$$\Gamma \rightarrow X \rightarrow M \rightarrow \Gamma$$

对角化哈密顿矩阵：
$$H(\vec{k})\psi_n(\vec{k}) = E_n(\vec{k})\psi_n(\vec{k})$$

得到五条能带：
$$E_1(\vec{k}), E_2(\vec{k}), E_3(\vec{k}), E_4(\vec{k}), E_5(\vec{k})$$

物理意义分析
1. **中心原子耦合**：$t'$ 控制中心原子 A1 与周边原子的耦合强度
2. **边缘原子耦合**：$t$ 决定小正方形顶点原子间的hopping强度
3. **几何对称性**：晶格的 $C_4$ 对称性反映在能带简并点
4. **拓扑性质**：狄拉克锥可能出现在高对称点（需进一步计算验证）

计算中采用：
$$t = t' = 1\text{ eV}$$

### 代码实现：
3D图：
```python
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具  
  
# 参数设置  
a = np.sqrt(2) + 1  # 大正方形边长  
b = 1  # 小正方形边长  
c = (a - b) / np.sqrt(2)  # 对角线键长  
t = 1  # 小正方形内跃迁  
t_prime = 1  # 中心原子到小正方形原子的跃迁  
  
# 设置k空间网格  
n_k = 50  # 每个方向上的点数  
kx = np.linspace(-np.pi/a, np.pi/a, n_k)  # 覆盖整个布里渊区  
ky = np.linspace(-np.pi/a, np.pi/a, n_k)  
KX, KY = np.meshgrid(kx, ky)  # 创建网格  
  
# 初始化能量网格 (5个能带)  
energies_3d = np.zeros((5, n_k, n_k))  
  
def Hamiltonian(k) :  
    kx, ky = k  
    k_vec = np.array([kx, ky])  
  
    # 最近邻矢量  
    d12 = np.array([(a - b) / 2, (a - b) / 2])  
    d13 = np.array([ - (a - b) / 2, (a - b) / 2])  
    d14 = np.array([ - (a - b) / 2, - (a - b) / 2])  
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
  
# 计算每个(kx,ky)点的能量  
for i in range(n_k):  
    for j in range(n_k):  
        k = np.array([KX[i, j], KY[i, j]])  
        H = Hamiltonian(k)  
        eigenvalues = np.linalg.eigvalsh(H)  
        energies_3d[:, i, j] = eigenvalues  # 存储所有能带  
  
# 创建3D图形  
fig = plt.figure(figsize=(12, 10))  
ax = fig.add_subplot(111, projection='3d')  
  
# 绘制每个能带的3D曲面  
for band in range(5):  
    ax.plot_surface(KX, KY, energies_3d[band],  
                      # 使用颜色映射表示能量高低  
                    alpha=0.7,       # 设置透明度  
                    rstride=2,       # 减少网格密度以提高性能  
                    cstride=2)  
  
# 设置坐标轴标签  
ax.set_xlabel('$k_x$', fontsize=12)  
ax.set_ylabel('$k_y$', fontsize=12)  
ax.set_zlabel('Energy (eV)', fontsize=12)  
  
# 设置视角以便更好地观察  
ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度  
  
# 添加标题  
plt.title('3D Band Structure of Special Monolayer Lattice', fontsize=14)  
plt.tight_layout()  
#plt.savefig('3D_Band_Structure.png')  
plt.show()
```

2D图：
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
# 参数  
a = np.sqrt(2) + 1 # 大正方形边长 （这样取边长使得b, c相等，对应的Hopping integral也会相等  
b = 1 # 小正方形边长  
c = (a - b) / np.sqrt(2) # 对角线键长  
'''A1 是大正方形上的原子，A2(左下),A3(右下),A4(右上),A5(左上)是小正方形上的原子'''  
t = 1 # Hopping integral among A2,A3,A4,A5 [eV]  
t_prime = 1 # Hopping integral A1--->(A2,A3,A4,A5)  
  
# 晶格基矢  
a1 = a * np.array([1, 0])  
a2 = a * np.array([0, 1])  
  
# 倒易空间基矢  
b1 = 2 * np.pi * np.array([1, 0]) / a # (2pi/(a), 0)  
b2 = 2 * np.pi * np.array([0, 1]) / a # (0, 2pi/(a))  
#print(b1, b2)  
  
# 高对称点  
Gamma = np.array([0, 0])  # Γ点  
X = np.array([ b1[0] / 2, 0])  # X点 (π/(a), 0)M = np.array([ b1[0] / 2, b2[1] / 2])  # M点 (2π/(a), 2π/(a))  
# 创建k空间路径：Γ-->X-->M-->Γ  
n_points = 50  
path_Gamma_X = np.linspace(Gamma, X, n_points, endpoint=False)  
path_X_M = np.linspace(X, M, n_points, endpoint=False)  
path_M_Gamma = np.linspace(M, Gamma, n_points, endpoint=False)  
k_path = np.concatenate([path_Gamma_X, path_X_M, path_M_Gamma], axis=0)  
  
# 定义哈密顿矩阵  
def Hamiltonian(k) :  
    kx, ky = k  
    k_vec = np.array([kx, ky])  
  
    # 最近邻矢量  
    d12 = np.array([(a - b) / 2, (a - b) / 2])  
    d13 = np.array([ - (a - b) / 2, (a - b) / 2])  
    d14 = np.array([ - (a - b) / 2, - (a - b) / 2])  
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
  
# 计算能带  
energies = []  
for k in k_path:  
    H = Hamiltonian(k)  
    eigenvalues = np.linalg.eigvalsh(H) # 计算本征值  
    energies.append(eigenvalues)  
energies = np.array(energies)  
  
# 绘制能带图（五条）  
plt.figure(figsize=(10,6))  
plt.plot(energies[:, 0], linewidth = 1.5)  
plt.plot(energies[:, 1], linewidth = 1.5)  
plt.plot(energies[:, 2], linewidth = 1.5)  
plt.plot(energies[:, 3], linewidth = 1.5)  
plt.plot(energies[:, 4], linewidth = 1.5)  
  
# 设置高对称点标记  
ticks_positions = [0, n_points, 2 * n_points, 3 * n_points]  
plt.xticks(ticks_positions, ['Γ', 'X', 'M', "Γ"])  
  
# 标签和标题  
plt.ylabel('Energy(eV)', fontsize = 12)  
plt.title('Band structure of Special Monolayer Lattice: Γ → X → M → Γ')  
plt.grid(alpha = 0.3)  
plt.legend(['1','2','3','4','5'], loc = 'upper right')  
plt.tight_layout()  
#plt.savefig('BSSML.png')  
plt.show()
```
### 结果：
![[BSSML.png|图7：特殊单层晶体能带结构二维图]]

![[Enhanced_3D_Band_Structure.png|图8：特殊单层晶体能带结构三维图]]
# 3.Chern number and Berry curvature
## 3.1概念
### Berry connection
$$
A(\mathbf{k})=i\langle u_{\mathbf{k}}|\nabla u_{\mathbf{k}}\rangle
$$
- $|u_{\mathbf{k}}\rangle$是**布洛赫波函数的周期部分**。
#### 布洛赫波：
描述电子在周期性势场中运动行为的波函数
$$
\psi(\mathbf{r})=e^{i\mathbf{k}\cdot\mathbf{r}}u(\mathbf{r})
$$
>布洛赫定理：对于在一个具有晶格平移对称性的势场 $V(\mathbf{r}) = V(\mathbf{r} + \mathbf{R})$ 中运动的粒子，其定态薛定谔方程的解必然具有如下形式：$\psi(\mathbf{r})=e^{i\mathbf{k}\cdot\mathbf{r}}u(\mathbf{r})$**其中，$u(\mathbf{r})$ 是一个与势场具有相同周期性的函数，即 $u(\mathbf{r}) = u(\mathbf{r} + \mathbf{R})$ 对于所有晶格矢量 $\mathbf{R}$ 成立。

### Berry curvature
$$
\begin{align}
\Omega(\mathbf{k})&=\nabla_\mathbf{k}\times A(\mathbf{k})\\
&=\partial_{k_x}A_y^{(n)}-\partial_{k_y}A_x^{(n)}
\end{align}
$$
### Chern number
$$
C=\frac{1}{2\pi}\int_{BZ}\Omega(\mathbf{k})d^2k
$$
#### 关键性质
1. 整数性：$C\in\mathbb{Z}$（拓扑量子化）
2. **规范不变性**：$A(\mathbf{k})$依赖规范选择，但$\Omega(\mathbf{k})$和$C$是规范不变的
3. 拓扑保护：陈数的值在能带不发生闭合（能隙不关闭）的连续形变下保持不变

##### 规范不变性：
**规范不变性指的是，尽管描述物理系统的“中间变量”（如势、波函数相位）依赖于人为的、非唯一的“规范”选择，但所有可观测的物理量（如电场、磁场、概率密度、能级、陈数）的测量结果必须与这种主观选择无关**
####  整数量子霍尔电导（QHE）
$$
\sigma_{xy}=C\frac{e^2}{h}
$$
#### 陈绝缘体
- 无外加磁场时，具有非零陈数的绝缘体，其边缘存在**手性拓扑边缘态**。
## 3.2定义法计算陈数
### 3.2.1理论推导
#### 1.基本理论框架
##### 1.1Berry phase and berry connection
在量子力学中，当系统参数变化（如动量k）绝热变化时，波函数会积累一个几何相位(berry phase):
$$
\gamma_n = \oint_C\mathbf{A}_n(\mathbf{k})\cdot d\mathbf{k}
$$
其中berry connection定义为：
$$
\mathbf{A}n(\mathbf{k})=i\langle u_{n\mathbf{k}}|\nabla_{\mathbf{k}}u_{n\mathbf{k}}\rangle
$$
这里$|u_{n\mathbf{k}\rangle}$是布洛赫哈密顿量的本征态
##### 1.2Berry curvature and chern number
贝利曲率是贝里联络的旋度：
$$
\Omega_n​(\mathbf{k})=\nabla_\mathbf{k}​×A_n​(\mathbf{k})=i⟨\nabla_\mathbf{k}​u_{n\mathbf{k}}​∣×∣\nabla_\mathbf{k}​u_{n\mathbf{k}}​⟩
$$
陈数则是贝里曲率在整个布里渊区域上的积分：
$$
C_n = \frac{1}{2\pi}\int_{BZ}\Omega_n(\mathbf{k})d^2k
$$
#### 2.石墨烯的紧束缚模型
石墨烯的紧束缚哈密顿量可以写为：
$$ H = -t \sum_{\langle i,j\rangle} (a_i^\dagger b_j + \text{h.c.}) $$
在动量空间中，这变为：
$$ H(\mathbf{k}) = \begin{pmatrix} 0 & h(\mathbf{k}) \\ h^*(\mathbf{k}) & 0 \end{pmatrix} $$
其中：
$$ h(\mathbf{k}) = -t \sum_{i=1}^3 e^{i\mathbf{k} \cdot \mathbf{d}_i} $$
哈密顿量的本征值为：

$$ E_{\pm}(\mathbf{k}) = \pm |h(\mathbf{k})| $$

**对应的本征态**为：

$$ |u_{\pm}(\mathbf{k})\rangle = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \ \pm \frac{h^*(\mathbf{k})}{|h(\mathbf{k})|} \end{pmatrix} $$
#### 3.数值计算方法
##### 3.1离散布里渊区
将布里渊区离散化为$N\times N$网络，每个k点坐标为
$$
\mathbf{k}_{ij}=(-\pi+\frac{2\pi i}{N},-\pi+\frac{2\pi j}{N}),\quad i,j=0,1,\ldots,N-1
$$
##### 3.2数值计算贝里联络
基于有限差分法,对于$x$分量
$$ 
A_x(\mathbf{k}) \approx i\frac{\langle u(\mathbf{k}) | u(\mathbf{k} + \Delta k_x \hat{x}) \rangle - 1}{\Delta k_x} 
$$

更稳定的中心差分格式
$$
A_x(\mathbf{k})\approx i\frac{\langle u(\mathbf{k})|u(\mathbf{k}+\Delta k_x\hat{x})\rangle-\langle  u(\mathbf{k})|u(\mathbf{k}-\Delta k_x\hat{x})\rangle}{2\Delta k_x}
$$
##### 3.3计算贝里曲率
$$
\Omega(\mathbf{k})= \frac{\partial A_y}{\partial k_x}-\frac{\partial A_x}{\partial k_y}
$$
使用中心差分法
$$
\Omega(\mathbf{k})\approx\frac{A_y(\mathbf{k}+\Delta k_x\hat{x})-A_y(\mathbf{k}-\Delta k_x\hat{x})}{2\Delta k_x}-\frac{A_x(\mathbf{k}+\Delta k_y\hat{y})-A_x(\mathbf{k}-\Delta k_y\hat{y})}{2\Delta k_y}
$$
##### 3.4陈数计算
$$
C=\frac{1}{2\pi}\sum_{i,j}\Omega(\mathbf{k}_{ij})\Delta k_x\Delta k_y
$$
其中$\Delta k_x=\Delta k_y=\frac{2\pi}{N}$
#### 4.相位固定
##### 4.1规范自由度
波函数的$U(1)$规范自由度意味着：
$$
|u'(\mathbf{k})\rangle=e^{i\theta(\mathbf{k})}|u(\mathbf{k})\rangle
$$
这会导致berry connection的变换：
$$
\mathbf{A}'(\mathbf{k})=\mathbf{A}(\mathbf{k})-\nabla_{\mathbf{k}}\theta(\mathbf{k})
$$
虽然berry curvature是规范不变的，但数值计算中直接使用任意相位的波函数会导致berry connection的不连续
##### 4.2相位固定方法
强制波函数的第一个分量为正实数：
$$
\text{如果}u(\mathbf{k})=\begin{pmatrix} a\\b\end{pmatrix},\text{则令}u'(\mathbf{k})=e^{-i\arg(a)}u(\mathbf{k})=\begin{pmatrix}|a|\\e^{-i\arg(a)}b\end{pmatrix}
$$
### 3.2.2 代码
#### 石墨烯陈数计算代码
##### 1.中心差分法
计算出来无能隙的时候是非0（很大，有点奇怪），有能隙的时候是0
```python
import numpy as np  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
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
```
##### 2.有限差分法
```python
import numpy as np  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
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
```
#### berry curvature分布可视化
```python
import numpy as np  
import matplotlib.pyplot as plt  
import time  
from matplotlib.colors import LinearSegmentedColormap  
  
# 物理参数  
a_0 = 0.142 # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
# 正空间基矢  
a1 = np.array([np.sqrt(3) * a / 2, a / 2])  
a2 = np.array([np.sqrt(3) * a / 2, -a / 2])  
  
# 倒空间基矢  
cross_a = np.cross(a1, a2)  
b1 = 2 * np.pi * np.array([a2[1], -a2[0]]) / cross_a  
b2 = 2 * np.pi * np.array([-a1[1], a1[0]]) / cross_a  
  
# 高对称点  
Gamma = np.array([0.0, 0.0])  
K = (2 * b1 + b2) / 3  # K点  
K_prime = (b1 + 2 * b2) / 3  # K'点  
M = (b1 + b2) / 2  # M点  
K_prime_prime = (b1 - b2) / 3 # K''点（方便绘制布里渊区）  
  
  
def graphene_hamiltonian(kx, ky):  
    # 三个最近邻矢量  
    d1 = np.array([a_0, 0])  
    d2 = np.array([-a_0 / 2, a_0 * np.sqrt(3) / 2])  
    d3 = np.array([-a_0 / 2, -a_0 * np.sqrt(3) / 2])  
  
    # 计算跃迁项  
    h = 0  
    for d in [d1, d2, d3]:  
        h += np.exp(1j * (kx * d[0] + ky * d[1]))  
  
    H = np.array([  
        [0, V_ppi_0 * h],  
        [V_ppi_0 * np.conj(h), 0]  
    ])  
    return H  
  
# 固定波函数相位（强制第一个分量为正实数）  
def fix_phase(vector):  
    phase = np.angle(vector[0])  
    return vector * np.exp(-1j * phase)  
  
# 主函数计算陈数并绘制图像  
def main():  
    start_time = time.time()  
    n = 100  # 积分密度  
    delta = 1e-9  # 有限差分步长  
    chern_number = 0  # 陈数初始化  
  
    # 生成k点网格  
    kx_list = np.linspace(-np.pi / a_0, np.pi / a_0, n, endpoint=False)  
    ky_list = np.linspace(-np.pi / a_0, np.pi / a_0, n, endpoint=False)  
    dk = 2 * np.pi / (a_0 * n)  # k空间步长  
  
    # 初始化贝里曲率数组  
    berry_curvature = np.zeros((n, n), dtype=complex)  
  
    for i, kx in enumerate(kx_list):  
        for j, ky in enumerate(ky_list):  
            # 中心点 (k)            H0 = graphene_hamiltonian(kx, ky)  
            eigvals, eigvecs = np.linalg.eigh(H0)  
            vector = fix_phase(eigvecs[:, 0])  # 价带波函数并固定相位  
  
            # 计算x方向贝里联络  
            H_minus_x = graphene_hamiltonian(kx - delta, ky)  
            _, eigvecs_minus_x = np.linalg.eigh(H_minus_x)  
            vec_minus_x = fix_phase(eigvecs_minus_x[:, 0])  
            A_x_minus = 1j * np.vdot(vector, vec_minus_x)  # <u(k)|u(k-dx)>  
  
            H_plus_x = graphene_hamiltonian(kx + delta, ky)  
            _, eigvecs_plus_x = np.linalg.eigh(H_plus_x)  
            vec_plus_x = fix_phase(eigvecs_plus_x[:, 0])  
            A_x_plus = 1j * np.vdot(vector, vec_plus_x)  # <u(k)|u(k+dx)>  
  
            # 计算y方向贝里联络  
            H_minus_y = graphene_hamiltonian(kx, ky - delta)  
            _, eigvecs_minus_y = np.linalg.eigh(H_minus_y)  
            vec_minus_y = fix_phase(eigvecs_minus_y[:, 0])  
            A_y_minus = 1j * np.vdot(vector, vec_minus_y)  # <u(k)|u(k-dy)>  
  
            H_plus_y = graphene_hamiltonian(kx, ky + delta)  
            _, eigvecs_plus_y = np.linalg.eigh(H_plus_y)  
            vec_plus_y = fix_phase(eigvecs_plus_y[:, 0])  
            A_y_plus = 1j * np.vdot(vector, vec_plus_y)  # <u(k)|u(k+dy)>  
  
            # 使用中心差分计算贝里曲率 (更稳定)  
            F = (A_y_plus - A_y_minus) / (2 * delta) - (A_x_plus - A_x_minus) / (2 * delta)  
  
            # 存储贝里曲率  
            berry_curvature[i, j] = F  
  
            # 累加到陈数  
            chern_number += F * dk ** 2  
  
    # 陈数公式 (除以2π)  
    chern_number = chern_number / (2 * np.pi * 1j)  
    print('Chern number = ', "%.11f" % np.real(chern_number))  
    end_time = time.time()  
    print('运行时间(min)=', (end_time - start_time) / 60)  
  
    # 绘制贝里曲率分布  
    plt.figure(figsize=(10, 8))  
  
    # 创建自定义颜色映射，以零为中心  
    colors = ['blue', 'white', 'red']  
    cmap = LinearSegmentedColormap.from_list('berry_cmap', colors, N=256)  
  
    # 绘制贝里曲率  
    im = plt.imshow(np.real(berry_curvature.T),  
                    extent=[-np.pi / a_0, np.pi / a_0, -np.pi / a_0, np.pi / a_0],  
                    origin='lower',  
                    cmap=cmap,  
                    aspect='equal')  
  
    # 添加颜色条  
    cbar = plt.colorbar(im)  
    cbar.set_label('Berry Curvature', fontsize=14)  
  
    # 标记高对称点  
    plt.scatter(Gamma[0], Gamma[1], color='black', marker='o', s=100, label='Γ', edgecolors='white')  
    plt.scatter(K[0], K[1], color='green', marker='^', s=100, label='K', edgecolors='white')  
    plt.scatter(K_prime[0], K_prime[1], color='purple', marker='v', s=100, label="K'", edgecolors='white')  
    plt.scatter(M[0], M[1], color='orange', marker='s', s=100, label='M', edgecolors='white')  
  
    # 在高对称点添加标签  
    plt.text(Gamma[0], Gamma[1], 'Γ', fontsize=16, ha='right', va='bottom', color='white', weight='bold')  
    plt.text(K[0], K[1], 'K', fontsize=16, ha='left', va='bottom', color='white', weight='bold')  
    plt.text(K_prime[0], K_prime[1], "K'", fontsize=16, ha='right', va='top', color='white', weight='bold')  
    plt.text(M[0], M[1], 'M', fontsize=16, ha='left', va='top', color='white', weight='bold')  
  
    # 绘制布里渊区边界 - 修正后的六边形  
    # 计算六边形顶点  
    vertices = [  
        K,  
        K_prime,  
        -K_prime_prime,  
        -K,  
        -K_prime,  
        K_prime_prime,  
        K  
        # 回到起点闭合  
    ]  
    vertices = np.array(vertices)  
    plt.plot(vertices[:, 0], vertices[:, 1], 'w-', linewidth=2.5, alpha=0.8)  
  
    # 设置标题和标签  
    plt.title('Berry Curvature Distribution in Graphene', fontsize=16)  
    plt.xlabel('$k_x$ (1/nm)', fontsize=14)  
    plt.ylabel('$k_y$ (1/nm)', fontsize=14)  
    plt.legend(loc='upper right')  
    plt.grid(True, alpha=0.3)  
  
    # 保存图像  
    # plt.savefig('berry_curvature_graphene.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
    # 绘制3D贝里曲率图  
    fig = plt.figure(figsize=(12, 8))  
    ax = fig.add_subplot(111, projection='3d')  
  
    KX, KY = np.meshgrid(kx_list, ky_list)  
  
    # 绘制表面图  
    surf = ax.plot_surface(KX, KY, np.real(berry_curvature.T),  
                           cmap=cmap,  
                           edgecolor='none',  
                           alpha=0.8)  
  
    # 添加颜色条  
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)  
  
    # 设置标签  
    ax.set_xlabel('$k_x$ (1/nm)', fontsize=12)  
    ax.set_ylabel('$k_y$ (1/nm)', fontsize=12)  
    ax.set_zlabel('Berry Curvature', fontsize=12)  
    ax.set_title('3D Berry Curvature Distribution', fontsize=14)  
  
    # plt.savefig('berry_curvature_3d.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
  
if __name__ == '__main__':  
    main()
```
## 3.3高效法计算石墨烯的陈数
### 3.3.1理论推导（Fukui-Hatsugai-Suzuki方法）
#### 离散化布里渊区
将布里渊区离散化$N\times N$的网格，网格点坐标为：
$$
k_{ij}=(k_x^i,k_y^j)=(\frac{2\pi i}{Na},\frac{2\pi j}{Na}),\quad i,j =0,1,\ldots,N-1
$$
#### $U(1)$链接变量
$$
U_\mu(k)=\frac{\langle u_n(k)|u_n(k+\hat{\mu})\rangle}{|\langle u_n(k)|u_n(k+\hat{\mu})\rangle|},\quad \mu=x,y
$$
>这些链接变量是规范不变的

#### 离散场强
$$
\tilde{F}_{12}(k)=ln[U_x(k)U_y(k+\hat{x})U_x^{-1}(k+\hat{y})U_y(k)]
$$
其中 $-\pi < \frac{1}{i}\tilde{F}_{12}(k) \leq \pi$
#### 离散陈数
$$
\tilde{C}_n = \frac{1}{2\pi i}\sum_k\tilde{F}_{12}(k)
$$
这个定义是规范不变的，对于足够细的网格，$\tilde{C}_n$等于连续情况下的陈数$C_n$
### 3.3.2 代码
```python
import numpy as np  
  
# 物理参数  
a_0 = 1  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
def graphene_hamiltonian(kx, ky, m=0.1):  
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
  
def main():  
    n = 200  
    delta = 2 * np.pi / n  
    chern_number = 0  # 陈数初始化  
    for kx in np.linspace(-np.pi, np.pi, n):  
        for ky in np.linspace(-np.pi, np.pi, n):  
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
            ux_y = ux_y / abs(ux)  
  
            uy_x = np.dot(np.conj(vector_delta_kx), vector_delta_kx_ky)  
            uy_x = uy_x / abs(uy_x)  
  
            F12 = np.log(ux * uy_x * (1 / ux_y) * (1 / uy))  
            # print(F12)  
  
            # 陈数(chern number)  
            chern_number += F12  
    chern_number = chern_number / (2 * np.pi * 1j)  
    print('Chern number = %.11f'%np.real(chern_number))  
  
if __name__ == '__main__':  
    main()
```
#### 绘制贝里曲率的分布
```python
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib import cm  
from mpl_toolkits.mplot3d import Axes3D  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
  
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
```
### 3.3.3理论补充（文献解读）
#### 核心思想
绕过直接计算规范依赖的贝里联络 $A_\mu$，通过构造一个**规范不变的威尔逊圈 (Wilson Loop)** 来定义离散的贝里曲率。该威尔逊圈代表波函数绕一个微小格点方圈平行输运后累积的相位，其相位角在连续极限下等于贝里曲率的通量。

#### 推导步骤

##### 1. 定义 U(1) 链变量 (Link Variable)
在离散布里渊区格点 $k_\ell$ 上，定义连接 $k_\ell$ 和其相邻点 $k_\ell + \hat{\mu}$ 的链变量：
$$
U_\mu(k_\ell) = \frac{ \langle n(k_\ell) | n(k_\ell + \hat{\mu}) \rangle }{ \mathcal{N}_\mu(k_\ell) }
$$
其中 $\mathcal{N}_\mu(k_\ell) = |\langle n(k_\ell) | n(k_\ell + \hat{\mu}) \rangle|$ 是归一化因子，确保 $|U_\mu(k_\ell)| = 1$。

- **物理意义**：代表波函数在两个相邻格点间移动的**相位变化**。
- **规范变换**： under $|n(k)\rangle \to e^{-i\lambda(k)} |n(k)\rangle$,
  $$U_\mu(k_\ell) \to e^{i\lambda(k_\ell)} U_\mu(k_\ell) e^{-i\lambda(k_\ell + \hat{\mu})}$$
  它不是规范不变的，但其变换形式是良好的。

##### 2. 构造威尔逊圈 (Wilson Loop)
考虑一个以 $k_\ell$ 为左下角的微小方格（Plaquette），其四个顶点为：
$k_\ell$ $\to$ $k_\ell + \hat{1}$ $\to$ $k_\ell + \hat{1} + \hat{2}$ $\to$ $k_\ell + \hat{2}$ $\to$ $k_\ell$

按**逆时针方向**将四个边的链变量相乘：
$$
U_{\text{plaquette}} = U_1(k_\ell) \cdot U_2(k_\ell + \hat{1}) \cdot U_1(k_\ell + \hat{2})^{-1} \cdot U_2(k_\ell)^{-1}
$$

**路径分解**:
1.  $U_1(k_\ell)$: $k_\ell \to k_\ell + \hat{1}$
2.  $U_2(k_\ell + \hat{1})$: $k_\ell + \hat{1} \to k_\ell + \hat{1} + \hat{2}$
3.  $U_1(k_\ell + \hat{2})^{-1}$: *(逆操作)* $k_\ell + \hat{1} + \hat{2} \to k_\ell + \hat{2}$
4.  $U_2(k_\ell)^{-1}$: *(逆操作)* $k_\ell + \hat{2} \to k_\ell$

- **物理意义**：$U_{\text{plaquette}}$ 是波函数绕该微小方格平行输运一周后获得的**总相位因子**。
- **关键性质（规范不变性）**：$U_{\text{plaquette}}$ 是**规范不变的**。所有由规范变换引入的相位因子 $e^{i\lambda(\cdots)}$ 在乘积中完全抵消：
$$
e^{i\lambda(k_\ell)} \cdot e^{-i\lambda(k_\ell+\hat{1})} \cdot e^{i\lambda(k_\ell+\hat{1})} \cdot e^{-i\lambda(k_\ell+\hat{1}+\hat{2})} \cdot e^{i\lambda(k_\ell+\hat{1}+\hat{2})} \cdot e^{-i\lambda(k_\ell+\hat{2})} \cdot e^{i\lambda(k_\ell+\hat{2})} \cdot e^{-i\lambda(k_\ell)} = 1
$$
  因此，$U_{\text{plaquette}}$ 的值是一个不依赖于规范选择的物理量。

##### 3. 定义离散场强 (Discrete Field Strength)
将规范不变的威尔逊圈相位定义为离散的贝里曲率通量：
$$
\tilde{F}_{12}(k_\ell) \equiv \ln\left[ U_{\text{plaquette}} \right] = \ln\left[ U_1(k_\ell) U_2(k_\ell + \hat{1}) U_1(k_\ell + \hat{2})^{-1} U_2(k_\ell)^{-1} \right]
$$
为确保值的唯一性，规定取复对数的主值 (Principal Value)，将其虚部限制在 $(-\pi, \pi]$ 区间内：
$$
-\pi < \frac{1}{i} \tilde{F}_{12}(k_\ell) \leq \pi
$$

- **物理意义**：$\frac{1}{i} \tilde{F}_{12}(k_\ell)$ 代表在该微小方格上积分所得的**规范不变的贝里曲率通量**。它与连续定义 $F_{12}(k) \delta k_1 \delta k_2$ 直接对应。
- **为何是曲率**：在连续极限下，根据斯托克斯定理 (Stokes' Theorem)，绕一个环路的贝里联络积分等于穿过该环路的贝里曲率通量：
  $$\oint_{\square} \mathbf{A} \cdot d\mathbf{l} \approx F_{12}(k) \delta k_1 \delta k_2$$
  威尔逊圈的相位 $\theta$ 正是这个环路积分，因此 $\theta \approx F_{12}(k) \delta k_1 \delta k_2$。离散场强 $\tilde{F}_{12}(k_\ell)$ 的虚部就是这个相位 $\theta$。

#### 总结
该方法通过以下流程实现了**规范不变**的离散贝里曲率计算：
**波函数 $\to$ 链变量 $U_\mu$ $\to$ 威尔逊圈 $U_{\text{plaquette}}$ $\to$ 离散场强 $\tilde{F}_{12}$**

其优势在于：
1.  **彻底规避了规范选择问题**，可直接使用数值计算得到的波函数。
2.  **物理图像清晰**，直接与贝里相位的几何解释相关联。
3.  最终陈数 $\tilde{c}_n = \frac{1}{2\pi i} \sum_\ell \tilde{F}_{12}(k_\ell)$ 是整数，即使对粗网格也有效。
## 3.4Wilson loop方法计算石墨烯陈数
### 3.4.1理论推导
#### Wilson loop解决规范依赖问题
$A(\mathbf{k})$本身是规范依赖的，但**其沿着一个闭环环路$\Gamma$的指数积分是规范不变的**。这个指数积分给出的相位称为贝里相位$\Phi_\Gamma$
$$
e^{i\Phi_\Gamma}=exp(i\oint_\Gamma \mathbf{A(k)}\cdot d\mathbf{k})
$$
规范不变性证明：
$$
\begin{align}
|u(\mathbf{k})'\rangle &=e^{i\theta(\mathbf{k})}|u(\mathbf{k})\rangle\\
A'(\mathbf{k}) &=A(\mathbf{k})-\nabla_k\theta(\mathbf{k})\\
e^{i\Phi'_\Gamma}&=exp(i\oint_\Gamma (\mathbf{A(k)}\cdot d\mathbf{k}-\nabla_\mathbf{k}\theta(\mathbf{k}))\\
\oint_\Gamma\nabla_\mathbf{k}\theta(\mathbf{k})d\mathbf{k} &=\theta(\mathbf{k}_{end})-\theta(\mathbf{k}_{start})=0
\end{align}
$$
#### 数值实现：离散化与内积近似
##### 1.离散化积分
环路积分可近似为沿着四条边的求和。利用贝利联络的定义，一小段路径上的积分可以近似为：
$$
i\int_{\mathbf{k}_{p}}^{\mathbf{k}_{p+1}}\langle u(\mathbf{k})|\nabla_{\mathbf{k}}u(\mathbf{k})\rangle\cdot d\mathbf{k}\approx i\langle u(\mathbf{k}_p)|(|u(\mathbf{k}_{p+1})\rangle-|u(\mathbf{k}_{p})\rangle)
$$
##### 2.构造规范不变量
$$
e^{i\Phi_{n}}\approx exp\left(i\sum_\text{边}\langle u(\mathbf{k}_p)|(|u(\mathbf{k}_{p+1})\rangle-|u(\mathbf{k}_{p})\rangle))\right)
$$
##### 3.内积近似
对于足够小的$\Delta k$,指数上求和可近似为连乘，并且有近似关系$e^x\approx1+x$
$$
exp\left(i\langle u(\mathbf{k}_p)|(|u(\mathbf{k}_{p+1})\rangle-|u(\mathbf{k}_{p})\rangle)\right)\approx1+i\langle u(\mathbf{k}_p)|(|u(\mathbf{k}_{p+1})\rangle-|u(\mathbf{k}_{p})\rangle)=\langle u(\mathbf{k_p})|u(\mathbf{k}_{p+1})\rangle
$$
则绕小方格一圈的规范不变相位因子可近似为四个内积的连乘：
$$
e^{i\Phi_n}\approx\langle u(\mathbf{k}_1)|u(\mathbf{k}_2)\rangle\cdot\langle u(\mathbf{k}_2)|u(\mathbf{k_3})\rangle\cdot\langle u(\mathbf{k}_3)|u(\mathbf{k}_4)\rangle\cdot\langle u(\mathbf{k}_4)|u(\mathbf{k}_1)\rangle
$$
##### 4.提取贝里相位
连乘结果是个复数，其辐角即为小方格的贝里相位
$$
\Phi_n=arg\left(\prod_{p=1}^4\langle u(\mathbf{k}_p)|u(\mathbf{k}_{p+1})\right)
$$
其中，$\mathbf{k}_1=\mathbf{k}_5$
**只需计算波函数在不同k点之间的内积，完全无需担心它们的相位或连续性**
##### 5.计算陈数
总的陈数近似为小格子的通量之和
$$
C\approx\frac{1}{2\pi}\sum_n\Phi_n
$$
#### 推广至多带系统
当系统有$N$个能带时，每个$k$点有多个波函数$|u_s(\mathbf{k})\rangle,s=1,2,\ldots,N$，为保证在任意的$U(N)$规范变换下结果不变，需使用**重叠矩阵(Overlap Matrix) **的行列式

##### 1.定义重叠矩阵
相邻两点间的重叠矩阵$S^{p,p+1}$是一个$N\times N$的矩阵，其矩阵元为：
$$
S^{p,p+1}_{ss'}=\langle u_s(\mathbf{k}_p)|u_{s'}(\mathbf{k}_{p+1})\rangle
$$
##### 2.多带系统的表达式
每个小格子的通量的公式推广为：
$$
e^{i\Phi_n}=\prod_{p=1}^{4}\text{det}\left(\mathbf{S}^{p,p+1}\right)
$$
##### 3.提取相位
$$
\Phi_n=arg\left(\prod_{p=1}^{4}\text{det}\left(\mathbf{S}^{p,p+1}\right)\right)
$$
行列式确保了计算在占据能带空间的任何幺正变换下保持不变，这是最高级别的规范不变性。
### 3.4.2代码实现
#### 陈数计算
```python
import numpy as np  
from scipy.linalg import eig  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
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
    k1 = np.linspace(0, 1, n_k, endpoint=False)  
    k2 = np.linspace(0, 1, n_k, endpoint=False)  
    K1, K2 = np.meshgrid(k1, k2, indexing='ij')  
  
    # 存储所有k点的波函数 (价带)  
    u_valence = np.zeros((n_k, n_k, 2), dtype=complex)  
  
    # 计算每个k点的波函数  
    for i in range(n_k):  
        for j in range(n_k):  
            # 将倒空间坐标转换为笛卡尔坐标  
            k_cart = K1[i, j] * b1 + K2[i, j] * b2  
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
```
#### 贝里曲率可视化
```python
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib import cm  
  
# 物理参数  
a_0 = 0.142  # C-C键长 (nm)a = np.sqrt(3) * a_0  # 晶格常数 (nm)V_ppi_0 = -2.7  # 跃迁能 (eV)  
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
```
### 3.4.3结果
![[FHS绘制的贝里曲率可视化.png]]
## 3.5Wilson loop法计算3-regula的贝利曲率和陈数
### 3.5.1代码
```python
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
  
    # --- 二维热图 ---    im = axes_2d[band].pcolormesh(KX, KY, bc_real.T, cmap='RdBu_r', shading='auto')  
    axes_2d[band].set_aspect('equal')  
    axes_2d[band].set_title(f'Band {band + 1}, Chern={chern_number[band]:.2f}', fontsize=10)  
    fig.colorbar(im, ax=axes_2d[band], shrink=0.6, pad=0.03, aspect=20)  # 紧凑colorbar  
  
    # --- 三维曲面 ---    surf = axes_3d[band].plot_surface(KX, KY, bc_real, cmap='RdBu_r',  
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
```
### 3.5.2结果
![[判断过后.png]]