"""
实验2：二维泊松方程（L形区域）
PDE: -Δu = f(x,y)
边界条件: u|_∂Ω = 0
右端项: f(x,y) = 1
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.sin(layer(x))  # 使用Sine激活函数
        x = self.layers[-1](x)
        return x

def poisson_residual(u, x, y):
    """
    计算Poisson方程残差: -Δu - f
    
    参数:
        u: shape=(N, 1)
        x: shape=(N, 1)
        y: shape=(N, 1)
    
    返回:
        residual: shape=(N, 1)
    """
    u.requires_grad_(True)
    xy = torch.cat([x, y], dim=1)
    
    # 计算梯度
    grad_u = torch.autograd.grad(
        outputs=u, inputs=xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    
    # 计算二阶导数
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    u_yy = torch.autograd.grad(
        outputs=u_y, inputs=xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]
    
    # Laplacian
    laplacian_u = u_xx + u_yy
    
    # 右端项
    f = torch.ones_like(u)
    
    # 残差
    residual = -laplacian_u - f
    return residual

def is_in_l_shape(x, y):
    """判断点是否在L形区域内"""
    return not (x > 0.5 and y > 0.5)

def sample_l_shape(N):
    """在L形区域均匀采样"""
    points = []
    while len(points) < N:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        if is_in_l_shape(x, y):
            points.append([x, y])
    return np.array(points)

def sample_boundary(N):
    """在L形边界上采样"""
    points = []
    # 下边界 (y=0, 0<=x<=1)
    n1 = N // 4
    x1 = np.random.uniform(0, 1, n1)
    y1 = np.zeros(n1)
    points.extend(np.column_stack([x1, y1]))
    
    # 左边界 (x=0, 0<=y<=1)
    n2 = N // 4
    x2 = np.zeros(n2)
    y2 = np.random.uniform(0, 1, n2)
    points.extend(np.column_stack([x2, y2]))
    
    # 右边界下段 (x=1, 0<=y<=0.5)
    n3 = N // 4
    x3 = np.ones(n3)
    y3 = np.random.uniform(0, 0.5, n3)
    points.extend(np.column_stack([x3, y3]))
    
    # 上边界右段 (y=0.5, 0.5<=x<=1)
    n4 = N - n1 - n2 - n3
    x4 = np.random.uniform(0.5, 1, n4)
    y4 = 0.5 * np.ones(n4)
    points.extend(np.column_stack([x4, y4]))
    
    return np.array(points)

# 创建模型
layers = [2, 100, 100, 100, 1]
model = PINN(layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

# 采样设置
N_r = 10000  # PDE残差点
N_b = 500    # 边界点

# 训练循环
loss_history = []
for epoch in range(15000):
    optimizer.zero_grad()
    
    # PDE残差点（内部）
    points_r = sample_l_shape(N_r)
    x_r = torch.tensor(points_r[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    y_r = torch.tensor(points_r[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
    xy_r = torch.cat([x_r, y_r], dim=1)
    u_r = model(xy_r)
    residual = poisson_residual(u_r, x_r, y_r)
    loss_r = torch.mean(residual**2)
    
    # 边界条件
    points_b = sample_boundary(N_b)
    x_b = torch.tensor(points_b[:, 0:1], dtype=torch.float32, device=device)
    y_b = torch.tensor(points_b[:, 1:2], dtype=torch.float32, device=device)
    xy_b = torch.cat([x_b, y_b], dim=1)
    u_b = model(xy_b)
    loss_b = torch.mean(u_b**2)
    
    # 总损失
    loss = loss_r + 10.0 * loss_b  # 边界条件权重更大
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if epoch % 2000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Loss_r: {loss_r.item():.6f}, '
              f'Loss_b: {loss_b.item():.6f}')

# 验证：在规则网格上评估
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
X_test, Y_test = np.meshgrid(x_test, y_test)

# 只保留L形区域内的点
mask = np.zeros_like(X_test, dtype=bool)
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        mask[i, j] = is_in_l_shape(X_test[i, j], Y_test[i, j])

X_test_flat = X_test[mask].flatten()
Y_test_flat = Y_test[mask].flatten()
XY_test = torch.tensor(np.column_stack([X_test_flat, Y_test_flat]), 
                       dtype=torch.float32, device=device)

model.eval()
with torch.no_grad():
    u_pred_flat = model(XY_test).cpu().numpy().flatten()

# 重构为网格形式（L形区域外设为NaN）
u_pred = np.full_like(X_test, np.nan)
u_pred[mask] = u_pred_flat

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 训练损失
ax1 = axes[0]
ax1.semilogy(loss_history)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

# 图2: 预测解
ax2 = axes[1]
im2 = ax2.contourf(X_test, Y_test, u_pred, levels=20, cmap='viridis')
ax2.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, fill=True, color='white', zorder=10))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('PINN预测解')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2)

# 图3: L形区域示意图
ax3 = axes[2]
ax3.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
ax3.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, fill=True, color='gray', alpha=0.5))
ax3.set_xlim(-0.1, 1.1)
ax3.set_ylim(-0.1, 1.1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('L形计算域')
ax3.set_aspect('equal')
ax3.grid(True)

plt.tight_layout()
plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment2_results.png', dpi=300, bbox_inches='tight')
print('\n结果已保存到 experiment2_results.png')

# 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 只绘制L形区域内的点
X_plot = X_test[mask]
Y_plot = Y_test[mask]
u_plot = u_pred[mask]

ax.plot_trisurf(X_plot.flatten(), Y_plot.flatten(), u_plot.flatten(), 
                cmap=cm.viridis, alpha=0.9)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
ax.set_title('PINN预测解（3D）')

plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment2_3d.png', dpi=300, bbox_inches='tight')
print('3D可视化已保存到 experiment2_3d.png')
