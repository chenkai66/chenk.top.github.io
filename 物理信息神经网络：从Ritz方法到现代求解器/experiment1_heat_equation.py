"""
实验1：一维热方程
PDE: ∂u/∂t = α ∂²u/∂x²
解析解: u(x,t) = e^(-απ²t) sin(πx)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# PINN网络架构
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
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

def heat_eq_residual(u, x, t, alpha=0.1):
    """
    计算热方程残差: ∂u/∂t - α ∂²u/∂x²
    
    参数:
        u: shape=(N, 1), 函数值
        x: shape=(N, 1), 空间坐标
        t: shape=(N, 1), 时间坐标
        alpha: 扩散系数
    
    返回:
        residual: shape=(N, 1)
    """
    u.requires_grad_(True)
    
    # 计算 ∂u/∂t
    u_t = torch.autograd.grad(
        outputs=u, inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    # 计算 ∂²u/∂x²
    u_x = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]
    
    # 残差
    residual = u_t - alpha * u_xx
    return residual

# 解析解
def analytical_solution(x, t, alpha=0.1):
    """解析解: u(x,t) = e^(-απ²t) sin(πx)"""
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

# 创建模型
layers = [2, 50, 50, 50, 1]
model = PINN(layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

# 采样设置
N_r = 10000  # PDE残差点
N_b = 100   # 边界点
N_i = 100   # 初始条件点

# 训练循环
loss_history = []
for epoch in range(10000):
    optimizer.zero_grad()
    
    # PDE残差点（内部）
    x_r = torch.rand(N_r, 1, device=device, requires_grad=True)
    t_r = torch.rand(N_r, 1, device=device, requires_grad=True)
    x_t_r = torch.cat([x_r, t_r], dim=1)
    u_r = model(x_t_r)
    residual = heat_eq_residual(u_r, x_r, t_r)
    loss_r = torch.mean(residual**2)
    
    # 边界条件（x=0和x=1）
    t_b = torch.rand(N_b, 1, device=device)
    x_b_0 = torch.zeros(N_b, 1, device=device)
    x_b_1 = torch.ones(N_b, 1, device=device)
    u_b_0 = model(torch.cat([x_b_0, t_b], dim=1))
    u_b_1 = model(torch.cat([x_b_1, t_b], dim=1))
    loss_b = torch.mean(u_b_0**2) + torch.mean(u_b_1**2)
    
    # 初始条件（t=0）
    x_i = torch.rand(N_i, 1, device=device)
    t_i = torch.zeros(N_i, 1, device=device)
    u_i = model(torch.cat([x_i, t_i], dim=1))
    u_i_true = torch.sin(np.pi * x_i.cpu().numpy())
    u_i_true = torch.tensor(u_i_true, dtype=torch.float32, device=device)
    loss_i = torch.mean((u_i - u_i_true)**2)
    
    # 总损失
    loss = loss_r + loss_b + loss_i
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Loss_r: {loss_r.item():.6f}, '
              f'Loss_b: {loss_b.item():.6f}, Loss_i: {loss_i.item():.6f}')

# 验证
x_test = np.linspace(0, 1, 100)
t_test = np.linspace(0, 1, 100)
X_test, T_test = np.meshgrid(x_test, t_test)
X_test_flat = torch.tensor(X_test.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
T_test_flat = torch.tensor(T_test.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
X_T_test = torch.cat([X_test_flat, T_test_flat], dim=1)

model.eval()
with torch.no_grad():
    u_pred = model(X_T_test).cpu().numpy().reshape(100, 100)

# 解析解
u_true = analytical_solution(X_test, T_test)

# 误差分析
error = np.abs(u_pred - u_true)
l2_error = np.sqrt(np.mean(error**2))
linf_error = np.max(error)
print(f'\n=== 误差分析 ===')
print(f'L2 error: {l2_error:.6f}')
print(f'L∞ error: {linf_error:.6f}')

# 可视化
fig = plt.figure(figsize=(15, 5))

# 图1: 训练损失曲线
ax1 = fig.add_subplot(131)
ax1.semilogy(loss_history)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

# 图2: 预测解 vs 真解（t=0.5时刻）
ax2 = fig.add_subplot(132)
t_slice = 50  # t=0.5
ax2.plot(x_test, u_pred[t_slice, :], 'b-', label='PINN预测', linewidth=2)
ax2.plot(x_test, u_true[t_slice, :], 'r--', label='解析解', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel('u(x, t=0.5)')
ax2.set_title('t=0.5时刻的解')
ax2.legend()
ax2.grid(True)

# 图3: 误差分布
ax3 = fig.add_subplot(133)
im = ax3.contourf(X_test, T_test, error, levels=20, cmap='hot')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_title('绝对误差分布')
plt.colorbar(im, ax=ax3)

plt.tight_layout()
plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment1_results.png', dpi=300, bbox_inches='tight')
print('\n结果已保存到 experiment1_results.png')

# 3D可视化
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X_test, T_test, u_pred, cmap=cm.viridis, alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u(x,t)')
ax1.set_title('PINN预测解')

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_test, T_test, u_true, cmap=cm.viridis, alpha=0.8)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x,t)')
ax2.set_title('解析解')

plt.tight_layout()
plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment1_3d.png', dpi=300, bbox_inches='tight')
print('3D可视化已保存到 experiment1_3d.png')
