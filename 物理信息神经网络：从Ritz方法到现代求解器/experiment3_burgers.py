"""
实验3：Burgers方程
PDE: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
边界条件: u(-1,t) = u(1,t) = 0
初始条件: u(x,0) = -sin(πx)
扩散系数: ν = 0.01/π
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

def burgers_residual(u, x, t, nu=0.01/np.pi):
    """
    计算Burgers方程残差: ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x²
    
    参数:
        u: shape=(N, 1)
        x: shape=(N, 1)
        t: shape=(N, 1)
        nu: 扩散系数
    
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
    
    # 计算 ∂u/∂x 和 ∂²u/∂x²
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
    residual = u_t + u * u_x - nu * u_xx
    return residual

def adaptive_sampling(model, N_new, x_min, x_max, t_min, t_max, nu=0.01/np.pi):
    """
    根据残差大小自适应采样
    
    参数:
        model: PINN模型
        N_new: 新增采样点数
        x_min, x_max: 空间范围
        t_min, t_max: 时间范围
        nu: 扩散系数
    
    返回:
        x_new, t_new: 新的采样点
    """
    # 候选点
    x_candidate = np.random.uniform(x_min, x_max, 10000)
    t_candidate = np.random.uniform(t_min, t_max, 10000)
    
    # 计算残差
    x_t_candidate = torch.tensor(
        np.column_stack([x_candidate, t_candidate]),
        dtype=torch.float32, device=device, requires_grad=True
    )
    
    x_cand_tensor = x_t_candidate[:, 0:1]
    t_cand_tensor = x_t_candidate[:, 1:2]
    
    model.eval()
    with torch.no_grad():
        u_candidate = model(x_t_candidate)
    
    # 需要梯度计算残差
    u_candidate.requires_grad_(True)
    residual = burgers_residual(u_candidate, x_cand_tensor, t_cand_tensor, nu)
    residual_norm = torch.abs(residual).cpu().detach().numpy().flatten()
    
    # 重要性采样：残差大的区域采样概率高
    prob = residual_norm / (residual_norm.sum() + 1e-10)
    prob = prob / prob.sum()  # 归一化
    
    indices = np.random.choice(len(x_candidate), N_new, p=prob)
    
    return x_candidate[indices], t_candidate[indices]

# 创建模型
layers = [2, 100, 100, 100, 1]
model = PINN(layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

# 采样设置
N_r = 10000  # PDE残差点
N_b = 100   # 边界点
N_i = 100   # 初始条件点
nu = 0.01 / np.pi

# 训练循环（带自适应采样）
loss_history = []
x_r_all = []
t_r_all = []

# 初始均匀采样
x_r = torch.tensor(np.random.uniform(-1, 1, N_r), dtype=torch.float32, 
                   device=device, requires_grad=True).reshape(-1, 1)
t_r = torch.tensor(np.random.uniform(0, 1, N_r), dtype=torch.float32, 
                   device=device, requires_grad=True).reshape(-1, 1)

for epoch in range(20000):
    optimizer.zero_grad()
    
    # 每2000个epoch进行一次自适应采样
    if epoch > 0 and epoch % 2000 == 0:
        print(f'Adaptive sampling at epoch {epoch}...')
        x_new, t_new = adaptive_sampling(model, N_r // 2, -1, 1, 0, 1, nu)
        x_r_new = torch.tensor(x_new, dtype=torch.float32, device=device, requires_grad=True).reshape(-1, 1)
        t_r_new = torch.tensor(t_new, dtype=torch.float32, device=device, requires_grad=True).reshape(-1, 1)
        # 合并新旧采样点
        x_r = torch.cat([x_r, x_r_new], dim=0)
        t_r = torch.cat([t_r, t_r_new], dim=0)
        # 随机选择N_r个点
        indices = np.random.choice(len(x_r), N_r, replace=False)
        x_r = x_r[indices]
        t_r = t_r[indices]
        x_r.requires_grad_(True)
        t_r.requires_grad_(True)
    
    # PDE残差点
    x_t_r = torch.cat([x_r, t_r], dim=1)
    u_r = model(x_t_r)
    residual = burgers_residual(u_r, x_r, t_r, nu)
    loss_r = torch.mean(residual**2)
    
    # 边界条件（x=-1和x=1）
    t_b = torch.rand(N_b, 1, device=device)
    x_b_m1 = -torch.ones(N_b, 1, device=device)
    x_b_1 = torch.ones(N_b, 1, device=device)
    u_b_m1 = model(torch.cat([x_b_m1, t_b], dim=1))
    u_b_1 = model(torch.cat([x_b_1, t_b], dim=1))
    loss_b = torch.mean(u_b_m1**2) + torch.mean(u_b_1**2)
    
    # 初始条件（t=0）
    x_i = torch.rand(N_i, 1, device=device) * 2 - 1  # [-1, 1]
    t_i = torch.zeros(N_i, 1, device=device)
    u_i = model(torch.cat([x_i, t_i], dim=1))
    u_i_true = -torch.sin(np.pi * x_i.cpu().numpy())
    u_i_true = torch.tensor(u_i_true, dtype=torch.float32, device=device)
    loss_i = torch.mean((u_i - u_i_true)**2)
    
    # 总损失
    loss = loss_r + 10.0 * loss_b + 10.0 * loss_i
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if epoch % 2000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Loss_r: {loss_r.item():.6f}, '
              f'Loss_b: {loss_b.item():.6f}, Loss_i: {loss_i.item():.6f}')

# 验证
x_test = np.linspace(-1, 1, 200)
t_test = np.linspace(0, 1, 100)
X_test, T_test = np.meshgrid(x_test, t_test)
X_test_flat = torch.tensor(X_test.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
T_test_flat = torch.tensor(T_test.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
X_T_test = torch.cat([X_test_flat, T_test_flat], dim=1)

model.eval()
with torch.no_grad():
    u_pred = model(X_T_test).cpu().numpy().reshape(100, 200)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 图1: 训练损失
ax1 = axes[0, 0]
ax1.semilogy(loss_history)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

# 图2: 不同时刻的解
ax2 = axes[0, 1]
t_slices = [0, 25, 50, 75, 99]
colors = plt.cm.viridis(np.linspace(0, 1, len(t_slices)))
for i, t_idx in enumerate(t_slices):
    ax2.plot(x_test, u_pred[t_idx, :], label=f't={T_test[t_idx, 0]:.2f}', 
             color=colors[i], linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel('u(x,t)')
ax2.set_title('不同时刻的解（激波演化）')
ax2.legend()
ax2.grid(True)

# 图3: 解的时空演化
ax3 = axes[1, 0]
im3 = ax3.contourf(X_test, T_test, u_pred, levels=30, cmap='viridis')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_title('解的时空演化')
plt.colorbar(im3, ax=ax3)

# 图4: 激波位置（最大值位置）
ax4 = axes[1, 1]
u_max_idx = np.argmax(u_pred, axis=1)
x_shock = x_test[u_max_idx]
ax4.plot(T_test[:, 0], x_shock, 'r-', linewidth=2, marker='o', markersize=4)
ax4.set_xlabel('t')
ax4.set_ylabel('激波位置 x')
ax4.set_title('激波位置随时间变化')
ax4.grid(True)

plt.tight_layout()
plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment3_results.png', dpi=300, bbox_inches='tight')
print('\n结果已保存到 experiment3_results.png')

# 计算L2误差（与数值解对比）
# 注意：这里简化处理，实际应该用高精度数值解作为参考
print(f'\n=== 结果分析 ===')
print(f'最大解值: {np.max(u_pred):.4f}')
print(f'最小解值: {np.min(u_pred):.4f}')
print(f'激波在t=1时的位置: x ≈ {x_shock[-1]:.4f}')
