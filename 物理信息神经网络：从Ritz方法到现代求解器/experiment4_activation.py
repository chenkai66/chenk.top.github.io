"""
实验4：激活函数对比
测试问题：二维Poisson方程
解析解: u(x,y) = sin(πx) sin(πy)
对比激活函数: Tanh, Sine, Swish, GELU
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义不同的激活函数
class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GELUActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

class PINN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation_name = activation
        
        # 选择激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sine':
            self.activation = SineActivation()
        elif activation == 'swish':
            self.activation = SwishActivation()
        elif activation == 'gelu':
            self.activation = GELUActivation()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            # Sine激活需要特殊初始化
            if activation == 'sine':
                nn.init.uniform_(self.layers[-1].weight, -1, 1)
            else:
                nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

def poisson_residual_2d(u, x, y):
    """
    计算二维Poisson方程残差: -Δu - f
    解析解: u(x,y) = sin(πx) sin(πy)
    右端项: f(x,y) = 2π² sin(πx) sin(πy)
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
    f = 2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    # 残差
    residual = -laplacian_u - f
    return residual

def train_pinn(activation='tanh', epochs=5000):
    """训练PINN并返回结果"""
    print(f'\n=== 训练 {activation.upper()} 激活函数 ===')
    
    # 创建模型
    layers = [2, 50, 50, 50, 1]
    model = PINN(layers, activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    # 采样设置
    N_r = 5000  # PDE残差点
    N_b = 200   # 边界点
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE残差点（内部）
        x_r = torch.rand(N_r, 1, device=device, requires_grad=True)
        y_r = torch.rand(N_r, 1, device=device, requires_grad=True)
        xy_r = torch.cat([x_r, y_r], dim=1)
        u_r = model(xy_r)
        residual = poisson_residual_2d(u_r, x_r, y_r)
        loss_r = torch.mean(residual**2)
        
        # 边界条件
        # 下边界 (y=0)
        x_b1 = torch.rand(N_b//4, 1, device=device)
        y_b1 = torch.zeros(N_b//4, 1, device=device)
        u_b1 = model(torch.cat([x_b1, y_b1], dim=1))
        loss_b1 = torch.mean(u_b1**2)
        
        # 上边界 (y=1)
        x_b2 = torch.rand(N_b//4, 1, device=device)
        y_b2 = torch.ones(N_b//4, 1, device=device)
        u_b2 = model(torch.cat([x_b2, y_b2], dim=1))
        loss_b2 = torch.mean(u_b2**2)
        
        # 左边界 (x=0)
        x_b3 = torch.zeros(N_b//4, 1, device=device)
        y_b3 = torch.rand(N_b//4, 1, device=device)
        u_b3 = model(torch.cat([x_b3, y_b3], dim=1))
        loss_b3 = torch.mean(u_b3**2)
        
        # 右边界 (x=1)
        x_b4 = torch.ones(N_b//4, 1, device=device)
        y_b4 = torch.rand(N_b//4, 1, device=device)
        u_b4 = model(torch.cat([x_b4, y_b4], dim=1))
        loss_b4 = torch.mean(u_b4**2)
        
        loss_b = loss_b1 + loss_b2 + loss_b3 + loss_b4
        
        # 总损失
        loss = loss_r + 10.0 * loss_b
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    training_time = time.time() - start_time
    
    # 验证
    x_test = np.linspace(0, 1, 50)
    y_test = np.linspace(0, 1, 50)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_test_flat = torch.tensor(X_test.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    Y_test_flat = torch.tensor(Y_test.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
    XY_test = torch.cat([X_test_flat, Y_test_flat], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(XY_test).cpu().numpy().reshape(50, 50)
    
    # 解析解
    u_true = np.sin(np.pi * X_test) * np.sin(np.pi * Y_test)
    
    # 误差分析
    error = np.abs(u_pred - u_true)
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(error)
    
    # 找到收敛迭代数（损失降到1e-4以下）
    converged_epoch = next((i for i, l in enumerate(loss_history) if l < 1e-4), epochs)
    
    return {
        'activation': activation,
        'model': model,
        'loss_history': loss_history,
        'u_pred': u_pred,
        'u_true': u_true,
        'l2_error': l2_error,
        'linf_error': linf_error,
        'training_time': training_time,
        'converged_epoch': converged_epoch
    }

# 训练所有激活函数
activations = ['tanh', 'sine', 'swish', 'gelu']
results = {}

for act in activations:
    results[act] = train_pinn(activation=act, epochs=5000)

# 结果汇总
print('\n=== 结果汇总 ===')
print(f"{'激活函数':<10} {'L2误差':<15} {'L∞误差':<15} {'训练时间(s)':<15} {'收敛迭代数':<15}")
print('-' * 70)
for act in activations:
    r = results[act]
    print(f"{act:<10} {r['l2_error']:<15.6e} {r['linf_error']:<15.6e} "
          f"{r['training_time']:<15.2f} {r['converged_epoch']:<15}")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 训练曲线
for i, act in enumerate(activations):
    ax = axes[0, i]
    ax.semilogy(results[act]['loss_history'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{act.upper()} - Training Loss')
    ax.grid(True)

# 预测解 vs 真解（沿对角线）
for i, act in enumerate(activations):
    ax = axes[1, i]
    x_diag = np.linspace(0, 1, 50)
    y_diag = x_diag
    u_pred_diag = results[act]['u_pred'][np.arange(50), np.arange(50)]
    u_true_diag = np.sin(np.pi * x_diag) * np.sin(np.pi * y_diag)
    
    ax.plot(x_diag, u_pred_diag, 'b-', label='PINN预测', linewidth=2)
    ax.plot(x_diag, u_true_diag, 'r--', label='解析解', linewidth=2)
    ax.set_xlabel('x (y=x)')
    ax.set_ylabel('u(x,x)')
    ax.set_title(f'{act.upper()} - 对角线切片')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment4_results.png', dpi=300, bbox_inches='tight')
print('\n结果已保存到 experiment4_results.png')

# 对比表格可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
for act in activations:
    r = results[act]
    table_data.append([
        act.upper(),
        f"{r['l2_error']:.2e}",
        f"{r['linf_error']:.2e}",
        f"{r['training_time']:.1f}s",
        f"{r['converged_epoch']}"
    ])

table = ax.table(cellText=table_data,
                 colLabels=['激活函数', 'L2误差', 'L∞误差', '训练时间', '收敛迭代数'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

plt.savefig('./物理信息神经网络：从Ritz方法到现代求解器/experiment4_table.png', dpi=300, bbox_inches='tight')
print('对比表格已保存到 experiment4_table.png')
