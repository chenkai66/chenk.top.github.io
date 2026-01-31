"""
实验1：简谐振子
验证HNN和SympNet的能量守恒性质
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

class StandardNN(nn.Module):
    """标准前馈神经网络"""
    def __init__(self, dim=2, hidden_dim=64):
        super(StandardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, z):
        return self.net(z)

class HNN(nn.Module):
    """哈密顿神经网络"""
    def __init__(self, dim=2, hidden_dim=64):
        super(HNN, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 辛矩阵
        self.J = torch.zeros(dim, dim)
        self.J[:dim//2, dim//2:] = torch.eye(dim//2)
        self.J[dim//2:, :dim//2] = -torch.eye(dim//2)
        
    def forward(self, z):
        """计算哈密顿量 H(z)"""
        return self.net(z).squeeze()
    
    def dynamics(self, z):
        """计算导数 dz/dt = J * grad H(z)"""
        z.requires_grad_(True)
        H = self.forward(z)
        grad_H = torch.autograd.grad(H, z, create_graph=True)[0]
        dzdt = torch.matmul(self.J, grad_H.unsqueeze(-1)).squeeze(-1)
        return dzdt

class GradientModule(nn.Module):
    """Gradient模块：q不变，p更新"""
    def __init__(self, dim_q, hidden_dim=64):
        super(GradientModule, self).__init__()
        self.dim_q = dim_q
        self.net = nn.Sequential(
            nn.Linear(dim_q, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        q, p = z[:, :self.dim_q], z[:, self.dim_q:]
        S = self.net(q).squeeze()
        grad_S = torch.autograd.grad(S.sum(), q, create_graph=True)[0]
        p_new = p + grad_S
        return torch.cat([q, p_new], dim=1)

class LiftModule(nn.Module):
    """Lift模块：p不变，q更新"""
    def __init__(self, dim_q, hidden_dim=64):
        super(LiftModule, self).__init__()
        self.dim_q = dim_q
        self.net = nn.Sequential(
            nn.Linear(dim_q, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        q, p = z[:, :self.dim_q], z[:, self.dim_q:]
        T = self.net(p).squeeze()
        grad_T = torch.autograd.grad(T.sum(), p, create_graph=True)[0]
        q_new = q + grad_T
        return torch.cat([q_new, p], dim=1)

class SympNet(nn.Module):
    """辛神经网络"""
    def __init__(self, dim=2, num_layers=4, hidden_dim=64):
        super(SympNet, self).__init__()
        self.dim = dim
        self.dim_q = dim // 2
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GradientModule(self.dim_q, hidden_dim))
            self.layers.append(LiftModule(self.dim_q, hidden_dim))
    
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z

def H_true(q, p):
    """真实哈密顿量"""
    return 0.5 * (p**2 + q**2)

def dynamics_true(z, t):
    """真实动力学"""
    q, p = z
    dqdt = p
    dpdt = -q
    return [dqdt, dpdt]

def generate_data(t_span, z0, dt=0.1):
    """生成训练数据"""
    t = np.arange(t_span[0], t_span[1], dt)
    z = odeint(dynamics_true, z0, t)
    dz = np.array([dynamics_true(z_i, None) for z_i in z])
    return t, z, dz

def train_model(model, z_train, dz_train, epochs=1000, lr=1e-3):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    z_tensor = torch.FloatTensor(z_train)
    dz_tensor = torch.FloatTensor(dz_train)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if isinstance(model, HNN):
            dz_pred = model.dynamics(z_tensor)
        elif isinstance(model, SympNet):
            # SympNet需要不同的训练方式：学习一步映射
            z_next = model(z_tensor)
            dz_pred = (z_next - z_tensor) / dt  # 近似导数
        else:
            dz_pred = model(z_tensor)
        
        loss = torch.mean((dz_pred - dz_tensor)**2)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses

def predict_trajectory(model, z0, t_test, dt):
    """预测轨迹"""
    z_pred = [z0]
    for _ in range(len(t_test) - 1):
        z_current = torch.FloatTensor([z_pred[-1]])
        
        if isinstance(model, HNN):
            dz = model.dynamics(z_current).detach().numpy()[0]
            z_next = z_pred[-1] + dz * dt
        elif isinstance(model, SympNet):
            z_next_tensor = model(z_current)
            z_next = z_next_tensor.detach().numpy()[0]
        else:
            dz = model(z_current).detach().numpy()[0]
            z_next = z_pred[-1] + dz * dt
        
        z_pred.append(z_next)
    
    return np.array(z_pred)

def compute_energy(model, z_traj):
    """计算能量"""
    if isinstance(model, HNN):
        energies = [model(torch.FloatTensor([z])).item() for z in z_traj]
    else:
        # 对于标准NN和SympNet，使用真实哈密顿量
        energies = [H_true(z[0], z[1]) for z in z_traj]
    return np.array(energies)

def main():
    """主函数"""
    # 参数设置
    z0 = [1.0, 0.0]  # 初始条件
    dt = 0.1  # 时间步长
    t_train_span = (0, 10)  # 训练时间范围
    t_test_span = (0, 100)  # 测试时间范围
    
    # 生成训练数据
    print("生成训练数据...")
    t_train, z_train, dz_train = generate_data(t_train_span, z0, dt)
    print(f"训练数据点数: {len(z_train)}")
    
    # 生成测试数据（真实解）
    t_test = np.linspace(t_test_span[0], t_test_span[1], 1000)
    z_test = odeint(dynamics_true, z0, t_test)
    E_true = np.array([H_true(z[0], z[1]) for z in z_test])
    
    # 训练模型
    models = {}
    models['StandardNN'] = StandardNN(dim=2, hidden_dim=64)
    models['HNN'] = HNN(dim=2, hidden_dim=64)
    models['SympNet'] = SympNet(dim=2, num_layers=4, hidden_dim=64)
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        train_model(model, z_train, dz_train, epochs=1000)
    
    # 预测轨迹
    dt_test = t_test[1] - t_test[0]
    trajectories = {}
    energies = {}
    
    for name, model in models.items():
        print(f"\n预测 {name} 轨迹...")
        z_pred = predict_trajectory(model, z0, t_test, dt_test)
        trajectories[name] = z_pred
        energies[name] = compute_energy(model, z_pred)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 能量时间演化
    ax = axes[0, 0]
    ax.plot(t_test, E_true, 'k-', label='True', linewidth=2)
    for name in models.keys():
        ax.plot(t_test, energies[name], '--', label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 能量误差
    ax = axes[0, 1]
    for name in models.keys():
        E_error = np.abs(energies[name] - E_true[0]) / np.abs(E_true[0])
        ax.semilogy(t_test, E_error, '--', label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Relative Energy Error', fontsize=12)
    ax.set_title('Energy Error', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 相空间轨迹
    ax = axes[1, 0]
    ax.plot(z_test[:, 0], z_test[:, 1], 'k-', label='True', linewidth=2)
    for name in models.keys():
        ax.plot(trajectories[name][:, 0], trajectories[name][:, 1], 
                '--', label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('q', fontsize=12)
    ax.set_ylabel('p', fontsize=12)
    ax.set_title('Phase Space Trajectory', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 位置时间演化
    ax = axes[1, 1]
    ax.plot(t_test, z_test[:, 0], 'k-', label='True', linewidth=2)
    for name in models.keys():
        ax.plot(t_test, trajectories[name][:, 0], 
                '--', label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Position q', fontsize=12)
    ax.set_title('Position Evolution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator_results.png', dpi=300, bbox_inches='tight')
    print("\n结果已保存到 harmonic_oscillator_results.png")
    plt.show()

if __name__ == '__main__':
    main()
