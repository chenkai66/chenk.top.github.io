"""
实验2：双摆系统
验证HNN在混沌系统中的表现
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

class HNN(nn.Module):
    """哈密顿神经网络"""
    def __init__(self, dim=4, hidden_dim=128):
        super(HNN, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 辛矩阵
        self.J = torch.zeros(dim, dim)
        n = dim // 2
        self.J[:n, n:] = torch.eye(n)
        self.J[n:, :n] = -torch.eye(n)
        
    def forward(self, z):
        return self.net(z).squeeze()
    
    def dynamics(self, z):
        z.requires_grad_(True)
        H = self.forward(z)
        grad_H = torch.autograd.grad(H, z, create_graph=True)[0]
        dzdt = torch.matmul(self.J, grad_H.unsqueeze(-1)).squeeze(-1)
        return dzdt

def H_double_pendulum(z, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8):
    """双摆系统的真实哈密顿量"""
    q1, q2, p1, p2 = z
    
    # 动能项（复杂形式）
    cos_diff = np.cos(q1 - q2)
    denom = m2 * l2**2 * (m1 + m2 * np.sin(q1 - q2)**2)
    
    T = (m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 
         - 2 * m2 * l1 * l2 * p1 * p2 * cos_diff) / (2 * denom)
    
    # 势能项
    V = -(m1 + m2) * g * l1 * np.cos(q1) - m2 * g * l2 * np.cos(q2)
    
    return T + V

def dynamics_double_pendulum(z, t, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8):
    """双摆系统的真实动力学"""
    q1, q2, p1, p2 = z
    
    # 使用数值方法计算哈密顿方程的导数
    eps = 1e-6
    H0 = H_double_pendulum(z, m1, m2, l1, l2, g)
    
    # 计算偏导数
    dH_dq1 = (H_double_pendulum([q1+eps, q2, p1, p2], m1, m2, l1, l2, g) - H0) / eps
    dH_dq2 = (H_double_pendulum([q1, q2+eps, p1, p2], m1, m2, l1, l2, g) - H0) / eps
    dH_dp1 = (H_double_pendulum([q1, q2, p1+eps, p2], m1, m2, l1, l2, g) - H0) / eps
    dH_dp2 = (H_double_pendulum([q1, q2, p1, p2+eps], m1, m2, l1, l2, g) - H0) / eps
    
    dq1dt = dH_dp1
    dq2dt = dH_dp2
    dp1dt = -dH_dq1
    dp2dt = -dH_dq2
    
    return [dq1dt, dq2dt, dp1dt, dp2dt]

def generate_training_data(num_trajectories=10, t_span=(0, 5), dt=0.05):
    """生成训练数据"""
    all_z = []
    all_dz = []
    
    for _ in range(num_trajectories):
        # 随机初始条件
        z0 = np.random.uniform([-np.pi, -np.pi, -2, -2], 
                               [np.pi, np.pi, 2, 2])
        t = np.arange(t_span[0], t_span[1], dt)
        z = odeint(dynamics_double_pendulum, z0, t)
        dz = np.array([dynamics_double_pendulum(z_i, None) for z_i in z])
        
        all_z.append(z)
        all_dz.append(dz)
    
    z_train = np.concatenate(all_z, axis=0)
    dz_train = np.concatenate(all_dz, axis=0)
    
    return z_train, dz_train

def train_hnn(model, z_train, dz_train, epochs=2000, lr=1e-3):
    """训练HNN"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    z_tensor = torch.FloatTensor(z_train)
    dz_tensor = torch.FloatTensor(dz_train)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        dz_pred = model.dynamics(z_tensor)
        loss = torch.mean((dz_pred - dz_tensor)**2)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses

def predict_trajectory(model, z0, t_test, dt):
    """预测轨迹"""
    z_pred = [z0]
    for _ in range(len(t_test) - 1):
        z_current = torch.FloatTensor([z_pred[-1]])
        dz = model.dynamics(z_current).detach().numpy()[0]
        z_next = z_pred[-1] + dz * dt
        z_pred.append(z_next)
    return np.array(z_pred)

def compute_energy(model, z_traj):
    """计算能量"""
    energies = [model(torch.FloatTensor([z])).item() for z in z_traj]
    return np.array(energies)

def visualize_double_pendulum(z_traj, title="Double Pendulum"):
    """可视化双摆轨迹"""
    q1, q2 = z_traj[:, 0], z_traj[:, 1]
    
    # 转换为笛卡尔坐标
    l1, l2 = 1.0, 1.0
    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)
    x2 = x1 + l2 * np.sin(q2)
    y2 = y1 - l2 * np.cos(q2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 相空间轨迹
    ax = axes[0]
    ax.plot(q1, q2, 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('$q_1$', fontsize=12)
    ax.set_ylabel('$q_2$', fontsize=12)
    ax.set_title(f'{title} - Phase Space', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 物理空间轨迹
    ax = axes[1]
    ax.plot(x1, y1, 'b-', label='Pendulum 1', linewidth=2, alpha=0.7)
    ax.plot(x2, y2, 'r-', label='Pendulum 2', linewidth=2, alpha=0.7)
    ax.plot([0, x1[0]], [0, y1[0]], 'ko-', markersize=8, linewidth=2)
    ax.plot([x1[0], x2[0]], [y1[0], y2[0]], 'ro-', markersize=8, linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{title} - Physical Space', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    # 生成训练数据
    print("生成训练数据...")
    z_train, dz_train = generate_training_data(num_trajectories=20, 
                                                t_span=(0, 5), dt=0.05)
    print(f"训练数据点数: {len(z_train)}")
    
    # 训练HNN
    model = HNN(dim=4, hidden_dim=128)
    print("\n训练HNN...")
    train_hnn(model, z_train, dz_train, epochs=2000)
    
    # 测试：规则运动
    print("\n测试规则运动...")
    z0_regular = [np.pi/4, np.pi/4, 0.0, 0.0]
    t_test = np.linspace(0, 20, 1000)
    z_true_regular = odeint(dynamics_double_pendulum, z0_regular, t_test)
    z_pred_regular = predict_trajectory(model, z0_regular, t_test, 
                                        t_test[1] - t_test[0])
    
    # 测试：混沌运动
    print("测试混沌运动...")
    z0_chaos = [np.pi/2, np.pi/2, 0.0, 0.0]
    z_true_chaos = odeint(dynamics_double_pendulum, z0_chaos, t_test)
    z_pred_chaos = predict_trajectory(model, z0_chaos, t_test, 
                                     t_test[1] - t_test[0])
    
    # 计算能量
    E_true_regular = np.array([H_double_pendulum(z) for z in z_true_regular])
    E_pred_regular = compute_energy(model, z_pred_regular)
    E_true_chaos = np.array([H_double_pendulum(z) for z in z_true_chaos])
    E_pred_chaos = compute_energy(model, z_pred_chaos)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 规则运动：能量
    ax = axes[0, 0]
    ax.plot(t_test, E_true_regular, 'k-', label='True', linewidth=2)
    ax.plot(t_test, E_pred_regular, 'b--', label='HNN', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Regular Motion - Energy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 规则运动：相空间
    ax = axes[0, 1]
    ax.plot(z_true_regular[:, 0], z_true_regular[:, 1], 'k-', 
            label='True', linewidth=2, alpha=0.7)
    ax.plot(z_pred_regular[:, 0], z_pred_regular[:, 1], 'b--', 
            label='HNN', linewidth=2, alpha=0.7)
    ax.set_xlabel('$q_1$', fontsize=12)
    ax.set_ylabel('$q_2$', fontsize=12)
    ax.set_title('Regular Motion - Phase Space', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 混沌运动：能量
    ax = axes[1, 0]
    ax.plot(t_test, E_true_chaos, 'k-', label='True', linewidth=2)
    ax.plot(t_test, E_pred_chaos, 'r--', label='HNN', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Chaotic Motion - Energy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 混沌运动：相空间
    ax = axes[1, 1]
    ax.plot(z_true_chaos[:, 0], z_true_chaos[:, 1], 'k-', 
            label='True', linewidth=1, alpha=0.5)
    ax.plot(z_pred_chaos[:, 0], z_pred_chaos[:, 1], 'r--', 
            label='HNN', linewidth=1, alpha=0.5)
    ax.set_xlabel('$q_1$', fontsize=12)
    ax.set_ylabel('$q_2$', fontsize=12)
    ax.set_title('Chaotic Motion - Phase Space', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('double_pendulum_results.png', dpi=300, bbox_inches='tight')
    print("\n结果已保存到 double_pendulum_results.png")
    plt.show()

if __name__ == '__main__':
    main()
