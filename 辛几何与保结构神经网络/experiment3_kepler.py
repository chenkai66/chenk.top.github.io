"""
实验3：Kepler问题（二体问题）
验证HNN保持能量和角动量守恒
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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

def H_kepler(z, G=1.0, M=1.0, m=1.0):
    """Kepler问题的真实哈密顿量（极坐标）"""
    r, theta, pr, ptheta = z
    H = (pr**2) / (2*m) + (ptheta**2) / (2*m*r**2) - G*M*m / r
    return H

def dynamics_kepler(z, t, G=1.0, M=1.0, m=1.0):
    """Kepler问题的真实动力学"""
    r, theta, pr, ptheta = z
    
    # 哈密顿方程
    drdt = pr / m
    dthetadt = ptheta / (m * r**2)
    dprdt = (ptheta**2) / (m * r**3) - G*M*m / r**2
    dpthetadt = 0  # 角动量守恒
    
    return [drdt, dthetadt, dprdt, dpthetadt]

def cartesian_to_polar(x, y, px, py):
    """笛卡尔坐标转极坐标"""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    pr = (x*px + y*py) / r
    ptheta = x*py - y*px
    return r, theta, pr, ptheta

def polar_to_cartesian(r, theta, pr, ptheta):
    """极坐标转笛卡尔坐标"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    px = pr * np.cos(theta) - ptheta * np.sin(theta) / r
    py = pr * np.sin(theta) + ptheta * np.cos(theta) / r
    return x, y, px, py

def generate_elliptic_orbit(a=1.0, e=0.5, num_points=100):
    """生成椭圆轨道数据"""
    # 初始条件（在近地点）
    r0 = a * (1 - e)
    theta0 = 0.0
    pr0 = 0.0
    
    # 计算角动量（从能量和角动量关系）
    E = -1.0 / (2*a)  # 椭圆轨道能量
    ptheta0 = np.sqrt(2 * a * (1 - e**2))  # 从轨道参数计算
    
    z0 = [r0, theta0, pr0, ptheta0]
    
    # 积分一个周期
    T = 2 * np.pi * np.sqrt(a**3)  # 开普勒第三定律
    t = np.linspace(0, T, num_points)
    z = odeint(dynamics_kepler, z0, t, args=(1.0, 1.0, 1.0))
    
    return t, z

def generate_training_data(num_orbits=5):
    """生成训练数据"""
    all_z = []
    all_dz = []
    
    for _ in range(num_orbits):
        # 随机椭圆轨道参数
        a = np.random.uniform(0.5, 2.0)
        e = np.random.uniform(0.1, 0.8)
        
        t, z = generate_elliptic_orbit(a, e, num_points=100)
        dz = np.array([dynamics_kepler(z_i, None) for z_i in z])
        
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
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        dz_pred = model.dynamics(z_tensor)
        loss = torch.mean((dz_pred - dz_tensor)**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

def predict_trajectory(model, z0, t_test, dt):
    """预测轨迹"""
    z_pred = [z0]
    for _ in range(len(t_test) - 1):
        z_current = torch.FloatTensor([z_pred[-1]])
        dz = model.dynamics(z_current).detach().numpy()[0]
        z_next = z_pred[-1] + dz * dt
        z_pred.append(z_next)
    return np.array(z_pred)

def compute_conserved_quantities(z_traj):
    """计算守恒量"""
    energies = []
    angular_momenta = []
    
    for z in z_traj:
        r, theta, pr, ptheta = z
        E = H_kepler(z)
        L = ptheta  # 角动量
        energies.append(E)
        angular_momenta.append(L)
    
    return np.array(energies), np.array(angular_momenta)

def main():
    """主函数"""
    # 生成训练数据
    print("生成训练数据...")
    z_train, dz_train = generate_training_data(num_orbits=10)
    print(f"训练数据点数: {len(z_train)}")
    
    # 训练HNN
    model = HNN(dim=4, hidden_dim=128)
    print("\n训练HNN...")
    train_hnn(model, z_train, dz_train, epochs=2000)
    
    # 测试：椭圆轨道
    print("\n测试椭圆轨道...")
    a, e = 1.0, 0.5
    r0 = a * (1 - e)
    theta0 = 0.0
    pr0 = 0.0
    ptheta0 = np.sqrt(2 * a * (1 - e**2))
    z0 = [r0, theta0, pr0, ptheta0]
    
    T = 2 * np.pi * np.sqrt(a**3)
    t_test = np.linspace(0, 3*T, 1000)  # 3个周期
    dt = t_test[1] - t_test[0]
    
    z_true = odeint(dynamics_kepler, z0, t_test, args=(1.0, 1.0, 1.0))
    z_pred = predict_trajectory(model, z0, t_test, dt)
    
    # 计算守恒量
    E_true, L_true = compute_conserved_quantities(z_true)
    E_pred, L_pred = compute_conserved_quantities(z_pred)
    
    # 转换为笛卡尔坐标可视化
    x_true, y_true, _, _ = zip(*[polar_to_cartesian(*z) for z in z_true])
    x_pred, y_pred, _, _ = zip(*[polar_to_cartesian(*z) for z in z_pred])
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 轨道
    ax = axes[0, 0]
    ax.plot(x_true, y_true, 'k-', label='True', linewidth=2)
    ax.plot(x_pred, y_pred, 'b--', label='HNN', linewidth=2, alpha=0.7)
    ax.plot(0, 0, 'ro', markersize=10, label='Central Body')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Orbit Trajectory', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 能量守恒
    ax = axes[0, 1]
    ax.plot(t_test, E_true, 'k-', label='True', linewidth=2)
    ax.plot(t_test, E_pred, 'b--', label='HNN', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 角动量守恒
    ax = axes[1, 0]
    ax.plot(t_test, L_true, 'k-', label='True', linewidth=2)
    ax.plot(t_test, L_pred, 'b--', label='HNN', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Angular Momentum', fontsize=12)
    ax.set_title('Angular Momentum Conservation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 能量误差
    ax = axes[1, 1]
    E_error = np.abs(E_pred - E_true[0]) / np.abs(E_true[0])
    ax.semilogy(t_test, E_error, 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Relative Energy Error', fontsize=12)
    ax.set_title('Energy Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kepler_results.png', dpi=300, bbox_inches='tight')
    print("\n结果已保存到 kepler_results.png")
    plt.show()

if __name__ == '__main__':
    main()
