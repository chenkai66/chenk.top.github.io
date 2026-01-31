"""
实验4：Flow Matching vs CNF生成质量对比
比较Flow Matching和CNF在生成任务上的性能
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error


class FlowMatching(nn.Module):
    """Flow Matching模型：学习速度场匹配目标速度场"""
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),  # +1 for time
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim)
        )
    
    def forward(self, t, z):
        t_vec = torch.ones(z.shape[0], 1).to(z) * t
        tz = torch.cat([z, t_vec], dim=1)
        return self.net(tz)


class CNF(nn.Module):
    """连续归一化流：用于对比"""
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, dim)
        )
    
    def forward(self, t, z):
        t_vec = torch.ones(z.shape[0], 1).to(z) * t
        tz = torch.cat([z, t_vec], dim=1)
        return self.net(tz)
    
    def divergence(self, t, z):
        """计算散度"""
        eps = torch.randn_like(z)
        z.requires_grad_(True)
        f = self.forward(t, z)
        grad_f = torch.autograd.grad(f, z, eps, create_graph=True)[0]
        div = (grad_f * eps).sum(dim=1)
        return div


def train_flow_matching(model, data, n_epochs=3000, lr=1e-3, batch_size=64):
    """训练Flow Matching模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 采样时间点
        t = torch.rand(batch_size, 1)
        
        # 线性插值路径：z_t = (1-t)z_0 + t*z_1
        z0 = torch.randn(batch_size, 2)  # 从先验分布采样
        idx = torch.randperm(len(data))[:batch_size]
        z1 = data[idx]  # 从数据分布采样
        
        z_t = (1 - t) * z0 + t * z1
        
        # 目标速度场：u_t = z_1 - z_0
        u_t = z1 - z0
        
        # 预测速度场：v_theta(t, z_t)
        v_t = model(t.squeeze(), z_t)
        
        # 损失：L2距离
        loss = torch.mean((v_t - u_t) ** 2)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses


def train_cnf(model, data, n_epochs=8000, lr=1e-4, batch_size=64):
    """训练CNF模型（简化版，用于对比）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 从数据分布采样
        idx = torch.randperm(len(data))[:batch_size]
        z1 = data[idx]
        
        # 反向ODE求解z0
        z0 = odeint(lambda t, z: -model(t, z), z1, torch.tensor([1., 0.]))
        z0 = z0[-1]
        
        # 前向ODE计算密度
        t_span = torch.linspace(0, 1, 10)
        z_traj = odeint(model, z0, t_span)
        
        # 计算散度积分
        div_integral = 0
        for i, t in enumerate(t_span):
            div = model.divergence(t, z_traj[i])
            div_integral += div * (t_span[1] - t_span[0])
        
        # 负对数似然
        log_p0 = -0.5 * (z0 ** 2).sum(dim=1) - np.log(2 * np.pi)
        log_p1 = log_p0 - div_integral
        loss = -log_p1.mean()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses


def generate_samples(model, n_samples=1000, method='flow_matching'):
    """生成样本"""
    z0 = torch.randn(n_samples, 2)
    
    if method == 'flow_matching':
        z1 = odeint(model, z0, torch.tensor([0., 1.]))
    else:  # CNF
        z1 = odeint(model, z0, torch.tensor([0., 1.]))
    
    return z1[-1].detach().numpy()


def compute_fid(real_samples, fake_samples):
    """计算简化的FID（Fréchet Inception Distance）"""
    # 简化版：使用均值和协方差
    mu_real = np.mean(real_samples, axis=0)
    mu_fake = np.mean(fake_samples, axis=0)
    sigma_real = np.cov(real_samples.T)
    sigma_fake = np.cov(fake_samples.T)
    
    # FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real * sigma_fake))
    diff = mu_real - mu_fake
    fid = np.sum(diff ** 2)
    
    # 简化：只计算迹的差异
    fid += np.trace(sigma_real + sigma_fake - 2 * np.sqrt(sigma_real * sigma_fake))
    
    return fid


def visualize_comparison(data, fm_samples, cnf_samples):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_title('Real Data (Moons)')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    axes[1].scatter(fm_samples[:, 0], fm_samples[:, 1], alpha=0.5, s=10, c='green')
    axes[1].set_title('Flow Matching Generated')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    axes[2].scatter(cnf_samples[:, 0], cnf_samples[:, 1], alpha=0.5, s=10, c='red')
    axes[2].set_title('CNF Generated')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('experiment4_comparison.png', dpi=150)
    print("Comparison visualization saved to experiment4_comparison.png")


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成Moons数据集
    print("Generating Moons dataset...")
    data, _ = make_moons(1000, noise=0.1)
    data = torch.tensor(data, dtype=torch.float32)
    
    # 可视化真实数据
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    plt.title('Real Data (Moons)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment4_real_data.png', dpi=150)
    print("Real data saved to experiment4_real_data.png\n")
    
    # 训练Flow Matching
    print("Training Flow Matching...")
    model_fm = FlowMatching()
    losses_fm = train_flow_matching(model_fm, data, n_epochs=3000, lr=1e-3)
    
    # 训练CNF
    print("\nTraining CNF...")
    model_cnf = CNF()
    losses_cnf = train_cnf(model_cnf, data, n_epochs=8000, lr=1e-4)
    
    # 生成样本
    print("\nGenerating samples...")
    fm_samples = generate_samples(model_fm, n_samples=1000, method='flow_matching')
    cnf_samples = generate_samples(model_cnf, n_samples=1000, method='cnf')
    
    # 评估（简化版）
    fid_fm = compute_fid(data.numpy(), fm_samples)
    fid_cnf = compute_fid(data.numpy(), cnf_samples)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Method':<30} {'FID (simplified)':<20}")
    print("-"*60)
    print(f"{'Flow Matching':<30} {fid_fm:<20.4f}")
    print(f"{'CNF':<30} {fid_cnf:<20.4f}")
    print("="*60)
    
    # 可视化
    visualize_comparison(data.numpy(), fm_samples, cnf_samples)
    
    # 绘制损失曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(losses_fm)
    axes[0].set_title('Flow Matching Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(losses_cnf)
    axes[1].set_title('CNF Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment4_losses.png', dpi=150)
    print("Loss curves saved to experiment4_losses.png")
