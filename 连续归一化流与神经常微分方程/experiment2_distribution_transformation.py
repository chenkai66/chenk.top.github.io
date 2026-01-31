"""
实验2：二维分布变换可视化
可视化连续归一化流如何将简单分布（高斯）变换为复杂分布（月牙形）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from scipy.stats import multivariate_normal


class CNF(nn.Module):
    """连续归一化流：学习速度场并计算散度"""
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),  # +1 for time
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, dim)
        )
    
    def forward(self, t, z):
        """速度场：dz/dt = f_theta(t, z)"""
        t_vec = torch.ones(z.shape[0], 1).to(z) * t
        tz = torch.cat([z, t_vec], dim=1)
        return self.net(tz)
    
    def divergence(self, t, z):
        """使用Hutchinson迹估计计算散度：div f = E[epsilon^T J_f epsilon]"""
        eps = torch.randn_like(z)
        z.requires_grad_(True)
        f = self.forward(t, z)
        # 计算梯度：df/dz
        grad_f = torch.autograd.grad(f, z, eps, create_graph=True)[0]
        # 散度估计：tr(J_f) ≈ epsilon^T J_f epsilon
        div = (grad_f * eps).sum(dim=1)
        return div


def target_distribution(n_samples):
    """生成月牙形分布：两个高斯的混合 + 非线性变换"""
    # 两个高斯的混合
    mix1 = multivariate_normal([-1, 0], [[0.5, 0], [0, 0.5]])
    mix2 = multivariate_normal([1, 0], [[0.5, 0], [0, 0.5]])
    samples1 = mix1.rvs(n_samples // 2)
    samples2 = mix2.rvs(n_samples // 2)
    samples = np.vstack([samples1, samples2])
    
    # 非线性变换形成月牙
    angle = np.arctan2(samples[:, 1], samples[:, 0])
    radius = np.linalg.norm(samples, axis=1)
    samples[:, 0] = radius * np.cos(angle + 0.3 * radius)
    samples[:, 1] = radius * np.sin(angle + 0.3 * radius)
    
    return torch.tensor(samples, dtype=torch.float32)


def train_cnf(model, target_samples, n_epochs=5000, lr=1e-4, batch_size=64):
    """训练连续归一化流"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 从目标分布采样batch
        idx = torch.randperm(len(target_samples))[:batch_size]
        z1 = target_samples[idx]
        
        # 反向ODE求解z0（从z1回到z0）
        z0 = odeint(lambda t, z: -model(t, z), z1, torch.tensor([1., 0.]))
        z0 = z0[-1]
        
        # 前向ODE计算密度演化
        t_span = torch.linspace(0, 1, 10)
        z_traj = odeint(model, z0, t_span)
        
        # 计算散度积分：int_0^1 div f dt
        div_integral = 0
        for i, t in enumerate(t_span):
            div = model.divergence(t, z_traj[i])
            div_integral += div * (t_span[1] - t_span[0])
        
        # 损失：负对数似然
        # log p_1(z_1) = log p_0(z_0) - int_0^1 div f dt
        log_p0 = -0.5 * (z0 ** 2).sum(dim=1) - np.log(2 * np.pi)
        log_p1 = log_p0 - div_integral
        loss = -log_p1.mean()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses


def visualize_transformation(model, n_samples=1000):
    """可视化分布变换：从高斯到月牙形"""
    # 从先验分布（标准高斯）采样
    z0_samples = torch.randn(n_samples, 2)
    
    # 前向ODE变换到目标分布
    z1_samples = odeint(model, z0_samples, torch.tensor([0., 1.]))
    z1_samples = z1_samples[-1]
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(z0_samples[:, 0].detach().numpy(), 
                    z0_samples[:, 1].detach().numpy(), 
                    alpha=0.5, s=10, c='blue')
    axes[0].set_title('Source Distribution (Gaussian)')
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z2')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    axes[1].scatter(z1_samples[:, 0].detach().numpy(), 
                    z1_samples[:, 1].detach().numpy(), 
                    alpha=0.5, s=10, c='red')
    axes[1].set_title('Transformed Distribution (Crescent)')
    axes[1].set_xlabel('z1')
    axes[1].set_ylabel('z2')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('experiment2_transformation.png', dpi=150)
    print("Transformation visualization saved to experiment2_transformation.png")


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成目标分布数据
    print("Generating target distribution (crescent shape)...")
    n_samples = 1000
    target_samples = target_distribution(n_samples)
    
    # 可视化目标分布
    plt.figure(figsize=(6, 6))
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, s=10)
    plt.title('Target Distribution (Crescent)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment2_target.png', dpi=150)
    print("Target distribution saved to experiment2_target.png")
    
    # 创建模型
    model = CNF(dim=2)
    
    # 训练
    print("\nTraining Continuous Normalizing Flow...")
    losses = train_cnf(model, target_samples, n_epochs=5000, lr=1e-4)
    
    # 可视化变换结果
    print("\nVisualizing transformation...")
    visualize_transformation(model, n_samples=1000)
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('experiment2_loss.png', dpi=150)
    print("Loss curve saved to experiment2_loss.png")
