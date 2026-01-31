"""
实验3：VI vs MCMC收敛性对比

目标：比较变分推断和Langevin MCMC的收敛速度。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm

def log_target(z):
    """目标分布的对数（未归一化）"""
    return -0.5 * z**2 - torch.cos(z)

def log_target_np(z):
    """目标分布的对数（numpy版本）"""
    return -0.5 * z**2 - np.cos(z)

def kl_divergence_gaussian(mu, sigma, log_target_func, n_samples=1000):
    """
    计算KL散度 KL(q || p)
    
    参数：
        mu: 均值
        sigma: 标准差
        log_target_func: 目标分布的对数函数
        n_samples: 蒙特卡洛采样数
    """
    # 从q中采样
    z_samples = mu + sigma * torch.randn(n_samples)
    
    # log q(z)
    log_q = -0.5 * np.log(2*np.pi) - torch.log(sigma) - 0.5 * ((z_samples - mu) / sigma)**2
    
    # log p(z)（未归一化，但KL散度中归一化常数会抵消）
    log_p = log_target_func(z_samples)
    
    # KL散度：E_q[log q - log p]
    kl = torch.mean(log_q - log_p)
    
    return kl

def variational_inference(n_iterations=1000, lr=0.01):
    """变分推断：使用高斯分布近似"""
    # 初始化参数
    mu = nn.Parameter(torch.tensor(0.0))
    log_sigma = nn.Parameter(torch.tensor(0.0))
    optimizer = optim.Adam([mu, log_sigma], lr=lr)
    
    kl_history = []
    mu_history = []
    sigma_history = []
    
    for step in range(n_iterations):
        sigma = torch.exp(log_sigma)
        kl = kl_divergence_gaussian(mu, sigma, log_target)
        
        optimizer.zero_grad()
        kl.backward()
        optimizer.step()
        
        kl_history.append(kl.item())
        mu_history.append(mu.item())
        sigma_history.append(sigma.item())
        
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: KL = {kl.item():.4f}, mu = {mu.item():.4f}, sigma = {sigma.item():.4f}")
    
    return {
        'kl_history': kl_history,
        'mu_history': mu_history,
        'sigma_history': sigma_history,
        'final_mu': mu.item(),
        'final_sigma': torch.exp(log_sigma).item()
    }

def langevin_mcmc(n_iterations=1000, epsilon=0.01, beta=1.0, n_particles=100):
    """Langevin MCMC采样"""
    # 初始化
    x = torch.randn(n_particles, requires_grad=True)
    
    samples = []
    kl_history = []
    
    # 目标分布（用于计算KL散度）
    z_grid = np.linspace(-5, 5, 200)
    p_true = np.exp(log_target_np(z_grid))
    p_true = p_true / np.trapz(p_true, z_grid)
    
    for step in range(n_iterations):
        # 计算log p(x)
        log_p = log_target(x)
        log_p_sum = log_p.sum()
        
        # 反向传播计算梯度
        if x.grad is not None:
            x.grad.zero_()
        log_p_sum.backward()
        
        # Langevin更新
        with torch.no_grad():
            noise = torch.randn_like(x) * np.sqrt(2 * epsilon / beta)
            x = x + epsilon * x.grad + noise
            x.requires_grad_(True)
        
        # 每10步计算一次KL散度（用KDE估计经验分布）
        if step % 10 == 0 and step > 100:  # 跳过预热期
            x_np = x.detach().numpy()
            # 使用KDE估计经验分布
            kde = gaussian_kde(x_np)
            q_est = kde(z_grid)
            q_est = q_est / np.trapz(q_est, z_grid)
            
            # 计算KL散度
            kl = np.trapz(q_est * np.log(q_est / (p_true + 1e-10) + 1e-10), z_grid)
            kl_history.append(kl)
            samples.append(x_np.copy())
    
    return {
        'kl_history': kl_history,
        'samples': samples,
        'final_samples': x.detach().numpy()
    }

def main():
    print("=" * 60)
    print("实验3：VI vs MCMC收敛性对比")
    print("=" * 60)
    
    # 变分推断
    print("\n运行变分推断...")
    vi_results = variational_inference(n_iterations=1000, lr=0.01)
    
    # Langevin MCMC
    print("\n运行Langevin MCMC...")
    mcmc_results = langevin_mcmc(n_iterations=2000, epsilon=0.01, beta=1.0, n_particles=100)
    
    # 可视化：KL散度对比
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # VI的KL散度
    axes[0].plot(vi_results['kl_history'], 'b-', linewidth=2, label='Variational Inference')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('KL Divergence', fontsize=12)
    axes[0].set_title('Variational Inference Convergence', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # MCMC的KL散度
    mcmc_steps = np.arange(len(mcmc_results['kl_history'])) * 10 + 100
    axes[1].plot(mcmc_steps, mcmc_results['kl_history'], 'r-', linewidth=2, label='Langevin MCMC')
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('KL Divergence', fontsize=12)
    axes[1].set_title('Langevin MCMC Convergence', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vi_vs_mcmc_convergence.png', dpi=300, bbox_inches='tight')
    print("\n收敛性对比图已保存：vi_vs_mcmc_convergence.png")
    
    # 参数演化（VI）
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(vi_results['mu_history'], 'b-', linewidth=2, label='μ')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('μ', fontsize=12)
    axes[0].set_title('Variational Inference: Mean Evolution', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(vi_results['sigma_history'], 'r-', linewidth=2, label='σ')
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('σ', fontsize=12)
    axes[1].set_title('Variational Inference: Std Evolution', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vi_parameter_evolution.png', dpi=300, bbox_inches='tight')
    print("VI参数演化图已保存：vi_parameter_evolution.png")
    
    # 最终分布对比
    z_grid = np.linspace(-5, 5, 200)
    p_true = np.exp(log_target_np(z_grid))
    p_true = p_true / np.trapz(p_true, z_grid)
    
    # VI的近似分布
    q_vi = norm.pdf(z_grid, vi_results['final_mu'], vi_results['final_sigma'])
    
    # MCMC的经验分布
    mcmc_samples = mcmc_results['final_samples'].flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_grid, p_true, 'k-', linewidth=2, label='True distribution', linestyle='--')
    ax.plot(z_grid, q_vi, 'b-', linewidth=2, label='VI (Gaussian)', alpha=0.7)
    ax.hist(mcmc_samples, bins=30, density=True, alpha=0.5, label='MCMC samples', color='red', edgecolor='black')
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Final Distribution Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("最终分布对比图已保存：final_distribution_comparison.png")
    
    plt.show()

if __name__ == '__main__':
    main()
