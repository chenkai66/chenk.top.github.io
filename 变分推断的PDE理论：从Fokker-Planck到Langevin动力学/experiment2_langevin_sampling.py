"""
实验2：Langevin动力学采样（多峰分布）

目标：用Langevin动力学从多峰分布中采样，比较不同温度参数的效果。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def U(x):
    """势能函数：双峰分布"""
    return 0.5 * x**2 + 2 * np.cos(2*x)

def dU(x):
    """势能梯度"""
    return x - 4 * np.sin(2*x)

def langevin_sampling(x0, n_steps, epsilon, beta, dU, burn_in=0):
    """
    Langevin动力学采样（Euler-Maruyama离散化）
    
    参数：
        x0: 初始位置
        n_steps: 步数
        epsilon: 步长
        beta: 温度参数（β = 1/kT）
        dU: 势能梯度函数
        burn_in: 预热步数
    """
    x = x0.copy()
    trajectory = []
    
    for i in range(n_steps + burn_in):
        # Euler-Maruyama更新
        noise = np.random.randn(*x.shape) * np.sqrt(2 * epsilon / beta)
        x = x - epsilon * dU(x) + noise
        
        if i >= burn_in:
            trajectory.append(x.copy())
    
    return np.array(trajectory)

def main():
    # 参数设置
    n_particles = 1000
    n_steps = 5000
    burn_in = 1000
    epsilon = 0.01
    beta_values = [0.5, 1.0, 2.0]
    
    # 初始位置：集中在原点附近
    np.random.seed(42)
    x0 = np.random.randn(n_particles) * 0.5
    
    # 真实分布
    x_grid = np.linspace(-5, 5, 200)
    p_true = np.exp(-beta_values[1] * U(x_grid))  # 使用beta=1.0作为参考
    p_true = p_true / np.trapz(p_true, x_grid)
    
    # 采样
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, beta in enumerate(beta_values):
        print(f"采样中：beta = {beta}")
        samples = langevin_sampling(x0, n_steps, epsilon, beta, dU, burn_in)
        final_samples = samples[-n_particles:]  # 使用最后n_particles个样本
        
        # 真实分布（对应当前beta）
        p_true_beta = np.exp(-beta * U(x_grid))
        p_true_beta = p_true_beta / np.trapz(p_true_beta, x_grid)
        
        # 直方图
        axes[idx].hist(final_samples.flatten(), bins=50, density=True, 
                       alpha=0.6, label='Samples', color='blue', edgecolor='black')
        axes[idx].plot(x_grid, p_true_beta, 'r-', linewidth=2, label='True distribution')
        axes[idx].set_xlabel('x', fontsize=12)
        axes[idx].set_ylabel('Density', fontsize=12)
        axes[idx].legend(fontsize=10)
        axes[idx].set_title(f'Beta = {beta}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('langevin_sampling_multimodal.png', dpi=300, bbox_inches='tight')
    print("采样结果图已保存：langevin_sampling_multimodal.png")
    
    # 轨迹可视化（单个粒子）
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for idx, beta in enumerate(beta_values):
        # 单个粒子的轨迹
        single_trajectory = langevin_sampling(np.array([0.0]), n_steps, epsilon, beta, dU, burn_in)
        time_steps = np.arange(len(single_trajectory))
        
        axes[idx].plot(time_steps, single_trajectory.flatten(), 'b-', alpha=0.7, linewidth=0.5)
        axes[idx].set_xlabel('Time step', fontsize=12)
        axes[idx].set_ylabel('Position', fontsize=12)
        axes[idx].set_title(f'Particle Trajectory (Beta = {beta})', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('langevin_trajectory.png', dpi=300, bbox_inches='tight')
    print("轨迹图已保存：langevin_trajectory.png")
    
    # 收敛性分析：计算有效样本数（ESS）
    def effective_sample_size(samples):
        """计算有效样本数（简化版本）"""
        # 使用自相关函数估计
        n = len(samples)
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]
        # 找到第一个自相关小于0.05的位置
        cutoff = np.where(autocorr < 0.05)[0]
        if len(cutoff) > 0:
            tau = cutoff[0]
        else:
            tau = n
        ess = n / (1 + 2 * tau)
        return ess
    
    # 比较不同beta的有效样本数
    beta_test = 1.0
    samples_test = langevin_sampling(x0[:100], n_steps, epsilon, beta_test, dU, burn_in)
    ess_values = []
    for i in range(100):
        ess = effective_sample_size(samples_test[:, i])
        ess_values.append(ess)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ess_values, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Effective Sample Size', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Effective Sample Size Distribution (Beta = {beta_test})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('effective_sample_size.png', dpi=300, bbox_inches='tight')
    print("有效样本数图已保存：effective_sample_size.png")
    
    plt.show()

if __name__ == '__main__':
    main()
