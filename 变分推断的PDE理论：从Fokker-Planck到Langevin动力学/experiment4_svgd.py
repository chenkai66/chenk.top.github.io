"""
实验4：SVGD粒子轨迹与密度估计

目标：可视化SVGD的粒子演化过程，展示粒子如何分散到目标分布的不同模式。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def U(x):
    """势能函数：双峰分布"""
    return 0.5 * x**2 + 2 * np.cos(2*x)

def dU(x):
    """势能梯度（对应log p的梯度）"""
    return -(x - 4 * np.sin(2*x))  # -dU/dx = d/dx log p

def rbf_kernel(x, y, h=1.0):
    """RBF核函数"""
    return np.exp(-0.5 * ((x - y) / h)**2)

def d_rbf_kernel(x, y, h=1.0):
    """RBF核函数的梯度（对y）"""
    return -(x - y) / h**2 * rbf_kernel(x, y, h)

def svgd_step(particles, log_target_grad, h=1.0, epsilon=0.01):
    """
    SVGD一步更新
    
    参数：
        particles: 粒子位置 (n_particles,)
        log_target_grad: log目标分布的梯度函数
        h: 核函数带宽
        epsilon: 步长
    """
    n = len(particles)
    updates = np.zeros_like(particles)
    
    # 计算每个粒子的更新
    for i in range(n):
        phi = 0.0
        for j in range(n):
            k_ij = rbf_kernel(particles[j], particles[i], h)
            dk_ij = d_rbf_kernel(particles[j], particles[i], h)
            # SVGD更新公式
            phi += k_ij * log_target_grad(particles[j]) + dk_ij
        updates[i] = epsilon * phi / n
    
    return particles + updates

def main():
    # 参数设置
    n_particles = 100
    n_steps = 500
    epsilon = 0.01
    h = 1.0
    
    # 初始粒子：集中在原点附近
    np.random.seed(42)
    particles = np.random.randn(n_particles) * 0.5
    
    # 目标分布
    x_grid = np.linspace(-5, 5, 200)
    p_true = np.exp(-U(x_grid))
    p_true = p_true / np.trapz(p_true, x_grid)
    
    # SVGD演化
    print("运行SVGD...")
    trajectories = [particles.copy()]
    
    for step in range(n_steps):
        particles = svgd_step(particles, dU, h, epsilon)
        if step % 50 == 0:
            trajectories.append(particles.copy())
            print(f"Step {step}: Mean = {np.mean(particles):.4f}, Std = {np.std(particles):.4f}")
    
    # 可视化：不同时刻的分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    time_indices = [0, len(trajectories)//3, 2*len(trajectories)//3, len(trajectories)-1]
    for idx, t_idx in enumerate(time_indices):
        particles_t = trajectories[t_idx]
        
        # 使用KDE估计密度
        kde = gaussian_kde(particles_t)
        p_est = kde(x_grid)
        p_est = p_est / np.trapz(p_est, x_grid)
        
        # 直方图
        axes[idx].hist(particles_t, bins=30, density=True, alpha=0.6, 
                       label='Particles', color='blue', edgecolor='black')
        axes[idx].plot(x_grid, p_est, 'g-', linewidth=2, label='KDE estimate', alpha=0.7)
        axes[idx].plot(x_grid, p_true, 'r--', linewidth=2, label='True distribution')
        axes[idx].set_xlabel('x', fontsize=12)
        axes[idx].set_ylabel('Density', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].set_title(f'Step = {t_idx * 50}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('svgd_evolution.png', dpi=300, bbox_inches='tight')
    print("SVGD演化图已保存：svgd_evolution.png")
    
    # 粒子轨迹可视化（前20个粒子）
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_show = min(20, n_particles)
    for i in range(n_show):
        trajectory_i = [traj[i] for traj in trajectories]
        ax.plot(np.arange(len(trajectory_i)) * 50, trajectory_i, 
                alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title('SVGD Particle Trajectories (First 20 Particles)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('svgd_trajectories.png', dpi=300, bbox_inches='tight')
    print("粒子轨迹图已保存：svgd_trajectories.png")
    
    # 粒子间距离的演化（展示粒子分散）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_distances = []
    for traj in trajectories:
        # 计算所有粒子对之间的平均距离
        distances = []
        for i in range(len(traj)):
            for j in range(i+1, len(traj)):
                distances.append(np.abs(traj[i] - traj[j]))
        mean_distances.append(np.mean(distances))
    
    ax.plot(np.arange(len(mean_distances)) * 50, mean_distances, 'b-', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Mean Pairwise Distance', fontsize=12)
    ax.set_title('Particle Dispersion Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svgd_dispersion.png', dpi=300, bbox_inches='tight')
    print("粒子分散图已保存：svgd_dispersion.png")
    
    # KL散度随时间的变化
    kl_divergence = []
    for traj in trajectories:
        # 使用KDE估计分布
        kde = gaussian_kde(traj)
        p_est = kde(x_grid)
        p_est = np.maximum(p_est, 1e-10)
        p_est = p_est / np.trapz(p_est, x_grid)
        
        # 计算KL散度
        p_true_safe = np.maximum(p_true, 1e-10)
        kl = np.trapz(p_est * np.log(p_est / p_true_safe), x_grid)
        kl_divergence.append(kl)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.arange(len(kl_divergence)) * 50, kl_divergence, 'b-', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('KL Divergence (log scale)', fontsize=12)
    ax.set_title('SVGD: KL Divergence Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svgd_kl_divergence.png', dpi=300, bbox_inches='tight')
    print("KL散度图已保存：svgd_kl_divergence.png")
    
    plt.show()

if __name__ == '__main__':
    main()
