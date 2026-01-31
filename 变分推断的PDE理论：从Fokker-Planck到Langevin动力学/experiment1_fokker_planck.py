"""
实验1：一维Fokker-Planck演化可视化

目标：可视化Fokker-Planck方程的解，展示概率密度如何演化到平衡分布。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

def U(x):
    """势能函数：双峰分布"""
    return 0.5 * x**2 + 2 * np.cos(2*x)

def dU(x):
    """势能梯度"""
    return x - 4 * np.sin(2*x)

def fokker_planck_rhs(rho, t, x, beta):
    """
    Fokker-Planck方程右端项
    
    方程：∂ρ/∂t = ∇·(ρ∇U) + β^{-1}Δρ
    """
    dx = x[1] - x[0]
    n = len(x)
    drho_dt = np.zeros_like(rho)
    
    # 扩散项: β^{-1} * ∂²ρ/∂x²
    drho_dt[1:-1] += (beta**-1) * (rho[2:] - 2*rho[1:-1] + rho[:-2]) / dx**2
    
    # 漂移项: ∂/∂x (ρ * dU/dx)
    dU_dx = dU(x)
    # 使用中心差分
    drho_dt[1:-1] += (rho[2:] * dU_dx[2:] - rho[:-2] * dU_dx[:-2]) / (2*dx)
    # 添加交叉项
    drho_dt[1:-1] += rho[1:-1] * (dU_dx[2:] - dU_dx[:-2]) / (2*dx)
    
    # 边界条件：零边界（反射边界）
    drho_dt[0] = 0
    drho_dt[-1] = 0
    
    return drho_dt

def main():
    # 参数设置
    x_min, x_max = -5, 5
    n_points = 200
    x = np.linspace(x_min, x_max, n_points)
    beta = 1.0
    
    # 初始分布：集中在 x=-2 附近的高斯分布
    rho0 = np.exp(-0.5 * ((x + 2) / 0.5)**2)
    rho0 = rho0 / np.trapz(rho0, x)  # 归一化
    
    # 平衡分布：Gibbs分布
    rho_inf = np.exp(-beta * U(x))
    rho_inf = rho_inf / np.trapz(rho_inf, x)
    
    # 时间演化
    t_span = np.linspace(0, 5, 100)
    sol = odeint(fokker_planck_rhs, rho0, t_span, args=(x, beta))
    
    # 静态可视化：不同时刻的分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    time_indices = [0, 10, 50, 99]
    for i, t_idx in enumerate(time_indices):
        axes[i].plot(x, sol[t_idx], 'b-', label=f't={t_span[t_idx]:.2f}', linewidth=2)
        axes[i].plot(x, rho_inf, 'r--', label='Equilibrium', linewidth=2)
        axes[i].set_xlabel('x', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'Time = {t_span[t_idx]:.2f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('fokker_planck_evolution.png', dpi=300, bbox_inches='tight')
    print("静态图已保存：fokker_planck_evolution.png")
    
    # 动态可视化：演化动画
    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot(x, sol[0], 'b-', linewidth=2, label='Current distribution')
    line2, = ax.plot(x, rho_inf, 'r--', linewidth=2, label='Equilibrium distribution')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_ylim([0, max(np.max(sol), np.max(rho_inf)) * 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Fokker-Planck Evolution', fontsize=14)
    
    def animate(frame):
        line1.set_ydata(sol[frame])
        ax.set_title(f'Fokker-Planck Evolution (t={t_span[frame]:.2f})', fontsize=14)
        return line1,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(t_span), 
                                    interval=50, blit=True, repeat=True)
    anim.save('fokker_planck_animation.gif', writer='pillow', fps=20)
    print("动画已保存：fokker_planck_animation.gif")
    
    # 计算KL散度随时间的变化
    kl_divergence = []
    for i in range(len(t_span)):
        # 避免log(0)
        rho_t = np.maximum(sol[i], 1e-10)
        rho_inf_safe = np.maximum(rho_inf, 1e-10)
        kl = np.trapz(rho_t * np.log(rho_t / rho_inf_safe), x)
        kl_divergence.append(kl)
    
    # 绘制KL散度
    plt.figure(figsize=(10, 6))
    plt.semilogy(t_span, kl_divergence, 'b-', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('KL Divergence (log scale)', fontsize=12)
    plt.title('KL Divergence: ρ_t || ρ_∞', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('kl_divergence.png', dpi=300, bbox_inches='tight')
    print("KL散度图已保存：kl_divergence.png")
    
    plt.show()

if __name__ == '__main__':
    main()
