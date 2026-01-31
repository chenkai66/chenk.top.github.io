"""
实验1：简单ODE系统拟合
验证神经ODE可以学习简单的ODE系统
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint

class ODENet(nn.Module):
    """神经ODE网络：学习速度场 f_theta(t, z)"""
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # +1 for time
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )
    
    def forward(self, t, z):
        """速度场：dz/dt = f_theta(t, z)"""
        t_vec = torch.ones(z.shape[0], 1).to(z) * t
        tz = torch.cat([z, t_vec], dim=1)
        return self.net(tz)


def true_ode(t, z):
    """真实ODE系统：dz/dt = Az，其中A是2x2矩阵"""
    A = torch.tensor([[-1., 1.], [-1., -1.]], dtype=torch.float32)
    return (A @ z.T).T


def generate_training_data(n_samples=100, t_end=2.0, n_points=100):
    """生成训练数据：真实ODE的轨迹"""
    t_span = torch.linspace(0, t_end, n_points)
    z0 = torch.randn(n_samples, 2) * 0.5  # 随机初始条件
    z_true = odeint(true_ode, z0, t_span)  # 求解真实ODE
    return z0, z_true, t_span


def train_neural_ode(model, z0, z_true, t_span, n_epochs=1000, lr=1e-3):
    """训练神经ODE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        z_pred = odeint(model, z0, t_span)  # 前向传播：求解ODE
        loss = torch.mean((z_pred - z_true) ** 2)  # MSE损失
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses


def visualize_results(model, z0, z_true, t_span):
    """可视化结果：真实轨迹 vs 预测轨迹"""
    z_pred = odeint(model, z0, t_span)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制几个样本轨迹
    n_trajectories = 5
    for i in range(n_trajectories):
        axes[0].plot(z_true[:, i, 0].detach().numpy(), 
                     z_true[:, i, 1].detach().numpy(), 
                     'b-', alpha=0.5, label='True' if i == 0 else '')
        axes[1].plot(z_pred[:, i, 0].detach().numpy(), 
                     z_pred[:, i, 1].detach().numpy(), 
                     'r-', alpha=0.5, label='Predicted' if i == 0 else '')
    
    axes[0].set_title('True ODE Trajectories')
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z2')
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_title('Neural ODE Predictions')
    axes[1].set_xlabel('z1')
    axes[1].set_ylabel('z2')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('experiment1_results.png', dpi=150)
    print("Results saved to experiment1_results.png")


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成训练数据
    print("Generating training data...")
    z0, z_true, t_span = generate_training_data(n_samples=100, t_end=2.0, n_points=100)
    
    # 创建模型
    model = ODENet(dim=2)
    
    # 训练
    print("\nTraining Neural ODE...")
    losses = train_neural_ode(model, z0, z_true, t_span, n_epochs=1000, lr=1e-3)
    
    # 评估
    z_pred = odeint(model, z0, t_span)
    final_error = torch.mean((z_pred - z_true) ** 2).item()
    print(f"\nFinal MSE Error: {final_error:.6f}")
    
    # 可视化
    visualize_results(model, z0, z_true, t_span)
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('experiment1_loss.png', dpi=150)
    print("Loss curve saved to experiment1_loss.png")
