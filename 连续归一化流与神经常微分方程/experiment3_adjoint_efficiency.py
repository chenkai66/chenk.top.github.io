"""
实验3：伴随方法 vs 反向传播效率对比
比较伴随方法和传统反向传播的内存和计算效率
"""

import torch
import torch.nn as nn
import time
from torchdiffeq import odeint, odeint_adjoint


class ODENet(nn.Module):
    """神经ODE网络：用于效率对比"""
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, dim)
        )
    
    def forward(self, t, z):
        t_vec = torch.ones(z.shape[0], 1).to(z) * t
        tz = torch.cat([z, t_vec], dim=1)
        return self.net(tz)


def benchmark_traditional_backprop(model, z0, t_span, device='cuda'):
    """传统反向传播：需要存储所有中间状态"""
    model = model.to(device)
    z0 = z0.to(device)
    t_span = t_span.to(device)
    
    # 重置内存统计
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # 前向传播：求解ODE（存储所有中间状态）
    z1 = odeint(model, z0, t_span)
    
    # 计算损失
    loss = z1[-1].sum()
    
    # 反向传播
    loss.backward()
    
    elapsed_time = time.time() - start_time
    
    # 获取峰值内存
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0  # CPU模式下不测量内存
    
    return elapsed_time, peak_memory


def benchmark_adjoint_method(model, z0, t_span, device='cuda'):
    """伴随方法：不需要存储中间状态"""
    model = model.to(device)
    z0 = z0.to(device)
    t_span = t_span.to(device)
    
    # 重置内存统计
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # 前向传播：使用伴随方法（不存储中间状态）
    z1 = odeint_adjoint(model, z0, t_span)
    
    # 计算损失
    loss = z1[-1].sum()
    
    # 反向传播（通过伴随ODE）
    loss.backward()
    
    elapsed_time = time.time() - start_time
    
    # 获取峰值内存
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0
    
    return elapsed_time, peak_memory


def run_comparison(dim=512, batch_size=32, n_steps=100, device='cuda'):
    """运行效率对比实验"""
    print(f"Running efficiency comparison...")
    print(f"Dimension: {dim}, Batch size: {batch_size}, ODE steps: {n_steps}")
    print(f"Device: {device}\n")
    
    # 创建模型和数据
    model1 = ODENet(dim=dim)
    model2 = ODENet(dim=dim)
    model2.load_state_dict(model1.state_dict())  # 相同参数
    
    z0 = torch.randn(batch_size, dim)
    t_span = torch.linspace(0, 1, n_steps)
    
    # 预热（避免第一次运行的初始化开销）
    if device == 'cuda':
        _ = benchmark_traditional_backprop(model1, z0, t_span, device)
        _ = benchmark_adjoint_method(model2, z0, t_span, device)
        torch.cuda.empty_cache()
    
    # 传统方法
    print("Benchmarking traditional backpropagation...")
    times_traditional = []
    memories_traditional = []
    for _ in range(5):  # 运行5次取平均
        time_t, memory_t = benchmark_traditional_backprop(model1, z0, t_span, device)
        times_traditional.append(time_t)
        memories_traditional.append(memory_t)
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    avg_time_traditional = np.mean(times_traditional)
    avg_memory_traditional = np.mean(memories_traditional)
    
    # 伴随方法
    print("Benchmarking adjoint method...")
    times_adjoint = []
    memories_adjoint = []
    for _ in range(5):
        time_a, memory_a = benchmark_adjoint_method(model2, z0, t_span, device)
        times_adjoint.append(time_a)
        memories_adjoint.append(memory_a)
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    avg_time_adjoint = np.mean(times_adjoint)
    avg_memory_adjoint = np.mean(memories_adjoint)
    
    # 打印结果
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Method':<30} {'Time (s)':<15} {'Memory (MB)':<15}")
    print("-"*60)
    print(f"{'Traditional Backprop':<30} {avg_time_traditional:<15.3f} {avg_memory_traditional:<15.1f}")
    print(f"{'Adjoint Method':<30} {avg_time_adjoint:<15.3f} {avg_memory_adjoint:<15.1f}")
    print("-"*60)
    
    if device == 'cuda':
        memory_reduction = (1 - avg_memory_adjoint / avg_memory_traditional) * 100
        time_overhead = (avg_time_adjoint / avg_time_traditional - 1) * 100
        print(f"\nMemory reduction: {memory_reduction:.1f}%")
        print(f"Time overhead: {time_overhead:.1f}%")
    
    return {
        'traditional': {'time': avg_time_traditional, 'memory': avg_memory_traditional},
        'adjoint': {'time': avg_time_adjoint, 'memory': avg_memory_adjoint}
    }


if __name__ == '__main__':
    import numpy as np
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, running on CPU (memory stats will be 0)")
    
    # 运行对比实验
    results = run_comparison(dim=512, batch_size=32, n_steps=100, device=device)
    
    # 可视化结果（如果有matplotlib）
    try:
        import matplotlib.pyplot as plt
        
        methods = ['Traditional\nBackprop', 'Adjoint\nMethod']
        times = [results['traditional']['time'], results['adjoint']['time']]
        memories = [results['traditional']['memory'], results['adjoint']['memory']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 时间对比
        ax1.bar(methods, times, color=['blue', 'green'])
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Computation Time Comparison')
        ax1.grid(True, alpha=0.3)
        
        # 内存对比
        if device == 'cuda':
            ax2.bar(methods, memories, color=['blue', 'green'])
            ax2.set_ylabel('Peak Memory (MB)')
            ax2.set_title('Memory Usage Comparison')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Memory stats\nnot available\non CPU', 
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Memory Usage Comparison')
        
        plt.tight_layout()
        plt.savefig('experiment3_efficiency.png', dpi=150)
        print("\nVisualization saved to experiment3_efficiency.png")
    except ImportError:
        print("Matplotlib not available, skipping visualization")
