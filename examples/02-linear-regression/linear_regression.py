"""
线性回归 - 第一个机器学习模型
==============================

线性回归是最简单的机器学习模型，用于预测连续值。

模型：y = w * x + b
- w 是权重（斜率）
- b 是偏置（截距）

目标：找到最优的 w 和 b，使得预测值与真实值尽可能接近

学习目标：
1. 理解机器学习的基本流程
2. 学会定义模型、损失函数、优化器
3. 理解训练循环
4. 可视化训练过程
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一步：准备数据")
print("=" * 60)

# 生成模拟数据：y = 3x + 0.8 + 噪声
# 假设真实的 w=3, b=0.8，我们让模型去学习这两个参数

torch.manual_seed(42)  # 设置随机种子，保证结果可复现

# 生成100个样本
X = torch.unsqueeze(torch.linspace(0, 10, 100), dim=1)  # shape: (100, 1)
y_true = 3 * X + 0.8  # 真实值

# 添加随机噪声
noise = torch.randn(X.shape) * 0.5
y = y_true + noise  # 观测值（带噪声）

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"真实参数: w=3, b=0.8")
print("我们的任务：让模型学习出 w≈3, b≈0.8")

print("\n" + "=" * 60)
print("第二步：定义模型")
print("=" * 60)

# 方法一：手动定义参数（推荐初学者理解原理）
w = torch.randn(1, requires_grad=True)  # 随机初始化权重
b = torch.randn(1, requires_grad=True)  # 随机初始化偏置

print(f"初始参数: w={w.item():.4f}, b={b.item():.4f}")

def model(x):
    """前向传播：y = w*x + b"""
    return w * x + b

print("\n" + "=" * 60)
print("第三步：定义损失函数和优化器")
print("=" * 60)

# 损失函数：均方误差 MSE = (1/n) * Σ(y_pred - y_true)^2
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# 优化器：随机梯度下降 SGD
learning_rate = 0.01  # 学习率（步长）

print(f"损失函数: MSE (均方误差)")
print(f"优化器: SGD (随机梯度下降)")
print(f"学习率: {learning_rate}")

print("\n" + "=" * 60)
print("第四步：训练循环")
print("=" * 60)

# 训练参数
epochs = 200  # 训练轮数

# 记录损失用于可视化
loss_history = []

for epoch in range(epochs):
    # === 前向传播 ===
    y_pred = model(X)  # 计算预测值

    # === 计算损失 ===
    loss = mse_loss(y_pred, y)

    # === 反向传播 ===
    loss.backward()  # 自动计算梯度

    # === 更新参数（手动实现梯度下降）===
    with torch.no_grad():  # 更新参数不需要梯度追踪
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # 清空梯度（非常重要！）
        w.grad.zero_()
        b.grad.zero_()

    # 记录损失
    loss_history.append(loss.item())

    # 每20轮打印一次
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

print("\n" + "=" * 60)
print("训练结果")
print("=" * 60)

print(f"学习到的参数: w={w.item():.4f}, b={b.item():.4f}")
print(f"真实参数: w=3.0000, b=0.8000")
print(f"误差: w误差={abs(w.item()-3):.4f}, b误差={abs(b.item()-0.8):.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图1：拟合结果
axes[0].scatter(X.numpy(), y.numpy(), s=10, alpha=0.5, label='训练数据')
axes[0].plot(X.numpy(), y_true.numpy(), 'g-', linewidth=2, label='真实直线 (y=3x+0.8)')
axes[0].plot(X.numpy(), model(X).detach().numpy(), 'r--', linewidth=2, label=f'学习直线 (y={w.item():.2f}x+{b.item():.2f})')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('线性回归拟合结果')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：损失曲线
axes[1].plot(loss_history)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('训练损失曲线')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=150)
plt.show()

print("\n图片已保存到 linear_regression_result.png")
