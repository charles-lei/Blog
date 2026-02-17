"""
多层神经网络 (MLP) - 深度学习的基础
======================================

多层神经网络 (Multi-Layer Perceptron, MLP) 也叫全连接神经网络。

为什么需要多层？
- 单个神经元只能解决线性可分问题
- 多层 + 非线性激活函数可以学习任意复杂模式

MLP 结构：
    输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
              ↓ReLU      ↓ReLU         ↓(无/softmax)

学习目标：
1. 理解为什么需要非线性激活函数
2. 理解如何构建多层网络
3. 学习解决非线性分类问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一部分：为什么需要非线性激活函数？")
print("=" * 60)

# 创建 XOR 数据（经典的非线性可分问题）
# 线性模型无法解决 XOR 问题！
X_xor = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("XOR 真值表：")
print("  输入 (0,0) → 输出 0")
print("  输入 (0,1) → 输出 1")
print("  输入 (1,0) → 输出 1")
print("  输入 (1,1) → 输出 0")
print("\n这是一个非线性可分问题，单层神经网络无法解决！")

# 尝试用线性模型（会失败）
print("\n尝试用线性模型解决 XOR...")

class LinearXOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

linear_model = LinearXOR()
criterion = nn.BCELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = linear_model(X_xor)
    loss = criterion(y_pred, y_xor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = (linear_model(X_xor) >= 0.5).float()
    accuracy = (predictions == y_xor).float().mean()

print(f"线性模型准确率: {accuracy.item()*100:.1f}%")
print("→ 线性模型无法完美解决 XOR 问题！\n")

print("=" * 60)
print("第二部分：使用多层神经网络")
print("=" * 60)

class MLP(nn.Module):
    """
    多层神经网络

    结构：
        输入 (2)
          ↓
        隐藏层1 (8个神经元) + ReLU
          ↓
        隐藏层2 (8个神经元) + ReLU
          ↓
        输出层 (1个神经元) + Sigmoid
    """

    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super().__init__()

        # 第一层：输入 -> 隐藏层1
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 第二层：隐藏层1 -> 隐藏层2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层：隐藏层2 -> 输出
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一隐藏层
        x = self.fc1(x)
        x = self.relu(x)

        # 第二隐藏层
        x = self.fc2(x)
        x = self.relu(x)

        # 输出层
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

# 创建 MLP 模型
mlp = MLP(input_dim=2, hidden_dim=8, output_dim=1)
print(f"MLP 结构:\n{mlp}")

# 训练 MLP
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.01)

print("\n训练 MLP 解决 XOR 问题...")
epochs = 2000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = mlp(X_xor)
    loss = criterion(y_pred, y_xor)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            predictions = (mlp(X_xor) >= 0.5).float()
            accuracy = (predictions == y_xor).float().mean()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.1f}%")

print("\n最终预测结果：")
with torch.no_grad():
    for i, (x, y_true) in enumerate(zip(X_xor, y_xor)):
        y_pred = mlp(x).item()
        predicted = 1 if y_pred >= 0.5 else 0
        print(f"  输入 {x.tolist()} → 真实: {y_true.item():.0f}, "
              f"预测概率: {y_pred:.4f}, 预测类别: {predicted}")

print("\n" + "=" * 60)
print("第三部分：更复杂的非线性分类问题")
print("=" * 60)

# 生成螺旋数据（更难的非线性问题）
def generate_spiral_data(n_samples=300):
    """生成两类螺旋数据"""
    np.random.seed(42)

    # 第一类：向内螺旋
    theta1 = np.linspace(0, 4*np.pi, n_samples//2)
    r1 = theta1 + np.random.randn(n_samples//2) * 0.2
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # 第二类：向外螺旋
    theta2 = np.linspace(0, 4*np.pi, n_samples//2)
    r2 = theta2 + np.random.randn(n_samples//2) * 0.2
    x2 = -r2 * np.cos(theta2)
    y2 = -r2 * np.sin(theta2)

    # 合并数据
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

    # 打乱
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_spiral, y_spiral = generate_spiral_data(300)

print(f"螺旋数据形状: X={X_spiral.shape}, y={y_spiral.shape}")
print(f"类别0样本数: {(y_spiral == 0).sum().item()}")
print(f"类别1样本数: {(y_spiral == 1).sum().item()}")

# 可视化数据
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_spiral[y_spiral.squeeze()==0, 0], X_spiral[y_spiral.squeeze()==0, 1],
                c='blue', label='类别 0', alpha=0.6)
axes[0].scatter(X_spiral[y_spiral.squeeze()==1, 0], X_spiral[y_spiral.squeeze()==1, 1],
                c='red', label='类别 1', alpha=0.6)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].set_title('螺旋数据（非线性可分）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 创建更深的网络解决螺旋问题
class DeepMLP(nn.Module):
    """
    更深的网络用于解决复杂分类问题
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# 训练
deep_mlp = DeepMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(deep_mlp.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = deep_mlp(X_spiral)
    loss = criterion(y_pred, y_spiral)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        with torch.no_grad():
            predictions = (deep_mlp(X_spiral) >= 0.5).float()
            accuracy = (predictions == y_spiral).float().mean()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%")

# 绘制决策边界
def plot_decision_boundary_spiral(model, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs.numpy(), levels=20, cmap='RdBu', alpha=0.7)
    ax.contour(xx, yy, probs.numpy(), levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[y.squeeze()==0, 0], X[y.squeeze()==0, 1],
               c='blue', label='类别 0', edgecolors='white', alpha=0.8)
    ax.scatter(X[y.squeeze()==1, 0], X[y.squeeze()==1, 1],
               c='red', label='类别 1', edgecolors='white', alpha=0.8)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title('MLP 决策边界')
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_decision_boundary_spiral(deep_mlp, X_spiral, y_spiral, axes[1])

plt.tight_layout()
plt.savefig('mlp_spiral_result.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("总结：多层神经网络的核心要点")
print("=" * 60)

print("""
1. 为什么需要多层？
   - 单层神经网络 = 线性模型，只能解决线性可分问题
   - 多层 + 非线性激活 = 可以逼近任意函数（通用近似定理）

2. 常用激活函数对比：

   ┌──────────┬──────────────────┬─────────────────┐
   │ 激活函数 │ 公式              │ 特点            │
   ├──────────┼──────────────────┼─────────────────┤
   │ Sigmoid  │ 1/(1+e^(-x))     │ 输出(0,1)，用于输出层 │
   │ ReLU     │ max(0, x)        │ 最常用，解决梯度消失 │
   │ Tanh     │ (e^x-e^(-x))/(e^x+e^(-x)) │ 输出(-1,1) │
   │ Softmax  │ exp(x)/Σexp(x)   │ 用于多分类输出   │
   └──────────┴──────────────────┴─────────────────┘

3. 网络设计原则：
   - 隐藏层数：2-4 层通常足够处理大多数问题
   - 隐藏层宽度：32, 64, 128, 256 等常见值
   - 输出层激活函数：
     * 二分类：Sigmoid
     * 多分类：Softmax
     * 回归：无激活函数

4. 训练技巧：
   - 使用 Adam 优化器（比 SGD 收敛更快）
   - 学习率：0.001 (Adam), 0.01 (SGD)
   - 训练足够多的 epoch 让损失收敛
""")

print("图片已保存到 mlp_spiral_result.png")
