"""
逻辑回归 - 二分类问题
======================

逻辑回归用于分类问题（不是回归！）。

与线性回归的区别：
- 线性回归：预测连续值 (y = wx + b)
- 逻辑回归：预测概率 (y = sigmoid(wx + b))，然后分类

Sigmoid 函数：
- 将任意实数映射到 (0, 1) 区间
- sigmoid(x) = 1 / (1 + e^(-x))
- 输出可以解释为"属于正类的概率"

学习目标：
1. 理解 Sigmoid 函数
2. 理解交叉熵损失（Binary Cross Entropy）
3. 实现二分类任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一部分：理解 Sigmoid 函数")
print("=" * 60)

# Sigmoid 函数可视化
x_sig = torch.linspace(-10, 10, 100)
y_sig = torch.sigmoid(x_sig)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_sig.numpy(), y_sig.numpy(), linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid 函数')
plt.grid(True, alpha=0.3)
plt.text(-8, 0.8, 'sigmoid(x) = 1/(1+e^(-x))\n将任意值映射到 (0,1)')

print("""
Sigmoid 函数的特点：
1. 输出范围：(0, 1)，可以解释为概率
2. 当 x=0 时，sigmoid(0) = 0.5
3. 当 x>0 时，sigmoid(x) > 0.5，预测为正类
4. 当 x<0 时，sigmoid(x) < 0.5，预测为负类
""")

print("\n" + "=" * 60)
print("第二步：准备分类数据")
print("=" * 60)

torch.manual_seed(42)

# 生成两类数据点
# 类别0：以 (2, 2) 为中心
class0_x = torch.randn(100, 2) + torch.tensor([2, 2])
class0_y = torch.zeros(100, 1)  # 标签 0

# 类别1：以 (-2, -2) 为中心
class1_x = torch.randn(100, 2) + torch.tensor([-2, -2])
class1_y = torch.ones(100, 1)   # 标签 1

# 合并数据
X = torch.cat([class0_x, class1_x], dim=0)  # shape: (200, 2)
y = torch.cat([class0_y, class1_y], dim=0)  # shape: (200, 1)

# 打乱数据
indices = torch.randperm(X.shape[0])
X = X[indices]
y = y[indices]

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"类别0样本数: {(y == 0).sum().item()}")
print(f"类别1样本数: {(y == 1).sum().item()}")

# 可视化数据
plt.subplot(1, 2, 2)
plt.scatter(class0_x[:, 0].numpy(), class0_x[:, 1].numpy(), c='blue', label='类别 0', alpha=0.6)
plt.scatter(class1_x[:, 0].numpy(), class1_x[:, 1].numpy(), c='red', label='类别 1', alpha=0.6)
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('训练数据分布')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_data.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("第三步：定义逻辑回归模型")
print("=" * 60)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输入2维，输出1维

    def forward(self, x):
        # 线性变换后接 sigmoid
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(input_dim=2)
print(f"模型结构:\n{model}")

# 损失函数：二元交叉熵
criterion = nn.BCELoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

print(f"\n损失函数: BCE Loss (二元交叉熵)")
print("公式: -[y*log(p) + (1-y)*log(1-p)]")

print("\n" + "=" * 60)
print("第四步：训练模型")
print("=" * 60)

epochs = 1000
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    # 五步法
    optimizer.zero_grad()

    y_pred = model(X)

    loss = criterion(y_pred, y)

    loss.backward()

    optimizer.step()

    # 计算准确率
    predictions = (y_pred >= 0.5).float()  # 概率>=0.5 预测为1，否则为0
    accuracy = (predictions == y).float().mean()

    loss_history.append(loss.item())
    accuracy_history.append(accuracy.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%")

print("\n" + "=" * 60)
print("第五步：可视化结果")
print("=" * 60)

# 绘制决策边界
def plot_decision_boundary(model, X, y):
    """绘制决策边界"""
    # 设置网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # 预测网格上每个点的类别
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape)

    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 图1：决策边界
    contour = axes[0].contourf(xx, yy, probs.numpy(), levels=20, cmap='RdBu', alpha=0.7)
    axes[0].contour(xx, yy, probs.numpy(), levels=[0.5], colors='black', linewidths=2)
    axes[0].scatter(X[y.squeeze()==0, 0].numpy(), X[y.squeeze()==0, 1].numpy(),
                    c='blue', label='类别 0', edgecolors='white')
    axes[0].scatter(X[y.squeeze()==1, 0].numpy(), X[y.squeeze()==1, 1].numpy(),
                    c='red', label='类别 1', edgecolors='white')
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')
    axes[0].set_title('决策边界（黑线为分界线）')
    axes[0].legend()
    plt.colorbar(contour, ax=axes[0], label='预测概率')

    # 图2：损失曲线
    axes[1].plot(loss_history)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (BCE)')
    axes[1].set_title('训练损失曲线')
    axes[1].grid(True, alpha=0.3)

    # 图3：准确率曲线
    axes[2].plot(accuracy_history)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('训练准确率曲线')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logistic_regression_result.png', dpi=150)
    plt.show()

plot_decision_boundary(model, X, y)

print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("""
逻辑回归 vs 线性回归：

┌─────────────┬────────────────┬────────────────┐
│             │ 线性回归        │ 逻辑回归        │
├─────────────┼────────────────┼────────────────┤
│ 任务类型    │ 回归（连续值）   │ 分类（离散值）   │
│ 激活函数    │ 无             │ Sigmoid        │
│ 输出范围    │ (-∞, +∞)       │ (0, 1)         │
│ 损失函数    │ MSE            │ BCE            │
│ 输出含义    │ 预测值          │ 属于正类的概率   │
└─────────────┴────────────────┴────────────────┘

关键公式：
- Sigmoid: σ(x) = 1 / (1 + e^(-x))
- BCE Loss: -[y*log(p) + (1-y)*log(1-p)]

决策规则：
- p >= 0.5 → 预测为正类(1)
- p < 0.5  → 预测为负类(0)
""")
