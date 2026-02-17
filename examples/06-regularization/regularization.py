"""
正则化技巧 - 防止过拟合
========================

过拟合问题：
- 训练集准确率很高
- 测试集准确率很低
- 模型"记住"了训练数据，而不是"学习"规律

正则化是防止过拟合的重要手段：
1. Dropout：随机丢弃部分神经元
2. Batch Normalization：批量归一化
3. Weight Decay (L2正则化)：惩罚大权重
4. 数据增强：增加训练数据多样性

学习目标：
1. 理解过拟合现象
2. 学会使用 Dropout
3. 学会使用 Batch Normalization
4. 理解各种正则化方法的效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一部分：制造过拟合问题")
print("=" * 60)

# 生成少量数据（容易过拟合的场景）
torch.manual_seed(42)
np.random.seed(42)

# 真实函数：y = sin(x) + 0.3x
X_train = torch.linspace(0, 2*np.pi, 20).unsqueeze(1)  # 只有20个点！
y_train = torch.sin(X_train) + 0.3 * X_train + torch.randn(X_train.shape) * 0.1

# 测试数据更密集
X_test = torch.linspace(0, 2*np.pi, 100).unsqueeze(1)
y_test = torch.sin(X_test) + 0.3 * X_test

print(f"训练集大小: {X_train.shape[0]} (数据很少，容易过拟合)")
print(f"测试集大小: {X_test.shape[0]}")

print("\n" + "=" * 60)
print("第二部分：对比不同模型")
print("=" * 60)

# 模型1：无正则化（容易过拟合）
class OverfitNet(nn.Module):
    """过深的网络，无正则化"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# 模型2：使用 Dropout
class DropoutNet(nn.Module):
    """使用 Dropout 防止过拟合"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout层

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout层

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout层

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# 模型3：使用 Batch Normalization
class BatchNormNet(nn.Module):
    """使用 Batch Normalization"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.BatchNorm1d(256),  # BN层
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# 模型4：组合多种正则化
class RegularizedNet(nn.Module):
    """组合多种正则化方法"""
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),  # 减少层数和宽度
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, model_name, epochs=1000, lr=0.01, use_weight_decay=False):
    """训练模型并记录训练过程"""
    criterion = nn.MSELoss()

    # weight_decay 是 L2 正则化
    weight_decay = 0.001 if use_weight_decay else 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # 评估
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()
            test_losses.append(test_loss)

    return train_losses, test_losses

# 训练所有模型
models = {
    '无正则化': OverfitNet(),
    'Dropout': DropoutNet(dropout_rate=0.3),
    'BatchNorm': BatchNormNet(),
    '组合正则化': RegularizedNet(dropout_rate=0.2)
}

results = {}
print("训练各个模型...")
for name, model in models.items():
    train_losses, test_losses = train_model(model, name, epochs=1000, lr=0.01)
    results[name] = {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }

# 打印结果
print("\n最终损失对比:")
print(f"{'模型':<12} {'训练损失':<12} {'测试损失':<12} {'泛化差距'}")
print("-" * 50)
for name, res in results.items():
    gap = res['final_test_loss'] - res['final_train_loss']
    print(f"{name:<12} {res['final_train_loss']:.4f}       {res['final_test_loss']:.4f}       {gap:.4f}")

print("\n" + "=" * 60)
print("第三步：理解 Dropout")
print("=" * 60)

print("""
Dropout 原理：
- 训练时：随机将一部分神经元的输出设为 0
- 测试时：使用所有神经元，但输出按比例缩放

作用：
1. 防止神经元共适应（co-adaptation）
2. 强迫网络学习冗余表示
3. 相当于训练多个子网络的集成

常见 dropout_rate：
- 0.2 - 0.5：大多数场景
- 0.3：推荐起始值
- > 0.5：可能导致欠拟合

放置位置：
- 通常放在激活函数之后
- 不在输出层使用
""")

# 可视化 Dropout 效果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 图1：训练曲线对比
ax = axes[0, 0]
for name, res in results.items():
    ax.plot(res['train_losses'], label=name, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('训练损失曲线')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 图2：测试曲线对比
ax = axes[0, 1]
for name, res in results.items():
    ax.plot(res['test_losses'], label=name, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss')
ax.set_title('测试损失曲线')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 图3：拟合效果对比 - 无正则化
ax = axes[0, 2]
model = results['无正则化']['model']
ax.scatter(X_train.numpy(), y_train.numpy(), c='blue', label='训练数据', alpha=0.6)
ax.plot(X_test.numpy(), y_test.numpy(), 'g-', label='真实曲线', linewidth=2)
with torch.no_grad():
    y_pred = model(X_test)
    ax.plot(X_test.numpy(), y_pred.numpy(), 'r--', label='预测曲线', linewidth=2)
ax.set_title('无正则化：过拟合')
ax.legend()
ax.grid(True, alpha=0.3)

# 图4：拟合效果对比 - Dropout
ax = axes[1, 0]
model = results['Dropout']['model']
ax.scatter(X_train.numpy(), y_train.numpy(), c='blue', alpha=0.6)
ax.plot(X_test.numpy(), y_test.numpy(), 'g-', linewidth=2)
with torch.no_grad():
    y_pred = model(X_test)
    ax.plot(X_test.numpy(), y_pred.numpy(), 'r--', linewidth=2)
ax.set_title('Dropout：平滑拟合')
ax.legend()
ax.grid(True, alpha=0.3)

# 图5：拟合效果对比 - 组合正则化
ax = axes[1, 1]
model = results['组合正则化']['model']
ax.scatter(X_train.numpy(), y_train.numpy(), c='blue', alpha=0.6)
ax.plot(X_test.numpy(), y_test.numpy(), 'g-', linewidth=2)
with torch.no_grad():
    y_pred = model(X_test)
    ax.plot(X_test.numpy(), y_pred.numpy(), 'r--', linewidth=2)
ax.set_title('组合正则化：最佳拟合')
ax.legend()
ax.grid(True, alpha=0.3)

# 图6：最终损失对比
ax = axes[1, 2]
names = list(results.keys())
final_test_losses = [results[name]['final_test_loss'] for name in names]
colors = ['red' if '无正则化' in name else 'green' for name in names]
bars = ax.bar(range(len(names)), final_test_losses, color=colors, alpha=0.7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('Test Loss')
ax.set_title('最终测试损失对比')
ax.grid(True, alpha=0.3, axis='y')
# 标注数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("总结：正则化方法对比")
print("=" * 60)

print("""
┌──────────────┬────────────────────────┬─────────────┐
│ 方法         │ 原理                    │ 使用场景    │
├──────────────┼────────────────────────┼─────────────┤
│ Dropout      │ 随机丢弃神经元          │ 全连接层    │
│ BatchNorm    │ 标准化每层输入          │ 深层网络    │
│ Weight Decay │ 惩罚大权重（L2正则）    │ 所有模型    │
│ 数据增强     │ 增加训练数据多样性      │ 图像/文本   │
│ 早停法       │ 监控验证集损失提前停止  │ 所有训练    │
└──────────────┴────────────────────────┴─────────────┘

使用建议：

1. Dropout (最常用)
   - rate = 0.2 - 0.5
   - 放在全连接层之后
   - 不在输出层使用

2. Batch Normalization
   - 放在激活函数之前
   - 允许使用更大的学习率
   - 减少对初始化的敏感性

3. Weight Decay
   - 在优化器中设置：weight_decay=0.001
   - 相当于 L2 正则化
   - 值越大，正则化越强

4. 组合使用
   - Dropout + BatchNorm + Weight Decay
   - 注意：可能需要调整学习率

5. 训练/评估模式切换
   model.train()   # 训练模式，启用 Dropout
   model.eval()    # 评估模式，禁用 Dropout

代码模板：
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),  # BN 在激活前
            nn.ReLU(),
            nn.Dropout(0.3),      # Dropout 在激活后

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

# 训练时
model.train()
# ... 训练代码 ...

# 评估时
model.eval()
with torch.no_grad():
    # ... 评估代码 ...
```
""")

print("\n图片已保存到 regularization_comparison.png")
