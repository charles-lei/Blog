"""
多分类问题 - Softmax 和交叉熵
==============================

多分类 vs 二分类：
- 二分类：是/否，猫/狗 (2个类别)
- 多分类：数字0-9，鸢尾花品种 (3个以上类别)

核心概念：
1. Softmax 函数：将输出转换为概率分布
2. 交叉熵损失：衡量预测概率分布与真实标签的差异

学习目标：
1. 理解 Softmax 函数
2. 学会处理多分类数据
3. 实现鸢尾花分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一部分：理解 Softmax 函数")
print("=" * 60)

# Softmax 公式: softmax(x_i) = exp(x_i) / Σexp(x_j)
# 作用：将任意数值转换为概率分布（和为1）

def softmax(x):
    """手动实现 Softmax"""
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])  # 数值稳定性
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# 示例：神经网络的原始输出（logits）
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [1.0, 3.0, 0.5],
                       [0.5, 0.5, 2.0]])

print("神经网络原始输出 (logits):")
print(logits)

probs = softmax(logits)
print("\nSoftmax 后 (概率分布):")
print(probs)
print(f"\n每行概率和: {probs.sum(dim=1)}")
print("→ 每行和为 1，可以解释为概率")

print("\n" + "=" * 60)
print("第二部分：理解交叉熵损失")
print("=" * 60)

# 交叉熵公式: H(p, q) = -Σp(x) * log(q(x))
# p: 真实分布 (one-hot 编码)
# q: 预测概率分布

print("""
交叉熵损失的作用：
- 衡量预测概率分布与真实标签的差异
- 值越小表示预测越准确
- 真实标签通常用整数表示 (0, 1, 2)，PyTorch 会自动转换

示例：
- 真实标签: 类别 1 (即 [0, 1, 0])
- 预测概率: [0.1, 0.7, 0.2]
- 交叉熵 = -log(0.7) = 0.357
""")

# PyTorch 的 CrossEntropyLoss
# 注意：CrossEntropyLoss 内部包含了 Softmax！
# 所以模型输出不需要加 Softmax 层

logits_example = torch.tensor([[1.0, 2.0, 0.5]])  # 原始输出
labels_example = torch.tensor([1])                   # 真实类别

criterion = nn.CrossEntropyLoss()
loss = criterion(logits_example, labels_example)
print(f"交叉熵损失: {loss.item():.4f}")

print("\n" + "=" * 60)
print("第三部分：鸢尾花多分类实战")
print("=" * 60)

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target  # 类别：0=setosa, 1=versicolor, 2=virginica

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"类别名称: {iris.target_names}")
print(f"特征名称: {iris.feature_names}")

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)  # 注意：分类标签用 long
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print(f"\n训练集大小: {X_train_t.shape[0]}")
print(f"测试集大小: {X_test_t.shape[0]}")

# 查看类别分布
print(f"\n训练集类别分布: {torch.bincount(y_train_t)}")
print(f"测试集类别分布: {torch.bincount(y_test_t)}")

print("\n" + "=" * 60)
print("第四步：定义多分类模型")
print("=" * 60)

class MultiClassClassifier(nn.Module):
    """
    多分类神经网络

    重要：
    - 输出层节点数 = 类别数 (3)
    - 不需要 Softmax 层！CrossEntropyLoss 会自动处理
    """

    def __init__(self, input_dim=4, hidden_dim=16, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出 logits（不需要 Softmax）
        return x

model = MultiClassClassifier(input_dim=4, hidden_dim=16, num_classes=3)
print(f"模型结构:\n{model}")

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\n损失函数: CrossEntropyLoss")
print("优化器: Adam, lr=0.01")

print("\n" + "=" * 60)
print("第五步：训练模型")
print("=" * 60)

epochs = 200
train_losses = []
train_accs = []
test_accs = []

for epoch in range(epochs):
    # 训练模式
    model.train()

    # 五步法
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 计算准确率
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            # 训练集准确率
            _, train_predicted = torch.max(model(X_train_t), 1)
            train_acc = (train_predicted == y_train_t).float().mean().item() * 100

            # 测试集准确率
            _, test_predicted = torch.max(model(X_test_t), 1)
            test_acc = (test_predicted == y_test_t).float().mean().item() * 100

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Loss: {loss.item():.4f}, "
                  f"训练准确率: {train_acc:.2f}%, "
                  f"测试准确率: {test_acc:.2f}%")

print("\n" + "=" * 60)
print("第六步：评估模型")
print("=" * 60)

# 最终评估
model.eval()
with torch.no_grad():
    # 训练集
    train_outputs = model(X_train_t)
    _, train_predicted = torch.max(train_outputs, 1)
    train_acc = (train_predicted == y_train_t).float().mean().item() * 100

    # 测试集
    test_outputs = model(X_test_t)
    _, test_predicted = torch.max(test_outputs, 1)
    test_acc = (test_predicted == y_test_t).float().mean().item() * 100

    # 获取预测概率
    test_probs = torch.softmax(test_outputs, dim=1)

print(f"训练集准确率: {train_acc:.2f}%")
print(f"测试集准确率: {test_acc:.2f}%")

# 查看一些预测结果
print("\n测试集预测示例:")
print("真实标签 | 预测标签 | 置信度 | 各类别概率")
print("-" * 60)
for i in range(min(10, len(X_test_t))):
    true_label = y_test_t[i].item()
    pred_label = test_predicted[i].item()
    probs = test_probs[i]
    confidence = probs[pred_label].item()

    print(f"{true_label:6d}   | {pred_label:6d}   | {confidence:.3f}   | "
          f"[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")

print("\n" + "=" * 60)
print("第七步：可视化结果")
print("=" * 60)

fig = plt.figure(figsize=(15, 10))

# 图1：损失曲线
ax1 = plt.subplot(2, 3, 1)
ax1.plot(train_losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练损失曲线')
ax1.grid(True, alpha=0.3)

# 图2：准确率曲线
ax2 = plt.subplot(2, 3, 2)
epochs_plot = range(20, epochs+1, 20)
ax2.plot(epochs_plot, train_accs, 'b-', label='训练准确率')
ax2.plot(epochs_plot, test_accs, 'r-', label='测试准确率')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('准确率曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：混淆矩阵
ax3 = plt.subplot(2, 3, 3)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_t.numpy(), test_predicted.numpy())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax3.set_xlabel('预测标签')
ax3.set_ylabel('真实标签')
ax3.set_title('混淆矩阵')

# 图4-6：特征对（使用原始数据）
from itertools import combinations
feature_indices = [0, 1, 2, 3]
feature_names = iris.feature_names
for idx, (i, j) in enumerate(combinations(feature_indices, 2)[:3]):
    ax = plt.subplot(2, 3, 4 + idx)

    # 绘制真实标签
    for class_idx in range(3):
        mask = y_test == class_idx
        ax.scatter(X_test[mask, i], X_test[mask, j],
                   label=iris.target_names[class_idx], alpha=0.7)

    # 标记预测错误的点
    wrong_mask = test_predicted.numpy() != y_test_t.numpy()
    ax.scatter(X_test[wrong_mask, i], X_test[wrong_mask, j],
               marker='x', s=100, c='red', label='预测错误')

    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_title(f'{feature_names[i]} vs {feature_names[j]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_classification_result.png', dpi=150)
plt.show()

# 保存模型
torch.save(model.state_dict(), 'iris_classifier.pth')
print("\n模型已保存到 iris_classifier.pth")

print("\n" + "=" * 60)
print("总结：多分类的关键要点")
print("=" * 60)

print("""
1. 模型结构：
   - 输出层节点数 = 类别数
   - 不需要 Softmax！CrossEntropyLoss 会自动处理

2. 损失函数：
   - 使用 nn.CrossEntropyLoss
   - 内部包含 LogSoftmax + NLLLoss
   - 标签类型：torch.long (整数 0, 1, 2, ...)

3. 预测：
   _, predicted = torch.max(outputs, 1)  # 获取预测类别
   probs = torch.softmax(outputs, dim=1)  # 获取概率

4. 评估指标：
   - 准确率：预测正确的比例
   - 混淆矩阵：查看各类别的分类情况

5. 与二分类对比：

   ┌────────────┬───────────────┬────────────────┐
   │            │ 二分类         │ 多分类          │
   ├────────────┼───────────────┼────────────────┤
   │ 输出层     │ 1 个节点       │ K 个节点        │
   │ 输出激活   │ Sigmoid       │ 无 (或 Softmax) │
   │ 损失函数   │ BCELoss       │ CrossEntropyLoss│
   │ 标签类型   │ float (0/1)   │ long (0,1,2...) │
   │ 预测方式   │ p >= 0.5      │ argmax(output)  │
   └────────────┴───────────────┴────────────────┘

6. 常见错误：
   ❌ 在输出层加 Softmax，然后用 CrossEntropyLoss
   → CrossEntropyLoss 已包含 Softmax，会重复计算

   ✅ 正确做法：输出层无激活，使用 CrossEntropyLoss
""")

print("图片已保存到 multi_classification_result.png")
