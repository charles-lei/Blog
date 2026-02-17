"""
线性回归 - 使用 PyTorch nn.Module（更规范的方式）
================================================

使用 PyTorch 提供的 nn.Module 来定义模型，
这是实际项目中更常用的方式。

学习目标：
1. 学会使用 nn.Linear 定义线性层
2. 学会使用 nn.MSELoss 定义损失函数
3. 学会使用 optim.SGD 定义优化器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一步：准备数据")
print("=" * 60)

torch.manual_seed(42)

X = torch.unsqueeze(torch.linspace(0, 10, 100), dim=1)
y_true = 3 * X + 0.8
noise = torch.randn(X.shape) * 0.5
y = y_true + noise

print(f"数据形状: X={X.shape}, y={y.shape}")

print("\n" + "=" * 60)
print("第二步：定义模型（使用 nn.Module）")
print("=" * 60)

class LinearRegression(nn.Module):
    """
    自定义模型类，继承自 nn.Module

    必须实现两个方法：
    1. __init__(): 定义模型中的层
    2. forward(): 定义前向传播
    """

    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features)
        # in_features: 输入特征数量
        # out_features: 输出特征数量
        self.linear = nn.Linear(1, 1)  # 一元线性回归：输入1维，输出1维

    def forward(self, x):
        """前向传播：定义如何从输入得到输出"""
        return self.linear(x)

# 创建模型实例
model = LinearRegression()

# 查看模型结构
print(f"模型结构:\n{model}")

# 查看初始参数
print(f"\n初始权重: {model.linear.weight.item():.4f}")
print(f"初始偏置: {model.linear.bias.item():.4f}")

print("\n" + "=" * 60)
print("第三步：定义损失函数和优化器")
print("=" * 60)

# 损失函数：PyTorch 提供的均方误差
criterion = nn.MSELoss()

# 优化器：PyTorch 提供的 SGD
# 需要传入：模型的参数、学习率
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

print("\n" + "=" * 60)
print("第四步：训练循环（标准模板）")
print("=" * 60)

epochs = 200
loss_history = []

for epoch in range(epochs):
    # === 步骤1：清空梯度 ===
    optimizer.zero_grad()

    # === 步骤2：前向传播 ===
    y_pred = model(X)

    # === 步骤3：计算损失 ===
    loss = criterion(y_pred, y)

    # === 步骤4：反向传播 ===
    loss.backward()

    # === 步骤5：更新参数 ===
    optimizer.step()

    # 记录损失
    loss_history.append(loss.item())

    if (epoch + 1) % 20 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w: {w:.4f}, b: {b:.4f}")

print("\n" + "=" * 60)
print("训练结果")
print("=" * 60)

w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"学习到的参数: w={w:.4f}, b={b:.4f}")
print(f"真实参数: w=3.0000, b=0.8000")

# 保存模型
torch.save(model.state_dict(), 'linear_model.pth')
print("\n模型已保存到 linear_model.pth")

# 加载模型
model_loaded = LinearRegression()
model_loaded.load_state_dict(torch.load('linear_model.pth'))
print("模型加载成功！")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X.numpy(), y.numpy(), s=10, alpha=0.5, label='训练数据')
axes[0].plot(X.numpy(), y_true.numpy(), 'g-', linewidth=2, label='真实直线')
axes[0].plot(X.numpy(), model(X).detach().numpy(), 'r--', linewidth=2, label='学习直线')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('线性回归拟合结果 (nn.Module版)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss_history)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('训练损失曲线')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_nn_result.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("总结：训练循环的五步法（必须牢记！）")
print("=" * 60)

print("""
for epoch in range(epochs):
    # 1. 清空梯度（防止梯度累加）
    optimizer.zero_grad()

    # 2. 前向传播（计算预测值）
    y_pred = model(X)

    # 3. 计算损失（衡量预测与真实值的差距）
    loss = criterion(y_pred, y)

    # 4. 反向传播（计算梯度）
    loss.backward()

    # 5. 更新参数（根据梯度调整参数）
    optimizer.step()

这个五步法适用于所有神经网络训练！
""")
