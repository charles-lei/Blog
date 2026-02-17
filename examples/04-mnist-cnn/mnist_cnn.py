"""
MNIST 手写数字识别 - 卷积神经网络(CNN)入门
===========================================

MNIST 是深度学习的 "Hello World"！
- 60,000 张训练图片，10,000 张测试图片
- 每张图片是 28x28 的灰度图
- 任务：识别图片中的数字(0-9)

CNN 核心概念：
1. 卷积层(Conv)：提取图像特征
2. 池化层(Pool)：降低维度，减少计算
3. 全连接层(FC)：分类决策

学习目标：
1. 学会使用 DataLoader 加载数据
2. 理解 CNN 的基本结构
3. 实现完整的图像分类流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

print("\n" + "=" * 60)
print("第一步：数据准备")
print("=" * 60)

# 数据预处理：将图片转换为Tensor并标准化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量，值范围从 [0,255] 变为 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST的均值和标准差）
])

# 下载并加载训练集
train_dataset = datasets.MNIST(
    root='./data',      # 数据保存路径
    train=True,         # 使用训练集
    download=True,      # 如果不存在则下载
    transform=transform  # 应用预处理
)

# 下载并加载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,        # 使用测试集
    download=True,
    transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"批次大小: {batch_size}")
print(f"训练批次数: {len(train_loader)}")

# 查看数据样例
images, labels = next(iter(train_loader))
print(f"\n一个批次的数据形状: images={images.shape}, labels={labels.shape}")
print("images shape 含义: (batch_size, channels, height, width)")
print("即 (64, 1, 28, 28) = 64张图片，1个颜色通道，28x28像素")

# 可视化一些样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f'标签: {labels[i].item()}')
    ax.axis('off')
plt.suptitle('MNIST 训练数据示例')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("第二步：定义 CNN 模型")
print("=" * 60)

class SimpleCNN(nn.Module):
    """
    简单的CNN结构：

    输入 (1, 28, 28)
        ↓
    Conv1: 1→16, 3x3 → (16, 26, 26)
    ReLU
    MaxPool 2x2 → (16, 13, 13)
        ↓
    Conv2: 16→32, 3x3 → (32, 11, 11)
    ReLU
    MaxPool 2x2 → (32, 5, 5)
        ↓
    Flatten → (32*5*5) = (800)
        ↓
    FC1: 800→128
    ReLU
        ↓
    FC2: 128→10 (10个数字类别)
    """

    def __init__(self):
        super().__init__()

        # 卷积层1：提取低级特征（边缘、角点等）
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 输入通道数（灰度图为1）
            out_channels=16,    # 输出通道数（卷积核数量）
            kernel_size=3,      # 卷积核大小 3x3
            stride=1,           # 步长
            padding=0           # 填充
        )

        # 卷积层2：提取高级特征
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        # 池化层：降维
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # 需要计算flatten后的维度
        self.fc2 = nn.Linear(128, 10)          # 10个类别

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一卷积块
        x = self.conv1(x)       # (batch, 16, 26, 26)
        x = self.relu(x)
        x = self.pool(x)        # (batch, 16, 13, 13)

        # 第二卷积块
        x = self.conv2(x)       # (batch, 32, 11, 11)
        x = self.relu(x)
        x = self.pool(x)        # (batch, 32, 5, 5)

        # 展平
        x = x.view(-1, 32 * 5 * 5)  # (batch, 800)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)         # (batch, 10)

        return x

# 创建模型
model = SimpleCNN().to(device)
print(f"\n模型结构:\n{model}")

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")

print("\n" + "=" * 60)
print("第三步：定义损失函数和优化器")
print("=" * 60)

# 交叉熵损失（多分类问题）
criterion = nn.CrossEntropyLoss()

# Adam优化器（比SGD收敛更快）
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"损失函数: CrossEntropyLoss (多分类交叉熵)")
print(f"优化器: Adam, 学习率=0.001")

print("\n" + "=" * 60)
print("第四步：训练模型")
print("=" * 60)

def train(model, device, train_loader, optimizer, criterion, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 五步法
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每100个batch打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    return running_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 测试不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'  测试集: 平均损失: {test_loss:.4f}, 准确率: {accuracy:.2f}%\n')
    return test_loss, accuracy


# 训练循环
epochs = 5
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

print("=" * 60)
print("第五步：可视化训练结果")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
axes[0].plot(range(1, epochs+1), train_losses, 'b-', label='训练损失')
axes[0].plot(range(1, epochs+1), test_losses, 'r-', label='测试损失')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练和测试损失')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(range(1, epochs+1), train_accs, 'b-', label='训练准确率')
axes[1].plot(range(1, epochs+1), test_accs, 'r-', label='测试准确率')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('训练和测试准确率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_training_curves.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("第六步：模型预测可视化")
print("=" * 60)

# 获取测试集的一些样本
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)

# 可视化预测结果
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].cpu().squeeze(), cmap='gray')
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f'预测: {predicted[i].item()}, 真实: {labels[i].item()}', color=color)
    ax.axis('off')

plt.suptitle('模型预测结果（绿色=正确，红色=错误）', fontsize=14)
plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=150)
plt.show()

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pth')
print(f"\n模型已保存到 mnist_cnn.pth")
print(f"最终测试准确率: {test_accs[-1]:.2f}%")

print("\n" + "=" * 60)
print("CNN 核心概念总结")
print("=" * 60)

print("""
1. 卷积层 (Conv2d)
   - 使用卷积核在图像上滑动，提取特征
   - 参数：in_channels, out_channels, kernel_size
   - 输出大小 = (输入大小 - 卷积核大小 + 2*padding) / stride + 1

2. 池化层 (MaxPool2d)
   - 降低特征图尺寸，减少计算量
   - 常用 2x2 池化，尺寸减半

3. 激活函数 (ReLU)
   - 引入非线性，使网络能学习复杂模式
   - ReLU(x) = max(0, x)

4. 全连接层 (Linear)
   - 将特征展平后进行分类
   - 最后输出节点数 = 类别数

5. 数据维度变化（MNIST示例）：
   输入: (batch, 1, 28, 28)
   Conv1: (batch, 16, 26, 26)
   Pool: (batch, 16, 13, 13)
   Conv2: (batch, 32, 11, 11)
   Pool: (batch, 32, 5, 5)
   Flatten: (batch, 800)
   FC1: (batch, 128)
   FC2: (batch, 10)
""")
