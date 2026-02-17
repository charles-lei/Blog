"""
迁移学习 (Transfer Learning)
============================

核心思想：
    利用在大数据集（如ImageNet）上预训练的模型，迁移到新任务上。
    可以显著减少训练时间和数据需求。

主要方法：
    1. 特征提取 (Feature Extraction): 冻结预训练模型，只训练分类头
    2. 微调 (Fine-tuning): 解冻部分或全部层，用较小学习率训练

适用场景：
    - 数据量较小
    - 预训练模型与新任务相关
    - 计算资源有限

常用预训练模型：
    - ResNet (18, 34, 50, 101, 152)
    - VGG (11, 13, 16, 19)
    - EfficientNet
    - Vision Transformer (ViT)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================
# 方法1: 特征提取 (Feature Extraction)
# ============================================

def create_feature_extractor(num_classes=10, pretrained=True):
    """
    创建特征提取模型
    - 冻结所有卷积层
    - 只训练最后的全连接层
    """
    # 加载预训练的ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 获取全连接层的输入特征数
    num_features = model.fc.in_features

    # 替换最后的全连接层（这一层会被训练）
    model.fc = nn.Linear(num_features, num_classes)

    return model


# ============================================
# 方法2: 微调 (Fine-tuning)
# ============================================

def create_finetune_model(num_classes=10, pretrained=True, freeze_backbone=True):
    """
    创建微调模型
    - 可以选择冻结骨干网络或全部解冻
    - 使用较小的学习率
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        # 冻结骨干网络，只训练分类头
        for param in model.parameters():
            param.requires_grad = False

    # 替换分类头
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


# ============================================
# 方法3: 渐进式解冻 (Progressive Unfreezing)
# ============================================

class ProgressiveUnfreezing:
    """
    渐进式解冻策略
    - 先训练分类头
    - 逐步解冻后面的层
    - 最后解冻全部网络
    """

    def __init__(self, model, num_stages=3):
        self.model = model
        self.num_stages = num_stages
        self.current_stage = 0

    def unfreeze_stage(self, stage):
        """解冻指定阶段"""
        # ResNet的层结构: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        backbone_layers = ['layer4', 'layer3', 'layer2', 'layer1']

        if stage == 0:
            # 阶段0: 只训练分类头（fc层）
            for name, param in self.model.named_parameters():
                param.requires_grad = 'fc' in name

        elif stage <= len(backbone_layers):
            # 阶段1-4: 逐步解冻后面的层
            layers_to_unfreeze = backbone_layers[:stage]
            for name, param in self.model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

        else:
            # 全部解冻
            for param in self.model.parameters():
                param.requires_grad = True

        self.current_stage = stage

    def get_trainable_params(self):
        """获取可训练参数的数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================
# 方法4: 差分学习率 (Discriminative Learning Rates)
# ============================================

def get_discriminative_params(model, base_lr=1e-3, lr_mult=0.1):
    """
    差分学习率
    - 浅层使用较小学习率（保留通用特征）
    - 深层使用较大学习率（适应新任务）
    - 分类头使用最大学习率
    """
    # 定义参数组
    param_groups = [
        # 浅层 (layer1, layer2): 最小学习率
        {'params': [], 'lr': base_lr * lr_mult * lr_mult, 'name': 'shallow'},
        # 中层 (layer3): 中等学习率
        {'params': [], 'lr': base_lr * lr_mult, 'name': 'middle'},
        # 深层 (layer4): 较大学习率
        {'params': [], 'lr': base_lr, 'name': 'deep'},
        # 分类头: 最大学习率
        {'params': [], 'lr': base_lr * 10, 'name': 'head'},
    ]

    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param_groups[0]['params'].append(param)
        elif 'layer3' in name:
            param_groups[1]['params'].append(param)
        elif 'layer4' in name:
            param_groups[2]['params'].append(param)
        elif 'fc' in name:
            param_groups[3]['params'].append(param)

    # 移除空的参数组
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    return param_groups


# ============================================
# 创建模拟数据
# ============================================

def create_mock_data(num_samples=500, num_classes=10, img_size=224):
    """创建模拟图像数据"""
    # 模拟RGB图像 (batch, channels, height, width)
    X = torch.randn(num_samples, 3, img_size, img_size)
    # 随机标签
    y = torch.randint(0, num_classes, (num_samples,))

    return X, y


def create_data_loaders(X, y, batch_size=32, train_ratio=0.8):
    """创建训练和测试数据加载器"""
    # 划分训练集和测试集
    train_size = int(len(X) * train_ratio)

    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    test_dataset = TensorDataset(X[train_size:], y[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================
# 训练和评估函数
# ============================================

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return total_loss / len(loader), correct / total


# ============================================
# 实验1: 特征提取 vs 微调 vs 从头训练
# ============================================

def experiment_comparison():
    """比较不同迁移学习策略"""
    print("\n" + "=" * 60)
    print("实验: 特征提取 vs 微调 vs 从头训练")
    print("=" * 60)

    # 创建数据
    X, y = create_mock_data(num_samples=500, num_classes=10)
    train_loader, test_loader = create_data_loaders(X, y)

    num_classes = 10
    epochs = 10

    results = {}

    # 1. 特征提取 (冻结骨干)
    print("\n1. 特征提取 (Feature Extraction)...")
    model_fe = create_feature_extractor(num_classes, pretrained=True)
    model_fe = model_fe.to(device)

    # 只优化分类头参数
    optimizer_fe = optim.Adam(filter(lambda p: p.requires_grad, model_fe.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    fe_train_losses = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model_fe, train_loader, criterion, optimizer_fe, device)
        test_loss, test_acc = evaluate(model_fe, test_loader, criterion, device)
        fe_train_losses.append(train_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")

    results['特征提取'] = fe_train_losses

    # 2. 微调 (全部解冻)
    print("\n2. 微调 (Fine-tuning)...")
    model_ft = create_finetune_model(num_classes, pretrained=True, freeze_backbone=False)
    model_ft = model_ft.to(device)

    # 使用较小学习率
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    ft_train_losses = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model_ft, train_loader, criterion, optimizer_ft, device)
        test_loss, test_acc = evaluate(model_ft, test_loader, criterion, device)
        ft_train_losses.append(train_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")

    results['微调'] = ft_train_losses

    # 3. 从头训练 (无预训练)
    print("\n3. 从头训练 (No Pretraining)...")
    model_scratch = create_finetune_model(num_classes, pretrained=False, freeze_backbone=False)
    model_scratch = model_scratch.to(device)

    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)

    scratch_train_losses = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model_scratch, train_loader, criterion, optimizer_scratch, device)
        test_loss, test_acc = evaluate(model_scratch, test_loader, criterion, device)
        scratch_train_losses.append(train_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")

    results['从头训练'] = scratch_train_losses

    # 绘制结果
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('迁移学习策略比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('09-transfer-learning/transfer_learning_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n图表已保存到 09-transfer-learning/transfer_learning_comparison.png")


# ============================================
# 实验2: 差分学习率
# ============================================

def experiment_discriminative_lr():
    """演示差分学习率的效果"""
    print("\n" + "=" * 60)
    print("实验: 差分学习率 (Discriminative Learning Rates)")
    print("=" * 60)

    X, y = create_mock_data(num_samples=500, num_classes=10)
    train_loader, test_loader = create_data_loaders(X, y)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # 获取差分学习率参数组
    param_groups = get_discriminative_params(model, base_lr=1e-4, lr_mult=0.1)

    print("\n参数组学习率配置:")
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: lr={group['lr']:.6f}, params={num_params:,}")

    # 创建优化器
    optimizer = optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n开始训练...")
    for epoch in range(5):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")


# ============================================
# 实验3: 渐进式解冻
# ============================================

def experiment_progressive_unfreezing():
    """演示渐进式解冻策略"""
    print("\n" + "=" * 60)
    print("实验: 渐进式解冻 (Progressive Unfreezing)")
    print("=" * 60)

    X, y = create_mock_data(num_samples=500, num_classes=10)
    train_loader, test_loader = create_data_loaders(X, y)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    unfreezing = ProgressiveUnfreezing(model)

    criterion = nn.CrossEntropyLoss()

    print("\n阶段0: 只训练分类头...")
    unfreezing.unfreeze_stage(0)
    print(f"  可训练参数: {unfreezing.get_trainable_params():,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    print("\n阶段1: 解冻layer4...")
    unfreezing.unfreeze_stage(1)
    print(f"  可训练参数: {unfreezing.get_trainable_params():,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    print("\n阶段2: 解冻layer3...")
    unfreezing.unfreeze_stage(2)
    print(f"  可训练参数: {unfreezing.get_trainable_params():,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")


# ============================================
# 实用函数: 查看模型结构
# ============================================

def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n模型信息:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {frozen_params:,}")


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 迁移学习教程")
    print("=" * 60)

    # 实验1: 比较不同策略
    experiment_comparison()

    # 实验2: 差分学习率
    experiment_discriminative_lr()

    # 实验3: 渐进式解冻
    experiment_progressive_unfreezing()

    print("\n" + "=" * 60)
    print("教程完成!")
    print("=" * 60)
    print("""
迁移学习最佳实践:

1. 数据量小时:
   - 使用特征提取（冻结骨干网络）
   - 只训练分类头

2. 数据量中等时:
   - 使用微调
   - 差分学习率
   - 渐进式解冻

3. 数据量大时:
   - 可以从头训练
   - 或使用较小学习率微调全部层

4. 学习率选择:
   - 预训练层: 1e-5 ~ 1e-4
   - 分类头: 1e-3 ~ 1e-2
   - 差分学习率: 深层 > 浅层

5. 常见错误:
   - 学习率太大导致预训练权重被破坏
   - 忘记冻结参数导致训练太慢
   - 没有调整输入图像的归一化
    """)
