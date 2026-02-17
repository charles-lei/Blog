"""
自定义Dataset和DataLoader (Custom Dataset & DataLoader)
=======================================================

PyTorch数据加载的核心组件:
    1. Dataset: 数据集的抽象类，定义如何获取单个样本
    2. DataLoader: 批量加载器，处理批处理、打乱、并行加载等

本教程涵盖:
    - 自定义Dataset的创建
    - 图像数据集处理
    - 文本数据集处理
    - 数据增强
    - 自定义采样器
    - 多进程数据加载

常用数据集类型:
    - 图像: ImageFolder,自定义图像Dataset
    - 文本: 自定义文本Dataset
    - 时序: 滑动窗口Dataset
    - 表格: 结合pandas使用
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
from collections import Counter

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================
# 1. 基础自定义Dataset
# ============================================

class SimpleDataset(Dataset):
    """
    最简单的自定义Dataset

    必须实现:
        - __len__: 返回数据集大小
        - __getitem__: 根据索引获取单个样本
    """

    def __init__(self, data, labels):
        """
        Args:
            data: 数据 (N, ...)
            labels: 标签 (N,)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取样本

        Returns:
            sample, label
        """
        return self.data[idx], self.labels[idx]


# ============================================
# 2. 图像数据集
# ============================================

class CustomImageDataset(Dataset):
    """
    自定义图像数据集

    支持:
        - 从文件路径加载图像
        - 数据增强
        - 缓存机制
    """

    def __init__(self, image_paths, labels, transform=None, cache=False):
        """
        Args:
            image_paths: 图像文件路径列表
            labels: 对应标签
            transform: 图像变换
            cache: 是否缓存图像到内存
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache = cache
        self.cached_data = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 检查缓存
        if self.cache and idx in self.cached_data:
            return self.cached_data[idx]

        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        if self.cache:
            self.cached_data[idx] = (image, label)

        return image, label


# ============================================
# 3. 文本数据集
# ============================================

class TextDataset(Dataset):
    """
    文本数据集

    支持:
        - 词汇表构建
        - 文本到索引转换
        - 序列padding
    """

    def __init__(self, texts, labels, max_length=100, vocab=None):
        """
        Args:
            texts: 文本列表
            labels: 标签列表
            max_length: 最大序列长度
            vocab: 预定义词汇表 (可选)
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # 构建词汇表
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

        self.unk_idx = self.vocab.get('<UNK>', 0)
        self.pad_idx = self.vocab.get('<PAD>', 1)

    def _build_vocab(self, texts):
        """构建词汇表"""
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())

        vocab = {'<UNK>': 0, '<PAD>': 1}
        for idx, (word, _) in enumerate(word_counts.most_common(), start=2):
            vocab[word] = idx

        return vocab

    def _text_to_indices(self, text):
        """将文本转换为索引序列"""
        words = text.lower().split()
        indices = [self.vocab.get(word, self.unk_idx) for word in words]

        # 截断或padding
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.pad_idx] * (self.max_length - len(indices))

        return indices

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_indices = self._text_to_indices(self.texts[idx])
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================
# 4. 时序数据集 (滑动窗口)
# ============================================

class TimeSeriesDataset(Dataset):
    """
    时序数据集 (滑动窗口)

    用于时间序列预测任务
    将连续数据转换为 (历史窗口, 未来目标) 对
    """

    def __init__(self, data, window_size, horizon=1, step=1):
        """
        Args:
            data: 时序数据 (T, features)
            window_size: 历史窗口大小
            horizon: 预测步长
            step: 滑动步长
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.step = step

        # 计算有效样本数
        self.n_samples = (len(data) - window_size - horizon) // step + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size

        # 历史窗口
        x = self.data[start:end]

        # 未来目标
        y = self.data[end:end + self.horizon]

        return x, y


# ============================================
# 5. 不平衡数据集处理
# ============================================

class ImbalancedDataset(Dataset):
    """
    不平衡数据集示例

    展示如何处理类别不平衡问题
    """

    def __init__(self, n_samples=1000, imbalance_ratio=0.1):
        """
        Args:
            n_samples: 总样本数
            imbalance_ratio: 少数类比例
        """
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority

        # 生成不平衡数据
        minority_data = torch.randn(n_minority, 10) + torch.tensor([1] * 10)
        majority_data = torch.randn(n_majority, 10) + torch.tensor([-1] * 10)

        self.data = torch.cat([minority_data, majority_data])
        self.labels = torch.cat([
            torch.ones(n_minority, dtype=torch.long),
            torch.zeros(n_majority, dtype=torch.long)
        ])

        # 打乱
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_class_weights(self):
        """计算类别权重，用于加权损失函数"""
        class_counts = torch.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights


# ============================================
# 6. 自定义采样器
# ============================================

class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器

    每个batch包含相同数量的各类别样本
    """

    def __init__(self, labels, batch_size, drop_last=True):
        """
        Args:
            labels: 标签数组
            batch_size: 批次大小 (应该是类别数的整数倍)
            drop_last: 是否丢弃最后不完整的batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 获取每个类别的索引
        self.class_indices = {}
        for cls in np.unique(self.labels):
            self.class_indices[cls] = np.where(self.labels == cls)[0].tolist()

        self.n_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.n_classes

        # 计算总批次数
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.n_batches = min_class_size // self.samples_per_class

        if not drop_last:
            self.n_batches = max(self.n_batches, 1)

    def __iter__(self):
        # 打乱每个类别的索引
        for cls in self.class_indices:
            random.shuffle(self.class_indices[cls])

        # 生成batch
        for batch_idx in range(self.n_batches):
            batch = []
            for cls in self.class_indices:
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(self.class_indices[cls][start:end])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ============================================
# 7. 数据增强
# ============================================

class CustomTransforms:
    """
    自定义数据增强

    展示常用变换和自定义变换
    """

    @staticmethod
    def get_train_transforms(img_size=224):
        """训练时的图像增强"""
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms(img_size=224):
        """验证时的变换 (不做增强)"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    class AddGaussianNoise:
        """添加高斯噪声 (自定义变换)"""

        def __init__(self, mean=0, std=0.1):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise

        def __repr__(self):
            return f"AddGaussianNoise(mean={self.mean}, std={self.std})"

    @staticmethod
    class Cutout:
        """Cutout数据增强"""

        def __init__(self, n_holes=1, length=16):
            self.n_holes = n_holes
            self.length = length

        def __call__(self, img):
            """
            Args:
                img: Tensor (C, H, W)
            """
            h, w = img.size(-2), img.size(-1)
            mask = torch.ones_like(img)

            for _ in range(self.n_holes):
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)

                y1 = max(0, y - self.length // 2)
                y2 = min(h, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(w, x + self.length // 2)

                mask[:, y1:y2, x1:x2] = 0

            return img * mask


# ============================================
# 8. 多任务数据集
# ============================================

class MultiTaskDataset(Dataset):
    """
    多任务学习数据集

    同时返回多个任务的标签
    """

    def __init__(self, data, task1_labels, task2_labels):
        """
        Args:
            data: 输入数据
            task1_labels: 任务1标签 (如分类)
            task2_labels: 任务2标签 (如回归)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.task1_labels = torch.tensor(task1_labels, dtype=torch.long)
        self.task2_labels = torch.tensor(task2_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input': self.data[idx],
            'task1': self.task1_labels[idx],
            'task2': self.task2_labels[idx]
        }


# ============================================
# 9. 演示和实验
# ============================================

def demo_basic_dataset():
    """演示基础Dataset"""
    print("\n" + "=" * 60)
    print("基础Dataset演示")
    print("=" * 60)

    # 创建数据
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)

    # 创建Dataset
    dataset = SimpleDataset(data, labels)

    print(f"\n数据集大小: {len(dataset)}")
    print(f"第一个样本: {dataset[0][0].shape}, 标签: {dataset[0][1]}")

    # 使用DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    print(f"\n批次数量: {len(dataloader)}")
    for batch_data, batch_labels in dataloader:
        print(f"批次数据形状: {batch_data.shape}, 标签形状: {batch_labels.shape}")
        break


def demo_timeseries_dataset():
    """演示时序数据集"""
    print("\n" + "=" * 60)
    print("时序数据集演示")
    print("=" * 60)

    # 创建时序数据
    t = np.linspace(0, 10, 200)
    data = np.sin(t).reshape(-1, 1)

    # 创建滑动窗口数据集
    dataset = TimeSeriesDataset(data, window_size=20, horizon=5, step=1)

    print(f"\n原始数据长度: {len(data)}")
    print(f"窗口大小: 20, 预测步长: 5")
    print(f"样本数量: {len(dataset)}")

    x, y = dataset[0]
    print(f"\n输入形状: {x.shape} (20个历史点)")
    print(f"目标形状: {y.shape} (5个未来点)")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[:50], data[:50], 'b-', label='原始数据', linewidth=2)
    ax.axvspan(0, t[19], alpha=0.3, color='blue', label='窗口 (输入)')
    ax.axvspan(t[20], t[24], alpha=0.3, color='red', label='预测目标')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    plt.title('滑动窗口数据集示意')
    plt.tight_layout()
    plt.savefig('14-custom-dataset/timeseries_window.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n时序窗口示意图已保存到 14-custom-dataset/timeseries_window.png")


def demo_imbalanced_dataset():
    """演示不平衡数据集处理"""
    print("\n" + "=" * 60)
    print("不平衡数据集处理演示")
    print("=" * 60)

    # 创建不平衡数据集
    dataset = ImbalancedDataset(n_samples=1000, imbalance_ratio=0.1)

    # 统计类别分布
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = Counter(labels)

    print(f"\n类别分布:")
    for cls, count in sorted(class_counts.items()):
        print(f"  类别 {cls}: {count} 样本 ({count/len(dataset)*100:.1f}%)")

    # 获取类别权重
    weights = dataset.get_class_weights()
    print(f"\n类别权重: {weights}")

    # 使用WeightedRandomSampler
    sample_weights = [weights[label] for _, label in dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    # 使用sampler的DataLoader
    balanced_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 检查平衡后的批次分布
    balanced_counts = Counter()
    for _, batch_labels in balanced_loader:
        balanced_counts.update(batch_labels.tolist())

    print(f"\n使用WeightedRandomSampler后的批次分布:")
    for cls, count in sorted(balanced_counts.items()):
        print(f"  类别 {cls}: {count} 次")


def demo_text_dataset():
    """演示文本数据集"""
    print("\n" + "=" * 60)
    print("文本数据集演示")
    print("=" * 60)

    # 示例文本
    texts = [
        "this movie is great and I love it",
        "terrible film waste of time",
        "amazing acting and great story",
        "boring and disappointing movie",
        "highly recommend this film",
        "worst movie I have ever seen"
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1: 正面, 0: 负面

    # 创建文本数据集
    dataset = TextDataset(texts, labels, max_length=10)

    print(f"\n词汇表大小: {len(dataset.vocab)}")
    print(f"词汇表样例: {list(dataset.vocab.items())[:10]}")

    print(f"\n样本数量: {len(dataset)}")
    x, y = dataset[0]
    print(f"文本索引形状: {x.shape}")
    print(f"原文本: '{texts[0]}'")
    print(f"索引序列: {x.tolist()}")


def demo_custom_collate():
    """演示自定义collate_fn"""
    print("\n" + "=" * 60)
    print("自定义collate_fn演示")
    print("=" * 60)

    def custom_collate(batch):
        """
        自定义批次整理函数

        可以处理:
            - 变长序列
            - 字典输出
            - 自定义padding
        """
        # 假设batch是 [(data1, label1), (data2, label2), ...]
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # 堆叠数据
        data = torch.stack(data)
        labels = torch.tensor(labels)

        return data, labels

    # 创建数据集
    dataset = SimpleDataset(np.random.randn(100, 10), np.random.randint(0, 3, 100))

    # 使用自定义collate
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate)

    for batch_data, batch_labels in dataloader:
        print(f"批次数据形状: {batch_data.shape}, 标签形状: {batch_labels.shape}")
        break


# ============================================
# 10. 完整训练示例
# ============================================

def train_with_custom_dataset():
    """使用自定义数据集的完整训练示例"""
    print("\n" + "=" * 60)
    print("完整训练示例")
    print("=" * 60)

    # 创建不平衡数据集
    full_dataset = ImbalancedDataset(n_samples=1000, imbalance_ratio=0.2)

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 创建模型
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    ).to(device)

    # 使用加权损失函数处理类别不平衡
    class_weights = full_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    print("\n开始训练...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(20):
        # 训练
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                output = model(batch_data)
                loss = criterion(output, batch_labels)
                val_loss += loss.item()

                _, predicted = output.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('14-custom-dataset/training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n训练曲线已保存到 14-custom-dataset/training_progress.png")


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 自定义Dataset & DataLoader 教程")
    print("=" * 60)

    # 创建输出目录
    os.makedirs('14-custom-dataset', exist_ok=True)

    # 演示各种Dataset
    demo_basic_dataset()
    demo_timeseries_dataset()
    demo_imbalanced_dataset()
    demo_text_dataset()
    demo_custom_collate()

    # 完整训练示例
    train_with_custom_dataset()

    print("\n" + "=" * 60)
    print("教程完成!")
    print("=" * 60)
    print("""
Dataset和DataLoader总结:

1. Dataset类型:
   - TensorDataset: 简单张量数据
   - ImageFolder: 图像文件夹结构
   - 自定义Dataset: 继承Dataset类

2. DataLoader参数:
   - batch_size: 批次大小
   - shuffle: 是否打乱
   - num_workers: 多进程加载
   - pin_memory: 加速GPU传输
   - drop_last: 丢弃不完整批次

3. 数据增强:
   - torchvision.transforms
   - 训练时: 多种随机变换
   - 验证时: 只做基本变换

4. 处理不平衡:
   - WeightedRandomSampler
   - 加权损失函数
   - 过采样/欠采样

5. 最佳实践:
   - 预处理尽量在Dataset中完成
   - 使用num_workers加速
   - 大数据集考虑内存映射
   - 复杂变换考虑预处理
    """)
