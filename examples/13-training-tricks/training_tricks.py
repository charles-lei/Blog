"""
训练技巧与学习率调度 (Training Tricks & Learning Rate Scheduling)
==================================================================

本教程涵盖:
    1. 学习率调度器 (Learning Rate Schedulers)
    2. 早停 (Early Stopping)
    3. 梯度裁剪 (Gradient Clipping)
    4. 混合精度训练 (Mixed Precision Training)
    5. 模型检查点 (Model Checkpointing)
    6. 梯度累积 (Gradient Accumulation)

这些技巧可以:
    - 加速训练
    - 提高最终性能
    - 稳定训练过程
    - 节省内存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================
# 1. 学习率调度器
# ============================================

def demo_lr_schedulers():
    """
    演示PyTorch内置的学习率调度器

    常用调度器:
        - StepLR: 每隔固定epoch降低学习率
        - MultiStepLR: 在指定的epoch降低学习率
        - ExponentialLR: 指数衰减
        - CosineAnnealingLR: 余弦退火
        - ReduceLROnPlateau: 当指标停止改善时降低学习率
        - OneCycleLR: One Cycle策略
        - CosineAnnealingWarmRestarts: 带热重启的余弦退火
    """
    print("\n" + "=" * 60)
    print("学习率调度器演示")
    print("=" * 60)

    # 模型参数
    model = nn.Linear(10, 1)
    initial_lr = 0.1
    epochs = 100

    # 定义不同的调度器
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR(
            optim.SGD(model.parameters(), lr=initial_lr),
            step_size=20, gamma=0.5
        ),
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(
            optim.SGD(model.parameters(), lr=initial_lr),
            gamma=0.95
        ),
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(
            optim.SGD(model.parameters(), lr=initial_lr),
            T_max=epochs, eta_min=0.001
        ),
        'OneCycleLR': optim.lr_scheduler.OneCycleLR(
            optim.SGD(model.parameters(), lr=initial_lr),
            max_lr=initial_lr, total_steps=epochs
        ),
        'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim.SGD(model.parameters(), lr=initial_lr),
            T_0=20, T_mult=2, eta_min=0.001
        ),
    }

    # 记录每个调度器的学习率变化
    lr_history = {name: [] for name in schedulers}

    # 模拟训练过程
    for epoch in range(epochs):
        for name, scheduler in schedulers.items():
            lr = scheduler.get_last_lr()[0]
            lr_history[name].append(lr)
            scheduler.step()

    # 可视化
    plt.figure(figsize=(12, 8))
    for name, lrs in lr_history.items():
        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('13-training-tricks/lr_schedulers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("学习率调度器对比图已保存到 13-training-tricks/lr_schedulers.png")


# ============================================
# 2. 自定义学习率调度器
# ============================================

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    带Warmup的余弦退火调度器

    常用于Transformer等大型模型

    学习率变化:
        1. Warmup阶段: 线性增加到max_lr
        2. 衰减阶段: 余弦衰减到min_lr
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup: 线性增加
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + alpha * (self.max_lr - self.min_lr) for _ in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                    for _ in self.base_lrs]


class NoamScheduler(optim.lr_scheduler._LRScheduler):
    """
    Noam 学习率调度器 (Transformer原论文)

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        scale = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [scale * base_lr for base_lr in self.base_lrs]


# ============================================
# 3. 早停 (Early Stopping)
# ============================================

class EarlyStopping:
    """
    早停机制

    当验证损失连续patience个epoch没有改善时，停止训练
    """

    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'min'表示指标越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def restore_best(self, model):
        """恢复最佳模型"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================
# 4. 梯度裁剪 (Gradient Clipping)
# ============================================

def demo_gradient_clipping():
    """
    演示梯度裁剪

    作用:
        - 防止梯度爆炸
        - 稳定RNN/Transformer训练

    方法:
        - clip_grad_norm_: 按范数裁剪
        - clip_grad_value_: 按值裁剪
    """
    print("\n" + "=" * 60)
    print("梯度裁剪演示")
    print("=" * 60)

    # 创建一个容易梯度爆炸的模型
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    ).to(device)

    # 模拟数据
    X = torch.randn(32, 10).to(device)
    y = torch.randn(32, 1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)  # 大学习率导致梯度爆炸

    # 不使用梯度裁剪
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()

    grad_norm_no_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    print(f"\n不裁剪的梯度范数: {grad_norm_no_clip:.4f}")

    # 使用梯度裁剪
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()

    max_norm = 1.0
    grad_norm_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    print(f"裁剪后的梯度范数 (max_norm={max_norm}): {grad_norm_clipped:.4f}")


# ============================================
# 5. 混合精度训练 (Mixed Precision Training)
# ============================================

def demo_mixed_precision():
    """
    演示混合精度训练

    优势:
        - 减少显存占用 (约50%)
        - 加速训练 (在支持Tensor Core的GPU上)
        - 保持模型精度

    使用方法:
        from torch.cuda.amp import autocast, GradScaler
    """
    print("\n" + "=" * 60)
    print("混合精度训练演示")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("混合精度训练需要CUDA支持，跳过演示")
        print("""
混合精度训练代码示例:

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    # 使用autocast进行前向传播
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # 使用scaler进行反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
        """)
        return

    # 实际演示
    from torch.cuda.amp import autocast, GradScaler

    model = nn.Linear(1000, 100).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scaler = GradScaler()

    X = torch.randn(64, 1000).cuda()
    y = torch.randn(64, 100).cuda()

    # 混合精度训练循环
    optimizer.zero_grad()
    with autocast():
        output = model(X)
        loss = criterion(output, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Loss: {loss.item():.4f}")


# ============================================
# 6. 模型检查点 (Model Checkpointing)
# ============================================

class ModelCheckpoint:
    """
    模型检查点保存

    功能:
        - 保存最佳模型
        - 定期保存
        - 保存完整训练状态 (用于恢复训练)
    """

    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        """
        Args:
            filepath: 保存路径模板 (可包含 {epoch}, {val_loss} 等占位符)
            monitor: 监控的指标名
            mode: 'min' 或 'max'
            save_best_only: 是否只保存最佳模型
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.counter = 0

    def __call__(self, model, optimizer, epoch, metrics):
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics: 指标字典，如 {'val_loss': 0.5, 'val_acc': 0.9}
        """
        score = metrics.get(self.monitor)

        # 格式化文件名
        filename = self.filepath.format(epoch=epoch, **metrics)

        if self.best_score is None:
            self.best_score = score
            self._save(model, optimizer, epoch, metrics, filename)
        elif self._is_improvement(score):
            self.best_score = score
            self._save(model, optimizer, epoch, metrics, filename)
            self.counter = 0
        else:
            self.counter += 1
            if not self.save_best_only:
                self._save(model, optimizer, epoch, metrics, filename)

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score

    def _save(self, model, optimizer, epoch, metrics, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, filename)
        print(f"  检查点已保存: {filename}")


def load_checkpoint(model, optimizer, filename):
    """加载检查点"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


# ============================================
# 7. 梯度累积 (Gradient Accumulation)
# ============================================

def demo_gradient_accumulation():
    """
    演示梯度累积

    用途:
        - 模拟更大的batch size
        - 在显存有限时训练大模型

    原理:
        - 多次前向传播和反向传播后才更新参数
        - 累积梯度，等效于使用更大的batch
    """
    print("\n" + "=" * 60)
    print("梯度累积演示")
    print("=" * 60)

    # 参数
    actual_batch_size = 16
    accumulation_steps = 4  # 等效batch size = 16 * 4 = 64
    epochs = 5

    # 创建模型和数据
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)

    X = torch.randn(200, 100).to(device)
    y = torch.randint(0, 10, (200,)).to(device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print(f"\n实际batch size: {actual_batch_size}")
    print(f"累积步数: {accumulation_steps}")
    print(f"等效batch size: {actual_batch_size * accumulation_steps}")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            output = model(batch_X)
            loss = criterion(output, batch_y)

            # 归一化loss (重要!)
            loss = loss / accumulation_steps

            loss.backward()
            total_loss += loss.item()

            # 每accumulation_steps步更新一次参数
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader) * accumulation_steps
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


# ============================================
# 8. 完整训练循环示例
# ============================================

def train_with_all_tricks():
    """
    综合使用所有训练技巧的完整示例
    """
    print("\n" + "=" * 60)
    print("综合训练技巧示例")
    print("=" * 60)

    # 参数
    epochs = 50
    batch_size = 32
    learning_rate = 0.01

    # 创建模型
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    ).to(device)

    # 创建数据
    X_train = torch.randn(400, 20).to(device)
    y_train = torch.randint(0, 3, (400,)).to(device)
    X_val = torch.randn(100, 20).to(device)
    y_val = torch.randint(0, 3, (100,)).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 早停
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    print("\n开始训练...")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # 早停检查
        early_stopping(val_loss, model)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.6f}")

        if early_stopping.early_stop:
            print(f"\n早停触发于 Epoch {epoch+1}")
            early_stopping.restore_best(model)
            break

    # 恢复最佳模型
    early_stopping.restore_best(model)

    # 可视化训练过程
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['lr'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)

    # 显示学习率与损失的关系
    axes[2].scatter(history['lr'], history['val_loss'], alpha=0.5)
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('Validation Loss')
    axes[2].set_title('LR vs Val Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('13-training-tricks/training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n训练过程图已保存到 13-training-tricks/training_progress.png")


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 训练技巧教程")
    print("=" * 60)

    # 创建输出目录
    os.makedirs('13-training-tricks', exist_ok=True)

    # 1. 学习率调度器
    demo_lr_schedulers()

    # 2. 梯度裁剪
    demo_gradient_clipping()

    # 3. 混合精度
    demo_mixed_precision()

    # 4. 梯度累积
    demo_gradient_accumulation()

    # 5. 综合示例
    train_with_all_tricks()

    print("\n" + "=" * 60)
    print("教程完成!")
    print("=" * 60)
    print("""
训练技巧总结:

1. 学习率调度:
   - 初期: 使用warmup
   - 中期: StepLR / CosineAnnealing
   - 后期: ReduceLROnPlateau

2. 防止过拟合:
   - Early Stopping
   - Dropout
   - Weight Decay
   - 数据增强

3. 训练稳定性:
   - 梯度裁剪 (RNN/Transformer)
   - Batch Normalization
   - Layer Normalization

4. 内存优化:
   - 梯度累积
   - 混合精度训练
   - 模型并行

5. 最佳实践:
   - 保存检查点
   - 记录训练日志
   - 可视化训练过程
   - 定期验证模型
    """)
