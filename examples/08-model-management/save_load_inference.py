"""
模型保存、加载和推理
===================

这是深度学习项目中的必备技能！

主要场景：
1. 训练中断后恢复（checkpoint）
2. 模型部署和推理
3. 迁移学习（使用预训练模型）
4. 模型版本管理

学习目标：
1. 掌握多种保存/加载方法
2. 理解 checkpoint 的使用
3. 学会模型推理的最佳实践
4. 了解 ONNX 导出（用于生产部署）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime

print("=" * 60)
print("第一部分：模型保存的三种方式")
print("=" * 60)

# 定义一个简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟训练过程
epoch = 50
loss = 0.123

print("""
方式1：只保存模型参数（推荐）
  torch.save(model.state_dict(), 'model.pth')
  - 文件小
  - 加载时需要先定义模型结构
  - 灵活性高，最常用

方式2：保存整个模型
  torch.save(model, 'model_whole.pth')
  - 文件大
  - 加载时不需要定义模型
  - 可能有兼容性问题，不推荐

方式3：保存 checkpoint（训练状态）
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
  }, 'checkpoint.pth')
  - 保存完整训练状态
  - 用于中断后恢复训练
""")

print("\n" + "=" * 60)
print("第二部分：完整示例 - 保存模型")
print("=" * 60)

# 创建目录
os.makedirs('checkpoints', exist_ok=True)

# 模拟训练一些步骤
print("模拟训练...")

# 1. 保存模型参数（推荐）
torch.save(model.state_dict(), 'checkpoints/model_weights.pth')
print("✓ 模型参数已保存到 checkpoints/model_weights.pth")

# 2. 保存完整模型（不推荐，但方便）
torch.save(model, 'checkpoints/model_whole.pth')
print("✓ 完整模型已保存到 checkpoints/model_whole.pth")

# 3. 保存 checkpoint（训练状态）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'model_config': {
        'input_size': 10,
        'hidden_size': 20,
        'output_size': 10
    },
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
torch.save(checkpoint, 'checkpoints/checkpoint_epoch50.pth')
print("✓ 训练 checkpoint 已保存到 checkpoints/checkpoint_epoch50.pth")

# 4. 保存最佳模型
best_loss = 0.100
torch.save({
    'model_state_dict': model.state_dict(),
    'loss': best_loss,
    'epoch': epoch
}, 'checkpoints/best_model.pth')
print(f"✓ 最佳模型已保存（loss={best_loss:.3f}）")

# 5. 保存训练历史（用于可视化）
history = {
    'train_losses': [0.5, 0.4, 0.3, 0.2, 0.123],
    'val_losses': [0.6, 0.5, 0.4, 0.3, 0.2],
    'epochs': list(range(1, 51))
}
with open('checkpoints/training_history.json', 'w') as f:
    json.dump(history, f)
print("✓ 训练历史已保存到 checkpoints/training_history.json")

print("\n" + "=" * 60)
print("第三部分：加载模型")
print("=" * 60)

print("\n方式1：加载模型参数（推荐）")
print("-" * 40)

# 必须先定义模型结构
loaded_model = SimpleNet()

# 加载参数
state_dict = torch.load('checkpoints/model_weights.pth')
loaded_model.load_state_dict(state_dict)
loaded_model.eval()  # 设置为评估模式

print("✓ 模型已加载")
print(f"  模型处于 {'训练' if loaded_model.training else '评估'} 模式")

print("\n方式2：从 checkpoint 加载")
print("-" * 40)

# 创建模型和优化器
new_model = SimpleNet()
new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)

# 加载 checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch50.pth')
new_model.load_state_dict(checkpoint['model_state_dict'])
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
loss = checkpoint['loss']

print(f"✓ Checkpoint 已加载")
print(f"  恢复训练从 epoch {start_epoch} 开始")
print(f"  之前损失: {loss:.4f}")
print(f"  保存时间: {checkpoint['timestamp']}")
print(f"  模型配置: {checkpoint['model_config']}")

print("\n方式3：加载完整模型")
print("-" * 40)

# 直接加载，不需要定义模型
loaded_whole = torch.load('checkpoints/model_whole.pth')
loaded_whole.eval()

print("✓ 完整模型已加载")
print("  注意：这种方式不推荐用于生产环境")

print("\n" + "=" * 60)
print("第四部分：模型推理（预测）")
print("=" * 60)

print("""
推理时的重要注意事项：

1. 设置为评估模式
   model.eval()
   - 禁用 Dropout
   - 禁用 BatchNorm 的训练行为

2. 使用 torch.no_grad()
   with torch.no_grad():
       predictions = model(inputs)
   - 不计算梯度，节省内存
   - 加快推理速度

3. 处理输入数据
   - 转换为 Tensor
   - 添加 batch 维度
   - 移动到正确设备（CPU/GPU）
   - 应用与训练时相同的预处理

4. 处理输出
   - 移回 CPU
   - 转换为 numpy
   - 反归一化（如果需要）
""")

# 完整的推理示例
def predict(model, input_data, device='cpu'):
    """
    标准的推理函数

    参数：
        model: 训练好的模型
        input_data: 输入数据（numpy array 或 tensor）
        device: 'cpu' 或 'cuda'
    """
    model.eval()  # 设置为评估模式

    # 处理输入
    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32)

    # 添加 batch 维度（如果需要）
    if input_data.dim() == 1:
        input_data = input_data.unsqueeze(0)

    # 移动到设备
    input_data = input_data.to(device)
    model = model.to(device)

    # 推理
    with torch.no_grad():
        output = model(input_data)

    # 移回 CPU 并转换为 numpy
    output = output.cpu().numpy()

    return output

# 测试推理
sample_input = torch.randn(1, 10)  # batch_size=1, input_dim=10
prediction = predict(loaded_model, sample_input)
print(f"\n示例推理：")
print(f"  输入形状: {sample_input.shape}")
print(f"  输出形状: {prediction.shape}")
print(f"  输出值: {prediction[0][:5]}... (前5个值)")

print("\n" + "=" * 60)
print("第五部分：批量推理")
print("=" * 60)

# 批量推理
batch_input = torch.randn(5, 10)  # 5个样本

def batch_predict(model, inputs, batch_size=32, device='cpu'):
    """
    批量推理（大数据集时很有用）
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size].to(device)
            output = model(batch)
            predictions.append(output.cpu())

    return torch.cat(predictions, dim=0)

predictions = batch_predict(loaded_model, batch_input, batch_size=2)
print(f"批量推理完成：{predictions.shape[0]} 个样本")

print("\n" + "=" * 60)
print("第六部分：模型导出为 ONNX（生产部署）")
print("=" * 60)

# 导出为 ONNX 格式
dummy_input = torch.randn(1, 10)

try:
    torch.onnx.export(
        loaded_model,                    # 模型
        dummy_input,                     # 示例输入
        "checkpoints/model.onnx",       # 输出文件
        export_params=True,             # 存储参数
        opset_version=14,                # ONNX 版本
        do_constant_folding=True,       # 优化常量
        input_names=['input'],           # 输入名称
        output_names=['output'],         # 输出名称
        dynamic_axes={                   # 动态维度
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✓ 模型已导出为 ONNX 格式：checkpoints/model.onnx")
    print("  ONNX 模型可以在其他框架（TensorFlow, Caffe）中使用")
except Exception as e:
    print(f"ONNX 导出失败: {e}")

print("\n" + "=" * 60)
print("第七部分：完整的训练循环（带 checkpoint）")
print("=" * 60)

def train_with_checkpointing(model, train_loader, val_loader, epochs, save_dir='checkpoints'):
    """
    带 checkpoint 保存的训练循环
    """
    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item()

        val_loss /= len(val_loader)

        # 保存 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{save_dir}/checkpoint_epoch{epoch+1}.pth')
            print(f'Epoch {epoch+1}: Checkpoint saved (val_loss={val_loss:.4f})')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch + 1
            }, f'{save_dir}/best_model.pth')
            print(f'  → New best model saved! (val_loss={val_loss:.4f})')

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")

print("""
def train_with_checkpointing(model, train_loader, val_loader, epochs, save_dir='checkpoints'):
    '''带 checkpoint 保存的训练循环'''
    # ... 完整实现见上方 ...

使用示例：
    model = SimpleNet()
    train_with_checkpointing(model, train_loader, val_loader, epochs=100)

恢复训练：
    checkpoint = torch.load('checkpoints/checkpoint_epoch50.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_with_checkpointing(model, train_loader, val_loader,
                             epochs=100, start_epoch=start_epoch)
""")

print("\n" + "=" * 60)
print("总结：最佳实践")
print("=" * 60)

print("""
┌─────────────────┬────────────────────────────────────────┐
│ 场景            │ 推荐方法                               │
├─────────────────┼────────────────────────────────────────┤
│ 保存最终模型     │ state_dict() + model_config.json       │
│ 训练中断恢复     │ 完整 checkpoint（包含优化器状态）       │
│ 部署到生产       │ ONNX + TorchScript                     │
│ 模型版本管理     │ 文件名包含时间戳或版本号                │
│ 实验跟踪         │ 使用 MLflow 或 Weights & Biases         │
└─────────────────┴────────────────────────────────────────┘

文件命名规范：
  - model_weights.pth              # 模型参数
  - model_config.json               # 模型配置
  - checkpoint_epoch{N}.pth         # 定期保存
  - best_model.pth                 # 最佳模型
  - best_model_val{loss:.4f}.pth    # 带验证损失
  - model_YYYYMMDD_HHMMSS.pth       # 时间戳

重要提醒：
  1. 始终保存模型配置（超参数）
  2. 定期保存 checkpoint（防训练中断）
  3. 推理前设置 model.eval()
  4. 推理时使用 torch.no_grad()
  5. 测试加载函数（确保能正确恢复）
""")

print("\n所有示例文件已保存在 checkpoints/ 目录")
