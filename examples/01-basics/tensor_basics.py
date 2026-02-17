"""
PyTorch 基础 - Tensor 操作
============================

Tensor（张量）是 PyTorch 的核心数据结构，可以理解为：
- 一个多维数组（类似 NumPy 的 ndarray）
- 但可以在 GPU 上运行
- 支持自动求导（autograd）

学习目标：
1. 创建 Tensor 的多种方式
2. Tensor 的基本运算
3. Tensor 与 NumPy 的转换
4. 索引和切片
"""

import torch
import numpy as np

print("=" * 60)
print("第一部分：创建 Tensor")
print("=" * 60)

# 1. 从列表创建
x1 = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建: {x1}")

# 2. 创建全0张量
x2 = torch.zeros(3, 4)  # 3行4列的全0矩阵
print(f"\n全0张量 (3x4):\n{x2}")

# 3. 创建全1张量
x3 = torch.ones(2, 3)  # 2行3列的全1矩阵
print(f"\n全1张量 (2x3):\n{x3}")

# 4. 创建随机张量（0-1均匀分布）
x4 = torch.rand(2, 3)
print(f"\n随机张量 (0-1均匀分布):\n{x4}")

# 5. 创建随机张量（标准正态分布）
x5 = torch.randn(2, 3)  # 均值0，方差1
print(f"\n随机张量 (标准正态分布):\n{x5}")

# 6. 创建指定范围的张量
x6 = torch.arange(0, 10, 2)  # 0到10，步长为2
print(f"\narange(0, 10, 2): {x6}")

# 7. 创建等间隔张量
x7 = torch.linspace(0, 1, 5)  # 0到1，分成5个点
print(f"linspace(0, 1, 5): {x7}")

print("\n" + "=" * 60)
print("第二部分：Tensor 属性")
print("=" * 60)

x = torch.randn(3, 4, 5)  # 三维张量
print(f"张量形状: {x.shape}")        # torch.Size([3, 4, 5])
print(f"张量维度: {x.dim()}")         # 3
print(f"元素总数: {x.numel()}")       # 3*4*5 = 60
print(f"数据类型: {x.dtype}")         # torch.float32
print(f"存储设备: {x.device}")        # cpu

print("\n" + "=" * 60)
print("第三部分：Tensor 运算")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"\n加法: a + b = {a + b}")
print(f"减法: a - b = {a - b}")
print(f"乘法 (逐元素): a * b = {a * b}")  # 注意：这是逐元素相乘，不是点积
print(f"除法: a / b = {a / b}")
print(f"幂运算: a ** 2 = {a ** 2}")

# 矩阵运算
A = torch.randn(2, 3)
B = torch.randn(3, 2)
print(f"\n矩阵 A (2x3):\n{A}")
print(f"\n矩阵 B (3x2):\n{B}")
print(f"\n矩阵乘法 A @ B (2x2):\n{A @ B}")  # 或 torch.mm(A, B) 或 torch.matmul(A, B)

# 常用统计操作
c = torch.randn(3, 4)
print(f"\n随机矩阵 c:\n{c}")
print(f"\n求和: {c.sum()}")
print(f"均值: {c.mean()}")
print(f"最大值: {c.max()}")
print(f"最小值: {c.min()}")
print(f"按列求均值 (dim=0): {c.mean(dim=0)}")  # 每列的均值
print(f"按行求均值 (dim=1): {c.mean(dim=1)}")  # 每行的均值

print("\n" + "=" * 60)
print("第四部分：索引和切片")
print("=" * 60)

x = torch.arange(12).reshape(3, 4)  # 0-11 排成 3x4
print(f"原始张量:\n{x}")

print(f"\n取第一行: x[0] = {x[0]}")
print(f"取第一列: x[:, 0] = {x[:, 0]}")
print(f"取第2-3行: x[1:3] =\n{x[1:3]}")
print(f"取特定元素 x[1, 2] = {x[1, 2]}")  # 第2行第3列

# 条件索引
print(f"\n大于5的元素: {x[x > 5]}")

print("\n" + "=" * 60)
print("第五部分：形状操作")
print("=" * 60)

x = torch.arange(12)
print(f"原始: {x}, 形状: {x.shape}")

# reshape: 改变形状
y = x.reshape(3, 4)
print(f"reshape(3, 4): 形状 {y.shape}")

# view: 类似 reshape，但共享内存
z = x.view(2, 6)
print(f"view(2, 6): 形状 {z.shape}")

# 转置
print(f"\n转置 (3x4 -> 4x3):\n{y.T}")

# squeeze: 去掉大小为1的维度
a = torch.randn(1, 3, 1, 4)
print(f"\n原始形状: {a.shape}")
print(f"squeeze后: {a.squeeze().shape}")  # 去掉所有大小为1的维度

# unsqueeze: 增加一个维度
b = torch.randn(3, 4)
print(f"\n原始形状: {b.shape}")
print(f"unsqueeze(0)后: {b.unsqueeze(0).shape}")  # 在第0维增加
print(f"unsqueeze(-1)后: {b.unsqueeze(-1).shape}")  # 在最后一维增加

print("\n" + "=" * 60)
print("第六部分：Tensor 与 NumPy 转换")
print("=" * 60)

# Tensor -> NumPy
tensor = torch.ones(5)
numpy_array = tensor.numpy()
print(f"Tensor: {tensor}")
print(f"NumPy: {numpy_array}")

# NumPy -> Tensor
np_array = np.ones(5)
tensor_from_np = torch.from_numpy(np_array)
print(f"\nNumPy: {np_array}")
print(f"Tensor: {tensor_from_np}")

# 注意：它们共享内存！修改一个会影响另一个
print("\n注意：它们共享内存！")
tensor[0] = 999
print(f"修改tensor后，numpy_array也变了: {numpy_array}")

print("\n" + "=" * 60)
print("第七部分：GPU 加速")
print("=" * 60)

if torch.cuda.is_available():
    print("GPU 可用！")
    x = torch.randn(3, 3)
    x_gpu = x.to("cuda")  # 或 x.cuda()
    print(f"CPU张量: {x.device}")
    print(f"GPU张量: {x_gpu.device}")

    # GPU运算后转回CPU
    x_back = x_gpu.cpu()  # 或 x_gpu.to("cpu")
else:
    print("GPU 不可用，使用 CPU 学习即可")

# Mac M1/M2 芯片
if torch.backends.mps.is_available():
    print("Apple Silicon GPU 可用！")
    x_mps = x.to("mps")
