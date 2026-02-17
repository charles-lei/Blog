"""
PyTorch 自动求导 (Autograd)
===========================

自动求导是 PyTorch 最重要的特性之一！

核心概念：
- 神经网络训练需要计算梯度（导数）
- 手动计算梯度非常繁琐且容易出错
- PyTorch 可以自动计算梯度

原理：
- 每个 Tensor 都有 .grad 属性存储梯度
- 设置 requires_grad=True 来追踪计算历史
- 调用 .backward() 自动计算梯度

学习目标：
1. 理解 requires_grad 的作用
2. 学会使用 backward() 计算梯度
3. 理解计算图的概念
"""

import torch

print("=" * 60)
print("第一部分：基本自动求导")
print("=" * 60)

# 创建一个需要追踪梯度的张量
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

# 定义一个函数 y = x^2
y = x ** 2
print(f"\ny = x^2 = {y}")

# 计算梯度（dy/dx = 2x）
y.backward()  # 反向传播
print(f"\n在 x=2 处，dy/dx = 2x = {x.grad}")  # 应该是 4

print("\n" + "=" * 60)
print("第二部分：更复杂的计算图")
print("=" * 60)

# 重置梯度
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 构建 y = w * x + b （这就是神经网络的基本操作！）
y = w * x + b
print(f"x = {x}, w = {w}, b = {b}")
print(f"y = w * x + b = {y}")

# 定义一个损失函数 loss = y^2
loss = y ** 2
print(f"loss = y^2 = {loss}")

# 反向传播
loss.backward()

print(f"\n梯度:")
print(f"d(loss)/d(x) = {x.grad}")  # 2y * w = 2*7*3 = 42
print(f"d(loss)/d(w) = {w.grad}")  # 2y * x = 2*7*2 = 28
print(f"d(loss)/d(b) = {b.grad}")  # 2y * 1 = 2*7 = 14

print("\n" + "=" * 60)
print("第三部分：多变量函数")
print("=" * 60)

# 损失函数通常涉及多个样本
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x = {x}")

# y = x^2 的和
y = (x ** 2).sum()
print(f"y = sum(x^2) = {y}")

y.backward()
print(f"\ndy/dx = {x.grad}")  # 应该是 [2, 4, 6]，即 2x

print("\n" + "=" * 60)
print("第四部分：梯度累加问题")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)

# 第一次计算
y1 = x ** 2
y1.backward()
print(f"第一次 backward 后 x.grad = {x.grad}")  # 4

# 第二次计算（梯度会累加！）
y2 = x ** 3
y2.backward()
print(f"第二次 backward 后 x.grad = {x.grad}")  # 4 + 12 = 16（累加了！）

# 正确做法：每次迭代前清空梯度
x.grad.zero_()  # 清空梯度
print(f"清空后 x.grad = {x.grad}")

print("\n" + "=" * 60)
print("第五部分：控制梯度追踪")
print("=" * 60)

x = torch.tensor([2.0], requires_grad=True)

# 方法1：with torch.no_grad() 临时禁用梯度追踪
with torch.no_grad():
    y = x ** 2
    print(f"在 no_grad() 内，y.requires_grad = {y.requires_grad}")

# 方法2：.detach() 创建一个不需要梯度的副本
y_detach = x.detach()
print(f"detach 后，y_detach.requires_grad = {y_detach.requires_grad}")

# 方法3：requires_grad_(False) 永久禁用
x.requires_grad_(False)
print(f"设置 False 后，x.requires_grad = {x.requires_grad}")

print("\n" + "=" * 60)
print("第六部分：理解计算图")
print("=" * 60)

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

# 构建计算图:
#       c = a * b
#       d = a + b
#       e = c * d
c = a * b
d = a + b
e = c * d

print("计算图结构:")
print(f"  a = {a.item()}, b = {b.item()}")
print(f"  c = a * b = {c.item()}")
print(f"  d = a + b = {d.item()}")
print(f"  e = c * d = {e.item()}")

e.backward()
print(f"\n梯度:")
print(f"  de/da = {a.grad}")  # 应该是 13
print(f"  de/db = {b.grad}")  # 应该是 8

print("""
验证：
  e = c * d = (a*b) * (a+b)
  de/da = b*(a+b) + (a*b)*1 = 3*5 + 6*1 = 15 + 6 = 21
  de/db = a*(a+b) + (a*b)*1 = 2*5 + 6*1 = 10 + 6 = 16
""")

print("\n" + "=" * 60)
print("总结：神经网络的梯度计算流程")
print("=" * 60)

print("""
训练神经网络的标准流程：

1. 初始化参数（requires_grad=True）
2. 前向传播：计算预测值
3. 计算损失
4. 反向传播：loss.backward() 自动计算所有梯度
5. 更新参数：optimizer.step()
6. 清空梯度：optimizer.zero_grad()（准备下一次迭代）

这个流程会在后续的线性回归和神经网络示例中反复使用！
""")
