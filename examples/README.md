# PyTorch 深度学习教程

从零开始学习 PyTorch 和深度学习的完整教程。

## 目录结构

```
pytorch-learning/
├── 01-basics/                   # PyTorch 基础
│   ├── tensor_basics.py         # 张量操作
│   └── autograd.py              # 自动求导
│
├── 02-linear-regression/        # 线性回归
│   ├── linear_regression.py     # 手动实现（理解原理）
│   └── linear_regression_nn.py  # 使用 nn.Module（标准方式）
│
├── 03-logistic-regression/      # 逻辑回归（二分类）
│   └── logistic_regression.py
│
├── 04-cnn/                      # 卷积神经网络
│   └── mnist_cnn.py             # MNIST 手写数字识别
│
├── 05-neural-network/           # 多层神经网络
│   ├── mlp_intro.py             # MLP 入门与非线性问题
│   └── multi_classification.py   # 多分类问题（鸢尾花）
│
├── 06-regularization/           # 正则化技巧
│   └── regularization.py        # Dropout, BatchNorm, Weight Decay
│
├── 07-sequence-models/          # 序列模型
│   └── rnn_lstm_intro.py        # RNN vs LSTM
│
├── 08-model-management/         # 模型管理
│   └── save_load_inference.py   # 保存、加载、推理
│
├── 09-transfer-learning/        # 迁移学习 (进阶)
│   └── transfer_learning.py     # 预训练模型、微调策略
│
├── 10-attention/                # 注意力机制 (进阶)
│   └── attention.py             # Self-Attention, Multi-Head
│
├── 11-transformer/              # Transformer (进阶)
│   └── transformer.py           # 完整Transformer实现
│
├── 12-gan/                      # 生成对抗网络 (进阶)
│   └── gan.py                   # DCGAN, Conditional GAN
│
├── 13-training-tricks/          # 训练技巧 (进阶)
│   └── training_tricks.py       # LR调度、早停、混合精度
│
├── 14-custom-dataset/           # 自定义数据集 (进阶)
│   └── custom_dataset.py        # Dataset、DataLoader、数据增强
│
└── requirements.txt              # 依赖包
```

## 学习路径

### 第一阶段：PyTorch 基础 (1-2周)

1. [张量基础](01-basics/tensor_basics.py) - 了解 PyTorch 的核心数据结构
2. [自动求导](01-basics/autograd.py) - 理解梯度计算原理

### 第二阶段：基础机器学习 (2-3周)

3. [线性回归](02-linear-regression/linear_regression.py) - 最简单的神经网络
4. [线性回归 nn.Module 版](02-linear-regression/linear_regression_nn.py) - 学习标准写法
5. [逻辑回归](03-logistic-regression/logistic_regression.py) - 二分类问题

### 第三阶段：深度学习入门 (3-4周)

6. [多层神经网络](05-neural-network/mlp_intro.py) - 理解非线性问题
7. [多分类问题](05-neural-network/multi_classification.py) - Softmax 和交叉熵
8. [CNN 图像分类](04-cnn/mnist_cnn.py) - 卷积神经网络入门

### 第四阶段：进阶技巧 (2-3周)

9. [正则化技巧](06-regularization/regularization.py) - 防止过拟合
10. [序列模型](07-sequence-models/rnn_lstm_intro.py) - RNN/LSTM
11. [模型管理](08-model-management/save_load_inference.py) - 保存、加载、部署

### 第五阶段：高级主题 (4-6周)

12. [迁移学习](09-transfer-learning/transfer_learning.py) - 预训练模型与微调策略
13. [注意力机制](10-attention/attention.py) - Self-Attention, Multi-Head Attention
14. [Transformer](11-transformer/transformer.py) - 完整Transformer实现
15. [生成对抗网络](12-gan/gan.py) - GAN, DCGAN, Conditional GAN
16. [训练技巧](13-training-tricks/training_tricks.py) - 学习率调度、早停、混合精度
17. [自定义数据集](14-custom-dataset/custom_dataset.py) - Dataset、DataLoader、数据增强

## 核心概念速查

### 训练循环五步法

```python
for epoch in range(epochs):
    # 1. 清空梯度
    optimizer.zero_grad()

    # 2. 前向传播
    y_pred = model(X)

    # 3. 计算损失
    loss = criterion(y_pred, y)

    # 4. 反向传播
    loss.backward()

    # 5. 更新参数
    optimizer.step()
```

### 常用激活函数

| 激活函数 | 使用场景 | 位置 |
|---------|---------|------|
| ReLU    | 隐藏层  | 最常用 |
| Sigmoid | 二分类输出 | 输出层 |
| Softmax | 多分类输出 | 输出层 |
| Tanh    | 隐藏层  | 输出 (-1,1) |
| None    | 回归输出 | 输出层 |

### 常用损失函数

| 任务 | 损失函数 | PyTorch |
|-----|---------|---------|
| 回归 | MSE | `nn.MSELoss()` |
| 二分类 | Binary Cross Entropy | `nn.BCELoss()` |
| 多分类 | Cross Entropy | `nn.CrossEntropyLoss()` |

### 常用优化器

| 优化器 | 特点 | 学习率 |
|-------|------|-------|
| SGD | 简单稳定 | 0.01-0.1 |
| Adam | 收敛快 | 0.001 |
| RMSprop | RNN 常用 | 0.001 |

### 正则化方法

```python
# 1. Dropout
nn.Dropout(p=0.3)

# 2. Batch Normalization
nn.BatchNorm1d(num_features)

# 3. Weight Decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# 4. 数据增强
transforms.Compose([...])
```

## 运行示例

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行单个文件

```bash
cd pytorch-learning
python 01-basics/tensor_basics.py
```

### 按顺序学习

推荐按以下顺序运行（每个文件 5-10 分钟）：

```bash
# 第一阶段
python 01-basics/tensor_basics.py
python 01-basics/autograd.py

# 第二阶段
python 02-linear-regression/linear_regression.py
python 02-linear-regression/linear_regression_nn.py
python 03-logistic-regression/logistic_regression.py

# 第三阶段
python 05-neural-network/mlp_intro.py
python 05-neural-network/multi_classification.py
python 04-cnn/mnist_cnn.py

# 第四阶段
python 06-regularization/regularization.py
python 07-sequence-models/rnn_lstm_intro.py
python 08-model-management/save_load_inference.py

# 第五阶段 (进阶)
python 09-transfer-learning/transfer_learning.py
python 10-attention/attention.py
python 11-transformer/transformer.py
python 12-gan/gan.py
python 13-training-tricks/training_tricks.py
python 14-custom-dataset/custom_dataset.py
```

## 学习建议

1. **理解原理，不要只跑代码**
   - 每个文件都有详细的注释和解释
   - 运行后查看生成的图片
   - 修改参数观察效果

2. **动手实验**
   - 修改超参数（学习率、层数等）
   - 尝试不同的数据集
   - 实现自己的想法

3. **循序渐进**
   - 不要跳过基础直接看复杂模型
   - 每个阶段充分理解再进入下一阶段
   - 遇到问题回顾前面的内容

4. **实践项目**
   - 学完基础后尝试自己的项目
   - 从简单问题开始（如房价预测）
   - 逐步挑战复杂任务

## 常见问题

### Q: GPU 不可用怎么办？
A: 所有示例都支持 CPU 训练，只是速度慢一些。Mac M1/M2 支持使用 MPS 加速。

### Q: 训练很慢怎么办？
A:
- 减少 batch_size
- 减少 model 层数和宽度
- 减少 epochs
- 使用更小的数据集

### Q: 过拟合怎么办？
A:
- 增加 Dropout
- 使用 Batch Normalization
- 使用 Weight Decay
- 增加训练数据
- 简化模型

### Q: 梯度消失怎么办？
A:
- 使用 ReLU 激活函数
- 使用 Batch Normalization
- 使用 LSTM/GRU（序列模型）
- 使用残差连接（ResNet）

## 扩展学习

### 推荐资源

**官方资源**
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 中文文档](https://pytorch.apachecn.org/)

**在线课程**
- Fast.ai 深度学习课程
- 吴恩达 Deep Learning Specialization
- 李宏毅机器学习课程

**实战项目**
- Kaggle 竞赛
- GitHub 上的开源项目
- 自己的 Kaggle 项目

### 进阶主题

本教程已包含以下进阶主题：

| 主题 | 文件 | 核心内容 |
|-----|------|---------|
| 迁移学习 | `09-transfer-learning/` | 预训练模型、微调、差分学习率 |
| 注意力机制 | `10-attention/` | Self-Attention, Multi-Head, 可视化 |
| Transformer | `11-transformer/` | 完整Encoder-Decoder, 位置编码 |
| 生成对抗网络 | `12-gan/` | GAN, DCGAN, Conditional GAN |
| 训练技巧 | `13-training-tricks/` | LR调度、早停、混合精度、梯度累积 |
| 自定义数据集 | `14-custom-dataset/` | Dataset、DataLoader、数据增强 |

进一步学习方向：
- 预训练语言模型 (BERT, GPT, T5)
- 扩散模型 (Diffusion Models)
- 强化学习 (Reinforcement Learning)
- 图神经网络 (Graph Neural Networks)
- 多模态学习 (Vision-Language Models)

## 版本信息

- PyTorch: 2.10.0
- Python: 3.8+

## 贡献

欢迎提出建议和改进！

---

祝你学习愉快！🚀
