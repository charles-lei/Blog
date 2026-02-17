"""
序列模型 - RNN 和 LSTM
========================

序列数据的特点：
- 数据有先后顺序
- 当前时刻的输入可能与之前的信息有关

常见序列数据：
- 文本（单词序列）
- 语音（声波序列）
- 股票价格（时间序列）
- 视频（帧序列）

RNN vs LSTM：
- RNN：简单的循环神经网络，容易梯度消失
- LSTM：长短期记忆网络，解决梯度消失问题

学习目标：
1. 理解 RNN 的工作原理
2. 理解 LSTM 的优势
3. 学会处理序列数据
4. 实现时间序列预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第一部分：理解序列数据和 RNN")
print("=" * 60)

print("""
RNN 核心思想：
- 引入"隐藏状态"（hidden state）
- 将上一步的信息传递到当前步骤
- 公式：
    h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
    y_t = W_y * h_t + c

问题：梯度消失
- 当序列很长时，早期信息难以传递到后期
- 无法学习长期依赖

LSTM 改进：
- 引入"门控机制"（遗忘门、输入门、输出门）
- 可以选择性保留或遗忘信息
- 能够学习长期依赖
""")

print("\n" + "=" * 60)
print("第二部分：创建序列数据（正弦波预测）")
print("=" * 60)

# 生成正弦波数据
def create_sine_wave(seq_length=1000):
    """创建正弦波数据"""
    t = np.linspace(0, 4*np.pi, seq_length)
    data = np.sin(t) + 0.1 * np.random.randn(seq_length)
    return torch.tensor(data, dtype=torch.float32)

seq_length = 1000
data = create_sine_wave(seq_length)

print(f"序列长度: {seq_length}")
print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")

# 可视化数据
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(data.numpy())
axes[0].set_xlabel('时间步')
axes[0].set_ylabel('值')
axes[0].set_title('完整正弦波序列')
axes[0].grid(True, alpha=0.3)

# 显示前100个点
axes[1].plot(data[:100].numpy(), marker='o', markersize=3)
axes[1].set_xlabel('时间步')
axes[1].set_ylabel('值')
axes[1].set_title('前100个时间步')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_data.png', dpi=150)
plt.show()

# 准备训练数据
def create_sequences(data, seq_len, pred_len):
    """
    将序列数据转换为训练样本

    参数：
        data: 完整序列
        seq_len: 输入序列长度
        pred_len: 预测序列长度
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return torch.stack(X), torch.stack(y)

# 使用前50个时间步预测后10个时间步
seq_input_len = 50
seq_output_len = 10

X, y = create_sequences(data, seq_input_len, seq_output_len)

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

print(f"\n训练样本数: {len(X_train)}")
print(f"测试样本数: {len(X_test)}")
print(f"输入序列长度: {seq_input_len}")
print(f"输出序列长度: {seq_output_len}")

print("\n" + "=" * 60)
print("第三部分：定义 RNN 和 LSTM 模型")
print("=" * 60)

class SimpleRNN(nn.Module):
    """简单的 RNN 模型"""
    def __init__(self, input_size=1, hidden_size=32, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # out shape: (batch, seq_len, hidden_size)
        # h_n shape: (1, batch, hidden_size)
        out, h_n = self.rnn(x)

        # 使用最后一个时间步的隐藏状态
        # 或者使用全序列的平均
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out

class LSTMModel(nn.Module):
    """LSTM 模型"""
    def __init__(self, input_size=1, hidden_size=32, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # out shape: (batch, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步
        out = self.fc(out[:, -1, :])
        return out

class BidirectionalLSTM(nn.Module):
    """双向 LSTM"""
    def __init__(self, input_size=1, hidden_size=32, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,
                           batch_first=True,
                           bidirectional=True)  # 双向
        # 双向输出维度是 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 打印模型结构
print("SimpleRNN 结构:")
rnn_model = SimpleRNN()
print(rnn_model)

print("\nLSTMModel 结构:")
lstm_model = LSTMModel()
print(lstm_model)

# 测试前向传播
test_input = X_train[:2]  # 取2个样本
print(f"\n测试输入形状: {test_input.shape}")  # (2, 50, 1)

rnn_output = rnn_model(test_input)
print(f"RNN 输出形状: {rnn_output.shape}")  # (2, 10)

lstm_output = lstm_model(test_input)
print(f"LSTM 输出形状: {lstm_output.shape}")  # (2, 10)

print("\n" + "=" * 60)
print("第四部分：训练和对比模型")
print("=" * 60)

def train_model(model, model_name, X_train, y_train, X_test, y_test, epochs=100):
    """训练序列模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train.unsqueeze(-1))  # 添加 feature 维度
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # 评估
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test.unsqueeze(-1)), y_test).item()
            test_losses.append(test_loss)

        if (epoch + 1) % 20 == 0:
            print(f"{model_name} - Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses

# 训练所有模型
models = {
    'RNN': SimpleRNN(),
    'LSTM': LSTMModel(),
    'BiLSTM': BidirectionalLSTM()
}

results = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    train_losses, test_losses = train_model(
        model, name, X_train, y_train, X_test, y_test, epochs=100
    )
    results[name] = {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses
    }

print("\n" + "=" * 60)
print("第五部分：可视化预测结果")
print("=" * 60)

# 预测并可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 选择一个测试样本进行可视化
sample_idx = 0  # 第一个测试样本
input_seq = X_test[sample_idx]
true_output = y_test[sample_idx]

# 获取各模型的预测
predictions = {}
for name, res in results.items():
    model = res['model']
    model.eval()
    with torch.no_grad():
        pred = model(input_seq.unsqueeze(0).unsqueeze(-1))
        predictions[name] = pred[0].numpy()

# 绘制完整的预测序列
ax = axes[0, 0]
ax.plot(input_seq.numpy(), 'b-o', markersize=3, label='输入序列', linewidth=2)
ax.plot(range(len(input_seq), len(input_seq) + len(true_output)),
        true_output.numpy(), 'g-o', markersize=4, label='真实输出', linewidth=2)
ax.plot(range(len(input_seq), len(input_seq) + len(true_output)),
        predictions['LSTM'], 'r--o', markersize=4, label='LSTM预测', linewidth=2)
ax.axvline(x=len(input_seq)-1, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('时间步')
ax.set_ylabel('值')
ax.set_title('单步预测可视化（LSTM）')
ax.legend()
ax.grid(True, alpha=0.3)

# 训练曲线对比
ax = axes[0, 1]
for name, res in results.items():
    ax.plot(res['train_losses'], label=name, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('训练损失对比')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 测试曲线对比
ax = axes[1, 0]
for name, res in results.items():
    ax.plot(res['test_losses'], label=name, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss')
ax.set_title('测试损失对比')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 对比所有模型的预测
ax = axes[1, 1]
ax.plot(range(len(input_seq), len(input_seq) + len(true_output)),
        true_output.numpy(), 'g-o', markersize=5, label='真实值', linewidth=2)
for name, pred in predictions.items():
    ax.plot(range(len(input_seq), len(input_seq) + len(true_output)),
            pred, '-o', markersize=4, label=name, alpha=0.7)
ax.set_xlabel('预测时间步')
ax.set_ylabel('值')
ax.set_title('各模型预测对比')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_lstm_comparison.png', dpi=150)
plt.show()

# 计算最终指标
print("\n最终测试损失:")
for name, res in results.items():
    final_loss = res['test_losses'][-1]
    print(f"  {name}: {final_loss:.6f}")

print("\n" + "=" * 60)
print("总结：RNN vs LSTM")
print("=" * 60)

print("""
┌──────────────┬─────────────────────┬──────────────────┐
│ 特性         │ RNN                 │ LSTM             │
├──────────────┼─────────────────────┼──────────────────┤
│ 结构复杂度   │ 简单                │ 复杂（有门控）    │
│ 参数量       │ 少                  │ 多               │
│ 训练速度     │ 快                  │ 慢               │
│ 长期依赖     │ 差（梯度消失）      │ 好               │
│ 适用场景     │ 短序列              │ 长序列           │
└──────────────┴─────────────────────┴──────────────────┘

LSTM 的门控机制：
1. 遗忘门 (Forget Gate)：决定丢弃什么信息
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. 输入门 (Input Gate)：决定存储什么信息
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

3. 输出门 (Output Gate)：决定输出什么信息
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

4. 候选值 (Candidate)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

5. 更新细胞状态
   C_t = f_t * C_{t-1} + i_t * C̃_t

6. 更新隐藏状态
   h_t = o_t * tanh(C_t)

使用建议：
- 短序列（< 20步）：RNN 足够
- 中长序列（20-100步）：LSTM
- 超长序列（> 100步）：LSTM + 注意力机制
- 需要上下文：使用双向 LSTM

常用参数设置：
- hidden_size: 32, 64, 128
- num_layers: 1-3（层数）
- dropout: 0.1-0.3（多层时使用）
- bidirectional: True/False（是否双向）

代码模板：
```python
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 最后时间步
        return out
```
""")

print("\n图片已保存到 rnn_lstm_comparison.png")
