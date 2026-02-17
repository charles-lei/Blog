"""
Transformer 模型
================

核心思想：
    完全基于注意力机制的序列到序列模型，不使用循环或卷积。
    通过自注意力机制并行处理序列，大大提高了训练效率。

主要组件：
    1. Encoder (编码器): 处理输入序列
    2. Decoder (解码器): 生成输出序列
    3. Positional Encoding: 添加位置信息

关键创新：
    - Multi-Head Self-Attention
    - Position-wise Feed-Forward Networks
    - Residual Connections + Layer Normalization

应用：
    - BERT, GPT, T5, etc.
    - 机器翻译
    - 文本摘要
    - 问答系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================
# 1. Positional Encoding
# ============================================

class PositionalEncoding(nn.Module):
    """
    位置编码

    公式：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    为什么用正弦/余弦？
        - 可以处理任意长度
        - 相对位置可以通过线性变换得到
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数位置用sin，奇数位置用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# 2. Scaled Dot-Product Attention
# ============================================

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    缩放点积注意力

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = query.size(-1)

    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用mask（用于decoder的自回归）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 应用到value
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# ============================================
# 3. Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: (batch, 1, seq_len_k) 或 (batch, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 分割成多头
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 缩放点积注意力
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )

        # 4. 合并多头
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 5. 输出投影
        output = self.W_o(attn_output)

        return output, attention_weights


# ============================================
# 4. Position-wise Feed-Forward Network
# ============================================

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    两个线性变换，中间有ReLU激活
    通常 d_ff = 4 * d_model
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ============================================
# 5. Encoder Layer
# ============================================

class EncoderLayer(nn.Module):
    """
    Transformer编码器层

    结构：
        x -> Multi-Head Attention -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention with Residual
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-Forward with Residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


# ============================================
# 6. Decoder Layer
# ============================================

class DecoderLayer(nn.Module):
    """
    Transformer解码器层

    结构：
        x -> Masked Self-Attention -> Add & Norm
          -> Cross-Attention -> Add & Norm
          -> FFN -> Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Self-Attention (带mask)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-Attention (query来自decoder, key/value来自encoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-Attention
        attn_out, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_out))

        # Feed-Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


# ============================================
# 7. Complete Transformer
# ============================================

class Transformer(nn.Module):
    """
    完整的Transformer模型

    用于序列到序列任务（如机器翻译）
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1,
                 max_len=5000):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        # Embedding + Positional Encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """解码器前向传播"""
        # Embedding + Positional Encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch, src_len) - 源序列
            tgt: (batch, tgt_len) - 目标序列
            src_mask: 源序列mask
            tgt_mask: 目标序列mask (用于自回归)

        Returns:
            output: (batch, tgt_len, tgt_vocab_size)
        """
        # Encode
        encoder_output = self.encode(src, src_mask)

        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.fc_out(decoder_output)

        return output

    def generate_square_subsequent_mask(self, sz):
        """
        生成用于自回归的三角形mask

        用于确保位置i只能看到位置0到i的信息
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask == 0  # 上三角为False，下三角为True
        return mask


# ============================================
# 8. Transformer for Classification (简化版)
# ============================================

class TransformerClassifier(nn.Module):
    """
    用于分类任务的Transformer

    只使用Encoder部分，使用[CLS] token进行分类
    """

    def __init__(self, vocab_size, num_classes, d_model=256, num_heads=8,
                 num_layers=4, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()

        self.d_model = d_model

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Encoder layers
        for layer in self.layers:
            x = layer(x)

        # Use [CLS] token for classification
        cls_output = x[:, 0, :]

        # Classify
        logits = self.classifier(cls_output)

        return logits


# ============================================
# 9. 实验和演示
# ============================================

def experiment_positional_encoding():
    """可视化位置编码"""
    print("\n" + "=" * 60)
    print("实验: 位置编码可视化")
    print("=" * 60)

    d_model = 128
    max_len = 100

    pe = PositionalEncoding(d_model, max_len, dropout=0)

    # 获取位置编码矩阵
    pe_matrix = pe.pe.squeeze(0).numpy()  # (max_len, d_model)

    # 可视化
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pe_matrix[:50, :].T, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Matrix')

    plt.subplot(1, 2, 2)
    # 显示前4个维度
    positions = range(50)
    for i in range(4):
        plt.plot(positions, pe_matrix[:50, i], label=f'Dim {i}')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding (First 4 Dimensions)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('11-transformer/positional_encoding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("位置编码可视化已保存到 11-transformer/positional_encoding.png")


def experiment_transformer_classifier():
    """测试Transformer分类器"""
    print("\n" + "=" * 60)
    print("实验: Transformer文本分类器")
    print("=" * 60)

    # 参数
    vocab_size = 1000
    num_classes = 3
    seq_len = 20
    batch_size = 32

    # 创建模型
    model = TransformerClassifier(
        vocab_size, num_classes,
        d_model=128, num_heads=4, num_layers=2
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 创建模拟数据
    num_samples = 500
    X = torch.randint(0, vocab_size, (num_samples, seq_len)).to(device)
    y = torch.randint(0, num_classes, (num_samples,)).to(device)

    # 划分数据集
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n训练中...")
    train_losses = []
    test_accuracies = []

    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 评估
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            _, predicted = test_logits.max(1)
            acc = predicted.eq(y_test).sum().item() / y_test.size(0)
            test_accuracies.append(acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Test Acc={acc:.4f}")

    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(test_accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11-transformer/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n训练曲线已保存到 11-transformer/training_curves.png")


def experiment_transformer_seq2seq():
    """测试完整的Transformer Seq2Seq"""
    print("\n" + "=" * 60)
    print("实验: Transformer Seq2Seq (序列反转任务)")
    print("=" * 60)

    # 参数
    vocab_size = 20  # 小词表便于演示
    d_model = 64
    num_heads = 4
    num_layers = 2

    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=128,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 创建序列反转数据集
    def create_reverse_data(num_samples, seq_len, vocab_size):
        # 输入序列（不包含特殊token）
        src = torch.randint(1, vocab_size, (num_samples, seq_len))
        # 目标序列 = 输入的反转
        tgt = src.flip(1)
        return src, tgt

    # 生成数据
    num_samples = 1000
    seq_len = 8
    X, y = create_reverse_data(num_samples, seq_len, vocab_size)
    X, y = X.to(device), y.to(device)

    # 划分
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n训练中 (学习序列反转)...")

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()

        # Teacher forcing: 使用真实目标作为decoder输入
        # 目标输入: 去掉最后一个token
        # 目标输出: 去掉第一个token
        tgt_input = y_train[:, :-1]
        tgt_output = y_train[:, 1:]

        # 生成自回归mask
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        output = model(X_train, tgt_input, tgt_mask=tgt_mask)

        # 计算loss
        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            # 评估
            model.eval()
            with torch.no_grad():
                test_output = model(X_test, y_test[:, :-1], tgt_mask=tgt_mask)
                _, predicted = test_output.max(-1)

                # 计算token级别准确率
                correct = predicted.eq(y_test[:, 1:]).sum().item()
                total = y_test[:, 1:].numel()
                acc = correct / total

            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    # 展示一些预测
    print("\n预测示例:")
    model.eval()
    with torch.no_grad():
        for i in range(3):
            src = X_test[i:i+1]
            tgt_input = y_test[i:i+1, :-1]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            output = model(src, tgt_input, tgt_mask=tgt_mask)
            _, predicted = output.max(-1)

            print(f"  输入: {src[0].tolist()}")
            print(f"  预测: {predicted[0].tolist()}")
            print(f"  真实: {y_test[i, 1:].tolist()}")
            print()


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Transformer 教程")
    print("=" * 60)

    # 实验1: 位置编码
    experiment_positional_encoding()

    # 实验2: Transformer分类器
    experiment_transformer_classifier()

    # 实验3: Seq2Seq
    experiment_transformer_seq2seq()

    print("\n" + "=" * 60)
    print("教程完成!")
    print("=" * 60)
    print("""
Transformer 核心要点:

1. 架构优势:
   - 并行化: 不像RNN需要顺序处理
   - 长距离依赖: 自注意力直接连接任意位置
   - 可扩展: 容易增加层数和参数

2. 关键组件:
   - Multi-Head Attention: 多角度关注输入
   - Positional Encoding: 注入位置信息
   - Layer Normalization: 稳定训练
   - Residual Connection: 缓解梯度消失

3. 训练技巧:
   - Warmup学习率调度
   - Label Smoothing
   - Dropout
   - Gradient Clipping

4. 常见变体:
   - BERT: Encoder-only, MLM预训练
   - GPT: Decoder-only, 自回归生成
   - T5: Encoder-Decoder, 文本到文本

5. 推荐进一步学习:
   - BERT (双向编码)
   - GPT (自回归生成)
   - Vision Transformer (ViT)
    """)
