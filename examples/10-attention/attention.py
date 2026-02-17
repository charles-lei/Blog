"""
注意力机制 (Attention Mechanism)
================================

核心思想：
    让模型学会"关注重点"，动态地为输入的不同部分分配不同的权重。
    这是一种通用的机制，可以应用于各种任务。

主要类型：
    1. Additive Attention (Bahdanau Attention)
    2. Dot-Product Attention (Luong Attention)
    3. Scaled Dot-Product Attention (Transformer)
    4. Self-Attention (自注意力)
    5. Multi-Head Attention (多头注意力)

应用场景：
    - 机器翻译 (Seq2Seq)
    - 图像描述 (Image Captioning)
    - 文本分类
    - 目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================
# 1. Additive Attention (Bahdanau Attention)
# ============================================

class AdditiveAttention(nn.Module):
    """
    加性注意力 (Bahdanau Attention)

    计算方式：
        score(h, s) = v^T * tanh(W_h * h + W_s * s)

    特点：
        - 使用可学习的权重矩阵
        - 计算量较大，但灵活
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, values):
        """
        Args:
            query: (batch_size, hidden_dim) - 解码器当前状态
            keys: (batch_size, seq_len, hidden_dim) - 编码器所有输出
            values: (batch_size, seq_len, hidden_dim) - 通常与keys相同

        Returns:
            context: (batch_size, hidden_dim) - 上下文向量
            weights: (batch_size, seq_len) - 注意力权重
        """
        # 扩展query维度以匹配keys
        # query: (batch, 1, hidden)
        query = query.unsqueeze(1)

        # 计算注意力分数
        # (batch, seq_len, hidden)
        score = torch.tanh(self.W_h(keys) + self.W_s(query))

        # (batch, seq_len, 1) -> (batch, seq_len)
        attention_weights = F.softmax(self.v(score), dim=1).squeeze(-1)

        # 计算上下文向量
        # (batch, 1, seq_len) @ (batch, seq_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


# ============================================
# 2. Dot-Product Attention (Luong Attention)
# ============================================

class DotProductAttention(nn.Module):
    """
    点积注意力 (Luong Attention)

    计算方式：
        score(h, s) = h^T * s

    特点：
        - 计算简单高效
        - 要求query和key维度相同
    """

    def __init__(self):
        super().__init__()

    def forward(self, query, keys, values):
        """
        Args:
            query: (batch_size, hidden_dim)
            keys: (batch_size, seq_len, hidden_dim)
            values: (batch_size, seq_len, hidden_dim)

        Returns:
            context: (batch_size, hidden_dim)
            weights: (batch_size, seq_len)
        """
        # (batch, hidden) @ (batch, hidden, seq_len) -> (batch, seq_len)
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


# ============================================
# 3. Scaled Dot-Product Attention (Transformer)
# ============================================

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 (Transformer Attention)

    计算方式：
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    特点：
        - 缩放因子防止点积值过大导致softmax梯度消失
        - Transformer的核心组件
    """

    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, num_heads, seq_len_q, d_k)
            key: (batch, num_heads, seq_len_k, d_k)
            value: (batch, num_heads, seq_len_v, d_v)
            mask: (batch, 1, seq_len_k) 用于mask某些位置

        Returns:
            output: (batch, num_heads, seq_len_q, d_v)
            attention: (batch, num_heads, seq_len_q, seq_len_k)
        """
        # 计算注意力分数
        # (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
        # -> (batch, heads, seq_q, seq_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention = F.softmax(scores, dim=-1)

        # 计算输出
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_v, d_v)
        # -> (batch, heads, seq_q, d_v)
        output = torch.matmul(attention, value)

        return output, attention


# ============================================
# 4. Self-Attention (自注意力)
# ============================================

class SelfAttention(nn.Module):
    """
    自注意力机制

    核心思想：
        序列中的每个元素都与序列中的所有其他元素计算注意力
        Query, Key, Value 都来自同一个输入

    应用：
        - 文本理解
        - 图像处理 (ViT)
    """

    def __init__(self, embed_dim, heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert embed_dim % heads == 0, "embed_dim必须能被heads整除"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(self.head_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, 1, seq_len) 可选

        Returns:
            out: (batch, seq_len, embed_dim)
            attention: (batch, heads, seq_len, seq_len)
        """
        batch_size = x.size(0)

        # 线性变换
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 分割成多头
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        out, attention = self.attention(q, k, v, mask)

        # 合并多头
        # (batch, heads, seq, head_dim) -> (batch, seq, embed)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 最终线性变换
        out = self.out_linear(out)

        return out, attention


# ============================================
# 5. Multi-Head Attention (多头注意力)
# ============================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    核心思想：
        将输入分割到多个"头"中，每个头独立计算注意力
        允许模型同时关注不同位置的不同表示子空间

    公式：
        MultiHead(Q, K, V) = Concat(head1, ..., headh) * W_o
        where head_i = Attention(Q * W_q, K * W_k, V * W_v)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V 的线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, embed_dim)
            key: (batch, seq_len_k, embed_dim)
            value: (batch, seq_len_v, embed_dim)
            mask: (batch, seq_len_q, seq_len_k) 可选

        Returns:
            output: (batch, seq_len_q, embed_dim)
            attention: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 分割多头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力到value
        output = torch.matmul(attention, v)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 输出投影
        output = self.out_proj(output)

        return output, attention


# ============================================
# 6. Attention with Position Encoding
# ============================================

class PositionalEncoding(nn.Module):
    """
    位置编码

    由于自注意力没有序列位置信息，需要添加位置编码
    使用正弦和余弦函数生成位置编码
    """

    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 生成位置编码
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# 7. 实际应用: 带注意力的序列分类器
# ============================================

class AttentionClassifier(nn.Module):
    """
    使用注意力机制的文本分类器

    结构：
        Embedding -> Position Encoding -> Self Attention -> Pooling -> Classifier
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, max_len=512, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len) - 词索引序列
            mask: (batch, seq_len) - 可选的mask

        Returns:
            logits: (batch, num_classes)
            attention: (batch, heads, seq_len, seq_len)
        """
        # Embedding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Self Attention with Residual
        attn_out, attention = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)

        # Feed Forward with Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Global Average Pooling
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits, attention


# ============================================
# 可视化注意力权重
# ============================================

def visualize_attention(attention, tokens=None, title="Attention Weights"):
    """
    可视化注意力权重矩阵

    Args:
        attention: (seq_len, seq_len) 或 (heads, seq_len, seq_len)
        tokens: token列表（用于标签）
        title: 图表标题
    """
    if attention.dim() == 3:
        # 多头注意力 - 只显示第一个头
        attention = attention[0]
        title += " (Head 1)"

    attention = attention.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention, cmap='Blues')

    ax.set_title(title)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


# ============================================
# 实验: 不同注意力机制的比较
# ============================================

def experiment_attention_comparison():
    """比较不同注意力机制"""
    print("\n" + "=" * 60)
    print("实验: 不同注意力机制的比较")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    hidden_dim = 64

    # 创建随机输入
    query = torch.randn(batch_size, hidden_dim).to(device)
    keys = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    values = torch.randn(batch_size, seq_len, hidden_dim).to(device)

    # 1. Additive Attention
    print("\n1. Additive Attention (Bahdanau):")
    additive_attn = AdditiveAttention(hidden_dim).to(device)
    context, weights = additive_attn(query, keys, values)
    print(f"   Context shape: {context.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   权重和: {weights[0].sum().item():.4f} (应该接近1)")

    # 2. Dot-Product Attention
    print("\n2. Dot-Product Attention (Luong):")
    dot_attn = DotProductAttention().to(device)
    context, weights = dot_attn(query, keys, values)
    print(f"   Context shape: {context.shape}")
    print(f"   Weights shape: {weights.shape}")

    # 3. Self-Attention
    print("\n3. Self-Attention:")
    x = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    self_attn = SelfAttention(hidden_dim, heads=4).to(device)
    out, attention = self_attn(x)
    print(f"   Output shape: {out.shape}")
    print(f"   Attention shape: {attention.shape}")

    # 可视化注意力
    fig = visualize_attention(attention[0], title="Self-Attention")
    plt.savefig('10-attention/self_attention_viz.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n   注意力可视化已保存到 10-attention/self_attention_viz.png")


# ============================================
# 实验: 注意力分类器
# ============================================

def experiment_attention_classifier():
    """测试注意力分类器"""
    print("\n" + "=" * 60)
    print("实验: 注意力分类器")
    print("=" * 60)

    # 参数
    vocab_size = 1000
    embed_dim = 64
    num_heads = 4
    num_classes = 3
    seq_len = 20
    batch_size = 16

    # 创建模型
    model = AttentionClassifier(
        vocab_size, embed_dim, num_heads, num_classes
    ).to(device)

    # 创建随机数据
    X = torch.randint(0, vocab_size, (batch_size * 5, seq_len)).to(device)
    y = torch.randint(0, num_classes, (batch_size * 5,)).to(device)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n训练中...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            _, predicted = logits.max(1)
            acc = predicted.eq(y).sum().item() / y.size(0)
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    # 可视化最后一个样本的注意力
    model.eval()
    with torch.no_grad():
        _, attention = model(X[:1])

    fig = visualize_attention(attention[0], title="Classifier Attention")
    plt.savefig('10-attention/classifier_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n分类器注意力可视化已保存到 10-attention/classifier_attention.png")


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 注意力机制教程")
    print("=" * 60)

    # 实验1: 比较不同注意力
    experiment_attention_comparison()

    # 实验2: 注意力分类器
    experiment_attention_classifier()

    print("\n" + "=" * 60)
    print("教程完成!")
    print("=" * 60)
    print("""
注意力机制总结:

1. Additive Attention (Bahdanau):
   - 灵活但计算量大
   - 适合query和key维度不同的情况

2. Dot-Product Attention (Luong):
   - 计算简单高效
   - 要求query和key维度相同

3. Scaled Dot-Product Attention:
   - Transformer的核心
   - 缩放因子防止梯度消失

4. Self-Attention:
   - 序列内部计算注意力
   - 捕捉长距离依赖

5. Multi-Head Attention:
   - 多个子空间并行计算
   - 捕捉不同类型的依赖关系

选择建议:
- 计算资源有限: Dot-Product
- 需要灵活性: Additive
- 追求性能: Multi-Head Scaled Dot-Product
    """)
