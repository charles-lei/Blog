"""
生成对抗网络 (Generative Adversarial Networks, GAN)
====================================================

核心思想：
    两个网络互相博弈：
    - Generator (生成器): 从噪声生成假数据，试图骗过判别器
    - Discriminator (判别器): 区分真假数据，试图不被骗

训练目标：
    - G 希望最大化 D(G(z)) （D认为假数据是真的）
    - D 希望最大化 D(x) 和最小化 D(G(z))

数学公式：
    min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

常见问题：
    - Mode Collapse: 生成器只产生几种样本
    - 训练不稳定: G和D能力不平衡
    - 梯度消失: D太强导致G无法学习

改进版本：
    - DCGAN: 使用卷积的GAN
    - WGAN: 使用Wasserstein距离
    - Conditional GAN: 条件生成
    - StyleGAN: 高质量图像生成
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================
# 1. 基础GAN (全连接网络)
# ============================================

class BasicGenerator(nn.Module):
    """
    基础生成器 (全连接)

    输入: 随机噪声 z
    输出: 生成的图像
    """

    def __init__(self, latent_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


class BasicDiscriminator(nn.Module):
    """
    基础判别器 (全连接)

    输入: 图像
    输出: 真实概率 [0, 1]
    """

    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ============================================
# 2. DCGAN (深度卷积GAN)
# ============================================

class DCGANGenerator(nn.Module):
    """
    DCGAN 生成器

    使用转置卷积进行上采样
    架构指南:
        - 使用转置卷积而非上采样+卷积
        - 使用BatchNorm
        - 使用ReLU激活
        - 输出层使用Tanh
    """

    def __init__(self, latent_dim, channels=1, feature_maps=64):
        super().__init__()

        self.model = nn.Sequential(
            # 输入: (batch, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            # (feature_maps*8, 4, 4)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # (feature_maps*4, 8, 8)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # (feature_maps*2, 16, 16)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # (feature_maps, 32, 32)
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (channels, 64, 64)
        )

    def forward(self, z):
        # reshape z to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        return self.model(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN 判别器

    架构指南:
        - 使用卷积而非池化
        - 使用LeakyReLU
        - 使用Dropout防止过拟合
        - 输出层使用Sigmoid
    """

    def __init__(self, channels=1, feature_maps=64):
        super().__init__()

        self.model = nn.Sequential(
            # 输入: (channels, 64, 64)
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps, 32, 32)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps*2, 16, 16)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps*4, 8, 8)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps*8, 4, 4)
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# ============================================
# 3. Conditional GAN (条件GAN)
# ============================================

class ConditionalGenerator(nn.Module):
    """
    条件生成器

    输入: 噪声 z + 类别标签 y
    输出: 指定类别的生成图像
    """

    def __init__(self, latent_dim, num_classes, output_dim):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 拼接噪声和标签embedding
        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)
        return self.model(x)


class ConditionalDiscriminator(nn.Module):
    """
    条件判别器

    输入: 图像 + 类别标签
    输出: 该图像是否属于该类别
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_emb = self.label_embedding(labels)
        x = torch.cat([x, label_emb], dim=1)
        return self.model(x)


# ============================================
# 4. Wasserstein GAN (WGAN)
# ============================================

class WGANDiscriminator(nn.Module):
    """
    WGAN 判别器 (称为Critic)

    关键区别:
        - 输出层没有Sigmoid (输出任意范围的分数)
        - 使用Gradient Penalty (WGAN-GP) 或 Weight Clipping
    """

    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
            # 注意：没有Sigmoid
        )

    def forward(self, x):
        return self.model(x)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    计算Gradient Penalty (WGAN-GP)

    用于强制满足Lipschitz约束
    """
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# ============================================
# 5. 训练函数
# ============================================

def train_basic_gan():
    """训练基础GAN"""
    print("\n" + "=" * 60)
    print("训练基础GAN")
    print("=" * 60)

    # 参数
    latent_dim = 100
    img_dim = 28 * 28  # MNIST
    batch_size = 64
    epochs = 50

    # 创建模型
    G = BasicGenerator(latent_dim, img_dim).to(device)
    D = BasicDiscriminator(img_dim).to(device)

    # 优化器
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 损失函数
    criterion = nn.BCELoss()

    # 创建模拟数据 (真实图像)
    # 这里用随机数据代替真实数据，实际应使用MNIST等数据集
    real_data = torch.randn(1000, img_dim).to(device)
    real_data = (real_data - real_data.min()) / (real_data.max() - real_data.min()) * 2 - 1

    dataloader = DataLoader(real_data, batch_size=batch_size, shuffle=True)

    # 训练
    G_losses = []
    D_losses = []

    print("\n开始训练...")
    for epoch in range(epochs):
        for batch_idx, real_imgs in enumerate(dataloader):
            batch_size_curr = real_imgs.size(0)

            # 真假标签
            real_labels = torch.ones(batch_size_curr, 1).to(device)
            fake_labels = torch.zeros(batch_size_curr, 1).to(device)

            # ---------------------
            # 训练判别器
            # ---------------------
            d_optimizer.zero_grad()

            # 真实图像的loss
            d_real = D(real_imgs)
            d_loss_real = criterion(d_real, real_labels)

            # 生成假图像
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake_imgs = G(z)

            # 假图像的loss
            d_fake = D(fake_imgs.detach())
            d_loss_fake = criterion(d_fake, fake_labels)

            # 总loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ---------------------
            # 训练生成器
            # ---------------------
            g_optimizer.zero_grad()

            # 生成假图像
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake_imgs = G(z)

            # G希望D认为假图像是真的
            g_output = D(fake_imgs)
            g_loss = criterion(g_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # 保存生成图像
    G.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        gen_imgs = G(z).view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i], cmap='gray')
        ax.axis('off')
    plt.suptitle('Basic GAN Generated Images')
    plt.savefig('12-gan/basic_gan_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("生成图像已保存到 12-gan/basic_gan_samples.png")

    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    plt.savefig('12-gan/basic_gan_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Loss曲线已保存到 12-gan/basic_gan_loss.png")


def train_conditional_gan():
    """训练条件GAN"""
    print("\n" + "=" * 60)
    print("训练条件GAN (Conditional GAN)")
    print("=" * 60)

    # 参数
    latent_dim = 100
    img_dim = 28 * 28
    num_classes = 10  # MNIST 10个数字
    batch_size = 64
    epochs = 50

    # 创建模型
    G = ConditionalGenerator(latent_dim, num_classes, img_dim).to(device)
    D = ConditionalDiscriminator(img_dim, num_classes).to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 创建模拟数据
    real_imgs = torch.randn(1000, img_dim).to(device)
    real_imgs = (real_imgs - real_imgs.min()) / (real_imgs.max() - real_imgs.min()) * 2 - 1
    real_labels_data = torch.randint(0, num_classes, (1000,)).to(device)

    dataset = torch.utils.data.TensorDataset(real_imgs, real_labels_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n开始训练...")
    for epoch in range(epochs):
        for batch_imgs, batch_labels in dataloader:
            batch_size_curr = batch_imgs.size(0)

            real_labels = torch.ones(batch_size_curr, 1).to(device)
            fake_labels = torch.zeros(batch_size_curr, 1).to(device)

            # 训练D
            d_optimizer.zero_grad()
            d_real = D(batch_imgs, batch_labels)
            d_loss_real = criterion(d_real, real_labels)

            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake_labels_gen = torch.randint(0, num_classes, (batch_size_curr,)).to(device)
            fake_imgs = G(z, fake_labels_gen)
            d_fake = D(fake_imgs.detach(), fake_labels_gen)
            d_loss_fake = criterion(d_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 训练G
            g_optimizer.zero_grad()
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake_labels_gen = torch.randint(0, num_classes, (batch_size_curr,)).to(device)
            fake_imgs = G(z, fake_labels_gen)
            g_output = D(fake_imgs, fake_labels_gen)
            g_loss = criterion(g_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # 生成每个类别的样本
    G.eval()
    fig, axes = plt.subplots(10, 5, figsize=(10, 20))
    with torch.no_grad():
        for class_idx in range(10):
            z = torch.randn(5, latent_dim).to(device)
            labels = torch.full((5,), class_idx, dtype=torch.long).to(device)
            gen_imgs = G(z, labels).view(-1, 28, 28).cpu()

            for i in range(5):
                axes[class_idx, i].imshow(gen_imgs[i], cmap='gray')
                axes[class_idx, i].axis('off')
                if i == 0:
                    axes[class_idx, i].set_ylabel(f'Class {class_idx}')

    plt.suptitle('Conditional GAN - Generated Samples by Class')
    plt.tight_layout()
    plt.savefig('12-gan/conditional_gan_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("条件GAN生成图像已保存到 12-gan/conditional_gan_samples.png")


# ============================================
# 6. GAN训练技巧总结
# ============================================

GAN_TIPS = """
GAN 训练技巧总结
================

1. 架构设计:
   - 使用LeakyReLU而非ReLU (防止梯度消失)
   - 生成器输出使用Tanh, 判别器输出使用Sigmoid
   - 使用BatchNorm (判别器输入层除外)

2. 优化技巧:
   - 使用Adam优化器, beta1=0.5
   - 学习率不宜太高 (0.0002左右)
   - G和D的训练频率可以调整 (如D训练多次,G训练一次)

3. 稳定训练:
   - WGAN: 使用Wasserstein距离
   - WGAN-GP: Gradient Penalty
   - Spectral Normalization
   - Feature Matching

4. 防止Mode Collapse:
   - Mini-batch discrimination
   - Unrolled GAN
   - 使用多个生成器

5. 评估指标:
   - Inception Score (IS)
   - Frechet Inception Distance (FID)
   - 可视化检查

6. 常见问题:
   - D太强: G无法学习 -> 减少D训练次数
   - G太强: D无法区分 -> 增加D容量
   - 训练震荡: 降低学习率
"""


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch GAN 教程")
    print("=" * 60)

    # 创建输出目录
    os.makedirs('12-gan', exist_ok=True)

    # 实验1: 基础GAN
    train_basic_gan()

    # 实验2: 条件GAN
    train_conditional_gan()

    print("\n" + GAN_TIPS)

    print("\n" + "=" * 60)
    print("教程完成!")
    print("=" * 60)
