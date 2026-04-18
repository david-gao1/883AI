from __future__ import annotations

import torch
from torch import nn


# ============================================================
# MLP 分类器
# 把图片拍平成一维向量，经过两个全连接层，输出 100 个类别的得分
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.3) -> None:
        super().__init__()
        # nn.Sequential 把多个层按顺序串起来，数据会依次经过每一层
        self.network = nn.Sequential(
            # 第 1 步：把 [batch, 3, 32, 32] 的图片拍平成 [batch, 3072] 的向量
            nn.Flatten(),
            # 第 2 步：全连接层，3072 维 → 1024 维（学习输入特征的组合）
            nn.Linear(32 * 32 * 3, 1024),
            # 第 3 步：ReLU 激活，负数归零，引入非线性
            nn.ReLU(inplace=True),
            # 第 4 步：Dropout，训练时随机关掉 30% 神经元，防止过拟合
            nn.Dropout(dropout),
            # 第 5 步：全连接层，1024 维 → 512 维（进一步压缩特征）
            nn.Linear(1024, 512),
            # 第 6 步：ReLU
            nn.ReLU(inplace=True),
            # 第 7 步：Dropout
            nn.Dropout(dropout),
            # 第 8 步：输出层，512 维 → 100 维（每个类别一个得分）
            # 注意：这里没有激活函数，因为交叉熵损失内部自带 softmax
            nn.Linear(512, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播：输入一批图片，输出每张图对 100 个类别的得分"""
        return self.network(inputs)


# ============================================================
# CNN 分类器
# 用卷积提取图像的局部空间特征，再用全连接层分类
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.3) -> None:
        super().__init__()

        # ---------- 特征提取部分：卷积 + 池化，逐步提取从低级到高级的特征 ----------
        self.features = nn.Sequential(
            # ===== 第一阶段：检测低级特征（边缘、颜色块）=====
            # 3→64：RGB 3通道输入，用 64 个 3×3 卷积核扫描，输出 64 张特征图
            # padding=1：四周补零，保持 32×32 不缩小
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # [batch, 64, 32, 32]
            # 批归一化：把这 64 个通道的数据分布校准到均值0、方差1，训练更稳定
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 再来一层卷积，64→64，在已有特征上进一步提取
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # [batch, 64, 32, 32]
            nn.ReLU(inplace=True),
            # 最大池化：每 2×2 区域取最大值，尺寸减半 32→16
            nn.MaxPool2d(kernel_size=2),                     # [batch, 64, 16, 16]
            # Dropout2d：随机关掉整个通道（不是单个像素），防过拟合
            nn.Dropout2d(dropout),

            # ===== 第二阶段：检测中级特征（纹理、局部形状）=====
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # [batch, 128, 16, 16]  通道数翻倍
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                     # [batch, 128, 8, 8]    尺寸再减半
            nn.Dropout2d(dropout),

            # ===== 第三阶段：检测高级特征（物体部件、语义）=====
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [batch, 256, 8, 8]    通道数再翻倍
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                     # [batch, 256, 4, 4]    尺寸再减半
        )
        # 到这里，一张 32×32×3 的图片已经变成了 256×4×4 = 4096 维的特征

        # ---------- 分类头：把提取到的特征变成 100 个类别的得分 ----------
        self.classifier = nn.Sequential(
            nn.Flatten(),                                    # [batch, 4096]  拍平
            nn.Linear(256 * 4 * 4, 512),                     # [batch, 512]   压缩
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),                      # [batch, 100]   输出得分
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播：图片 → 卷积提取特征 → 全连接分类"""
        features = self.features(inputs)    # 经过所有卷积+池化层
        return self.classifier(features)    # 经过全连接层得到分类得分


def build_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """根据名字创建对应的模型。train.py 中通过命令行参数 --models mlp cnn 来调用。"""
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLPClassifier(num_classes=num_classes)
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
