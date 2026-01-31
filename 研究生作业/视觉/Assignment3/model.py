"""
CNN语义分割模型定义
使用U-Net架构进行煤矸石分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net语义分割网络
    
    编码器-解码器架构，用于像素级分类
    """
    
    def __init__(self, n_channels=3, n_classes=1):
        """
        Args:
            n_channels: 输入通道数（RGB图像为3）
            n_classes: 输出类别数（1=二分类sigmoid，3=背景/煤/矸石用CrossEntropyLoss）
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes
        
        # 编码器（下采样路径）
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 解码器（上采样路径）
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
    
    @staticmethod
    def crop_tensor(tensor, target_size):
        """
        裁剪tensor到目标尺寸（用于处理尺寸不匹配问题）
        
        Args:
            tensor: 要裁剪的tensor
            target_size: 目标尺寸 (H, W)
        
        Returns:
            cropped_tensor: 裁剪后的tensor
        """
        _, _, h, w = tensor.size()
        target_h, target_w = target_size
        
        # 计算裁剪位置（居中裁剪）
        diff_h = h - target_h
        diff_w = w - target_w
        
        start_h = diff_h // 2
        start_w = diff_w // 2
        
        return tensor[:, :, start_h:start_h+target_h, start_w:start_w+target_w]
        
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        e5 = self.enc5(p4)
        
        # 解码器（带跳跃连接，处理尺寸不匹配）
        d1 = self.up1(e5)
        # 如果尺寸不匹配，裁剪e4以匹配d1
        if d1.shape[2:] != e4.shape[2:]:
            e4 = self.crop_tensor(e4, d1.shape[2:])
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        # 如果尺寸不匹配，裁剪e3以匹配d2
        if d2.shape[2:] != e3.shape[2:]:
            e3 = self.crop_tensor(e3, d2.shape[2:])
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        # 如果尺寸不匹配，裁剪e2以匹配d3
        if d3.shape[2:] != e2.shape[2:]:
            e2 = self.crop_tensor(e2, d3.shape[2:])
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        # 如果尺寸不匹配，裁剪e1以匹配d4
        if d4.shape[2:] != e1.shape[2:]:
            e1 = self.crop_tensor(e1, d4.shape[2:])
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)
        
        out = self.out(d4)
        if self.n_classes == 1:
            out = torch.sigmoid(out)
        # n_classes >= 2 时输出 logits，由 CrossEntropyLoss 内建 softmax
        return out


class SimpleCNN(nn.Module):
    """
    简化的CNN模型（备选方案）
    用于对比实验
    """
    
    def __init__(self, n_channels=3, n_classes=1):
        super(SimpleCNN, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )
        self.n_classes = n_classes
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.n_classes == 1:
            x = torch.sigmoid(x)
        return x


def get_model(model_name='unet', n_channels=3, n_classes=1):
    """
    获取模型实例
    
    Args:
        model_name: 模型名称 ('unet' 或 'simple_cnn')
        n_channels: 输入通道数
        n_classes: 输出类别数
    
    Returns:
        model: 模型实例
    """
    if model_name == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'simple_cnn':
        model = SimpleCNN(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
