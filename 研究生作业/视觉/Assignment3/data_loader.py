"""
数据加载和预处理模块
用于加载原始图像和groundtruth标注，并进行数据增强和预处理
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# 统一 resize 尺寸，必须能被 16 整除（U-Net 有 4 次下采样）
TARGET_SIZE = (256, 256)

# 三分类：背景=0, 煤=1, 矸石=2（与 Ground-Truth 标注一致：黑=背景，红=煤，绿=矸石）
# 数据集中 PNG 灰度值通常为 0, 38, 75，对应 背景/煤/矸石（红暗绿亮，故 38→煤 75→矸石）
NUM_CLASSES = 3
GT_VALUE_BACKGROUND = 0
GT_VALUE_COAL = 38   # 红色在灰度图中较暗
GT_VALUE_GANGUE = 75  # 绿色在灰度图中较亮


class CoalDataset(Dataset):
    """煤矸石数据集类"""
    
    def __init__(self, raw_data_dir, groundtruth_dir, image_ids, transform=None, is_training=True, target_size=None):
        """
        Args:
            raw_data_dir: 原始图像目录
            groundtruth_dir: 标注图像目录
            image_ids: 图像ID列表（如[1,2,...,200]）
            transform: 数据增强变换
            is_training: 是否为训练模式
            target_size: 统一 resize 尺寸 (H, W)，必须能被 16 整除，默认 (256, 256)
        """
        self.raw_data_dir = raw_data_dir
        self.groundtruth_dir = groundtruth_dir
        self.image_ids = image_ids
        self.transform = transform
        self.is_training = is_training
        self.target_size = target_size if target_size is not None else TARGET_SIZE
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # 加载原始图像
        img_path = os.path.join(self.raw_data_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        # 加载标注图像
        mask_path = os.path.join(self.groundtruth_dir, f"{image_id}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # 转为灰度图
        else:
            # 如果标注不存在，创建全零mask
            mask = Image.new('L', image.size, 0)
        
        image = np.array(image)
        mask = np.array(mask)
        
        # 三分类：0=背景, 1=煤, 2=矸石。支持两种标注格式：0/1/2 或 0/38/75（黑/红/绿转灰度）
        uniq = np.unique(mask)
        if set(uniq).issubset({0, 1, 2}):
            out_mask = mask.astype(np.uint8)
        else:
            out_mask = np.zeros_like(mask, dtype=np.uint8)
            out_mask[mask == GT_VALUE_BACKGROUND] = 0
            out_mask[mask == GT_VALUE_COAL] = 1
            out_mask[mask == GT_VALUE_GANGUE] = 2
            out_mask[(mask != GT_VALUE_BACKGROUND) & (mask != GT_VALUE_COAL) & (mask != GT_VALUE_GANGUE)] = 0
        
        image = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(out_mask)  # 0,1,2 用 NEAREST 缩放不会混
        
        if self.transform and self.is_training:
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask_pil = TF.hflip(mask_pil)
            if np.random.random() > 0.5:
                image = TF.vflip(image)
                mask_pil = TF.vflip(mask_pil)
        
        image = TF.resize(image, self.target_size, interpolation=2)
        mask_pil = TF.resize(mask_pil, self.target_size, interpolation=0)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask_np = np.array(mask_pil).astype(np.int64)
        mask = torch.from_numpy(mask_np).long()
        return image, mask


def get_data_loaders(raw_data_dir, groundtruth_dir, train_ids, test_ids, batch_size=8, num_workers=2):
    """
    获取训练和测试数据加载器
    
    Args:
        raw_data_dir: 原始图像目录
        groundtruth_dir: 标注图像目录
        train_ids: 训练集图像ID列表（可为空，仅评估时只传 test_ids）
        test_ids: 测试集图像ID列表
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, test_loader（当 train_ids 为空时 train_loader 为 None）
    """
    train_loader = None
    if len(train_ids) > 0:
        # 训练集（带数据增强）
        train_dataset = CoalDataset(
            raw_data_dir,
            groundtruth_dir,
            train_ids,
            transform=None,
            is_training=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # 测试集（无数据增强）
    test_dataset = CoalDataset(
        raw_data_dir,
        groundtruth_dir,
        test_ids,
        transform=None,
        is_training=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def calculate_coal_ratio(mask):
    """
    计算 mask 中煤的占比（相对整图）：煤像素数 / 整图像素数。
    支持二值 mask (0/1) 或三分类 mask (0=背景, 1=煤, 2=矸石)。
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    total_pixels = mask.size
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        coal_pixels = np.sum(mask > 0.5)
    else:
        coal_pixels = np.sum(mask == 1)
    return (coal_pixels / total_pixels) if total_pixels > 0 else 0.0


def calculate_gangue_ratio(mask):
    """三分类 mask 下矸石占比（相对整图）：矸石像素数 / 整图像素数。"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    total_pixels = mask.size
    gangue_pixels = np.sum(mask == 2)
    return (gangue_pixels / total_pixels) if total_pixels > 0 else 0.0


def calculate_coal_ratio_in_object(prob_coal, prob_gangue=None, object_threshold=0.1):
    """
    计算煤在物体区域内的占比：煤像素数 / 物体像素数。
    三分类时：物体 = 煤 + 矸石；prob_coal 为 class=1 的概率图，prob_gangue 为 class=2 的概率图（可选）。
    若仅传 prob_coal（单通道），则物体 = prob_coal + (1 - prob_coal) 中大于 object_threshold 的部分，即前景。
    """
    if isinstance(prob_coal, torch.Tensor):
        prob_coal = prob_coal.cpu().numpy()
    if prob_gangue is not None and isinstance(prob_gangue, torch.Tensor):
        prob_gangue = prob_gangue.cpu().numpy()
    if prob_gangue is not None:
        object_pixels = np.sum((prob_coal > object_threshold) | (prob_gangue > object_threshold))
        coal_pixels = np.sum(prob_coal > 0.5)
    else:
        object_pixels = np.sum(prob_coal > object_threshold)
        coal_pixels = np.sum(prob_coal > 0.5)
    return (coal_pixels / object_pixels) if object_pixels > 0 else 0.0
