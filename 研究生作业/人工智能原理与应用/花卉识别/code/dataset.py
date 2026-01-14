#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集模块
"""

import os
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional
from torchvision import transforms


class FlowerDataset(Dataset):
    """花卉数据集类"""
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = sorted(os.listdir(data_dir))
        self.class_to_idx: dict = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        self._load_data()
    
    def _load_data(self):
        """加载所有图像和标签"""
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
    
    def _remap_labels(self):
        """重新映射标签，使用当前的class_to_idx"""
        remapped_labels = []
        for img_path in self.images:
            # 从路径中提取类别名
            class_name = os.path.basename(os.path.dirname(img_path))
            if class_name in self.class_to_idx:
                remapped_labels.append(self.class_to_idx[class_name])
            else:
                # 如果类别不在映射中，使用第一个类别（fallback）
                remapped_labels.append(0)
        self.labels = remapped_labels
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (image, label): 图像和标签
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names
    
    def get_class_to_idx(self) -> dict:
        """获取类别到索引的映射"""
        return self.class_to_idx


class DataTransform:
    """数据变换类"""
    
    @staticmethod
    def get_train_transform(image_size: int = 224) -> transforms.Compose:
        """
        获取训练集数据增强（增强版，缓解过拟合）
        
        Args:
            image_size: 图像尺寸
            
        Returns:
            数据变换组合
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),  # 增加到30度
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 增强
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加平移
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_val_test_transform(resize_size: int = 256, image_size: int = 224) -> transforms.Compose:
        """
        获取验证/测试集数据变换
        
        Args:
            resize_size: 调整大小
            image_size: 裁剪尺寸
            
        Returns:
            数据变换组合
        """
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

