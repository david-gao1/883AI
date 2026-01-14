#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """训练配置类"""
    
    # 数据路径（相对于code文件夹，需要回到上一级目录）
    data_root: str = '../data'
    train_dir: str = os.path.join(data_root, 'train')
    val_dir: str = os.path.join(data_root, 'val')
    test_dir: str = os.path.join(data_root, 'test')
    
    # 模型配置
    model_name: str = 'resnet18'  # 'resnet18' or 'resnet34'
    num_classes: int = 5
    pretrained: bool = True
    
    # 训练配置
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0001  # 降低学习率，缓解过拟合
    weight_decay: float = 0.001  # 增加权重衰减
    
    # 学习率调度
    lr_scheduler_step_size: int = 7
    lr_scheduler_gamma: float = 0.1
    
    # 早停配置
    early_stopping_patience: int = 5  # 验证集5个epoch不提升就停止
    
    # 数据增强
    image_size: int = 224
    resize_size: int = 256
    
    # 设备配置
    device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    # 注意：Mac ARM架构使用 'mps' 或 'auto' 以使用Apple Silicon GPU
    num_workers: int = 0  # Mac上建议设为0，避免多进程问题
    
    # 随机种子
    random_seed: int = 42
    
    # 输出路径（相对于code文件夹，保存到项目根目录）
    model_save_path: str = '../flower_classifier.pth'
    results_save_path: str = '../results.json'
    training_curves_path: str = '../training_curves.png'
    confusion_matrix_path: str = '../confusion_matrix.png'
    
    def __post_init__(self):
        """验证配置"""
        if self.model_name not in ['resnet18', 'resnet34']:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        
        if not os.path.exists(self.val_dir):
            raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")

