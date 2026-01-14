#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型定义模块
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class FlowerClassifier(nn.Module):
    """花卉分类模型"""
    
    def __init__(self, num_classes: int = 5, model_name: str = 'resnet18', 
                 pretrained: bool = True):
        """
        初始化模型
        
        Args:
            num_classes: 分类数量
            model_name: 模型名称 ('resnet18' or 'resnet34')
            pretrained: 是否使用预训练权重
        """
        super(FlowerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained
        
        self._build_model()
    
    def _build_model(self):
        """构建模型"""
        if self.model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=self.pretrained)
            num_features = self.backbone.fc.in_features
            # 添加Dropout层缓解过拟合
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.num_classes)
            )
        elif self.model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=self.pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        output = self.backbone(x)
        # 确保输出维度正确
        if output.size(1) != self.num_classes:
            raise ValueError(f"Output size {output.size(1)} does not match num_classes {self.num_classes}")
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'total_params': total_params,
            'trainable_params': trainable_params
        }

