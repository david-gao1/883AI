#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估器模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from typing import Dict, List
import numpy as np


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        初始化评估器
        
        Args:
            model: 模型
            device: 设备
        """
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader: DataLoader, class_names: List[str]) -> Dict:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            class_names: 类别名称列表
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        num_classes = len(class_names)
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                preds = predicted.cpu().numpy()
                labs = labels.cpu().numpy()
                
                # 确保预测值在有效范围内（0到num_classes-1）
                preds = np.clip(preds, 0, num_classes - 1)
                
                # 检查标签是否在有效范围内
                if np.any((labs < 0) | (labs >= num_classes)):
                    print(f"Warning: Found invalid labels: {labs}")
                    labs = np.clip(labs, 0, num_classes - 1)
                
                all_preds.extend(preds.tolist())
                all_labels.extend(labs.tolist())
        
        # 计算准确率
        accuracy = accuracy_score(all_labels, all_preds)
        
        # 分类报告
        # 确保标签和预测都在有效范围内
        unique_labels = sorted(list(set(all_labels + all_preds)))
        valid_labels = [l for l in unique_labels if 0 <= l < len(class_names)]
        
        if len(valid_labels) != len(class_names):
            print(f"Warning: Found {len(unique_labels)} unique labels, expected {len(class_names)}")
            print(f"Unique labels: {unique_labels}")
            print(f"Valid labels: {valid_labels}")
        
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            labels=list(range(len(class_names))),
            output_dict=True,
            zero_division=0
        )
        
        # 混淆矩阵（确保使用正确的标签范围）
        all_labels_array = np.array(all_labels)
        all_preds_array = np.array(all_preds)
        cm = confusion_matrix(all_labels_array, all_preds_array, labels=list(range(len(class_names))))
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def print_results(self, results: Dict, class_names: List[str]):
        """
        打印评估结果
        
        Args:
            results: 评估结果
            class_names: 类别名称列表
        """
        print(f'\nTest Accuracy: {results["accuracy"]*100:.2f}%')
        print('\nClassification Report:')
        print(classification_report(
            results['labels'],
            results['predictions'],
            target_names=class_names
        ))

