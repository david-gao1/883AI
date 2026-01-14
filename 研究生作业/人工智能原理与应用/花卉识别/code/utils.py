#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import json
from datetime import datetime


class Visualizer:
    """可视化工具类"""
    
    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = 'training_curves.png'):
        """
        绘制训练曲线
        
        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_accs'], label='Train Acc')
        ax2.plot(history['val_accs'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training curves saved to {save_path}')
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                              save_path: str = 'confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to {save_path}')


class ResultSaver:
    """结果保存工具类"""
    
    @staticmethod
    def save_results(results: Dict, history: Dict, class_names: List[str],
                    save_path: str = 'results.json'):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            history: 训练历史
            class_names: 类别名称列表
            save_path: 保存路径
        """
        results_dict = {
            'test_accuracy': float(results['accuracy']),
            'best_val_accuracy': float(history.get('best_val_acc', 0)),
            'classification_report': results['report'],
            'class_names': class_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f'Results saved to {save_path}')


def set_random_seed(seed: int):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Mac MPS不需要特殊的随机种子设置

