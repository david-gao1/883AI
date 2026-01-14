#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import numpy as np


class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, config, device: torch.device):
        """
        初始化训练器
        
        Args:
            model: 模型
            config: 配置对象
            device: 设备
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_scheduler_step_size,
            gamma=config.lr_scheduler_gamma
        )
        
        # 训练历史
        self.history = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # 早停机制
        self.patience = getattr(config, 'early_stopping_patience', 5)
        self.wait = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标字典
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        return {
            'loss': train_loss,
            'accuracy': train_acc
        }
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史字典
        """
        print('Starting training...\n')
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_losses'].append(train_metrics['loss'])
            self.history['train_accs'].append(train_metrics['accuracy'])
            
            # 验证
            val_metrics = self.validate(val_loader, epoch)
            self.history['val_losses'].append(val_metrics['loss'])
            self.history['val_accs'].append(val_metrics['accuracy'])
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印结果
            print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
            
            # 早停检查
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                self.wait = 0
                print(f'  ✓ Best validation accuracy: {self.best_val_acc:.2f}%')
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f'\nEarly stopping at epoch {epoch+1} (patience={self.patience})')
                    print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
                    break
            print()
        
        # 加载最佳模型
        self.model.load_state_dict(self.best_model_state)
        
        self.history['best_val_acc'] = self.best_val_acc
        
        return self.history
    
    def save_model(self, save_path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

