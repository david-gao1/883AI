#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序入口
"""

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import FlowerDataset, DataTransform
from model import FlowerClassifier
from trainer import Trainer
from evaluator import Evaluator
from utils import Visualizer, ResultSaver, set_random_seed


def create_data_loaders(config: Config):
    """
    创建数据加载器
    
    Args:
        config: 配置对象
        
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    # 数据变换
    train_transform = DataTransform.get_train_transform(config.image_size)
    val_test_transform = DataTransform.get_val_test_transform(
        config.resize_size, config.image_size
    )
    
    # 加载数据集（确保使用相同的类别映射）
    print('Loading datasets...')
    train_dataset = FlowerDataset(config.train_dir, transform=train_transform)
    # 使用训练集的类别映射，确保一致性
    class_names = train_dataset.get_class_names()
    class_to_idx = train_dataset.get_class_to_idx()
    
    val_dataset = FlowerDataset(config.val_dir, transform=val_test_transform)
    test_dataset = FlowerDataset(config.test_dir, transform=val_test_transform)
    
    # 确保所有数据集使用相同的类别顺序
    # 重新映射验证集和测试集的标签
    val_dataset.class_names = class_names
    val_dataset.class_to_idx = class_to_idx
    test_dataset.class_names = class_names
    test_dataset.class_to_idx = class_to_idx
    
    # 重新映射标签
    val_dataset._remap_labels()
    test_dataset._remap_labels()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    print(f'Classes: {class_names}')
    print(f'Class mapping: {class_to_idx}')
    print()
    
    return train_loader, val_loader, test_loader, class_names


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_random_seed(config.random_seed)
    
    # 设备配置（自动检测）
    if config.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using device: CUDA (GPU)')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Using device: MPS (Apple Silicon GPU)')
        else:
            device = torch.device('cpu')
            print('Using device: CPU')
    elif config.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using device: CUDA (GPU)')
        else:
            device = torch.device('cpu')
            print('CUDA not available, using CPU')
    elif config.device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Using device: MPS (Apple Silicon GPU)')
        else:
            device = torch.device('cpu')
            print('MPS not available, using CPU')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')
    
    print()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, class_names = create_data_loaders(config)
    
    # 创建模型
    model = FlowerClassifier(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained
    )
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    print('Model Information:')
    for key, value in model_info.items():
        print(f'  {key}: {value}')
    print()
    
    # 创建训练器
    trainer = Trainer(model, config, device)
    
    # 训练模型
    history = trainer.train(train_loader, val_loader)
    
    # 绘制训练曲线
    visualizer = Visualizer()
    visualizer.plot_training_curves(history, config.training_curves_path)
    
    # 评估模型
    print('\nEvaluating on test set...')
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate(test_loader, class_names)
    evaluator.print_results(results, class_names)
    
    # 绘制混淆矩阵
    import numpy as np
    cm = np.array(results['confusion_matrix'])
    visualizer.plot_confusion_matrix(cm, class_names, config.confusion_matrix_path)
    
    # 保存结果
    result_saver = ResultSaver()
    result_saver.save_results(results, history, class_names, config.results_save_path)
    
    # 保存模型
    trainer.save_model(config.model_save_path)
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

