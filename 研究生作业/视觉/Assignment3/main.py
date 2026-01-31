"""
主程序入口
整合训练和评估流程
"""

import argparse
import torch
import os

from train import train_model
from evaluate import main as evaluate_main
from data_loader import get_data_loaders, NUM_CLASSES
from model import get_model


def main():
    parser = argparse.ArgumentParser(description='煤矸石分割项目')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                        help='运行模式: train(训练), eval(评估), both(训练+评估)')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'simple_cnn'],
                        help='模型类型')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 配置参数
    RAW_DATA_DIR = './raw_data'
    GROUNDTRUTH_DIR = './groundtruth'
    TRAIN_IDS = list(range(1, 201))  # 1-200
    TEST_IDS = list(range(201, 237))  # 201-236
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*50)
        print("开始训练")
        print("="*50)
        
        # 创建数据加载器
        train_loader, test_loader = get_data_loaders(
            RAW_DATA_DIR,
            GROUNDTRUTH_DIR,
            TRAIN_IDS,
            TEST_IDS,
            batch_size=args.batch_size
        )
        
        model = get_model(model_name=args.model, n_channels=3, n_classes=NUM_CLASSES)
        model = model.to(DEVICE)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型: {args.model}")
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
        # 训练模型
        history = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=DEVICE,
            save_dir='./checkpoints'
        )
        
        print("\n训练完成！")
    
    if args.mode in ['eval', 'both']:
        print("\n" + "="*50)
        print("开始评估")
        print("="*50)
        
        evaluate_main()


if __name__ == '__main__':
    main()
