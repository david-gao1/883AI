"""
训练脚本
用于训练CNN语义分割模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import get_data_loaders, calculate_coal_ratio
from model import get_model
from metrics import calculate_iou, calculate_iou_multiclass, calculate_pixel_accuracy


class DiceLoss(nn.Module):
    """Dice损失函数，常用于语义分割任务"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """组合损失：BCE + Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_epoch(model, train_loader, criterion, optimizer, device, num_classes=3):
    """训练一个 epoch。num_classes=3 时使用 CrossEntropyLoss 与多分类 IoU。"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        target_size = masks.shape[-2:]
        if outputs.shape[2:] != target_size:
            outputs = F.interpolate(outputs, size=target_size, mode='bilinear', align_corners=False)
        
        if num_classes >= 3:
            loss = criterion(outputs, masks)
            pred_masks = outputs.argmax(dim=1).long()
            iou = calculate_iou_multiclass(pred_masks, masks, num_classes=num_classes)
            acc = calculate_pixel_accuracy(pred_masks, masks)
        else:
            loss = criterion(outputs, masks.unsqueeze(1))
            pred_masks = (outputs > 0.5).float()
            iou = calculate_iou(pred_masks, masks.unsqueeze(1))
            acc = calculate_pixel_accuracy(pred_masks, masks.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += iou if isinstance(iou, float) else iou.item()
        total_acc += acc.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}' if isinstance(iou, float) else f'{iou.item():.4f}',
            'acc': f'{acc.item():.4f}'
        })
    
    n = len(train_loader)
    return total_loss / n, total_iou / n, total_acc / n


def validate(model, val_loader, criterion, device, num_classes=3):
    """验证模型。num_classes=3 时使用多分类 IoU/Acc。"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            target_size = masks.shape[-2:]
            if outputs.shape[2:] != target_size:
                outputs = F.interpolate(outputs, size=target_size, mode='bilinear', align_corners=False)
            
            if num_classes >= 3:
                loss = criterion(outputs, masks)
                pred_masks = outputs.argmax(dim=1).long()
                iou = calculate_iou_multiclass(pred_masks, masks, num_classes=num_classes)
                acc = calculate_pixel_accuracy(pred_masks, masks)
            else:
                loss = criterion(outputs, masks.unsqueeze(1))
                pred_masks = (outputs > 0.5).float()
                iou = calculate_iou(pred_masks, masks.unsqueeze(1))
                acc = calculate_pixel_accuracy(pred_masks, masks.unsqueeze(1))
            
            total_loss += loss.item()
            total_iou += iou if isinstance(iou, float) else iou.item()
            total_acc += acc.item()
    
    n = len(val_loader)
    return total_loss / n, total_iou / n, total_acc / n


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-4,
    device='cuda',
    save_dir='./checkpoints'
):
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备（'cuda' 或 'cpu'）
        save_dir: 模型保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_classes = getattr(model, 'n_classes', 1)
    if num_classes >= 3:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        num_classes_for_metrics = num_classes
    else:
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        num_classes_for_metrics = 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_iou': [],
        'train_acc': [],
        'val_loss': [],
        'val_iou': [],
        'val_acc': []
    }
    
    best_val_iou = 0.0
    
    print(f"开始训练，设备: {device}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        train_loss, train_iou, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes=num_classes_for_metrics
        )
        val_loss, val_iou, val_acc = validate(
            model, val_loader, criterion, device, num_classes=num_classes_for_metrics
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"保存最佳模型 (Val IoU: {val_iou:.4f})")
    
    # 保存训练历史
    np.save(os.path.join(save_dir, 'history.npy'), history)
    
    # 绘制训练曲线
    plot_training_history(history, save_dir)
    
    print("\n训练完成！")
    return history


def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU曲线
    axes[1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[1].set_title('Training and Validation IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    # Accuracy曲线
    axes[2].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[2].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[2].set_title('Training and Validation Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 配置参数
    RAW_DATA_DIR = './raw_data'
    GROUNDTRUTH_DIR = './groundtruth'
    TRAIN_IDS = list(range(1, 201))  # 1-200
    TEST_IDS = list(range(201, 237))  # 201-236
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建数据加载器
    train_loader, test_loader = get_data_loaders(
        RAW_DATA_DIR,
        GROUNDTRUTH_DIR,
        TRAIN_IDS,
        TEST_IDS,
        batch_size=BATCH_SIZE
    )
    
    from data_loader import NUM_CLASSES
    model = get_model(model_name='unet', n_channels=3, n_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # 训练模型
    history = train_model(
        model,
        train_loader,
        test_loader,  # 使用测试集作为验证集
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_dir='./checkpoints'
    )
