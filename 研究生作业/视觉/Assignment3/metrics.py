"""
评估指标计算模块
包括IoU、像素准确率、ROC曲线、PR曲线等
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def calculate_iou(pred, target):
    """
    二分类 IoU：pred/target 为 (B,1,H,W) 或 (B,H,W)，取值 0/1 或 bool。
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum(dim=(1, 2))
    union = (pred | target).float().sum(dim=(1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def calculate_iou_multiclass(pred, target, num_classes=3):
    """
    多分类 IoU：pred/target 为 (B,H,W) long，取值 0..num_classes-1。
    返回 mean IoU（对各类 IoU 取平均）。
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).float().sum(dim=(1, 2))
        union = (pred_c | target_c).float().sum(dim=(1, 2))
        iou_c = (inter + 1e-6) / (union + 1e-6)
        ious.append(iou_c.mean().item())
    return sum(ious) / num_classes if num_classes else 0.0


def calculate_pixel_accuracy(pred, target):
    """
    像素准确率：pred/target 可为 (B,1,H,W) 或 (B,H,W)，支持二值或多分类。
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    if pred.dtype == torch.bool or target.dtype == torch.bool:
        pred = pred.long()
        target = target.long()
    correct = (pred == target).float().sum(dim=(1, 2))
    total = target.shape[1] * target.shape[2]
    return (correct / total).mean()


def calculate_coal_ratio_batch(pred_masks, num_classes=3):
    """
    批量计算煤的占比。二分类时 pred_masks 为 0/1 或概率>0.5；三分类时 pred_masks 为 0/1/2，煤=1。
    """
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if pred_masks.ndim == 4:
        pred_masks = pred_masks.squeeze(1)
    ratios = []
    for mask in pred_masks:
        total = mask.size
        if num_classes >= 3:
            coal_pixels = np.sum(mask == 1)
        else:
            coal_pixels = np.sum(mask > 0.5)
        ratios.append((coal_pixels / total) if total > 0 else 0.0)
    return ratios


def calculate_roc_pr_curves(model, data_loader, device, save_path='./results'):
    """
    计算并绘制ROC和PR曲线
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        save_path: 保存路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            if outputs.shape[1] >= 3:
                pred_probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
                target_binary = (masks == 1).cpu().numpy().flatten().astype(np.float32)
            else:
                pred_probs = outputs.cpu().numpy().flatten()
                target_binary = masks.cpu().numpy().flatten()
            all_preds.extend(pred_probs)
            all_targets.extend(target_binary)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    n_pos = int(np.sum(all_targets > 0.5))
    n_neg = int(np.sum(all_targets <= 0.5))
    no_positive = n_pos == 0
    no_negative = n_neg == 0

    if no_positive or no_negative:
        # 无正类或无负类时 ROC/PR 无定义，避免调用 sklearn 产生警告
        roc_auc = np.nan
        pr_auc = 0.0
        fpr, tpr = [0, 1], [0, 1]
        recall, precision = [0, 1], [0, 0]  # 占位，保证绘图不报错
    else:
        fpr, tpr, roc_thresholds = roc_curve(all_targets, all_preds)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
    
    # 绘制ROC曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})' if not np.isnan(roc_auc) else 'ROC curve (AUC = N/A, no pos/neg class)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if np.isnan(roc_auc):
        print("ROC AUC: N/A (no positive or no negative samples in test set)")
    else:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC (AP): {pr_auc:.4f}")
    
    return roc_auc, pr_auc


def evaluate_model(model, data_loader, device, num_classes=3):
    """
    全面评估模型性能。支持三分类（背景/煤/矸石）：num_classes=3，输出 mean IoU、煤占比 MAE/RMSE。
    """
    from data_loader import calculate_coal_ratio
    
    model.eval()
    total_iou = 0.0
    total_acc = 0.0
    all_coal_ratios_pred = []
    all_coal_ratios_target = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if outputs.shape[1] >= 3:
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
                    )
                pred_masks = outputs.argmax(dim=1).long()
                iou = calculate_iou_multiclass(pred_masks, masks, num_classes=num_classes)
                acc = calculate_pixel_accuracy(pred_masks, masks)
                total_iou += iou
                total_acc += acc.item()
                for i in range(pred_masks.shape[0]):
                    pred_ratio = (pred_masks[i] == 1).float().sum().item() / pred_masks[i].numel()
                    target_ratio = (masks[i] == 1).float().sum().item() / masks[i].numel()
                    all_coal_ratios_pred.append(pred_ratio)
                    all_coal_ratios_target.append(target_ratio)
            else:
                pred_masks = (outputs > 0.5).float()
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
                    )
                    pred_masks = (outputs > 0.5).float()
                iou = calculate_iou(pred_masks, masks.unsqueeze(1))
                acc = calculate_pixel_accuracy(pred_masks, masks.unsqueeze(1))
                total_iou += iou.item()
                total_acc += acc.item()
                for i in range(pred_masks.shape[0]):
                    pred_ratio = calculate_coal_ratio(pred_masks[i, 0].cpu().numpy())
                    target_ratio = calculate_coal_ratio(masks[i].cpu().numpy())
                    all_coal_ratios_pred.append(pred_ratio)
                    all_coal_ratios_target.append(target_ratio)
    
    n_batches = len(data_loader)
    avg_iou = total_iou / n_batches if n_batches else 0.0
    avg_acc = total_acc / n_batches if n_batches else 0.0
    all_coal_ratios_pred = np.array(all_coal_ratios_pred)
    all_coal_ratios_target = np.array(all_coal_ratios_target)
    mae = np.mean(np.abs(all_coal_ratios_pred - all_coal_ratios_target))
    rmse = np.sqrt(np.mean((all_coal_ratios_pred - all_coal_ratios_target) ** 2))
    
    return {
        'iou': avg_iou,
        'pixel_accuracy': avg_acc,
        'coal_ratio_mae': mae,
        'coal_ratio_rmse': rmse,
        'coal_ratios_pred': all_coal_ratios_pred,
        'coal_ratios_target': all_coal_ratios_target,
    }
