"""
评估脚本
用于评估训练好的模型在测试集上的性能
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from data_loader import get_data_loaders, calculate_coal_ratio, calculate_coal_ratio_in_object, NUM_CLASSES
from model import get_model
from metrics import (
    evaluate_model,
    calculate_roc_pr_curves,
    calculate_coal_ratio_batch,
)
import torch.nn.functional as F


def visualize_predictions(model, data_loader, device, save_dir='./results/visualizations', num_samples=10):
    """
    可视化预测结果
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 可视化样本数量
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    count = 0
    
    multiclass = getattr(model, 'n_classes', 1) >= 3
    with torch.no_grad():
        for images, masks in data_loader:
            if count >= num_samples:
                break
            images = images.to(device)
            outputs = model(images)
            if multiclass and outputs.shape[1] >= 3:
                pred_masks = outputs.argmax(dim=1).cpu().numpy()
                prob_coal = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            else:
                pred_masks = (outputs > 0.5).float().squeeze(1).cpu().numpy()
                prob_coal = outputs.squeeze(1).cpu().numpy()
            for i in range(images.shape[0]):
                if count >= num_samples:
                    break
                img = images[i].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = TF.to_pil_image(img)
                pred_mask = pred_masks[i]
                target_mask = masks[i].cpu().numpy()
                pred_ratio = (np.sum(pred_mask == 1) / pred_mask.size) if multiclass else calculate_coal_ratio(pred_mask)
                target_ratio = (np.sum(target_mask == 1) / target_mask.size) if multiclass else calculate_coal_ratio(target_mask)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                axes[1].imshow(target_mask, cmap='gray', vmin=0, vmax=2)
                axes[1].set_title(f'Groundtruth (Coal: {target_ratio:.4f})')
                axes[1].axis('off')
                axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=2)
                axes[2].set_title(f'Prediction (Coal: {pred_ratio:.4f})')
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{count+1}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                count += 1


def visualize_experiment_display(model, data_loader, device, save_path='./results/experiment_display.png', num_samples=6, grid_shape=(2, 3), grid_n=28):
    """
    实验结果展示一：三分类时绿=煤、红=矸石；仅在物体区域（煤+矸石）绘网格，背景不绘。
    左上角：煤占比(%)、矸石占比(%)（物体内）、耗时。
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    model.eval()
    multiclass = getattr(model, 'n_classes', 1) >= 3
    collected = []  # (img_gray, pred_mask, prob_coal, prob_gangue, coal_ratio_obj, gangue_ratio_obj, time_s)

    with torch.no_grad():
        for images, _ in data_loader:
            if len(collected) >= num_samples:
                break
            images_gpu = images.to(device)
            for i in range(images.shape[0]):
                if len(collected) >= num_samples:
                    break
                single = images_gpu[i:i+1]
                t0 = time.perf_counter()
                out = model(single)
                t1 = time.perf_counter()
                time_s = t1 - t0

                if multiclass and out.shape[1] >= 3:
                    probs = F.softmax(out[0], dim=0).cpu().numpy()  # (3,H,W)
                    prob_coal = probs[1]
                    prob_gangue = probs[2]
                    pred_mask = out.argmax(dim=1)[0].cpu().numpy()
                    obj_pixels = np.sum((pred_mask == 1) | (pred_mask == 2))
                    coal_pixels = np.sum(pred_mask == 1)
                    gangue_pixels = np.sum(pred_mask == 2)
                    coal_ratio_obj = (coal_pixels / obj_pixels) if obj_pixels > 0 else 0.0
                    gangue_ratio_obj = (gangue_pixels / obj_pixels) if obj_pixels > 0 else 0.0
                else:
                    prob_coal = out[0, 0].cpu().numpy()
                    prob_gangue = None
                    pred_mask = (out > 0.5).float()[0, 0].cpu().numpy()
                    coal_ratio_obj = calculate_coal_ratio_in_object(prob_coal)
                    gangue_ratio_obj = max(0.0, 1.0 - coal_ratio_obj)

                img = images[i].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img_np = img.permute(1, 2, 0).numpy()
                img_gray = np.clip(np.dot(img_np[..., :3], [0.299, 0.587, 0.114]), 0, 1)
                collected.append((img_gray, pred_mask, prob_coal, prob_gangue, coal_ratio_obj, gangue_ratio_obj, time_s))

    nrows, ncols = grid_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, (img_gray, pred_mask, prob_coal, prob_gangue, coal_ratio_obj, gangue_ratio_obj, time_s) in enumerate(collected):
        ax = axes.flat[idx]
        ax.imshow(img_gray, cmap='gray')
        h, w = prob_coal.shape
        dy, dx = max(1, h // grid_n), max(1, w // grid_n)
        for gy in range(grid_n):
            for gx in range(grid_n):
                y1, y2 = min(gy * dy, h - 1), min((gy + 1) * dy, h)
                x1, x2 = min(gx * dx, w - 1), min((gx + 1) * dx, w)
                if y2 <= y1 or x2 <= x1:
                    continue
                if prob_gangue is not None:
                    mean_coal = np.mean(prob_coal[y1:y2, x1:x2])
                    mean_gangue = np.mean(prob_gangue[y1:y2, x1:x2])
                    if mean_coal <= 0.2 and mean_gangue <= 0.2:
                        continue
                    if mean_coal >= mean_gangue and mean_coal > 0.5:
                        border_color = 'lime'
                    elif mean_gangue > 0.5:
                        border_color = 'red'
                    else:
                        border_color = 'orange'
                else:
                    cell = prob_coal[y1:y2, x1:x2]
                    mean_val = np.mean(cell) if cell.size else 0
                    if mean_val <= 0.2:
                        continue
                    border_color = 'lime' if mean_val > 0.5 else 'red'
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=border_color, linewidth=0.8)
                ax.add_patch(rect)
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                ax.plot(cx, cy, 'o', color=border_color, markersize=1.2, markeredgecolor=border_color)
        ax.text(0.02, 0.98, f'coal: {coal_ratio_obj*100:.1f}%, gangue: {gangue_ratio_obj*100:.1f}%, time: {time_s:.2f}s', transform=ax.transAxes,
                fontsize=10, color='red', va='top', ha='left')
        ax.axis('off')
    plt.suptitle('Experiment display: coal vs gangue ratio (in object) and time (s)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存实验结果展示一: {save_path}")


def plot_coal_ratio_comparison(metrics, save_path='./results'):
    """
    绘制煤占比预测对比图
    
    Args:
        metrics: 评估指标字典
        save_path: 保存路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    pred_ratios = metrics['coal_ratios_pred']
    target_ratios = metrics['coal_ratios_target']
    
    # 散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(target_ratios, pred_ratios, alpha=0.6, s=50)
    
    # 对角线
    min_val = min(min(target_ratios), min(pred_ratios))
    max_val = max(max(target_ratios), max(pred_ratios))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Groundtruth Coal Ratio', fontsize=12)
    plt.ylabel('Predicted Coal Ratio', fontsize=12)
    plt.title(f'Coal Ratio Prediction Comparison\nMAE: {metrics["coal_ratio_mae"]:.4f}, RMSE: {metrics["coal_ratio_rmse"]:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'coal_ratio_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 误差分布直方图
    errors = pred_ratios - target_ratios
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (Predicted - Groundtruth)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Coal Ratio Prediction Error Distribution', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主评估函数"""
    # 配置参数
    RAW_DATA_DIR = './raw_data'
    GROUNDTRUTH_DIR = './groundtruth'
    TEST_IDS = list(range(201, 237))  # 201-236
    
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_PATH = './checkpoints/best_model.pth'
    RESULTS_DIR = './results'
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 创建测试数据加载器
    _, test_loader = get_data_loaders(
        RAW_DATA_DIR,
        GROUNDTRUTH_DIR,
        [],  # 不需要训练集
        TEST_IDS,
        batch_size=BATCH_SIZE
    )
    
    model = get_model(model_name='unet', n_channels=3, n_classes=NUM_CLASSES)
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型检查点 (Epoch {checkpoint['epoch']}, Val IoU: {checkpoint['val_iou']:.4f})")
    else:
        print(f"警告: 检查点文件不存在: {CHECKPOINT_PATH}")
        print("使用未训练的模型进行评估")
    
    model = model.to(DEVICE)
    
    print(f"\n开始评估，设备: {DEVICE}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 全面评估
    print("\n计算评估指标...")
    metrics = evaluate_model(model, test_loader, DEVICE)
    
    print("\n评估结果:")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Coal Ratio MAE: {metrics['coal_ratio_mae']:.4f}")
    print(f"Coal Ratio RMSE: {metrics['coal_ratio_rmse']:.4f}")
    
    # 计算ROC和PR曲线
    print("\n计算ROC和PR曲线...")
    roc_auc, pr_auc = calculate_roc_pr_curves(model, test_loader, DEVICE, RESULTS_DIR)
    
    # 绘制煤占比对比图
    print("\n绘制煤占比对比图...")
    plot_coal_ratio_comparison(metrics, RESULTS_DIR)
    
    # 可视化预测结果
    print("\n生成可视化结果...")
    visualize_predictions(model, test_loader, DEVICE, os.path.join(RESULTS_DIR, 'visualizations'), num_samples=20)

    # 实验结果展示一：2×3 网格（测试集前 6 张），煤占比 + 处理耗时
    print("\n生成实验结果展示一...")
    visualize_experiment_display(
        model, test_loader, DEVICE,
        save_path=os.path.join(RESULTS_DIR, 'experiment_display.png'),
        num_samples=6, grid_shape=(2, 3)
    )

    # 实验结果展示一（指定图 1–9）：用 raw_data 的 1.jpg–9.jpg 重新算，3×3 网格
    CUSTOM_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    _, loader_custom = get_data_loaders(
        RAW_DATA_DIR, GROUNDTRUTH_DIR,
        [], CUSTOM_IDS,
        batch_size=1
    )
    if loader_custom is not None and len(loader_custom.dataset) > 0:
        print("\n生成实验结果展示一（图 1–9）...")
        visualize_experiment_display(
            model, loader_custom, DEVICE,
            save_path=os.path.join(RESULTS_DIR, 'experiment_display_1-9.png'),
            num_samples=9, grid_shape=(3, 3)
        )
    
    # 保存详细结果
    results_summary = {
        'iou': float(metrics['iou']),
        'pixel_accuracy': float(metrics['pixel_accuracy']),
        'coal_ratio_mae': float(metrics['coal_ratio_mae']),
        'coal_ratio_rmse': float(metrics['coal_ratio_rmse']),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'coal_ratios_pred': metrics['coal_ratios_pred'].tolist(),
        'coal_ratios_target': metrics['coal_ratios_target'].tolist()
    }
    
    import json
    with open(os.path.join(RESULTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n评估完成！结果已保存到 {RESULTS_DIR}")
    print(f"- ROC/PR曲线: {os.path.join(RESULTS_DIR, 'roc_pr_curves.png')}")
    print(f"- 煤占比对比图: {os.path.join(RESULTS_DIR, 'coal_ratio_comparison.png')}")
    print(f"- 误差分布图: {os.path.join(RESULTS_DIR, 'error_distribution.png')}")
    print(f"- 实验结果展示一: {os.path.join(RESULTS_DIR, 'experiment_display.png')}")
    print(f"- 实验结果展示一（图1–9）: {os.path.join(RESULTS_DIR, 'experiment_display_1-9.png')}")
    print(f"- 可视化结果: {os.path.join(RESULTS_DIR, 'visualizations')}")
    print(f"- 详细结果JSON: {os.path.join(RESULTS_DIR, 'evaluation_results.json')}")


if __name__ == '__main__':
    main()
