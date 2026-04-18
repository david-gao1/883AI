"""
实验主入口：遍历"模型 × 优化器 × 学习率"的所有组合，逐个运行训练和评估。

用法：
  完整实验：uv run train.py --epochs 10
  快速验证：uv run train.py --epochs 1 --subset-ratio 0.05

默认跑 2×2×3 = 12 组实验：
  模型:    [mlp, cnn]
  优化器:  [sgd, adam]
  学习率:  [0.1, 0.01, 0.001]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# 从 src 包中导入所有需要的函数（定义在 models.py 和 trainer.py 中）
from src import (
    build_model,
    build_optimizer,
    fit_model,
    get_cifar100_loaders,
    get_device,
    plot_curves,
    plot_predictions,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，所有实验配置都可以通过命令行调整。"""
    parser = argparse.ArgumentParser(description="CIFAR-100 classification homework runner")
    parser.add_argument("--data-root", type=Path, default=Path("data"))              # 数据集存放目录
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))          # 实验结果输出目录
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn"])                # 要跑哪些模型
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "adam"])            # 要试哪些优化器
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.1, 0.01, 0.001])  # 要试哪些学习率
    parser.add_argument("--epochs", type=int, default=10)                             # 训练轮数
    parser.add_argument("--batch-size", type=int, default=128)                        # 每批多少张图
    parser.add_argument("--num-workers", type=int, default=2)                         # 数据加载的并行进程数
    parser.add_argument("--subset-ratio", type=float, default=1.0)                    # 用多少比例的数据（1.0=全部）
    parser.add_argument("--val-ratio", type=float, default=0.1)                       # 从训练集切出多少比例做验证集
    parser.add_argument("--seed", type=int, default=42)                               # 随机种子
    parser.add_argument("--weight-decay", type=float, default=1e-4)                   # 权重衰减（L2 正则化）
    parser.add_argument("--momentum", type=float, default=0.9)                        # SGD 的动量系数
    parser.add_argument("--max-visualizations", type=int, default=12)                 # 预测可视化展示几张图
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)       # 固定随机种子，保证实验可复现
    device = get_device()     # 自动选 GPU 或 CPU

    # ===== 第一步：加载数据 =====
    # 下载 CIFAR-100，切分训练/验证/测试集，创建 DataLoader
    train_loader, val_loader, test_loader, class_names = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        subset_ratio=args.subset_ratio,
    )

    # ===== 第二步：生成所有实验组合 =====
    # 笛卡尔积：模型 × 优化器 × 学习率 → 比如 ("mlp", "sgd", 0.1)
    records: list[dict[str, float | str | int]] = []
    experiments = [
        (model_name, optimizer_name, learning_rate)
        for model_name in args.models
        for optimizer_name in args.optimizers
        for learning_rate in args.learning_rates
    ]

    # ===== 第三步：逐个运行实验 =====
    print(f"Running {len(experiments)} experiments on device={device}")
    for model_name, optimizer_name, learning_rate in tqdm(experiments, desc="experiments"):
        # 给这组实验起个名字，比如 "cnn_adam_lr0.001"
        experiment_name = f"{model_name}_{optimizer_name}_lr{learning_rate:g}"
        experiment_dir = args.output_root / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型（MLP 或 CNN）
        model = build_model(model_name)
        # 创建优化器（SGD 或 Adam）
        optimizer = build_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
        # 训练 + 评估：返回训练历史和最佳模型参数
        history, best_state = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
        )

        # ===== 第四步：保存本组实验的所有结果 =====
        # 保存最佳模型参数（.pt 文件）
        model_path = experiment_dir / "best_model.pt"
        torch.save(best_state, model_path)
        # 保存训练曲线图（loss + accuracy 随 epoch 的变化）
        plot_curves(history, experiment_dir / "loss_curve.png", title=experiment_name)
        # 保存测试集预测可视化（模型预测 vs 真实标签）
        plot_predictions(
            model=model,
            loader=test_loader,
            class_names=class_names,
            device=device,
            output_path=experiment_dir / "prediction_examples.png",
            max_images=args.max_visualizations,
        )

        # 记录这组实验的关键指标
        record = {
            "model": model_name,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "subset_ratio": args.subset_ratio,
            "best_val_acc": float(history["best_val_acc"]),
            "test_loss": float(history["test_loss"]),
            "test_acc": float(history["test_acc"]),
        }
        records.append(record)
        # 保存详细指标到 JSON（包含每个 epoch 的记录）
        save_json({**record, "history": history}, experiment_dir / "metrics.json")

    # ===== 第五步：汇总所有实验 =====
    # 把所有实验结果整理成一张表，按测试准确率从高到低排序
    summary = pd.DataFrame(records).sort_values(by="test_acc", ascending=False)
    summary_path = args.output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
