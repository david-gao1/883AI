"""
自动报告生成器：读取 outputs/ 下所有实验的 metrics.json，
生成一份 Markdown 格式的汇总报告。

用法：uv run summarize_results.py
前提：先跑完 train.py，outputs/ 下有实验结果
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """把 pandas 表格转成 Markdown 表格字符串（用于写入 .md 报告）。"""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",          # 表头
        "| " + " | ".join(["---"] * len(headers)) + " |",  # 分隔线
    ]
    for row in df.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")    # 每行数据
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CIFAR-100 experiment results")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))                # 实验输出目录
    parser.add_argument("--report-path", type=Path, default=Path("reports/自动实验报告.md"))  # 报告保存路径
    return parser.parse_args()


def load_metrics(output_root: Path) -> list[dict]:
    """扫描 outputs/ 下所有子目录的 metrics.json，读取并返回列表。"""
    metrics_files = sorted(output_root.glob("*/metrics.json"))
    records = []
    for metrics_file in metrics_files:
        with metrics_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records.append(payload)
    return records


def main() -> None:
    args = parse_args()

    # 读取所有实验结果
    records = load_metrics(args.output_root)
    if not records:
        raise SystemExit("No experiment metrics found. Please run train.py first.")

    # 转成表格，按测试准确率从高到低排序
    df = pd.DataFrame(records)
    df = df.sort_values(by="test_acc", ascending=False)
    best = df.iloc[0]  # 取最优的那组实验

    # 拼接 Markdown 报告内容
    lines = [
        "# CIFAR-100 分类实验自动报告",
        "",
        "## 最优实验",
        "",
        f"- 模型：`{best['model']}`",
        f"- 优化器：`{best['optimizer']}`",
        f"- 学习率：`{best['learning_rate']}`",
        f"- 验证集最佳准确率：`{best['best_val_acc']:.4f}`",
        f"- 测试集准确率：`{best['test_acc']:.4f}`",
        f"- 测试集损失：`{best['test_loss']:.4f}`",
        "",
        "## 全部实验结果",
        "",
        dataframe_to_markdown(
            df[
                [
                    "model",
                    "optimizer",
                    "learning_rate",
                    "epochs",
                    "batch_size",
                    "subset_ratio",
                    "best_val_acc",
                    "test_loss",
                    "test_acc",
                ]
            ]
        ),
        "",
        "## 分析建议",
        "",
        "1. 对比 `MLP` 与 `CNN` 的测试准确率，说明卷积结构对图像局部特征提取更有效。",
        "2. 对比不同学习率下 loss 曲线的下降速度与稳定性，分析学习率过大或过小的现象。",
        "3. 对比 `SGD` 与 `Adam` 的收敛情况，结合实验结果说明两者优缺点。",
        "4. 将各实验目录中的 `loss_curve.png` 和 `prediction_examples.png` 插入报告正文完成可视化分析。",
    ]

    # 写入报告文件
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
