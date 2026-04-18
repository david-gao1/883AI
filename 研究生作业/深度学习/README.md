# CIFAR-100 分类作业

本目录用于完成深度学习课程中的 `CIFAR-100` 图像分类实验，覆盖以下要求：

- 分别使用 `MLP` 与 `CNN` 进行图像分类
- 对比不同学习率、不同优化器对模型表现的影响
- 绘制训练集与验证集的 `loss/accuracy` 曲线
- 生成测试集预测可视化图，支撑实验分析

## 目录结构

```text
深度学习/
├── pyproject.toml
├── train.py
├── summarize_results.py
├── src/
│   ├── __init__.py
│   ├── models.py
│   └── trainer.py
├── outputs/              # 每组实验输出
├── reports/
│   ├── 实验报告.md
│   └── 自动实验报告.md
└── figures/              # 可选，手动补充图片
```

## 环境安装

推荐使用 `uv`：

```bash
uv sync
```

如果你使用 `pip`，可手动安装：

```bash
pip install matplotlib numpy pandas torch torchvision tqdm
```

## 运行实验

完整实验：

```bash
uv run train.py --epochs 10 --batch-size 128 --models mlp cnn --optimizers sgd adam --learning-rates 0.1 0.01 0.001
```

快速验证：

```bash
uv run train.py --epochs 1 --subset-ratio 0.05 --batch-size 64 --models mlp cnn --optimizers adam --learning-rates 0.001
```

生成自动报告：

```bash
uv run summarize_results.py
```

## 输出说明

- `outputs/<实验名>/metrics.json`：每轮训练记录和最终指标
- `outputs/<实验名>/loss_curve.png`：训练/验证曲线
- `outputs/<实验名>/prediction_examples.png`：测试集预测可视化
- `outputs/<实验名>/best_model.pt`：最佳模型参数
- `outputs/summary.csv`：全部实验汇总
- `reports/自动实验报告.md`：根据实验结果自动生成的报告摘要

## 交作业建议

1. 运行完整实验，保留 `outputs/` 中的曲线和预测图。
2. 在 `reports/实验报告.md` 中补充你的实验结论与截图。
3. 按要求打包为 `学号-姓名-课程项目一.zip` 后提交。
