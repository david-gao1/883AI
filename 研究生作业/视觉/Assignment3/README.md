# 煤矸石图像分割与占比计算项目

基于CNN的煤矸石图像语义分割项目，用于自动计算图像中煤的占比。

## 项目概述

本项目实现了一个基于U-Net架构的**三分类语义分割**模型，能够：
- **区分煤与矸石**：输出三类——背景(0)、煤(1)、矸石(2)
- 自动分割煤矸石图像中的煤区域与矸石区域
- 计算每张图像中煤的占比（及矸石占比）
- 提供完整的评估指标和可视化结果（绿=煤，红=矸石）

## 项目结构

```
Assignment3/
├── data_loader.py          # 数据加载和预处理模块
├── model.py                # 模型定义（U-Net）
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── metrics.py              # 评估指标计算
├── main.py                 # 主程序入口
├── README.md               # 项目说明文档
├── 项目报告.md              # 完整项目报告
├── raw_data/               # 原始图像目录（1-236.jpg）
├── groundtruth/            # 标注图像目录（1-236.png）
├── checkpoints/            # 模型检查点保存目录
└── results/                # 结果输出目录
    ├── roc_pr_curves.png           # ROC和PR曲线
    ├── coal_ratio_comparison.png   # 煤占比对比图
    ├── error_distribution.png      # 误差分布图
    ├── visualizations/             # 可视化结果
    └── evaluation_results.json     # 详细评估结果
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- scikit-learn
- PIL/Pillow
- tqdm

### 安装依赖

**方法1：使用conda环境（推荐）**

```bash
# 激活conda环境
conda activate 883ai

# 安装依赖
pip install -r requirements.txt
```

**方法2：直接安装**

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow tqdm
```

**注意**：如果遇到 `ModuleNotFoundError: No module named 'torch'` 错误，请确保：
1. 已正确激活conda环境：`conda activate 883ai`
2. 或使用提供的启动脚本：`./run.sh --mode train --model unet --batch_size 8 --epochs 50 --lr 1e-4`

## 数据集

- **训练集**：图像 1–200（共 200 张）
- **测试集**：图像 201–236（共 36 张）
- **数据格式**：
  - 原始图像：JPG 格式，存储在 `raw_data/` 目录
  - 标注图像：PNG 格式（三分类 mask），存储在 `groundtruth/` 目录；**约定**：黑色=背景(0)，红色=煤(1)，绿色=矸石(2)。灰度值通常为 0、38、75，对应背景/煤/矸石；若标注为 0/1/2 则直接使用

**重要**：本项目已升级为三分类（背景/煤/矸石）。若之前保存的是二分类检查点，需**重新训练**后再评估；评估时模型会按 `n_classes=3` 加载。

## 使用方法

### 0. 环境设置（重要！）

**确保已激活conda环境**：

```bash
# 方法1：手动激活环境
conda activate 883ai
python main.py --mode train --model unet --batch_size 8 --epochs 50 --lr 1e-4

# 方法2：使用启动脚本（自动处理环境）
./run.sh --mode train --model unet --batch_size 8 --epochs 50 --lr 1e-4
```

### 1. 训练模型

```bash
# 使用默认参数训练
python main.py --mode train

# 自定义参数训练
python main.py --mode train --model unet --batch_size 8 --epochs 50 --lr 1e-4
```

参数说明：
- `--mode`: 运行模式（train/eval/both）
- `--model`: 模型类型（unet/simple_cnn）
- `--batch_size`: 批次大小（默认8）
- `--epochs`: 训练轮数（默认50）
- `--lr`: 学习率（默认1e-4）

### 2. 评估模型

```bash
python main.py --mode eval
```

评估结果将保存在`results/`目录：
- `roc_pr_curves.png`: ROC和PR曲线
- `coal_ratio_comparison.png`: 煤占比预测对比图
- `error_distribution.png`: 误差分布直方图
- `visualizations/`: 可视化预测结果
- `evaluation_results.json`: 详细评估指标

### 3. 训练+评估

```bash
python main.py --mode both
```

## 模型架构

### U-Net

- **编码器**：5个下采样块，通道数：64→128→256→512→1024
- **解码器**：4个上采样块，带跳跃连接
- **输出**：单通道概率图（经过sigmoid激活）

### 损失函数

组合损失：`L = 0.5 × BCE_Loss + 0.5 × Dice_Loss`

## 评估指标

### 分割精度
- **IoU** (Intersection over Union)
- **Pixel Accuracy** (像素准确率)

### 煤占比计算
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)

### 分类性能
- **ROC AUC**
- **PR AUC** (Average Precision)

## 结果示例

训练完成后运行 `python main.py --mode eval`，将生成/更新以下结果（均在 `results/` 下）：

| 路径 | 说明 |
|------|------|
| `checkpoints/training_history.png` | 训练历史：Loss、IoU、Accuracy 曲线 |
| `results/roc_pr_curves.png` | ROC 与 PR 曲线（ROC AUC、PR AUC） |
| `results/coal_ratio_comparison.png` | 煤占比预测 vs 真实值散点图（含 MAE、RMSE） |
| `results/error_distribution.png` | 煤占比预测误差分布直方图 |
| **`results/experiment_display.png`** | **实验结果展示一**：2×3 网格，每张图标注煤占比与处理耗时 (s)，原图叠加绿色分割边界与红/绿网格 |
| `results/visualizations/sample_1.png` … `sample_20.png` | 20 张测试样本：原图 / 标注 / 预测对比 |
| `results/evaluation_results.json` | 评估标量（IoU、Acc、MAE、RMSE、ROC AUC、PR AUC）及每张图的 `coal_ratios_pred` / `coal_ratios_target` |

重新执行评估会覆盖上述图片与 JSON，以当前 `checkpoints/best_model.pth` 更新结果。本次运行（Epoch 19, Val IoU 0.9384）已更新 `results/` 下所有图片与 `evaluation_results.json`。

## 项目报告

完整的项目报告请参见`项目报告.md`，包含：
- Abstract（摘要）
- Introduction（引言）
- Related Research（相关工作）
- Methods（方法）
- Results（实验结果）
- Conclusion（结论）

## 注意事项

1. **GPU支持**：如果有CUDA GPU，代码会自动使用GPU加速训练
2. **内存要求**：建议至少8GB内存，batch_size可根据内存调整
3. **训练时间**：在 CPU 上训练 20 个 epoch 约 47 分钟（每 epoch 约 2 分 20 秒）；GPU 或更多 epoch 时相应增加
4. **数据路径**：确保`raw_data/`和`groundtruth/`目录存在且包含数据

## 常见问题

### Q: 训练时出现内存不足错误
A: 减小`batch_size`参数，例如：`--batch_size 4`

### Q: 如何查看训练进度？
A: 训练过程中会显示进度条和实时指标，训练历史保存在`checkpoints/history.npy`

### Q: 模型保存在哪里？
A: 最佳模型保存在`checkpoints/best_model.pth`

### Q: 如何只评估特定模型？
A: 修改`evaluate.py`中的`CHECKPOINT_PATH`变量指向你的模型文件

### Q: 出现 `ModuleNotFoundError: No module named 'torch'` 错误
A: 这表示没有正确激活conda环境。解决方法：
   - 运行 `conda activate 883ai` 激活环境
   - 或使用 `./run.sh` 脚本自动处理环境
   - 确认当前Python路径：`which python` 应该指向conda环境的Python

### Q: 出现 OpenMP 错误（OMP: Error #179）
A: 这是macOS上的常见问题，启动脚本已自动处理。如果仍有问题，可以设置：
   ```bash
   export OMP_NUM_THREADS=1
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

## 训练记录（参考）

最近一次训练（修正标注二值化 mask > 0 后）：`python3 main.py --mode train --model unet --batch_size 8 --epochs 20 --lr 1e-4`（CPU）。  
- **训练**：Epoch 20 时 Train Loss≈0.21、Train IoU≈0.96、Val Loss≈0.29、**Val IoU=0.9384**、**Val Acc=0.9950**；最佳检查点 Epoch 19。  
- **评估**（`python3 main.py --mode eval`）：**测试集 IoU 0.9384**、**Pixel Accuracy 0.9950**、**Coal Ratio MAE 0.0028**、**RMSE 0.0033**、**ROC AUC 0.9992**、**PR AUC 0.9938**。  
详见 `项目报告.md` 第 4 节。

## 作者

本项目完成于2026年1月29日

## 许可证

本项目仅供学习和研究使用
