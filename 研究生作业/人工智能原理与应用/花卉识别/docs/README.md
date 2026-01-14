# 花卉识别项目

基于深度学习的花卉分类系统，使用ResNet18进行迁移学习。

## 项目结构

```
花卉识别/
├── data/              # 数据集目录
│   ├── train/        # 训练集
│   ├── val/          # 验证集
│   └── test/         # 测试集
├── config.py         # 配置文件
├── dataset.py        # 数据集模块
├── model.py          # 模型定义
├── trainer.py        # 训练器
├── evaluator.py      # 评估器
├── utils.py          # 工具函数
├── main.py           # 主程序入口
├── requirements.txt  # 依赖包
├── README.md         # 说明文档
└── 实验报告.md       # 实验报告
```

## 模块说明

- **config.py**: 配置类，集中管理所有超参数和路径
- **dataset.py**: 数据集类和数据变换类
- **model.py**: 模型定义类
- **trainer.py**: 训练器类，封装训练逻辑
- **evaluator.py**: 评估器类，封装评估逻辑
- **utils.py**: 工具函数（可视化、结果保存等）
- **main.py**: 主程序，组织整个训练流程

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 检查设备支持

**Mac ARM架构（M1/M2/M3）**：
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

**NVIDIA GPU（Linux/Windows）**：
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**注意**：Mac ARM架构会自动使用MPS（Metal Performance Shaders）进行GPU加速，无需CUDA。

## 使用方法

### 训练模型

```bash
python main.py
```

训练过程会：
1. 加载数据集
2. 训练模型（20个epoch）
3. 在验证集上评估
4. 保存最佳模型
5. 生成训练曲线和混淆矩阵

### 输出文件

训练完成后会生成以下文件：
- `flower_classifier.pth` - 训练好的模型权重
- `training_curves.png` - 训练曲线图
- `confusion_matrix.png` - 混淆矩阵图
- `results.json` - 详细评估结果

## 数据集

数据集包含5种花卉：
- Daisy（雏菊）
- Dandelion（蒲公英）
- Roses（玫瑰）
- Sunflowers（向日葵）
- Tulips（郁金香）

数据已按train/val/test划分好。

## 模型架构

- **基础模型**：ResNet18（ImageNet预训练）
- **输出类别**：5类
- **输入尺寸**：224×224
- **优化器**：Adam (lr=0.001)
- **学习率调度**：每7个epoch衰减为0.1倍

## 实验结果

运行训练脚本后，会在控制台输出：
- 每个epoch的训练和验证损失、准确率
- 测试集上的整体准确率
- 各类别的精确率、召回率、F1分数

详细结果保存在`results.json`中。

## 注意事项

1. 确保数据路径正确（`data/train`, `data/val`, `data/test`）
2. 如果内存不足，可以减小batch_size（在config.py中修改）
3. 训练时间取决于硬件配置：
   - **Mac ARM（M1/M2/M3）**：自动使用MPS GPU加速
   - **NVIDIA GPU**：使用CUDA加速
   - **CPU**：速度较慢，但可以正常运行
4. 首次运行会下载ResNet18的预训练权重
5. **Mac用户**：如果遇到多进程问题，config.py中`num_workers`已设为0

## 故障排除

**问题1：内存不足（CUDA/MPS out of memory）**
- 解决：减小batch_size（在config.py中修改，如改为16或8）

**问题2：Mac上多进程错误**
- 解决：config.py中`num_workers`已设为0，如果还有问题，确保使用Python 3.8+

**问题2：找不到数据文件**
- 解决：检查data目录路径是否正确

**问题3：依赖包安装失败**
- 解决：使用conda环境或虚拟环境

