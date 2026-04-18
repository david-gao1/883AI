from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

# ============================================================
# 环境配置：让 matplotlib 的缓存目录指向项目内，避免权限问题
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# CIFAR-100 数据集的全局像素均值和标准差（预先统计好的常量）
# 标准化时会用到：pixel = (pixel - mean) / std
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed: int) -> None:
    """固定所有随机种子，保证实验可复现。"""
    random.seed(seed)           # Python 内置随机数
    np.random.seed(seed)        # NumPy 随机数
    torch.manual_seed(seed)     # PyTorch CPU 随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU 随机数


def get_device() -> torch.device:
    """自动选择计算设备：优先 Mac GPU (mps) → NVIDIA GPU (cuda) → CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _subset_dataset(dataset: Any, subset_ratio: float, seed: int) -> Any:
    """从数据集中随机取一个子集（用于快速验证时只取一小部分数据）。"""
    if subset_ratio >= 1.0:
        return dataset  # 比例为 1 就是用全部数据
    subset_size = max(1, int(len(dataset) * subset_ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


# ============================================================
# 数据加载
# ============================================================

def get_cifar100_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float = 0.1,
    subset_ratio: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    下载 CIFAR-100 并创建三个数据加载器（训练/验证/测试）。

    返回: (train_loader, val_loader, test_loader, 100个类别名列表)
    """

    # ---------- 数据预处理 ----------

    # 训练集的预处理：加了数据增强（随机裁剪+随机翻转），让模型见到更多变体
    train_transform = transforms.Compose(
        [
            # 先在四周补 4 像素，再随机裁剪回 32×32，相当于随机平移
            transforms.RandomCrop(32, padding=4),
            # 50% 概率水平翻转（左右镜像）
            transforms.RandomHorizontalFlip(),
            # PIL 图片 → PyTorch 张量，像素值从 [0,255] 变为 [0,1]
            transforms.ToTensor(),
            # 标准化：减均值、除标准差，把分布拉到均值 0 附近
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    # 验证集和测试集的预处理：不做增强，只做标准化，保证评估结果稳定
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    # ---------- 加载数据集 ----------

    # download=True：第一次运行时自动下载
    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    eval_train_dataset = datasets.CIFAR100(root=data_root, train=True, download=False, transform=eval_transform)
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_transform)

    # ---------- 切分训练集和验证集 ----------

    # 从 50000 张训练图里切出 10%（5000 张）做验证集
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    # 如果只想用一部分数据（快速验证），这里会截取子集
    train_subset = _subset_dataset(train_subset, subset_ratio, seed)
    val_subset = _subset_dataset(val_subset, subset_ratio, seed + 1)
    test_dataset = _subset_dataset(test_dataset, subset_ratio, seed + 2)

    # ---------- 创建 DataLoader ----------

    # DataLoader 负责：把数据分成一批一批（batch）、打乱顺序、多进程预取
    pin_memory = torch.cuda.is_available()  # 用 GPU 时开启，加速数据搬运
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,    # 每批 128 张图
        shuffle=True,             # 训练集要打乱顺序，防止模型记住数据的排列
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,            # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # eval_train_dataset.classes 是一个列表，包含 100 个类别的英文名
    return train_loader, val_loader, test_loader, eval_train_dataset.classes


# ============================================================
# 优化器
# ============================================================

def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    根据名字创建优化器。

    - SGD：经典梯度下降 + 动量，需要精心调学习率
    - Adam：自适应学习率，开箱即用
    - weight_decay：权重衰减，防止参数过大（L2 正则化）
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),       # 告诉优化器要更新哪些参数
            lr=learning_rate,         # 学习率：每步走多大
            momentum=momentum,        # 动量：记住之前的方向，不容易被小坑困住
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# ============================================================
# 训练/评估核心逻辑
# ============================================================

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """
    跑一个 epoch（完整过一遍数据集）。

    - 传了 optimizer → 训练模式（会更新参数）
    - 没传 optimizer → 评估模式（只算指标，不更新参数）

    返回: (平均损失, 准确率)
    """
    is_train = optimizer is not None
    # model.train(True)：开启 Dropout 和 BatchNorm 的训练行为
    # model.train(False) 即 model.eval()：关闭 Dropout，BatchNorm 用全局统计量
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 训练时计算梯度，评估时关闭梯度（省内存、加速）
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            # 把数据搬到 GPU/CPU 上（和模型在同一设备）
            inputs = inputs.to(device)    # 一批图片，形状 [batch, 3, 32, 32]
            targets = targets.to(device)  # 一批标签，形状 [batch]，每个值是 0-99 的类别编号

            if is_train:
                # 清空上一步残留的梯度（PyTorch 默认会累加梯度）
                optimizer.zero_grad(set_to_none=True)

            # ① 前向传播：图片送入模型，得到 100 个类别的得分
            logits = model(inputs)        # 形状 [batch, 100]
            # ② 计算损失：预测得分 vs 真实标签
            loss = criterion(logits, targets)

            if is_train:
                # ③ 反向传播：从损失出发，计算每个参数的梯度
                loss.backward()
                # ④ 更新参数：优化器根据梯度调整模型权重
                optimizer.step()

            # 累计统计量（用于最后算平均值）
            total_loss += loss.item() * inputs.size(0)  # loss.item() 是这一批的平均损失
            # argmax(dim=1)：取 100 个得分中最大的那个的下标，作为预测类别
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, list[float] | float], dict[str, torch.Tensor]]:
    """
    完整的训练流程：训练多个 epoch，每个 epoch 结束后在验证集上评估，
    记录最好的模型，最后在测试集上做最终评估。

    返回: (训练历史记录, 最佳模型参数)
    """
    # 交叉熵损失：分类任务的标配损失函数
    criterion = nn.CrossEntropyLoss()

    # history 记录每个 epoch 的指标，用于画曲线
    history: dict[str, list[float] | float] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_state: dict[str, torch.Tensor] = {}
    best_val_acc = -1.0

    model.to(device)  # 把模型搬到 GPU/CPU

    for _ in range(epochs):
        # 训练一个 epoch（传了 optimizer → 会更新参数）
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        # 验证一个 epoch（没传 optimizer → 只评估，不更新）
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 如果这个 epoch 的验证准确率是目前最好的，就保存模型参数
        # （Early Stopping 思想：最后一轮不一定最好，保留验证集上最优的那一轮）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    # 训练结束，加载验证集上最好的那一轮参数
    model.load_state_dict(best_state)
    # 用最好的模型在测试集上做最终评估
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, device)
    history["best_val_acc"] = best_val_acc
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    return history, best_state


# ============================================================
# 可视化
# ============================================================

def plot_curves(history: dict[str, list[float] | float], output_path: str | Path, title: str) -> None:
    """画训练曲线：左图是 loss 随 epoch 变化，右图是 accuracy 随 epoch 变化。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    # 左图：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()

    # 右图：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _denormalize(image: torch.Tensor) -> torch.Tensor:
    """把标准化后的图片还原回 [0,1] 范围，用于可视化展示。"""
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    image = image.cpu() * std + mean       # 逆标准化：pixel = pixel * std + mean
    return image.clamp(0.0, 1.0)           # 裁剪到 [0,1]，防止数值溢出


def plot_predictions(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_path: str | Path,
    max_images: int = 12,
) -> None:
    """从测试集取几张图，展示模型的预测结果 vs 真实标签。"""
    model.eval()  # 切换到评估模式（关闭 Dropout）
    images_to_show: list[torch.Tensor] = []
    titles: list[str] = []

    with torch.no_grad():  # 不需要梯度，省内存
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 取 100 个得分中最大的下标作为预测类别
            predictions = model(inputs).argmax(dim=1)

            for image, target, prediction in zip(inputs, targets, predictions):
                # 还原图片并记录预测/真实标签
                images_to_show.append(_denormalize(image))
                titles.append(
                    f"pred: {class_names[prediction.item()]}\ntrue: {class_names[target.item()]}"
                )
                if len(images_to_show) >= max_images:
                    break
            if len(images_to_show) >= max_images:
                break

    # 画网格图
    columns = 3
    rows = int(np.ceil(len(images_to_show) / columns))
    plt.figure(figsize=(12, 4 * rows))
    for index, (image, title) in enumerate(zip(images_to_show, titles), start=1):
        plt.subplot(rows, columns, index)
        # PyTorch 的图片格式是 [通道, 高, 宽]，matplotlib 需要 [高, 宽, 通道]
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.axis("off")
        plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    """把字典保存为 JSON 文件（用于记录每组实验的指标）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
