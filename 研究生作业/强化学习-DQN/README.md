# 强化学习第二次作业：DQN 打游戏

这个目录给出一个可直接运行的 DQN 作业实现，使用 `CartPole-v1` 作为实验环境。

## 目录说明

- `dqn_cartpole.py`：DQN 主程序，包含经验回放、目标网络、训练、评估、保存模型与出图
- `pyproject.toml`：依赖配置
- `results/`：运行后自动生成，用来保存模型和训练曲线

## 算法要点

DQN（Deep Q-Network）用神经网络近似动作价值函数 `Q(s, a)`，核心思路包括：

1. 用 `epsilon-greedy` 在探索和利用之间做平衡
2. 用经验回放池打乱样本，降低样本相关性
3. 用目标网络稳定 TD 目标，避免训练震荡

更新目标为：

```text
y = r + gamma * max_a' Q_target(s', a')    （终止状态时后项为 0）
```

损失函数为：

```text
L = SmoothL1Loss(Q_online(s, a), y)
```

## 运行方式

先进入当前目录安装依赖：

```bash
uv sync
```

开始训练：

```bash
uv run python dqn_cartpole.py
```

自定义训练回合数：

```bash
uv run python dqn_cartpole.py --episodes 500 --eval-episodes 30
```

训练完成后会在 `results/` 下生成：

- `dqn_cartpole_model.pt`
- `dqn_cartpole_curve.png`

## 常用参数

- `--episodes`：训练回合数
- `--lr`：学习率
- `--batch-size`：每次更新的样本数
- `--epsilon-start` / `--epsilon-end`：探索率起止值
- `--epsilon-decay-steps`：探索率衰减步数
- `--target-update-interval`：目标网络同步间隔
- `--render-eval`：训练后渲染评估过程

## 结果解读

如果训练正常，`CartPole-v1` 的训练回报会逐步上升，贪心评估的平均回报通常也会明显提高。训练曲线图包含三部分：

1. 每回合回报与滑动平均
2. 每回合平均 TD 损失
3. epsilon 探索率衰减过程

## 可直接写进作业报告的描述

本实验使用 DQN 算法在 `CartPole-v1` 环境中训练智能体。实现中采用了经验回放机制和目标网络机制，以提高样本利用效率并稳定训练过程。训练结束后，利用贪心策略进行多回合评估，并输出平均回报、平均步数以及训练曲线图，用于分析 DQN 的收敛效果。
