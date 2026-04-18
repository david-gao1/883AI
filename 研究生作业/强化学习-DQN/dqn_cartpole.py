"""
使用 DQN 在 CartPole-v1 环境上训练智能体。

运行方式：
    uv run python dqn_cartpole.py

常用参数：
    uv run python dqn_cartpole.py --episodes 400 --eval-episodes 20
    uv run python dqn_cartpole.py --render-eval

输出：
    - results/dqn_cartpole_curve.png：训练曲线
    - results/dqn_cartpole_model.pt：模型参数
"""

from __future__ import annotations

import argparse
import os
import platform
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = PROJECT_DIR / ".mplconfig"
XDG_CACHE_DIR = PROJECT_DIR / ".cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_matplotlib_cjk() -> None:
    """让 matplotlib 优先尝试常见中文字体，避免图中文字乱码。"""
    system = platform.system()
    if system == "Darwin":
        candidates = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
    elif system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei"]
    else:
        candidates = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback"]
    plt.rcParams["font.sans-serif"] = candidates + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """计算滑动平均，用于平滑训练曲线。"""
    if window <= 1:
        return values.astype(np.float64).copy()
    result = np.full(len(values), np.nan, dtype=np.float64)
    cumsum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    result[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return result


def select_device(device_name: str) -> torch.device:
    """根据参数自动选择计算设备。"""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DQNConfig:
    """DQN 训练配置。"""

    env_name: str = "CartPole-v1"
    seed: int = 42
    episodes: int = 400
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 20000
    min_buffer_size: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000
    train_frequency: int = 1
    target_update_interval: int = 200
    eval_episodes: int = 20
    max_steps_per_episode: int = 500
    ma_window: int = 20
    device_name: str = "auto"
    plot_dir: Path = PROJECT_DIR / "results"

    @property
    def device(self) -> torch.device:
        return select_device(self.device_name)


class ReplayBuffer:
    """经验回放池。"""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=device
        )
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )


class QNetwork(nn.Module):
    """Q 网络：输入状态，输出每个动作的 Q 值。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN 智能体，维护在线网络和目标网络。"""

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig) -> None:
        self.config = config
        self.device = config.device
        self.online_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.get_action_dim())
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_action_dim(self) -> int:
        last_layer = self.online_net.net[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("QNetwork 的最后一层应为线性层。")
        return last_layer.out_features

    def update(self, replay_buffer: ReplayBuffer) -> float:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = replay_buffer.sample(self.config.batch_size, self.device)

        q_values = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())


def make_env(config: DQNConfig, render_mode: str | None = None) -> gym.Env:
    """创建 Gymnasium 环境。"""
    return gym.make(config.env_name, render_mode=render_mode)


def set_seed(seed: int) -> None:
    """统一设置随机种子，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_step(step: int, config: DQNConfig) -> float:
    """线性衰减 epsilon。"""
    progress = min(1.0, step / max(1, config.epsilon_decay_steps))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def evaluate_policy(agent: DQNAgent, config: DQNConfig, render: bool = False) -> Tuple[float, float]:
    """用贪心策略评估当前模型表现。"""
    env = make_env(config, render_mode="human" if render else None)
    returns: List[float] = []
    lengths: List[int] = []
    for episode in range(config.eval_episodes):
        state, _ = env.reset(seed=10000 + config.seed + episode)
        state = np.asarray(state, dtype=np.float32)
        episode_return = 0.0
        steps = 0
        done = False

        while not done and steps < config.max_steps_per_episode:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = np.asarray(next_state, dtype=np.float32)
            episode_return += reward
            steps += 1
            done = terminated or truncated

        returns.append(episode_return)
        lengths.append(steps)

    env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


def save_training_plot(
    returns: List[float],
    losses: List[float],
    epsilons: List[float],
    eval_return: float,
    eval_length: float,
    out_path: Path,
    ma_window: int,
) -> None:
    """保存训练曲线图。"""
    configure_matplotlib_cjk()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    returns_np = np.asarray(returns, dtype=np.float64)
    losses_np = np.asarray(losses, dtype=np.float64)
    epsilons_np = np.asarray(epsilons, dtype=np.float64)
    episodes = np.arange(1, len(returns) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")

    axes[0].plot(episodes, returns_np, color="#87CEEB", linewidth=0.8, label="每回合回报")
    axes[0].plot(
        episodes,
        moving_average(returns_np, max(2, ma_window)),
        color="#FF8C00",
        linewidth=2.0,
        label=f"{ma_window} 回合滑动平均",
    )
    axes[0].set_title("DQN 训练回报（CartPole-v1）")
    axes[0].set_xlabel("回合")
    axes[0].set_ylabel("回报")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[0].text(
        0.02,
        0.98,
        f"贪心评估平均回报: {eval_return:.2f}\n贪心评估平均步数: {eval_length:.2f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.85},
    )

    axes[1].plot(episodes, losses_np, color="#C44E52", linewidth=1.2)
    axes[1].set_title("每回合平均 TD 损失")
    axes[1].set_xlabel("回合")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, epsilons_np, color="#4C72B0", linewidth=1.5)
    axes[2].set_title("探索率 epsilon 衰减")
    axes[2].set_xlabel("回合")
    axes[2].set_ylabel("epsilon")
    axes[2].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_dqn(config: DQNConfig) -> Tuple[DQNAgent, List[float], List[float], List[float]]:
    """训练 DQN 智能体并返回训练日志。"""
    set_seed(config.seed)
    env = make_env(config)
    env.action_space.seed(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    replay_buffer = ReplayBuffer(config.replay_capacity)

    episode_returns: List[float] = []
    episode_losses: List[float] = []
    episode_epsilons: List[float] = []
    global_step = 0

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        state = np.asarray(state, dtype=np.float32)
        done = False
        total_reward = 0.0
        losses_this_episode: List[float] = []
        epsilon = epsilon_by_step(global_step, config)

        for _ in range(config.max_steps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = terminated or truncated

            replay_buffer.add(state, action, float(reward), next_state, done)
            state = next_state
            total_reward += reward
            global_step += 1
            epsilon = epsilon_by_step(global_step, config)

            if (
                len(replay_buffer) >= config.min_buffer_size
                and global_step % config.train_frequency == 0
            ):
                loss = agent.update(replay_buffer)
                losses_this_episode.append(loss)

            if global_step % config.target_update_interval == 0:
                agent.sync_target()

            if done:
                break

        episode_returns.append(total_reward)
        episode_losses.append(
            float(np.mean(losses_this_episode)) if losses_this_episode else float("nan")
        )
        episode_epsilons.append(epsilon)

        if episode % 20 == 0 or episode == 1 or episode == config.episodes:
            recent = episode_returns[-20:]
            print(
                f"Episode {episode:4d}/{config.episodes} | "
                f"最近{len(recent):2d}局平均回报: {np.mean(recent):7.2f} | "
                f"epsilon: {epsilon:6.3f} | "
                f"buffer: {len(replay_buffer):5d}"
            )

    env.close()
    agent.sync_target()
    return agent, episode_returns, episode_losses, episode_epsilons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=400, help="训练回合数")
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层宽度")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--replay-capacity", type=int, default=20000, help="经验池容量")
    parser.add_argument("--min-buffer-size", type=int, default=1000, help="开始训练前经验池最小容量")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="最终探索率")
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=8000, help="epsilon 线性衰减步数"
    )
    parser.add_argument("--train-frequency", type=int, default=1, help="每多少步训练一次")
    parser.add_argument(
        "--target-update-interval", type=int, default=200, help="每多少步同步一次目标网络"
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="贪心评估回合数")
    parser.add_argument("--max-steps", type=int, default=500, help="每回合最大步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ma-window", type=int, default=20, help="滑动平均窗口")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="训练设备",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="结果输出目录",
    )
    parser.add_argument("--render-eval", action="store_true", help="评估时渲染一局")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNConfig(
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_buffer_size=args.min_buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        train_frequency=args.train_frequency,
        target_update_interval=args.target_update_interval,
        eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        ma_window=args.ma_window,
        device_name=args.device,
        plot_dir=args.plot_dir,
    )

    print(f"环境: {config.env_name}")
    print(f"设备: {config.device}")
    print(f"训练回合数: {config.episodes}")
    print("-" * 60)

    agent, returns, losses, epsilons = train_dqn(config)
    mean_eval_return, mean_eval_length = evaluate_policy(
        agent, config, render=args.render_eval
    )

    config.plot_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.plot_dir / "dqn_cartpole_model.pt"
    plot_path = config.plot_dir / "dqn_cartpole_curve.png"
    torch.save(agent.online_net.state_dict(), model_path)
    save_training_plot(
        returns=returns,
        losses=losses,
        epsilons=epsilons,
        eval_return=mean_eval_return,
        eval_length=mean_eval_length,
        out_path=plot_path,
        ma_window=config.ma_window,
    )

    print(f"贪心评估平均回报: {mean_eval_return:.2f}")
    print(f"贪心评估平均步数: {mean_eval_length:.2f}")
    print(f"最近20局平均训练回报: {np.mean(returns[-20:]):.2f}")
    print(f"模型已保存: {model_path}")
    print(f"曲线已保存: {plot_path}")


if __name__ == "__main__":
    main()
