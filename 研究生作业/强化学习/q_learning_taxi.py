"""
Taxi-v3 上的表格 Q-learning（Gymnasium 离散环境，等价于“格子世界”类任务）。

更新（off-policy TD）：
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
终止状态不引导：若 s' 为终止，则 max_a' Q(s',a') 取 0。

参考：Gymnasium 文档 https://gymnasium.org.cn/
环境：Taxi-v3（Toy Text），观测离散 500，动作离散 6。

依赖：在本目录执行 `uv sync`，运行 `uv run python q_learning_taxi.py`。
训练结束会在 `results/` 下保存回报曲线图（可用 `--plot-dir` / `--no-plot` 调整）。
与课件类似的四宫格图（FrozenLake 8×8 + Taxi）：`uv run python q_learning_report.py`。
概念主线见 `什么是Q-learning.md`，代码逐段对照见 `代码讲解与强化学习理论.md`。
"""

from __future__ import annotations

import argparse
import platform
import random
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _configure_matplotlib_cjk() -> None:
    """让 matplotlib 优先用系统里的中文字体，避免图上中文变方框。"""
    system = platform.system()
    if system == "Darwin":
        candidates = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
    elif system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei"]
    else:
        candidates = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback"]
    plt.rcParams["font.sans-serif"] = candidates + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def epsilon_greedy(
    q_row: np.ndarray, n_actions: int, epsilon: float, rng: random.Random
) -> int:
    """ε-greedy：以概率 epsilon 随机探索，否则在当前 Q 行里贪心选最大值对应动作。"""
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    return int(np.argmax(q_row))


def q_learning_train(
    env: gym.Env,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    seed: int,
) -> Tuple[np.ndarray, list]:
    """表格 Q-learning 训练。

    返回:
        q: 形状 (状态数, 动作数)，Taxi-v3 下为 (500, 6)。
        episode_returns: 每局累计 reward，用于画曲线，不参与 TD 公式。
    """
    rng = random.Random(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # 初值全 0；也可改为小随机数，全 0 最常见
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    episode_returns: list = []

    epsilon = epsilon_start
    for ep in range(n_episodes):
        # 新一局：state 是环境给的「现场」编号，用作 q 的行下标
        state, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            # 按当前行 q[state] 选动作（带探索）
            action = epsilon_greedy(q[state], n_actions, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD 目标：非结束时 r + γ·max_a Q(s',a)；结束时不再 bootstrap，只用 r
            if done:
                target = reward
            else:
                target = reward + gamma * float(np.max(q[next_state]))

            # Q(s,a) ← Q(s,a) + α * (目标 − Q(s,a))，α 控制「挪一小步」
            q[state, action] += alpha * (target - q[state, action])
            ep_return += reward
            state = next_state

        episode_returns.append(ep_return)
        # 本局结束：略减探索率，但不低于 epsilon_end
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q, episode_returns


def evaluate_greedy(
    env: gym.Env, q: np.ndarray, n_episodes: int, seed: int
) -> Tuple[float, float]:
    """用纯贪心（每步 argmax Q）跑若干局，看训练后的策略平均回报与平均步数。"""
    returns = []
    lengths = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=10000 + seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = int(np.argmax(q[state]))  # 无随机，只选当前行最大列
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
        returns.append(total)
        lengths.append(steps)
    return float(np.mean(returns)), float(np.mean(lengths))


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """滑动平均：平滑每局回报曲线，前 window-1 个点为 nan。"""
    if window < 2:
        return arr.astype(np.float64).copy()
    out = np.full(len(arr), np.nan, dtype=np.float64)
    c = np.cumsum(np.insert(arr.astype(np.float64), 0, 0.0))
    out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def save_result_plots(
    episode_returns: List[float],
    plot_dir: Path,
    mean_eval_return: float,
    mean_eval_len: float,
    window: int = 100,
) -> Path:
    """保存训练曲线图：原始每局回报 + 滑动平均，并标注贪心评估摘要。"""
    _configure_matplotlib_cjk()
    plot_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(episode_returns) + 1)
    returns = np.asarray(episode_returns, dtype=np.float64)
    ma = moving_average(returns, window)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.plot(episodes, returns, color="#87CEEB", linewidth=0.4, alpha=0.9, label="每回合回报")
    ax.plot(episodes, ma, color="#FF8C00", linewidth=2.0, label=f"{window} 回合滑动平均")
    ax.set_xlabel("回合")
    ax.set_ylabel("回报")
    ax.set_title("Q-learning 训练曲线（Taxi-v3）")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    summary = (
        f"贪心评估（独立回合）\n"
        f"平均回报: {mean_eval_return:.2f}\n"
        f"平均步数: {mean_eval_len:.1f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.85},
    )
    path = plot_dir / "q_learning_taxi_results.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """流程：创建 Taxi-v3 → 训练得 q → 贪心评估 → 打印指标 → 可选保存曲线图 / 渲染一局。"""
    parser = argparse.ArgumentParser(description="Q-learning on Taxi-v3 (Gymnasium)")
    parser.add_argument("--episodes", type=int, default=20000, help="训练回合数")
    parser.add_argument("--alpha", type=float, default=0.1, help="步长 / 学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=1.0, help="初始 epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="epsilon 下限")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="每回合 epsilon 乘数")
    parser.add_argument("--eval-episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-eval", action="store_true", help="评估时用 human 渲染一局")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="保存曲线图的目录（默认 ./results）",
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    parser.add_argument("--ma-window", type=int, default=100, help="滑动平均窗口（回合）")
    args = parser.parse_args()

    # 训练阶段：用 ε-greedy 与环境交互，更新 q
    train_env = gym.make("Taxi-v3")
    q, returns = q_learning_train(
        train_env,
        n_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    train_env.close()

    # 评估阶段：不探索，只看当前 q 的贪心表现
    eval_env = gym.make("Taxi-v3")
    mean_ret, mean_len = evaluate_greedy(eval_env, q, args.eval_episodes, args.seed)
    eval_env.close()

    print(f"训练回合: {args.episodes}")
    print(f"评估 ({args.eval_episodes} 局, 贪心策略): 平均回报 = {mean_ret:.2f}, 平均步数 = {mean_len:.1f}")
    tail = min(100, len(returns))
    print(f"最近 {tail} 局训练回报均值: {float(np.mean(returns[-tail:])):.2f}")

    if not args.no_plot:
        out = save_result_plots(
            returns,
            args.plot_dir,
            mean_ret,
            mean_len,
            window=max(2, args.ma_window),
        )
        print(f"已保存图片: {out}")

    if args.render_eval:
        # 弹窗演示一局贪心过程（需图形界面）
        render_env = gym.make("Taxi-v3", render_mode="human")
        state, _ = render_env.reset(seed=999)
        done = False
        while not done:
            action = int(np.argmax(q[state]))
            state, _, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
        render_env.close()


if __name__ == "__main__":
    main()
