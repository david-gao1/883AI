"""
生成与常见作业示例类似的 2×2 图：
  左上 FrozenLake 8×8 训练曲线
  右上 max_a Q(s,a) 热力图
  左下 贪心策略箭头
  右下 Taxi-v3 训练曲线

依赖 q_learning_taxi 中的训练与工具函数。运行：
  uv run python q_learning_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_learning_taxi import _configure_matplotlib_cjk, moving_average, q_learning_train


# FrozenLake: 0 left, 1 down, 2 right, 3 up — 在「行向下增大」的网格坐标中的单位方向
_ACTION_VEC = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, -1.0)]


def _plot_training_curve(
    ax,
    returns: list,
    title: str,
    window: int = 100,
    ylabel: str = "Reward",
) -> None:
    eps = np.arange(1, len(returns) + 1)
    r = np.asarray(returns, dtype=np.float64)
    ma = moving_average(r, max(2, window))
    ax.plot(eps, r, color="#87CEEB", linewidth=0.35, alpha=0.9)
    ax.plot(
        eps,
        ma,
        color="#FF8C00",
        linewidth=2.0,
        label=f"Rolling mean ({window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_q_heatmap(ax, q: np.ndarray, desc: np.ndarray) -> None:
    nrow, ncol = desc.shape
    qmax = np.max(q, axis=1).reshape(nrow, ncol).astype(np.float64)
    hole = desc == b"H"
    qmax_display = qmax.copy()
    qmax_display[hole] = np.nan
    im = ax.imshow(qmax_display, cmap="YlOrRd", origin="upper", interpolation="nearest")
    ax.set_title(r"$\max_a Q(s,a)$")
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_greedy_arrows(ax, q: np.ndarray, desc: np.ndarray) -> None:
    nrow, ncol = desc.shape
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.set_title("Greedy policy (arrows)")

    for r in range(nrow):
        for c in range(ncol):
            cell = desc[r, c]
            if cell in (b"H",):
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor="0.75",
                        edgecolor="0.5",
                        linewidth=0.5,
                    )
                )
                continue
            s = r * ncol + c
            a = int(np.argmax(q[s]))
            dx, dy = _ACTION_VEC[a]
            x, y = float(c), float(r)
            ax.annotate(
                "",
                xy=(x + 0.32 * dx, y + 0.32 * dy),
                xytext=(x - 0.32 * dx, y - 0.32 * dy),
                arrowprops=dict(
                    arrowstyle="->",
                    color="royalblue",
                    lw=1.4,
                    mutation_scale=12,
                ),
                annotation_clip=False,
            )


def build_report_figure(
    fl_returns: list,
    fl_q: np.ndarray,
    fl_desc: np.ndarray,
    taxi_returns: list,
    ma_window: int,
) -> plt.Figure:
    _configure_matplotlib_cjk()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), layout="constrained")

    _plot_training_curve(
        axes[0, 0],
        fl_returns,
        "FrozenLake 8×8 — Training curve",
        window=ma_window,
    )
    _plot_q_heatmap(axes[0, 1], fl_q, fl_desc)
    _plot_greedy_arrows(axes[1, 0], fl_q, fl_desc)
    _plot_training_curve(
        axes[1, 1],
        taxi_returns,
        "Taxi-v3 — Training curve",
        window=ma_window,
    )

    fig.suptitle("Q-learning (tabular)", fontsize=14, y=1.02)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="2×2 Q-learning report figure")
    parser.add_argument("--fl-episodes", type=int, default=5000)
    parser.add_argument("--taxi-episodes", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ma-window", type=int, default=100)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--slippery",
        action="store_true",
        help="FrozenLake 使用随机打滑（默认 False，便于策略与热力图可读）",
    )
    args = parser.parse_args()

    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "q_learning_report_2x2.png"

    fl_env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=args.slippery,
    )
    fl_q, fl_returns = q_learning_train(
        fl_env,
        n_episodes=args.fl_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    fl_desc = np.asarray(fl_env.unwrapped.desc)
    fl_env.close()

    taxi_env = gym.make("Taxi-v3")
    taxi_q, taxi_returns = q_learning_train(
        taxi_env,
        n_episodes=args.taxi_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed + 7,
    )
    taxi_env.close()

    fig = build_report_figure(
        fl_returns,
        fl_q,
        fl_desc,
        taxi_returns,
        ma_window=max(2, args.ma_window),
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存: {out_path}")
    print(
        f"FrozenLake 末段回报均值(最后100局): {float(np.mean(fl_returns[-100:])):.3f}"
    )
    print(
        f"Taxi 末段回报均值(最后100局): {float(np.mean(taxi_returns[-100:])):.3f}"
    )


if __name__ == "__main__":
    main()
