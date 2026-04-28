"""
plot_results.py
===============
Plot training curves and save figures to results/.

Run:
    python plot_results.py
    python plot_results.py --stats logs/training_stats.json
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def moving_average(x: list, window: int = 20) -> np.ndarray:
    """Compute moving average over list x."""
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    pad    = np.full(window - 1, x[0])
    return np.convolve(np.concatenate([pad, x]), kernel, mode="valid")


def plot_training(stats_path: str = "logs/training_stats.json",
                  output_dir: str = "results") -> None:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(stats_path):
        print(f"  Stats file not found: {stats_path}")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    ep_rewards = stats.get("episode_rewards", [])
    ep_lengths = stats.get("episode_lengths", [])
    losses     = stats.get("losses", [])
    epsilons   = stats.get("epsilons", [])
    eval_log   = stats.get("eval_log", [])

    if not HAS_MPL:
        # ASCII fallback
        print("  [matplotlib not available — printing ASCII summary]")
        if ep_rewards:
            n = len(ep_rewards)
            print(f"  Episodes     : {n}")
            print(f"  Final reward : {ep_rewards[-1]:.1f}")
            window = min(100, n)
            print(f"  Avg last {window:3d} : {np.mean(ep_rewards[-window:]):.1f}")
        return

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("DQN Training Results — CartPole", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Episode Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    if ep_rewards:
        ax1.plot(ep_rewards, alpha=0.3, color="#4C9BE8", linewidth=0.8, label="Episode")
        ma = moving_average(ep_rewards, 20)
        ax1.plot(range(len(ma)), ma, color="#1A5FAD", linewidth=2, label="MA-20")
        ax1.axhline(475, color="green", linestyle="--", alpha=0.7, label="Solve (475)")
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    # 2. Training Loss
    ax2 = fig.add_subplot(gs[0, 1])
    if losses:
        step_idx = np.linspace(0, len(ep_rewards) if ep_rewards else 1, len(losses))
        ax2.plot(step_idx, losses, alpha=0.3, color="#E84C4C", linewidth=0.5)
        ma_loss = moving_average(losses, 200)
        ax2.plot(np.linspace(0, step_idx[-1], len(ma_loss)),
                 ma_loss, color="#A01010", linewidth=2, label="MA-200")
        ax2.set_title("Training Loss (MSE)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_yscale("log")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    # 3. Epsilon Decay
    ax3 = fig.add_subplot(gs[1, 0])
    if epsilons:
        ax3.plot(epsilons, color="#E8A14C", linewidth=1.5)
        ax3.set_title("Exploration ε Decay")
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("Epsilon")
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)

    # 4. Eval Rewards
    ax4 = fig.add_subplot(gs[1, 1])
    if eval_log:
        ep_x    = [e["episode"]      for e in eval_log]
        mean_r  = [e["mean_reward"]  for e in eval_log]
        std_r   = [e["std_reward"]   for e in eval_log]
        mean_r  = np.array(mean_r)
        std_r   = np.array(std_r)
        ax4.plot(ep_x, mean_r, color="#4CAF50", linewidth=2, marker="o",
                 markersize=4, label="Mean (greedy)")
        ax4.fill_between(ep_x, mean_r - std_r, mean_r + std_r,
                         alpha=0.2, color="#4CAF50")
        ax4.axhline(475, color="green", linestyle="--", alpha=0.7)
        ax4.set_title("Evaluation Reward (Greedy Policy)")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Mean Reward")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved → {out_path}")

    # ── Second figure: reward distribution ───────────────────────────
    if ep_rewards and len(ep_rewards) >= 20:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle("Reward Distribution Analysis", fontweight="bold")

        # Histogram
        axes[0].hist(ep_rewards, bins=30, color="#4C9BE8", edgecolor="white",
                     alpha=0.8, density=True)
        axes[0].axvline(np.mean(ep_rewards), color="red", linestyle="--",
                        label=f"Mean={np.mean(ep_rewards):.1f}")
        axes[0].set_title("Reward Distribution")
        axes[0].set_xlabel("Episode Reward")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Quartile box over training
        n_bins = min(10, len(ep_rewards) // 20)
        if n_bins >= 2:
            chunk_size = len(ep_rewards) // n_bins
            positions  = []
            data       = []
            for i in range(n_bins):
                chunk = ep_rewards[i * chunk_size: (i + 1) * chunk_size]
                positions.append(i * chunk_size + chunk_size // 2)
                data.append(chunk)
            axes[1].boxplot(data, positions=positions,
                            widths=chunk_size * 0.7, patch_artist=True,
                            boxprops=dict(facecolor="#4C9BE855"))
            axes[1].set_title("Reward Quartiles Over Training")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Episode Reward")
            axes[1].grid(True, alpha=0.3)

        dist_path = os.path.join(output_dir, "reward_distribution.png")
        plt.savefig(dist_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Reward distribution saved → {dist_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stats",  type=str, default="logs/training_stats.json")
    p.add_argument("--output", type=str, default="results")
    args = p.parse_args()
    plot_training(args.stats, args.output)
