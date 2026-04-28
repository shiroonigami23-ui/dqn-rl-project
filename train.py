"""
train.py
========
Full DQN training loop for CartPole.

Run:
    python train.py
    python train.py --episodes 500 --seed 7 --double_dqn

Features
--------
- Configurable via CLI args or direct import
- Progress bar (ASCII fallback if tqdm absent)
- Periodic evaluation rollouts (pure-greedy, no exploration)
- Checkpoint saving every N episodes and at completion
- JSON stats export for plotting
- Pretrained model warm-start support
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Make sure model/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

from model.environment   import CartPoleEnv
from model.dqn_agent     import DQNAgent, DQNConfig
from model.neural_network import NeuralNetwork


# ──────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────

def evaluate_agent(agent: DQNAgent, n_episodes: int = 10, seed: int = 999) -> dict:
    """Run agent in greedy mode, return stats dict."""
    env     = CartPoleEnv(seed=seed)
    rewards = []
    lengths = []
    for ep in range(n_episodes):
        obs    = env.reset(seed=seed + ep)
        total_r = 0.0
        length  = 0
        while True:
            action = agent.select_action_greedy(obs)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            length  += 1
            if term or trunc:
                break
        rewards.append(total_r)
        lengths.append(length)
    return {
        "mean_reward"  : float(np.mean(rewards)),
        "std_reward"   : float(np.std(rewards)),
        "max_reward"   : float(np.max(rewards)),
        "min_reward"   : float(np.min(rewards)),
        "mean_length"  : float(np.mean(lengths)),
    }


# ──────────────────────────────────────────────────────────────────────
# ASCII progress bar (no tqdm needed)
# ──────────────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, width: int = 30) -> str:
    pct  = current / total
    done = int(pct * width)
    bar  = "█" * done + "░" * (width - done)
    return f"[{bar}] {current}/{total} ({pct*100:.0f}%)"


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────

def train(
    n_episodes         : int   = 300,
    max_steps_per_ep   : int   = 500,
    seed               : int   = 42,
    checkpoint_dir     : str   = "checkpoints",
    log_dir            : str   = "logs",
    checkpoint_every   : int   = 50,
    eval_every         : int   = 25,
    eval_episodes      : int   = 10,
    pretrained_path    : str   = None,
    # DQN hyperparams
    hidden_sizes       : list  = None,
    lr                 : float = 1e-3,
    batch_size         : int   = 64,
    gamma              : float = 0.99,
    buffer_capacity    : int   = 100_000,
    min_buffer_size    : int   = 1_000,
    target_update_freq : int   = 100,
    eps_start          : float = 1.0,
    eps_end            : float = 0.01,
    eps_decay_steps    : int   = 10_000,
    double_dqn         : bool  = True,
) -> DQNAgent:

    hidden_sizes = hidden_sizes or [128, 128]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    # ── Build config ─────────────────────────────────────────────────
    cfg = DQNConfig(
        hidden_sizes       = hidden_sizes,
        learning_rate      = lr,
        batch_size         = batch_size,
        gamma              = gamma,
        buffer_capacity    = buffer_capacity,
        min_buffer_size    = min_buffer_size,
        target_update_freq = target_update_freq,
        eps_start          = eps_start,
        eps_end            = eps_end,
        eps_decay_steps    = eps_decay_steps,
        double_dqn         = double_dqn,
        seed               = seed,
    )
    print(cfg)

    # ── Create / load agent ──────────────────────────────────────────
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\n  🔄 Loading pretrained model from: {pretrained_path}")
        agent = DQNAgent.load_checkpoint(pretrained_path)
        # Reset exploration if fine-tuning
        agent.epsilon = max(agent.epsilon, 0.1)
    else:
        agent = DQNAgent(state_dim=4, n_actions=2, config=cfg)
        print(agent.summary())

    env = CartPoleEnv(seed=seed)

    # ── Tracking ─────────────────────────────────────────────────────
    eval_log      = []
    start_time    = time.time()
    best_reward   = -np.inf
    solve_episode = None           # episode where avg100 >= 475

    print("\n" + "=" * 65)
    print("  Starting DQN Training on CartPole")
    print("=" * 65)

    # ── Training loop ────────────────────────────────────────────────
    for episode in range(1, n_episodes + 1):
        obs          = env.reset(seed=seed + episode)
        ep_reward    = 0.0
        ep_length    = 0
        ep_losses    = []

        for step in range(max_steps_per_ep):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, terminated)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            ep_length += 1
            obs = next_obs

            if done:
                break

        agent.log_episode(ep_reward, ep_length)

        # ── Print progress ───────────────────────────────────────
        avg_loss   = float(np.mean(ep_losses)) if ep_losses else 0.0
        mean_r100  = agent.mean_reward(100)
        pb         = progress_bar(episode, n_episodes)

        if episode % 10 == 0 or episode <= 5:
            print(f"  Ep {episode:4d}/{n_episodes}  "
                  f"reward={ep_reward:6.1f}  avg100={mean_r100:6.1f}  "
                  f"ε={agent.epsilon:.3f}  loss={avg_loss:.4f}  "
                  f"steps={ep_length:3d}  buf={len(agent.buffer):6,}")

        # ── Check solve condition ────────────────────────────────
        if mean_r100 >= 475.0 and solve_episode is None and episode >= 100:
            solve_episode = episode
            print(f"\n  ★ SOLVED at episode {episode}! "
                  f"(avg100={mean_r100:.1f})\n")

        # ── Evaluation rollout ───────────────────────────────────
        if episode % eval_every == 0:
            eval_stats = evaluate_agent(agent, n_episodes=eval_episodes)
            eval_stats["episode"] = episode
            eval_stats["wall_time"] = round(time.time() - start_time, 1)
            eval_log.append(eval_stats)
            print(f"\n  ── Eval @ ep {episode:4d}: "
                  f"mean={eval_stats['mean_reward']:.1f} ± {eval_stats['std_reward']:.1f}  "
                  f"max={eval_stats['max_reward']:.0f} ──\n")

            if eval_stats["mean_reward"] > best_reward:
                best_reward = eval_stats["mean_reward"]
                best_path   = os.path.join(checkpoint_dir, "best_model.pkl")
                agent.save_checkpoint(best_path)

        # ── Periodic checkpoint ──────────────────────────────────
        if episode % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pkl")
            agent.save_checkpoint(ckpt_path)

    # ── Final save ───────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final_model.pkl")
    agent.save_checkpoint(final_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  Training complete!")
    print(f"  Episodes  : {n_episodes}")
    print(f"  Total steps: {agent.total_steps:,}")
    print(f"  Best eval reward: {best_reward:.1f}")
    print(f"  Solved at episode: {solve_episode or 'N/A'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 65)

    # ── Save stats JSON ──────────────────────────────────────────────
    stats_path = os.path.join(log_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "config"         : cfg.to_dict(),
            "episode_rewards": agent.stats["episode_rewards"],
            "episode_lengths": agent.stats["episode_lengths"],
            "losses"         : agent.stats["losses"][-5000:],   # last 5000
            "epsilons"       : agent.stats["epsilons"][-5000:],
            "eval_log"       : eval_log,
            "solve_episode"  : solve_episode,
            "best_reward"    : best_reward,
            "total_steps"    : agent.total_steps,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"  Stats saved → {stats_path}")

    return agent


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on CartPole")
    p.add_argument("--episodes",          type=int,   default=300)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--eps_decay_steps",   type=int,   default=10_000)
    p.add_argument("--target_update_freq",type=int,   default=100)
    p.add_argument("--double_dqn",        action="store_true", default=True)
    p.add_argument("--checkpoint_dir",    type=str,   default="checkpoints")
    p.add_argument("--log_dir",           type=str,   default="logs")
    p.add_argument("--pretrained",        type=str,   default=None,
                   help="Path to pretrained checkpoint for warm-start")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        n_episodes         = args.episodes,
        seed               = args.seed,
        lr                 = args.lr,
        batch_size         = args.batch_size,
        gamma              = args.gamma,
        eps_decay_steps    = args.eps_decay_steps,
        target_update_freq = args.target_update_freq,
        double_dqn         = args.double_dqn,
        checkpoint_dir     = args.checkpoint_dir,
        log_dir            = args.log_dir,
        pretrained_path    = args.pretrained,
    )
