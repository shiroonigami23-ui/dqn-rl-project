"""
evaluate.py
===========
Load a saved DQN agent and evaluate it on CartPole.

Run:
    python evaluate.py --model checkpoints/final_model.pkl
    python evaluate.py --model checkpoints/best_model.pkl --episodes 20
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.environment import CartPoleEnv
from model.dqn_agent   import DQNAgent


def evaluate(model_path: str, n_episodes: int = 20, seed: int = 1000,
             verbose: bool = True) -> dict:
    """
    Evaluate a saved DQN agent.

    Returns
    -------
    dict with mean/std/min/max reward and episode lengths
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    agent = DQNAgent.load_checkpoint(model_path)
    env   = CartPoleEnv(seed=seed)

    rewards = []
    lengths = []

    if verbose:
        print(f"\n  Evaluating: {model_path}")
        print(f"  Episodes  : {n_episodes}")
        print("-" * 55)

    for ep in range(1, n_episodes + 1):
        obs      = env.reset(seed=seed + ep)
        total_r  = 0.0
        length   = 0
        while True:
            action = agent.select_action_greedy(obs)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            length  += 1
            if term or trunc:
                break
        rewards.append(total_r)
        lengths.append(length)
        if verbose:
            status = "✓" if total_r >= 200 else "✗"
            print(f"  Ep {ep:3d}: reward={total_r:6.1f}  steps={length:4d}  {status}")

    stats = {
        "mean_reward"  : float(np.mean(rewards)),
        "std_reward"   : float(np.std(rewards)),
        "max_reward"   : float(np.max(rewards)),
        "min_reward"   : float(np.min(rewards)),
        "mean_length"  : float(np.mean(lengths)),
        "solve_rate"   : float(np.mean([r >= 475 for r in rewards])),
        "n_episodes"   : n_episodes,
    }

    if verbose:
        print("-" * 55)
        print(f"  Mean reward : {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
        print(f"  Max / Min   : {stats['max_reward']:.0f} / {stats['min_reward']:.0f}")
        print(f"  Solve rate  : {stats['solve_rate']*100:.0f}%  (reward ≥ 475)")
        print("-" * 55)

    return stats


def render_episode(model_path: str, seed: int = 42) -> None:
    """Render one episode with ASCII output."""
    agent = DQNAgent.load_checkpoint(model_path)
    env   = CartPoleEnv(seed=seed)
    obs   = env.reset(seed=seed)
    total = 0.0
    print("\n  ASCII Render:")
    print("  " + "=" * 50)
    while True:
        action = agent.select_action_greedy(obs)
        obs, r, term, trunc, info = env.step(action)
        total += r
        print(f"  {env.render()}")
        if term or trunc:
            break
    print("  " + "=" * 50)
    print(f"  Total reward: {total:.0f}  steps: {env.step_count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, default="checkpoints/final_model.pkl")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed",     type=int, default=1000)
    p.add_argument("--render",   action="store_true")
    args = p.parse_args()

    stats = evaluate(args.model, args.episodes, args.seed)

    if args.render:
        render_episode(args.model, seed=args.seed)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n  Results saved → results/eval_results.json")
