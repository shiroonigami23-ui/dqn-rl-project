"""
pretrain.py
===========
Generate a pretrained DQN checkpoint using supervised pre-training.

Strategy
--------
A hand-crafted heuristic policy (optimal for CartPole) is used to
generate demonstration trajectories.  The Q-network is then trained
via supervised learning to imitate the value function of this expert,
giving a strong warm-start before RL fine-tuning begins.

This is sometimes called "Imitation Learning" or "Behavioural Cloning"
and dramatically accelerates convergence.

Run:
    python pretrain.py
Output:
    checkpoints/pretrained_model.pkl
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.environment    import CartPoleEnv
from model.neural_network import NeuralNetwork
from model.replay_buffer  import ReplayBuffer
from model.dqn_agent      import DQNAgent, DQNConfig


# ──────────────────────────────────────────────────────────────────────
# Heuristic expert policy
# ──────────────────────────────────────────────────────────────────────

def expert_policy(state: np.ndarray) -> int:
    """
    Simple heuristic that balances the pole:
    Push in the direction the pole is leaning.
    Augmented with cart-position correction.
    """
    x, x_dot, theta, theta_dot = state
    # Primary: follow pole angle
    score = theta + 0.1 * theta_dot + 0.05 * x + 0.01 * x_dot
    return 1 if score > 0 else 0


def expert_q_values(state: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Approximate Q-values for the expert policy.
    Q(s, expert_action) ≈ 1/(1-γ)  (infinite horizon)
    Q(s, other_action)  ≈ 0 (rough approximation)
    """
    expert_a = expert_policy(state)
    q = np.zeros(2, dtype=np.float32)
    q[expert_a]     = 1.0 / (1.0 - gamma)   # ≈ 100 for γ=0.99
    q[1 - expert_a] = q[expert_a] * 0.5      # suboptimal but non-zero
    return q


# ──────────────────────────────────────────────────────────────────────
# Generate demonstration dataset
# ──────────────────────────────────────────────────────────────────────

def generate_demos(n_episodes: int = 200, seed: int = 42) -> tuple:
    """
    Run the expert policy for n_episodes, collect (state, q_target) pairs.
    """
    env    = CartPoleEnv(seed=seed)
    states = []
    q_tgts = []

    for ep in range(n_episodes):
        obs  = env.reset(seed=seed + ep)
        done = False
        while not done:
            q_target = expert_q_values(obs)
            states.append(obs.copy())
            q_tgts.append(q_target)
            action = expert_policy(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

    X = np.array(states, dtype=np.float32)
    Y = np.array(q_tgts, dtype=np.float32)
    print(f"  Generated {len(X):,} demonstration steps from {n_episodes} episodes")
    return X, Y


# ──────────────────────────────────────────────────────────────────────
# Supervised pre-training
# ──────────────────────────────────────────────────────────────────────

def pretrain_network(X: np.ndarray, Y: np.ndarray,
                     hidden_sizes: list = None,
                     lr: float = 5e-4,
                     epochs: int = 500,
                     batch_size: int = 64,
                     seed: int = 42) -> NeuralNetwork:
    hidden_sizes = hidden_sizes or [128, 128]
    layer_sizes  = [4] + hidden_sizes + [2]
    activations  = ["relu"] * len(hidden_sizes) + ["linear"]
    net          = NeuralNetwork(layer_sizes, activations, seed=seed)

    n      = len(X)
    rng    = np.random.default_rng(seed)
    losses = []

    print(f"\n  Supervised pre-training:")
    print(f"  {net.summary()}")
    print(f"  Dataset: {n} samples  |  epochs={epochs}  "
          f"batch={batch_size}  lr={lr}")
    print()

    for epoch in range(1, epochs + 1):
        idx         = rng.permutation(n)
        epoch_loss  = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start: start + batch_size]
            x_b       = X[batch_idx]
            y_b       = Y[batch_idx]
            loss      = net.train_step(x_b, y_b, lr=lr)
            epoch_loss.append(loss)

        mean_loss = float(np.mean(epoch_loss))
        losses.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  loss={mean_loss:.4f}")

    print(f"\n  Pre-training complete. Final loss: {losses[-1]:.4f}")
    return net, losses


# ──────────────────────────────────────────────────────────────────────
# Package into DQNAgent checkpoint
# ──────────────────────────────────────────────────────────────────────

def create_pretrained_checkpoint(
        output_path: str = "checkpoints/pretrained_model.pkl",
        n_demo_episodes: int = 200,
        pretrain_epochs: int = 500,
        seed: int = 42) -> str:
    """
    Full pipeline:
    1. Generate expert demos
    2. Pretrain Q-network via supervised learning
    3. Wrap in DQNAgent + save checkpoint
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("  DQN Pretrained Model Generator")
    print("=" * 60)

    # Step 1: generate demos
    X, Y = generate_demos(n_demo_episodes, seed=seed)

    # Step 2: supervised pretraining
    net, losses = pretrain_network(X, Y,
                                   hidden_sizes=[128, 128],
                                   lr=5e-4,
                                   epochs=pretrain_epochs,
                                   seed=seed)

    # Step 3: transfer weights to DQNAgent
    cfg   = DQNConfig(
        hidden_sizes       = [128, 128],
        learning_rate      = 1e-3,
        batch_size         = 64,
        eps_start          = 0.3,    # start lower epsilon (already pretrained)
        eps_end            = 0.01,
        eps_decay_steps    = 5_000,
        seed               = seed,
    )
    agent = DQNAgent(state_dim=4, n_actions=2, config=cfg)

    # Copy pretrained weights into agent's Q-network and target network
    net.copy_weights_to(agent.q_net)      # <- directly via weight transfer
    # (net and agent.q_net share same architecture, so we copy layer by layer)
    for src_l, dst_l in zip(net.layers, agent.q_net.layers):
        dst_l.set_params(src_l.get_params())
    for src_l, dst_l in zip(net.layers, agent.target_net.layers):
        dst_l.set_params(src_l.get_params())

    # Quick eval to show pretrain quality
    from train import evaluate_agent
    eval_stats = evaluate_agent(agent, n_episodes=10)
    print(f"\n  Pretrained agent eval (greedy): "
          f"mean={eval_stats['mean_reward']:.1f}  "
          f"max={eval_stats['max_reward']:.0f}")

    # Save checkpoint
    agent.save_checkpoint(output_path)
    print(f"\n  Pretrained checkpoint → {output_path}")
    print("=" * 60)

    # Also save the loss curve
    import json
    loss_path = os.path.join(os.path.dirname(output_path), "pretrain_losses.json")
    with open(loss_path, "w") as f:
        json.dump({"pretrain_losses": losses}, f)

    return output_path


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_pretrained_checkpoint(
        output_path    = "checkpoints/pretrained_model.pkl",
        n_demo_episodes = 200,
        pretrain_epochs = 500,
        seed           = 42
    )
