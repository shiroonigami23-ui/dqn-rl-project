"""
dqn_agent.py
============
Deep Q-Network (DQN) Agent — NumPy implementation.

Implements the full DQN algorithm from:
  "Human-level control through deep reinforcement learning"
  Mnih et al., Nature 2015

Key features
------------
✓  Online  Q-network (trained every step)
✓  Target  Q-network (hard-copied every C steps)
✓  ε-greedy exploration with linear / exponential decay
✓  Experience replay via ReplayBuffer
✓  Double DQN (optional) — reduces overestimation bias
✓  Gradient clipping
✓  Checkpoint save / load
✓  Full training statistics logging
"""

import numpy as np
import os
import pickle
import json
import sys

# local imports
sys.path.insert(0, os.path.dirname(__file__))
from neural_network import NeuralNetwork
from replay_buffer  import ReplayBuffer


# ──────────────────────────────────────────────────────────────────────
class DQNConfig:
    """All hyperparameters in one place."""
    # Network
    hidden_sizes    : list  = None   # hidden layer sizes, e.g. [128, 128]

    # Training
    learning_rate   : float = 1e-3
    batch_size      : int   = 64
    gamma           : float = 0.99   # discount factor
    grad_clip       : float = 10.0

    # Replay buffer
    buffer_capacity : int   = 100_000
    min_buffer_size : int   = 1_000  # warm-up before training starts

    # Target network
    target_update_freq: int = 100    # hard update every N steps

    # Exploration (ε-greedy)
    eps_start       : float = 1.0
    eps_end         : float = 0.01
    eps_decay_steps : int   = 10_000 # linear decay over N steps

    # Double DQN
    double_dqn      : bool  = True

    # Misc
    seed            : int   = 42

    def __init__(self, **kwargs):
        self.hidden_sizes = [128, 128]
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self):
        lines = ["DQNConfig {"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}: {v}")
        lines.append("}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network Agent
    =====================

    Usage
    -----
    >>> agent = DQNAgent(state_dim=4, n_actions=2, config=DQNConfig())
    >>> obs = env.reset()
    >>> action = agent.select_action(obs)
    >>> agent.store_transition(obs, action, reward, next_obs, done)
    >>> loss = agent.train_step()
    """

    def __init__(self, state_dim: int, n_actions: int, config: DQNConfig = None):
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.config     = config or DQNConfig()
        cfg             = self.config

        # Build Q-network and target network
        layer_sizes = [state_dim] + cfg.hidden_sizes + [n_actions]
        activations = ["relu"] * len(cfg.hidden_sizes) + ["linear"]

        self.q_net      = NeuralNetwork(layer_sizes, activations, seed=cfg.seed)
        self.target_net = NeuralNetwork(layer_sizes, activations, seed=cfg.seed + 1)
        # Initialise target = q
        self.q_net.copy_weights_to(self.target_net)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity    = cfg.buffer_capacity,
            state_shape = (state_dim,),
            seed        = cfg.seed
        )

        # Counters & stats
        self.total_steps   = 0
        self.train_steps   = 0
        self.episodes      = 0
        self.epsilon       = cfg.eps_start

        self.stats = {
            "episode_rewards"  : [],
            "episode_lengths"  : [],
            "losses"           : [],
            "epsilons"         : [],
            "mean_q_values"    : [],
        }

        self._rng = np.random.default_rng(cfg.seed)

    # ── Epsilon ──────────────────────────────────────────────────────
    def _update_epsilon(self) -> None:
        cfg = self.config
        progress = min(1.0, self.total_steps / cfg.eps_decay_steps)
        self.epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * progress

    # ── Action selection ─────────────────────────────────────────────
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        q_vals = self.q_net.predict(state.reshape(1, -1))
        return int(np.argmax(q_vals[0]))

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Pure greedy — used for evaluation."""
        return self.select_action(state, greedy=True)

    # ── Store transition ─────────────────────────────────────────────
    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._update_epsilon()

    # ── Training step ────────────────────────────────────────────────
    def train_step(self) -> float | None:
        """
        Sample a batch and perform one gradient update on Q-network.
        Returns the loss value, or None if buffer is not warm yet.
        """
        cfg = self.config
        if not self.buffer.is_ready(cfg.batch_size):
            return None

        # Sample
        batch = self.buffer.sample(cfg.batch_size)
        S  = batch.state                     # (B, state_dim)
        A  = batch.action                    # (B,)
        R  = batch.reward                    # (B,)
        S2 = batch.next_state                # (B, state_dim)
        D  = batch.done                      # (B,)

        # ── Compute TD target ─────────────────────────────────────
        if cfg.double_dqn:
            # Double DQN: action chosen by Q-net, value from target-net
            online_q_next  = self.q_net.predict(S2)          # (B, n_act)
            best_actions   = np.argmax(online_q_next, axis=1) # (B,)
            target_q_next  = self.target_net.predict(S2)      # (B, n_act)
            q_next_vals    = target_q_next[np.arange(cfg.batch_size), best_actions]
        else:
            target_q_next  = self.target_net.predict(S2)
            q_next_vals    = target_q_next.max(axis=1)

        td_target = R + cfg.gamma * q_next_vals * (1 - D)    # (B,)

        # ── Build training target ─────────────────────────────────
        q_pred   = self.q_net.forward(S)                      # (B, n_act)
        td_error = q_pred[np.arange(cfg.batch_size), A] - td_target

        # Only update the Q-values for the taken actions
        target_q      = q_pred.copy()
        target_q[np.arange(cfg.batch_size), A] = td_target

        # ── Gradient step ────────────────────────────────────────
        loss, d_out = NeuralNetwork.mse_loss(q_pred, target_q)

        # Gradient clipping (clip d_out)
        d_out = np.clip(d_out, -cfg.grad_clip, cfg.grad_clip)

        self.q_net.backward(d_out)
        self.q_net.adam_update(lr=cfg.learning_rate)

        # ── Target network hard update ───────────────────────────
        self.train_steps += 1
        if self.train_steps % cfg.target_update_freq == 0:
            self.q_net.copy_weights_to(self.target_net)

        # ── Log ──────────────────────────────────────────────────
        mean_q = float(q_pred.max(axis=1).mean())
        self.stats["losses"].append(loss)
        self.stats["epsilons"].append(self.epsilon)
        self.stats["mean_q_values"].append(mean_q)

        return loss

    # ── Episode logging ──────────────────────────────────────────────
    def log_episode(self, reward: float, length: int) -> None:
        self.episodes += 1
        self.stats["episode_rewards"].append(reward)
        self.stats["episode_lengths"].append(length)

    def mean_reward(self, last_n: int = 100) -> float:
        r = self.stats["episode_rewards"]
        if not r:
            return 0.0
        return float(np.mean(r[-last_n:]))

    # ── Checkpoints ──────────────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "config"       : self.config.to_dict(),
            "state_dim"    : self.state_dim,
            "n_actions"    : self.n_actions,
            "q_net_params" : [l.get_params() for l in self.q_net.layers],
            "tgt_net_params": [l.get_params() for l in self.target_net.layers],
            "total_steps"  : self.total_steps,
            "train_steps"  : self.train_steps,
            "episodes"     : self.episodes,
            "epsilon"      : self.epsilon,
            "stats"        : self.stats,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  ✓ Checkpoint saved → {path}  "
              f"(ep={self.episodes}, ε={self.epsilon:.3f}, "
              f"mean_r={self.mean_reward():.1f})")

    @classmethod
    def load_checkpoint(cls, path: str) -> "DQNAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = DQNConfig(**{k: v for k, v in data["config"].items()
                           if k not in ("hidden_sizes",)})
        cfg.hidden_sizes = data["config"]["hidden_sizes"]
        agent = cls(data["state_dim"], data["n_actions"], cfg)
        for layer, params in zip(agent.q_net.layers, data["q_net_params"]):
            layer.set_params(params)
        for layer, params in zip(agent.target_net.layers, data["tgt_net_params"]):
            layer.set_params(params)
        agent.total_steps = data["total_steps"]
        agent.train_steps = data["train_steps"]
        agent.episodes    = data["episodes"]
        agent.epsilon     = data["epsilon"]
        agent.stats       = data["stats"]
        print(f"  ✓ Checkpoint loaded ← {path}  "
              f"(ep={agent.episodes}, ε={agent.epsilon:.3f})")
        return agent

    # ── Summary ──────────────────────────────────────────────────────
    def summary(self) -> str:
        return (
            f"DQNAgent\n"
            f"  State dim  : {self.state_dim}\n"
            f"  Actions    : {self.n_actions}\n"
            f"  Steps      : {self.total_steps:,}\n"
            f"  Episodes   : {self.episodes}\n"
            f"  Epsilon    : {self.epsilon:.4f}\n"
            f"  Buffer     : {len(self.buffer):,} / {self.config.buffer_capacity:,}\n"
            f"  Mean reward: {self.mean_reward():.2f}\n"
            + self.q_net.summary()
        )


# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from environment import CartPoleEnv

    env   = CartPoleEnv(seed=0)
    cfg   = DQNConfig(hidden_sizes=[64, 64], batch_size=32, min_buffer_size=100,
                      eps_decay_steps=500)
    agent = DQNAgent(state_dim=4, n_actions=2, config=cfg)
    print(agent.summary())

    obs = env.reset()
    for step in range(200):
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, term or trunc)
        loss = agent.train_step()
        obs = next_obs if not (term or trunc) else env.reset()

    print(f"  Final ε={agent.epsilon:.3f}  buffer={len(agent.buffer)}")
    print("  DQNAgent smoke-test passed ✓")
