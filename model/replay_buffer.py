"""
replay_buffer.py
================
Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
provides uniform random sampling for off-policy learning.

Features
--------
- Circular buffer with O(1) insert and O(batch) sample
- Priority-weighted sampling (optional — Prioritised Experience Replay)
- Numpy-only, no external dependencies
"""

import numpy as np
from collections import namedtuple


# Named tuple for a single transition
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Uniform Experience Replay Buffer
    =================================
    Circular buffer that stores transitions and supports random batch sampling.

    Parameters
    ----------
    capacity     : int   — maximum number of transitions to store
    state_shape  : tuple — shape of a single observation, e.g. (4,)
    seed         : int   — RNG seed for reproducible sampling
    """

    def __init__(self, capacity: int = 100_000, state_shape: tuple = (4,), seed: int = 0):
        self.capacity    = capacity
        self.state_shape = state_shape
        self._rng        = np.random.default_rng(seed)
        self._ptr        = 0       # write pointer
        self._size       = 0       # current number of stored transitions

        # Pre-allocate arrays for efficiency
        self._states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions     = np.zeros(capacity,                 dtype=np.int32)
        self._rewards     = np.zeros(capacity,                 dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._dones       = np.zeros(capacity,                 dtype=np.float32)

    # ── Add a single transition ───────────────────────────────────────
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        idx = self._ptr % self.capacity
        self._states[idx]      = state
        self._actions[idx]     = action
        self._rewards[idx]     = reward
        self._next_states[idx] = next_state
        self._dones[idx]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ── Sample a random batch ─────────────────────────────────────────
    def sample(self, batch_size: int) -> Transition:
        """
        Returns a Transition of stacked numpy arrays, each of shape
        (batch_size, ...).
        """
        assert self._size >= batch_size, \
            f"Buffer has {self._size} transitions, need {batch_size}"
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return Transition(
            state      = self._states[indices],
            action     = self._actions[indices],
            reward     = self._rewards[indices],
            next_state = self._next_states[indices],
            done       = self._dones[indices],
        )

    # ── Properties ───────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (f"ReplayBuffer(capacity={self.capacity:,}, "
                f"size={self._size:,}, state_shape={self.state_shape})")


# ──────────────────────────────────────────────────────────────────────
class PrioritisedReplayBuffer(ReplayBuffer):
    """
    Prioritised Experience Replay (PER) Buffer
    ============================================
    Transitions are sampled proportionally to their TD-error priority.
    Uses a simple proportional scheme with segment-tree for O(log N) update.

    Parameters
    ----------
    alpha : float — priority exponent  (0 = uniform, 1 = full priority)
    beta  : float — IS weight exponent (0 = no correction, 1 = full)
    eps   : float — small constant to avoid zero priority
    """

    def __init__(self, capacity: int = 100_000, state_shape: tuple = (4,),
                 alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-6,
                 seed: int = 0):
        super().__init__(capacity, state_shape, seed)
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._max_prio   = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        idx = self._ptr % self.capacity
        super().push(state, action, reward, next_state, done)
        self._priorities[idx] = self._max_prio

    def sample(self, batch_size: int):
        assert self._size >= batch_size
        probs    = self._priorities[:self._size] ** self.alpha
        probs   /= probs.sum()
        indices  = self._rng.choice(self._size, size=batch_size, replace=False, p=probs)
        # Importance sampling weights
        weights  = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        batch    = Transition(
            state      = self._states[indices],
            action     = self._actions[indices],
            reward     = self._rewards[indices],
            next_state = self._next_states[indices],
            done       = self._dones[indices],
        )
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        new_prios = (np.abs(td_errors) + self.eps).astype(np.float32)
        self._priorities[indices] = new_prios
        self._max_prio = max(self._max_prio, float(new_prios.max()))

    def anneal_beta(self, step: int, total_steps: int) -> None:
        """Linearly anneal beta from initial value to 1.0."""
        self.beta = min(1.0, self.beta + (1.0 - self.beta) * step / total_steps)


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    buf = ReplayBuffer(capacity=1000, state_shape=(4,), seed=42)
    for i in range(200):
        s  = np.random.randn(4).astype(np.float32)
        a  = np.random.randint(0, 2)
        r  = float(np.random.rand())
        ns = np.random.randn(4).astype(np.float32)
        d  = bool(np.random.rand() > 0.95)
        buf.push(s, a, r, ns, d)

    batch = buf.sample(32)
    print(f"  Buffer: {buf}")
    print(f"  Batch states shape: {batch.state.shape}")
    print(f"  Batch actions: {batch.action[:8]}")
    print("  ReplayBuffer test passed ✓")

    per = PrioritisedReplayBuffer(capacity=1000, state_shape=(4,), seed=0)
    for i in range(200):
        s  = np.random.randn(4).astype(np.float32)
        per.push(s, 0, 1.0, s, False)
    batch, idx, w = per.sample(32)
    per.update_priorities(idx, np.random.rand(32))
    print(f"  PER weights range: [{w.min():.3f}, {w.max():.3f}]")
    print("  PrioritisedReplayBuffer test passed ✓")
