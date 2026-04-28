"""
environment.py
==============
Custom CartPole-style environment built from scratch using only NumPy.
Mimics the OpenAI Gymnasium interface: reset(), step(), render().

Physics: Classic cart-pole inverted pendulum.
State : [cart_pos, cart_vel, pole_angle, pole_angular_vel]
Action: 0 = push left, 1 = push right
"""

import numpy as np
import math


class CartPoleEnv:
    """
    CartPole Inverted Pendulum Environment
    =======================================
    A cart moves left/right on a frictionless track.
    Goal: keep the pole balanced upright.

    Observation Space: Box(4,) — [x, x_dot, theta, theta_dot]
    Action Space     : Discrete(2) — {0: left, 1: right}
    Reward           : +1.0 every step the pole is upright
    Termination      : pole angle > 12°, cart out of bounds, or 500 steps
    """

    # Physics constants
    GRAVITY        = 9.8
    CART_MASS      = 1.0
    POLE_MASS      = 0.1
    POLE_HALF_LEN  = 0.5         # half-length of pole (meters)
    FORCE_MAG      = 10.0
    TAU            = 0.02        # seconds per step

    # Termination thresholds
    THETA_THRESHOLD = 12 * math.pi / 180  # 12 degrees in radians
    X_THRESHOLD     = 2.4
    MAX_STEPS       = 500

    # Derived
    TOTAL_MASS      = CART_MASS + POLE_MASS
    POLE_MASS_LEN   = POLE_MASS * POLE_HALF_LEN

    def __init__(self, seed: int = None):
        self.observation_space_shape = (4,)
        self.action_space_n          = 2
        self._rng = np.random.default_rng(seed)
        self.state       = None
        self.step_count  = 0
        self.total_reward = 0.0

        # Observation bounds (used for clipping / display)
        self.obs_high = np.array([
            self.X_THRESHOLD * 2,
            np.finfo(np.float32).max,
            self.THETA_THRESHOLD * 2,
            np.finfo(np.float32).max,
        ], dtype=np.float32)
        self.obs_low = -self.obs_high

    # ------------------------------------------------------------------
    def reset(self, seed: int = None) -> np.ndarray:
        """Reset environment. Returns initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state        = self._rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.step_count   = 0
        self.total_reward = 0.0
        return self.state.copy()

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Apply action, advance physics one time-step.

        Returns
        -------
        obs        : np.ndarray (4,)
        reward     : float
        terminated : bool
        truncated  : bool
        info       : dict
        """
        assert action in (0, 1), f"Invalid action {action}"
        x, x_dot, theta, theta_dot = self.state

        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Equations of motion (Euler integration)
        temp = (force + self.POLE_MASS_LEN * theta_dot**2 * sin_theta) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.POLE_HALF_LEN * (4.0 / 3.0 - self.POLE_MASS * cos_theta**2 / self.TOTAL_MASS)
        )
        x_acc = temp - self.POLE_MASS_LEN * theta_acc * cos_theta / self.TOTAL_MASS

        # Euler update
        x         = x         + self.TAU * x_dot
        x_dot     = x_dot     + self.TAU * x_acc
        theta     = theta     + self.TAU * theta_dot
        theta_dot = theta_dot + self.TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.step_count += 1

        terminated = bool(
            x < -self.X_THRESHOLD or x > self.X_THRESHOLD
            or theta < -self.THETA_THRESHOLD or theta > self.THETA_THRESHOLD
        )
        truncated = self.step_count >= self.MAX_STEPS

        reward = 1.0 if not terminated else 0.0
        self.total_reward += reward

        info = {
            "step"        : self.step_count,
            "total_reward": self.total_reward,
            "cart_pos"    : float(x),
            "pole_angle"  : float(math.degrees(theta)),
        }
        return self.state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> str:
        """ASCII render of current state."""
        if self.state is None:
            return "[env not reset]"
        x, _, theta, _ = self.state
        bar_len = 40
        pos = int((x / self.X_THRESHOLD + 1) / 2 * bar_len)
        pos = max(0, min(bar_len, pos))
        bar = [" "] * bar_len
        bar[pos] = "|"
        angle_str = f"{math.degrees(theta):+.1f}°"
        return f"[{''.join(bar)}]  angle={angle_str}  step={self.step_count}"

    # ------------------------------------------------------------------
    def sample_action(self) -> int:
        """Sample a random action."""
        return int(self._rng.integers(0, self.action_space_n))


# ------------------------------------------------------------------
# Tiny test
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = CartPoleEnv(seed=42)
    obs = env.reset()
    print(f"Initial obs: {obs}")
    for _ in range(5):
        a   = env.sample_action()
        obs, r, term, trunc, info = env.step(a)
        print(f"  action={a}  reward={r:.1f}  {env.render()}")
        if term or trunc:
            break
    print("Environment test passed ✓")
