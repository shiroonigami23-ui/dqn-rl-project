import os
import tempfile
from typing import List

import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from huggingface_hub import hf_hub_download

from model.environment import CartPoleEnv
from model.dqn_agent import DQNAgent

MODEL_REPO_ID = "ShiroOnigami23/dqn-cartpole-numpy"
MODEL_CANDIDATES = [
    "best_model_finetuned_5000ep.pkl",
    "best_model.pkl",
    "final_model_finetuned_5000ep.pkl",
]


def load_agent():
    last_err = None
    for fname in MODEL_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=fname)
            agent = DQNAgent.load_checkpoint(path)
            return agent, fname
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any checkpoint from {MODEL_REPO_ID}: {last_err}")


AGENT, ACTIVE_FILE = load_agent()


def _rollout(seed: int, max_steps: int, policy: str = "dqn"):
    env = CartPoleEnv(seed=seed)
    obs = env.reset(seed=seed)

    total_reward = 0.0
    rewards = []
    angles = []
    positions = []
    actions = []
    q_left = []
    q_right = []
    states = []

    for _ in range(max_steps):
        if policy == "dqn":
            q = AGENT.q_net.predict(obs.reshape(1, -1))[0]
            action = int(np.argmax(q))
        else:
            q = np.array([np.nan, np.nan], dtype=np.float32)
            action = env.sample_action()

        next_obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        rewards.append(total_reward)
        angles.append(float(next_obs[2]))
        positions.append(float(next_obs[0]))
        actions.append(int(action))
        q_left.append(float(q[0]))
        q_right.append(float(q[1]))
        states.append(next_obs.copy())

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": len(rewards),
        "rewards": rewards,
        "angles": angles,
        "positions": positions,
        "actions": actions,
        "q_left": q_left,
        "q_right": q_right,
        "states": states,
    }


def _draw_lane(draw: ImageDraw.ImageDraw, x: float, theta: float, lane_y: int, w: int, label: str, color: tuple):
    track_left, track_right = 40, w - 40
    draw.line((track_left, lane_y, track_right, lane_y), fill=(140, 165, 220), width=4)

    x_norm = (x + 2.4) / 4.8
    cart_cx = int(track_left + x_norm * (track_right - track_left))
    cart_w, cart_h = 76, 32
    cart_x0, cart_y0 = cart_cx - cart_w // 2, lane_y - cart_h
    cart_x1, cart_y1 = cart_cx + cart_w // 2, lane_y
    draw.rounded_rectangle((cart_x0, cart_y0, cart_x1, cart_y1), radius=8, fill=color, outline=(235, 242, 255), width=2)

    pivot_x, pivot_y = cart_cx, cart_y0
    pole_len = 105
    pole_x = int(pivot_x + pole_len * np.sin(theta))
    pole_y = int(pivot_y - pole_len * np.cos(theta))
    draw.line((pivot_x, pivot_y, pole_x, pole_y), fill=(255, 134, 94), width=6)
    draw.ellipse((pivot_x - 4, pivot_y - 4, pivot_x + 4, pivot_y + 4), fill=(255, 255, 255))

    draw.text((18, lane_y - 130), f"{label}", fill=(235, 241, 255))
    draw.text((18, lane_y - 108), f"x={x:+.3f}  theta={theta:+.3f} rad", fill=(200, 214, 245))


def _make_dual_video(dqn_states: List[np.ndarray], rnd_states: List[np.ndarray], out_path: str):
    n = max(len(dqn_states), len(rnd_states))
    if n == 0:
        img = Image.new("RGB", (960, 540), (10, 16, 30))
        imageio.mimsave(out_path, [np.array(img)], fps=24)
        return out_path

    # keep render size efficient
    sample = max(1, n // 220)
    frames = []
    for i in range(0, n, sample):
        sd = dqn_states[min(i, len(dqn_states)-1)] if dqn_states else np.zeros(4)
        sr = rnd_states[min(i, len(rnd_states)-1)] if rnd_states else np.zeros(4)

        img = Image.new("RGB", (960, 540), (9, 14, 28))
        d = ImageDraw.Draw(img)
        d.text((20, 16), "CartPole Arena: DQN vs Random", fill=(242, 246, 255))
        d.text((20, 40), f"Frame {i+1}/{n}", fill=(190, 206, 240))

        _draw_lane(d, float(sd[0]), float(sd[2]), lane_y=230, w=960, label="DQN Agent", color=(62, 102, 234))
        _draw_lane(d, float(sr[0]), float(sr[2]), lane_y=470, w=960, label="Random Baseline", color=(120, 120, 120))

        frames.append(np.array(img))

    imageio.mimsave(out_path, frames, fps=30, codec="libx264", quality=8)
    return out_path


def _benchmark(seed: int, episodes: int = 20, max_steps: int = 500):
    dqn_rewards, rnd_rewards = [], []
    for i in range(episodes):
        d = _rollout(seed + i, max_steps, policy="dqn")
        r = _rollout(seed + i, max_steps, policy="random")
        dqn_rewards.append(d["total_reward"])
        rnd_rewards.append(r["total_reward"])
    return np.array(dqn_rewards, dtype=float), np.array(rnd_rewards, dtype=float)


def run_episode(seed: int, max_steps: int):
    dqn = _rollout(seed, max_steps, policy="dqn")
    rnd = _rollout(seed, max_steps, policy="random")

    dqn_rewards, rnd_rewards = _benchmark(seed=seed + 1000, episodes=20, max_steps=max_steps)

    solved = dqn["total_reward"] >= 475
    left_ratio = float(np.mean(np.array(dqn["actions"]) == 0)) if dqn["actions"] else 0.0
    right_ratio = 1.0 - left_ratio if dqn["actions"] else 0.0

    # Arena video
    video_path = os.path.join(tempfile.gettempdir(), f"cartpole_arena_{seed}_{max_steps}.mp4")
    _make_dual_video(dqn["states"], rnd["states"], video_path)

    # Reward curve (single episode)
    fig1, ax1 = plt.subplots(figsize=(8.8, 3.5))
    ax1.plot(dqn["rewards"], label="DQN policy", linewidth=2.4)
    ax1.plot(rnd["rewards"], label="Random policy", linewidth=1.9, alpha=0.9)
    ax1.set_title("Same Seed Episode: DQN vs Random Reward")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Reward")
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    # Benchmark bars
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax2.bar(["DQN", "Random"], [dqn_rewards.mean(), rnd_rewards.mean()], color=["#3b82f6", "#9ca3af"])
    ax2.set_title("20-Episode Mean Reward")
    ax2.grid(axis="y", alpha=0.3)

    ax3.boxplot([dqn_rewards, rnd_rewards], labels=["DQN", "Random"], patch_artist=True)
    ax3.set_title("Reward Distribution (20 episodes)")
    ax3.grid(axis="y", alpha=0.3)
    fig2.tight_layout()

    # Q-values + actions
    fig3, (ax4, ax5) = plt.subplots(2, 1, figsize=(8.8, 5.3), sharex=True)
    ax4.plot(dqn["q_left"], label="Q(left)", linewidth=1.8)
    ax4.plot(dqn["q_right"], label="Q(right)", linewidth=1.8)
    ax4.set_ylabel("Q-value")
    ax4.grid(alpha=0.3)
    ax4.legend()

    ax5.plot(dqn["actions"], linewidth=1.5)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(["Left", "Right"])
    ax5.set_xlabel("Step")
    ax5.set_title("Chosen Actions")
    ax5.grid(alpha=0.3)
    fig3.tight_layout()

    # Decision trace
    n_rows = min(25, dqn["steps"])
    lines = [
        "### First 25 Decision Steps",
        "| step | action | q_left | q_right | cart_x | theta |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for i in range(n_rows):
        s = dqn["states"][i]
        lines.append(
            f"| {i+1} | {dqn['actions'][i]} | {dqn['q_left'][i]:.3f} | {dqn['q_right'][i]:.3f} | {float(s[0]):+.3f} | {float(s[2]):+.3f} |"
        )
    trace_md = "\n".join(lines)

    stats_md = (
        f"### Episode Summary\n"
        f"- Checkpoint: `{ACTIVE_FILE}`\n"
        f"- DQN reward (seed {seed}): **{dqn['total_reward']:.1f}** in **{dqn['steps']}** steps\n"
        f"- Random reward (same seed): **{rnd['total_reward']:.1f}** in **{rnd['steps']}** steps\n"
        f"- Solved (>=475): **{'Yes' if solved else 'No'}**\n"
        f"- Action mix: Left **{left_ratio*100:.1f}%**, Right **{right_ratio*100:.1f}%**\n"
        f"- 20-episode benchmark mean: DQN **{dqn_rewards.mean():.1f} ± {dqn_rewards.std():.1f}**, Random **{rnd_rewards.mean():.1f} ± {rnd_rewards.std():.1f}**"
    )

    return stats_md, video_path, fig1, fig2, fig3, trace_md


CSS = """
.gradio-container {max-width: 1240px !important}
h1, h2, h3 {letter-spacing: .2px}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# DQN CartPole Arena
A real side-by-side game replay: trained DQN controller versus random baseline on the same physics world.
"""
    )

    with gr.Row():
        seed = gr.Slider(0, 10000, value=42, step=1, label="Random Seed")
        max_steps = gr.Slider(100, 500, value=500, step=10, label="Max Episode Steps")

    run_btn = gr.Button("Run Arena", variant="primary")

    stats = gr.Markdown()
    arena_video = gr.Video(label="Arena Replay (Top: DQN, Bottom: Random)")
    reward_plot = gr.Plot(label="Episode Reward Curves")
    bench_plot = gr.Plot(label="20-Episode Benchmark")
    q_plot = gr.Plot(label="Q-values and Actions")
    trace = gr.Markdown()

    run_btn.click(
        run_episode,
        inputs=[seed, max_steps],
        outputs=[stats, arena_video, reward_plot, bench_plot, q_plot, trace],
    )

if __name__ == "__main__":
    demo.launch()
