import os
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
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


def run_episode(seed: int, max_steps: int):
    env = CartPoleEnv(seed=seed)
    obs = env.reset(seed=seed)

    rewards = []
    angles = []
    cart_positions = []
    actions = []

    total_reward = 0.0
    for _ in range(max_steps):
        action = AGENT.select_action_greedy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        rewards.append(total_reward)
        angles.append(float(obs[2]))
        cart_positions.append(float(obs[0]))
        actions.append(int(action))

        if terminated or truncated:
            break

    steps = len(rewards)
    solved = total_reward >= 475

    fig1, ax1 = plt.subplots(figsize=(8, 3.2))
    ax1.plot(rewards, linewidth=2)
    ax1.set_title("Cumulative Reward")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Reward")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()

    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax2.plot(cart_positions, color="#2563eb", linewidth=1.8)
    ax2.set_ylabel("Cart Position")
    ax2.grid(alpha=0.3)

    ax3.plot(angles, color="#dc2626", linewidth=1.8)
    ax3.set_ylabel("Pole Angle (rad)")
    ax3.set_xlabel("Step")
    ax3.grid(alpha=0.3)
    fig2.tight_layout()

    left_ratio = float(np.mean(np.array(actions) == 0)) if actions else 0.0
    right_ratio = 1.0 - left_ratio if actions else 0.0

    stats_md = (
        f"### Episode Summary\n"
        f"- Checkpoint: `{ACTIVE_FILE}`\n"
        f"- Steps survived: **{steps}**\n"
        f"- Total reward: **{total_reward:.1f}**\n"
        f"- Solved (>=475): **{'Yes' if solved else 'No'}**\n"
        f"- Action mix: Left **{left_ratio*100:.1f}%**, Right **{right_ratio*100:.1f}%**"
    )

    return stats_md, fig1, fig2


CSS = """
.gradio-container {max-width: 1100px !important}
h1, h2, h3 {letter-spacing: .2px}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# DQN CartPole Live Demo
Interactive inference for a NumPy-only Double DQN agent trained on custom CartPole.
"""
    )

    with gr.Row():
        seed = gr.Slider(0, 10000, value=42, step=1, label="Random Seed")
        max_steps = gr.Slider(50, 500, value=500, step=10, label="Max Episode Steps")

    run_btn = gr.Button("Run Model", variant="primary")

    stats = gr.Markdown()
    reward_plot = gr.Plot(label="Reward Curve")
    traj_plot = gr.Plot(label="Cart & Pole Trajectory")

    run_btn.click(run_episode, inputs=[seed, max_steps], outputs=[stats, reward_plot, traj_plot])

if __name__ == "__main__":
    demo.launch()
