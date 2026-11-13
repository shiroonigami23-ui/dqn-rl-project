# 🗺 Roadmap — DQN CartPole — NumPy From Scratch

> Back to the [README](../README.md).

## Where the project stands

Status: **active**.

## Milestones

### v1.0 — Core agent (done)

- [x] NumPy MLP with manual backprop + Adam
- [x] Double DQN with target network & replay
- [x] 12-chart analysis suite, notebooks and technical PDF
- [x] Hugging Face Space, model card and GitHub Pages showcase

### v1.1 — Algorithmic upgrades

*Goal:* Add the most impactful Rainbow components, still in pure NumPy.

- [ ] Dueling network head
- [ ] Prioritised Experience Replay (SumTree)
- [ ] n-step returns and a Huber-vs-MSE loss comparison
- [ ] Multi-seed evaluation with confidence intervals

### v1.2 — More environments

- [ ] MountainCar and Acrobot ports of the custom environment
- [ ] Gymnasium adapter so the agent also runs on official envs
- [ ] Unit tests for gradients (finite-difference check) + GitHub Actions CI

### v2.0 — Teaching edition

- [ ] Step-by-step "build DQN yourself" notebook series
- [ ] Interactive in-browser training (Pyodide) on the showcase page

## 🔭 Long-term Vision

Become the clearest from-scratch reference for value-based deep RL: every equation in the paper mapped to a line of
NumPy, with visual tools that let a student watch the Q-function learn.

## 💡 Ideas Backlog (unscheduled)

- Export trained weights to ONNX for the Unity scene
- Side-by-side comparison against a PyTorch reference implementation

## How to help

Pick any unchecked item, open an issue to discuss the approach, then send a PR. Items in the earliest unfinished milestone are the highest priority.


---
*Last updated: 2026-11-13*
