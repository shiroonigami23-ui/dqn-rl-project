# Changelog

All notable changes to this project are documented here.
Format: [Version] — Date — Description

---

## [3.0.0] — 2025-04-28 — Full Release

### Added
- Complete 500-episode training run with all metrics collected
- `extended_train.py` with TD-error tracking and Q-value snapshots
- 14 production-quality charts covering every aspect of training
- `generate_all_charts.py` master chart generation script
- `DQN_Complete.ipynb` — comprehensive notebook (14 sections)
- `technical_documentation.pdf` — 20-page technical reference
- Correlation heatmap (Chart 13) across all training metrics
- Replay buffer utilization analysis (Chart 14)
- Algorithm flow diagram in dark mode (Chart 12)
- Hyperparameter impact table (Chart 11)
- Weight distribution heatmaps (Chart 10)
- Trajectory comparison: trained vs random agent (Chart 09)
- Pretrain analysis with phase-by-phase performance (Chart 08)
- Neural network architecture diagram (Chart 07)
- Algorithm radar/bar comparison (Chart 06)
- State-space policy decision boundary map (Chart 05)
- Q-value evolution heatmap over training (Chart 04)
- Confusion matrix with classification metrics (Chart 03)
- Reward distribution analysis (Chart 02)
- Full training dashboard 6-panel (Chart 01)

### Changed
- Enriched `training_stats.json` with TD errors, Q-snapshots, pretrain data
- README expanded with full chart reference table

---

## [2.0.0] — 2025-04-27 — Training Complete

### Added
- Full DQN training pipeline producing `best_model.pkl` (score: 500/500)
- `pretrained_model.pkl` via supervised behavioural cloning
- `checkpoint_ep100/200/300/400.pkl` snapshots
- `training_stats.json` with 200 episodes of real RL data
- `DQN_CartPole.ipynb` — initial notebook
- `evaluate.py` with greedy rollouts and ASCII rendering
- `plot_results.py` basic chart generator
- `training_curves.png` and `reward_distribution.png`

### Fixed
- Replay buffer: corrected circular pointer wraparound
- DQN agent: epsilon update now happens inside `store_transition`
- Neural network: fixed bias gradient accumulation across batches

---

## [1.0.0] — 2025-04-27 — Core Implementation

### Added
- `model/environment.py` — CartPole physics from scratch (Euler integration)
- `model/neural_network.py` — FC network with He init, ReLU, Adam
- `model/replay_buffer.py` — Uniform + Prioritised Experience Replay
- `model/dqn_agent.py` — Double DQN agent with checkpoint support
- `train.py` — CLI training script
- `pretrain.py` — standalone behavioural cloning pretrainer
- `model/__init__.py` — package exports

### Architecture
- Q-Network: 4 → 128 → 128 → 2 (17,410 params)
- Activation: ReLU (hidden), Linear (output)
- Loss: MSE on Bellman targets
- Optimiser: Adam (lr=1e-3)
- Double DQN enabled by default
- Target net hard-copy every 50 training steps

---

## [0.1.0] — 2025-04-26 — Prototype

### Added
- Initial CartPole physics prototype
- Single-layer Q-network test
- Basic training loop without replay buffer
- Proof-of-concept showing gradient flow works in pure NumPy
