# Publishing Workflow

## GitHub

```bash
git add .
git commit -m "Initial DQN RL project import with docs and notebook refresh"
git branch -M main
gh repo create dqn-rl-project --public --source . --remote origin --push
```

## Kaggle Notebook Versioning

```bash
kaggle kernels push -p kaggle_kernel
```

This creates/updates a Kaggle notebook and versions it.

## Hugging Face Model Upload

```bash
hf repo create dqn-cartpole-numpy --type model -y
hf upload dqn-cartpole-numpy checkpoints/best_model.pkl best_model.pkl
hf upload dqn-cartpole-numpy checkpoints/final_model.pkl final_model.pkl
```
