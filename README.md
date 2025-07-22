# Crypto Price Prediction

This repository contains several experimental approaches for forecasting minute level cryptocurrency price movements.

## Repository structure

- `crypto_MDP/` – Environment and feature engineering utilities.
- `RL/` – Reinforcement learning agents and training scripts.
- `matt_initial_coding/` – Early experiments and baselines.

## Getting started

1. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Place `train.parquet` and `test.parquet` in the project root.

3. Train a model:

```bash
python RL/scripts/train.py --data train.parquet --agent sac --output runs/
```

4. Generate predictions for the test set:

```bash
python RL/scripts/test.py --model runs/<model_dir>/final_model.pth --data test.parquet --output submission.csv
```

The configuration options used for training are defined in `RL/configs/default_config.yaml` and can be adjusted as needed.
