#!/usr/bin/env python3
"""
nexus-ml/scripts/train.py
==========================
STEP 2 OF 4 — Run this after prepare_data.py.

Trains the Dueling DQN agent using 3-stage curriculum learning.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --resume checkpoints/ckpt_epoch010_r42.50.pt

What it does:
    1. Loads train.npz and val.npz from data/processed/
    2. Runs Stage 1 (easy), Stage 2 (medium), Stage 3 (full) training
    3. Validates every 2 epochs, saves best checkpoints
    4. Writes TensorBoard logs to logs/tensorboard/
    5. Saves training_summary.json to results/

Monitor training live:
    tensorboard --logdir logs/tensorboard

After training:
    Run: python scripts/evaluate.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import yaml
    def load_config(path):
        with open(path) as f:
            return yaml.safe_load(f)
except ImportError:
    import json
    def load_config(path):
        # Fallback: try JSON, otherwise use hardcoded defaults
        try:
            with open(path.replace(".yaml", ".json")) as f:
                return json.load(f)
        except FileNotFoundError:
            return get_default_config()


def get_default_config():
    """Hardcoded defaults matching configs/config.yaml."""
    return {
        "paths": {
            "data_processed": "data/processed",
            "checkpoints":    "checkpoints",
            "logs":           "logs/tensorboard",
            "results":        "results",
        },
        "environment": {"episode_length": 500},
        "model": {
            "input_dim": 15, "hidden_dims": [128, 128, 64],
            "value_hidden": 32, "advantage_hidden": 32, "num_actions": 7,
        },
        "training": {
            "curriculum": {
                "stage1_epochs": 10, "stage2_epochs": 15,
                "stage3_epochs": 25, "total_epochs": 50
            },
            "learning_rate": 3e-4, "batch_size": 256,
            "replay_buffer_size": 50000, "min_replay_size": 1000,
            "target_update_freq": 500, "gamma": 0.99,
            "epsilon_start": 1.0, "epsilon_end": 0.05,
            "epsilon_decay_steps": 20000, "gradient_clip": 10.0,
            "eval_freq": 1000, "checkpoint_keep": 3,
            "early_stopping_patience": 5, "early_stopping_min_delta": 0.005,
        },
        "reward": {
            "true_positive": 1.0, "true_negative": 0.5,
            "false_positive": -1.5, "false_negative": -2.0,
            "ml_waste": -0.1, "ml_bonus": 0.3,
            "latency_penalty": 0.2, "latency_budget": 0.5,
        },
        "evaluation": {
            "target_precision": 0.92, "target_recall": 0.90,
            "target_fpr": 0.05, "target_ml_invocation_rate": 0.30,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train NexusWAF RL agent")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"[WARN] Config not found: {args.config}. Using defaults.")
        config = get_default_config()

    # Import trainer (deferred to avoid import errors if torch missing)
    try:
        from src.training.trainer import Trainer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the nexus-ml/ directory")
        print("and all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    trainer = Trainer(config)

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.agent.load(args.resume)
        print(f"Resumed from: {args.resume}")

    trainer.train()


if __name__ == "__main__":
    main()
