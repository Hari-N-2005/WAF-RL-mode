#!/usr/bin/env python3
"""
nexus-ml/scripts/evaluate.py
==============================
STEP 3 OF 4 — Run this after train.py completes.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/ckpt_epoch050_r85.20.pt
"""

import sys, os, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def find_best_checkpoint(ckpt_dir: str) -> str:
    pattern = os.path.join(ckpt_dir, "*.pt")
    files   = glob.glob(pattern)
    if not files:
        return None
    # Parse reward from filename: ckpt_epochXXX_rNN.NN.pt
    def extract_reward(f):
        try:
            return float(f.split("_r")[-1].replace(".pt", ""))
        except Exception:
            return -999.0
    return max(files, key=extract_reward)


def get_default_config():
    return {
        "paths": {
            "data_processed": "data/processed",
            "checkpoints":    "checkpoints",
            "logs":           "logs/tensorboard",
            "results":        "results",
        },
        "environment": {"episode_length": 500},
        "model": {"input_dim": 15, "hidden_dims": [128, 128, 64], "num_actions": 7},
        "training": {"learning_rate": 3e-4, "batch_size": 256,
                     "replay_buffer_size": 50000, "gamma": 0.99,
                     "epsilon_start": 1.0, "epsilon_end": 0.05,
                     "epsilon_decay_steps": 20000, "gradient_clip": 10.0,
                     "target_update_freq": 500,
                     "checkpoint_keep": 3,
                     "early_stopping_patience": 5,
                     "early_stopping_min_delta": 0.005,
                     "curriculum": {"stage1_epochs": 10, "stage2_epochs": 15,
                                    "stage3_epochs": 25, "total_epochs": 50}},
        "reward": {"true_positive": 1.0, "true_negative": 0.5,
                   "false_positive": -1.5, "false_negative": -2.0,
                   "ml_waste": -0.1, "ml_bonus": 0.3,
                   "latency_penalty": 0.2, "latency_budget": 0.5},
        "evaluation": {"target_precision": 0.92, "target_recall": 0.90,
                       "target_fpr": 0.05, "target_ml_invocation_rate": 0.30},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()

    config = get_default_config()  # Always start from defaults

    try:
        import yaml
        if os.path.exists(args.config):
            with open(args.config) as f:
                config.update(yaml.safe_load(f))
    except ImportError:
        pass

    ckpt = args.checkpoint
    if not ckpt:
        ckpt = find_best_checkpoint(config["paths"]["checkpoints"])
    if not ckpt:
        print("ERROR: No checkpoint found. Run train.py first.")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt}")

    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator(config, ckpt)
    report = evaluator.evaluate()


if __name__ == "__main__":
    main()
