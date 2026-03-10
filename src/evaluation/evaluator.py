"""
nexus-ml/src/evaluation/evaluator.py
=====================================
Comprehensive evaluation of a trained DQN agent on the held-out test set.

Computes all 8 metrics from SRS FR-9:
  - Accuracy, Precision, Recall, F1, FPR
  - ML invocation rate
  - Per-attack-type breakdown
  - Confusion matrix
  - Comparison vs. static threshold baseline

Run with: python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""

import os
import json
import numpy as np
from typing import Dict, List
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/seaborn not installed. Plots disabled.")

from ..environment.waf_env import NexusWAFEnv, Action
from ..model.dueling_dqn import DQNAgent


class Evaluator:
    """
    Runs the trained agent on the test dataset and produces a full
    evaluation report including:
      - Overall metrics (accuracy, precision, recall, F1, FPR)
      - Per-attack-type confusion breakdown
      - ML invocation efficiency analysis
      - Comparison against static-threshold baseline
      - Saved plots and JSON report
    """

    def __init__(self, config: Dict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.results_dir = config["paths"]["results"]
        os.makedirs(self.results_dir, exist_ok=True)

        # Load test data
        test_path = os.path.join(config["paths"]["data_processed"], "test.npz")
        data = np.load(test_path, allow_pickle=True)
        self.test_states       = data["states"].astype(np.float32)
        self.test_labels       = data["labels"].astype(np.int32)
        self.test_attack_types = data["attack_types"]

        # Build environment (no curriculum filtering — full test set)
        rc = config["reward"]
        self.env = NexusWAFEnv(
            states         = self.test_states,
            labels         = self.test_labels,
            attack_types   = self.test_attack_types,
            episode_length = len(self.test_states),  # Full test in one episode
            reward_config  = rc,
            curriculum_stage = 3,
            seed           = 0,
        )

        # Load agent
        mc = config["model"]
        tc = config["training"]
        self.agent = DQNAgent(
            state_dim   = mc["input_dim"],
            num_actions = mc["num_actions"],
            hidden_dims = mc["hidden_dims"],
            device      = "cpu",
        )
        self.agent.load(checkpoint_path)
        self.agent.epsilon = 0.0  # No exploration during evaluation

    # -------------------------------------------------------------------------
    # Main evaluation
    # -------------------------------------------------------------------------

    def evaluate(self) -> Dict:
        """
        Run full evaluation. Returns the complete metrics dictionary.
        """
        print(f"\n{'='*60}")
        print(f"  NexusWAF RL Model — Test Set Evaluation")
        print(f"  Checkpoint: {os.path.basename(self.checkpoint_path)}")
        print(f"  Test samples: {len(self.test_states)}")
        print(f"{'='*60}\n")

        # --- Run agent on full test set --------------------------------------
        records = self._run_full_test()

        # --- Compute metrics -------------------------------------------------
        metrics = self._compute_metrics(records)

        # --- Baseline comparison (static threshold) -------------------------
        baseline = self._static_threshold_baseline()

        # --- Print report ----------------------------------------------------
        self._print_report(metrics, baseline)

        # --- Save plots -------------------------------------------------------
        if HAS_PLOT:
            self._plot_confusion_matrix(records)
            self._plot_per_attack_type(records)
            self._plot_action_distribution(records)
            self._plot_reward_curve(records)

        # --- Save JSON report ------------------------------------------------
        full_report = {
            "model_metrics":    metrics,
            "baseline_metrics": baseline,
            "improvement":      {
                k: metrics.get(k, 0) - baseline.get(k, 0)
                for k in ["precision", "recall", "f1", "accuracy"]
            },
            "checkpoint": self.checkpoint_path,
        }
        report_path = os.path.join(self.results_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"\n  Full report saved: {report_path}")

        return full_report

    # -------------------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------------------

    def _run_full_test(self) -> List[Dict]:
        """Run the agent through all test samples and collect per-step records."""
        obs, _ = self.env.reset(seed=0)
        records = []
        done = False

        while not done:
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            records.append({
                "action":      action,
                "action_name": Action.NAMES[action],
                "label":       info["label"],
                "attack_type": info["attack_type"],
                "outcome":     info["outcome"],
                "reward":      reward,
                "risk_score":  float(obs[0]),   # Feature 0 = risk_score
            })
            obs = next_obs

        return records

    # -------------------------------------------------------------------------
    # Metric computation
    # -------------------------------------------------------------------------

    def _compute_metrics(self, records: List[Dict]) -> Dict:
        tp = sum(1 for r in records if r["outcome"] == "tp")
        tn = sum(1 for r in records if r["outcome"] == "tn")
        fp = sum(1 for r in records if r["outcome"] == "fp")
        fn = sum(1 for r in records if r["outcome"] == "fn")
        total = len(records)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        fpr       = fp / max(fp + tn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy  = (tp + tn) / total
        ml_rate   = sum(1 for r in records if r["action"] == Action.INVOKE_ML) / total
        total_reward = sum(r["reward"] for r in records)

        # Per-attack-type recall
        type_metrics = defaultdict(lambda: {"tp": 0, "fn": 0})
        for r in records:
            if r["label"] == 1:
                if r["outcome"] == "tp":
                    type_metrics[r["attack_type"]]["tp"] += 1
                else:
                    type_metrics[r["attack_type"]]["fn"] += 1

        per_type_recall = {
            atype: vals["tp"] / max(vals["tp"] + vals["fn"], 1)
            for atype, vals in type_metrics.items()
        }

        return {
            "accuracy":         round(accuracy,  4),
            "precision":        round(precision, 4),
            "recall":           round(recall,    4),
            "f1":               round(f1,        4),
            "fpr":              round(fpr,       4),
            "ml_invocation_rate": round(ml_rate, 4),
            "total_reward":     round(total_reward, 2),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "total_samples":    total,
            "per_attack_type_recall": per_type_recall,
        }

    # -------------------------------------------------------------------------
    # Baseline: static risk threshold (no RL)
    # -------------------------------------------------------------------------

    def _static_threshold_baseline(self, threshold: float = 0.7) -> Dict:
        """
        Simulate the behaviour without RL: block if risk_score >= threshold.
        This is the baseline that RL must beat.
        """
        tp = tn = fp = fn = 0
        for i in range(len(self.test_states)):
            risk_score = float(self.test_states[i][0])
            true_label = int(self.test_labels[i])
            decision_block = risk_score >= threshold

            if true_label == 1 and decision_block:     tp += 1
            elif true_label == 0 and not decision_block: tn += 1
            elif true_label == 0 and decision_block:   fp += 1
            else:                                       fn += 1

        total     = len(self.test_states)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        fpr       = fp / max(fp + tn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy  = (tp + tn) / total

        return {
            "accuracy":   round(accuracy,  4),
            "precision":  round(precision, 4),
            "recall":     round(recall,    4),
            "f1":         round(f1,        4),
            "fpr":        round(fpr,       4),
            "ml_invocation_rate": 1.0,  # Static baseline calls ML on everything above threshold
            "description": f"Static threshold = {threshold}"
        }

    # -------------------------------------------------------------------------
    # Report printing
    # -------------------------------------------------------------------------

    def _print_report(self, metrics: Dict, baseline: Dict):
        ec = self.config["evaluation"]

        def check(val, target, lower_is_better=False):
            ok = (val <= target) if lower_is_better else (val >= target)
            return "✓" if ok else "✗"

        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │                  EVALUATION RESULTS                  │")
        print("  ├──────────────────┬──────────────┬──────────┬────────┤")
        print("  │ Metric           │  RL Agent    │ Baseline │ Target │")
        print("  ├──────────────────┼──────────────┼──────────┼────────┤")

        rows = [
            ("Accuracy",        "accuracy",         0.95,  False),
            ("Precision",       "precision",        ec["target_precision"], False),
            ("Recall",          "recall",           ec["target_recall"],    False),
            ("F1 Score",        "f1",               0.91,  False),
            ("FPR",             "fpr",              ec["target_fpr"],       True),
            ("ML Invoke Rate",  "ml_invocation_rate", ec["target_ml_invocation_rate"], True),
        ]

        for name, key, target, lower_is_better in rows:
            rl_val  = metrics.get(key, 0)
            bl_val  = baseline.get(key, 0)
            c       = check(rl_val, target, lower_is_better)
            target_str = f"{'≤' if lower_is_better else '≥'}{target:.2f}"
            print(f"  │ {name:<16} │  {rl_val:>8.4f}    │  {bl_val:>6.4f}  │ {target_str:>6} {c} │")

        print("  ├──────────────────┼──────────────┼──────────┼────────┤")
        print(f"  │ TP/TN/FP/FN      │  {metrics['tp']:>4}/{metrics['tn']:>4}/{metrics['fp']:>4}/{metrics['fn']:>4} │          │        │")
        print("  └──────────────────┴──────────────┴──────────┴────────┘")

        print("\n  Per-attack-type recall:")
        for atype, rec in sorted(metrics["per_attack_type_recall"].items()):
            bar = "█" * int(rec * 20)
            print(f"    {atype:<20} {rec:.3f}  {bar}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    def _plot_confusion_matrix(self, records: List[Dict]):
        if not HAS_PLOT: return
        tp = sum(1 for r in records if r["outcome"] == "tp")
        tn = sum(1 for r in records if r["outcome"] == "tn")
        fp = sum(1 for r in records if r["outcome"] == "fp")
        fn = sum(1 for r in records if r["outcome"] == "fn")

        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted Allow", "Predicted Block"],
                    yticklabels=["True Benign", "True Attack"], ax=ax)
        ax.set_title("Confusion Matrix — RL Agent")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "confusion_matrix.png"), dpi=120)
        plt.close()
        print("  Plot saved: confusion_matrix.png")

    def _plot_per_attack_type(self, records: List[Dict]):
        if not HAS_PLOT: return
        type_data = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
        for r in records:
            if r["label"] == 1:
                if r["outcome"] == "tp": type_data[r["attack_type"]]["tp"] += 1
                else:                    type_data[r["attack_type"]]["fn"] += 1

        if not type_data: return

        types    = list(type_data.keys())
        recalls  = [type_data[t]["tp"] / max(type_data[t]["tp"] + type_data[t]["fn"], 1)
                    for t in types]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(types, recalls, color=["#2E75B6", "#1F7A4D", "#C75B00", "#A82020", "#5B2C8D"][:len(types)])
        ax.axhline(0.90, color="red", linestyle="--", label="Target (0.90)")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Recall")
        ax.set_title("Recall per Attack Type")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "per_attack_recall.png"), dpi=120)
        plt.close()
        print("  Plot saved: per_attack_recall.png")

    def _plot_action_distribution(self, records: List[Dict]):
        if not HAS_PLOT: return
        from collections import Counter
        action_counts = Counter(r["action_name"] for r in records)
        names  = list(action_counts.keys())
        values = [action_counts[n] for n in names]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, values, color="#2E75B6")
        ax.set_ylabel("Count")
        ax.set_title("Action Distribution")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "action_distribution.png"), dpi=120)
        plt.close()
        print("  Plot saved: action_distribution.png")

    def _plot_reward_curve(self, records: List[Dict]):
        if not HAS_PLOT: return
        window = 50
        rewards = [r["reward"] for r in records]
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(smoothed, color="#2E75B6", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Request index")
        ax.set_ylabel(f"Reward (rolling mean, window={window})")
        ax.set_title("Per-request Reward on Test Set")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "test_reward_curve.png"), dpi=120)
        plt.close()
        print("  Plot saved: test_reward_curve.png")
