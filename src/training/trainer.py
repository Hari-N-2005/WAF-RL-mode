"""
nexus-ml/src/training/trainer.py
=================================
Complete DQN training loop with:
  - 3-stage curriculum learning
  - TensorBoard logging
  - Checkpoint management (keep best N)
  - Early stopping
  - Live console progress
  - Reproducible seeding

Usage (called by scripts/train.py):
    trainer = Trainer(config)
    trainer.train()
"""

import os
import json
import time
import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch not installed. Run: pip install torch")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[WARN] TensorBoard not available. Install: pip install tensorboard")
    class SummaryWriter:  # Stub
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def add_scalars(self, *a, **kw): pass
        def close(self): pass

from ..environment.waf_env import NexusWAFEnv, Action
from ..model.dueling_dqn import DQNAgent


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Manages the full DQN training pipeline for NexusWAF.

    Curriculum stages are switched automatically at the configured epoch
    boundaries. Validation happens every eval_freq training steps.
    """

    def __init__(self, config: Dict):
        self.config   = config
        self.device   = self._detect_device()
        self._step    = 0       # global training steps
        self._best_reward_history: List[Tuple[float, str]] = []  # (reward, path)

        # Create output directories
        Path(config["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
        Path(config["paths"]["logs"]).mkdir(parents=True, exist_ok=True)
        Path(config["paths"]["results"]).mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=config["paths"]["logs"])

        # Load datasets
        self.train_data = self._load_split("train")
        self.val_data   = self._load_split("val")

        # Build environments
        self.train_env = self._build_env(self.train_data, stage=1)
        self.val_env   = self._build_env(self.val_data,   stage=3)  # Full for validation

        # Build agent
        tc = config["training"]
        mc = config["model"]
        self.agent = DQNAgent(
            state_dim          = mc["input_dim"],
            num_actions        = mc["num_actions"],
            hidden_dims        = mc["hidden_dims"],
            learning_rate      = tc["learning_rate"],
            gamma              = tc["gamma"],
            epsilon_start      = tc["epsilon_start"],
            epsilon_end        = tc["epsilon_end"],
            epsilon_decay      = tc["epsilon_decay_steps"],
            batch_size         = tc["batch_size"],
            replay_capacity    = tc["replay_buffer_size"],
            target_update_freq = tc["target_update_freq"],
            gradient_clip      = tc["gradient_clip"],
            device             = self.device,
        )

        # Metrics tracking
        self._recent_losses:   deque = deque(maxlen=100)
        self._recent_rewards:  deque = deque(maxlen=20)   # episode rewards
        self._val_rewards:     List[float] = []
        self._no_improve_count = 0

        print(f"\n{'='*60}")
        print(f"  NexusWAF RL Trainer")
        print(f"  Device:  {self.device}")
        print(f"  Train:   {len(self.train_data['states'])} samples")
        print(f"  Val:     {len(self.val_data['states'])} samples")
        print(f"  Actions: {Action.N}")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def train(self):
        """Run the complete curriculum training."""
        tc = config = self.config["training"]
        cc = config["curriculum"]

        curriculum = [
            (cc["stage1_epochs"], 1, "Stage 1: Obvious cases only"),
            (cc["stage2_epochs"], 2, "Stage 2: Ambiguous cases added"),
            (cc["stage3_epochs"], 3, "Stage 3: Full adversarial mix"),
        ]

        total_start = time.time()
        epoch = 0

        for num_epochs, stage, stage_name in curriculum:
            print(f"\n{'─'*60}")
            print(f"  {stage_name}")
            print(f"{'─'*60}")

            self.train_env.set_curriculum_stage(stage)

            for _ in range(num_epochs):
                epoch += 1
                ep_reward, ep_metrics = self._run_episode(training=True)

                self._recent_rewards.append(ep_reward)
                mean_reward = np.mean(list(self._recent_rewards))

                # Log to TensorBoard
                self.writer.add_scalar("train/episode_reward", ep_reward, epoch)
                self.writer.add_scalar("train/mean_reward_20ep", mean_reward, epoch)
                self.writer.add_scalar("train/epsilon", self.agent.epsilon, epoch)
                self.writer.add_scalar("train/precision", ep_metrics["precision"], epoch)
                self.writer.add_scalar("train/recall",    ep_metrics["recall"],    epoch)
                self.writer.add_scalar("train/fpr",       ep_metrics["fpr"],       epoch)
                self.writer.add_scalar("train/ml_rate",   ep_metrics["ml_rate"],   epoch)
                if self._recent_losses:
                    mean_loss = np.mean(list(self._recent_losses))
                    self.writer.add_scalar("train/td_loss", mean_loss, epoch)

                # Console output
                self._print_progress(epoch, ep_reward, mean_reward, ep_metrics)

                # Validation
                if epoch % 2 == 0:  # Validate every 2 epochs
                    val_reward, val_metrics = self._validate()
                    self.writer.add_scalar("val/episode_reward", val_reward, epoch)
                    self.writer.add_scalars("val/metrics", {
                        "precision": val_metrics["precision"],
                        "recall":    val_metrics["recall"],
                        "f1":        val_metrics["f1"],
                    }, epoch)

                    self._val_rewards.append(val_reward)
                    self._checkpoint(val_reward, epoch)

                    # Early stopping
                    if self._check_early_stopping(val_reward):
                        print(f"\n  ⏹  Early stopping triggered at epoch {epoch}")
                        break

        elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed/60:.1f} minutes")
        print(f"  Total gradient updates: {self.agent.total_updates:,}")
        self._save_final_results()
        self.writer.close()

    # -------------------------------------------------------------------------
    # Episode runner
    # -------------------------------------------------------------------------

    def _run_episode(self, training: bool = True) -> Tuple[float, Dict]:
        """Run one full episode (episode_length steps)."""
        obs, _ = self.train_env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # Action selection
            action = self.agent.select_action(obs)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            ep_reward += reward

            if training:
                # Store transition
                self.agent.replay.push(
                    torch.tensor(obs,      dtype=torch.float32),
                    action,
                    reward,
                    torch.tensor(next_obs, dtype=torch.float32),
                    done,
                )

                # Gradient update
                loss = self.agent.update()
                if loss > 0:
                    self._recent_losses.append(loss)

                # Epsilon decay
                self.agent.decay_epsilon()
                self._step += 1

            obs = next_obs

        metrics = self.train_env.episode_metrics()
        return ep_reward, metrics

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self, n_episodes: int = 5) -> Tuple[float, Dict]:
        """Run N validation episodes and average the results."""
        rewards  = []
        all_metrics: List[Dict] = []

        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # No exploration during validation

        for _ in range(n_episodes):
            obs, _ = self.val_env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action = self.agent.select_action(obs)
                obs, reward, terminated, truncated, info = self.val_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            rewards.append(ep_reward)
            all_metrics.append(self.val_env.episode_metrics())

        self.agent.epsilon = original_epsilon  # Restore

        mean_reward = float(np.mean(rewards))
        mean_metrics = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]
        }

        ec = self.config["evaluation"]
        target_met = (
            mean_metrics["precision"] >= ec["target_precision"] and
            mean_metrics["recall"]    >= ec["target_recall"]    and
            mean_metrics["fpr"]       <= ec["target_fpr"]
        )

        status = "✓ TARGET MET" if target_met else "  below target"
        print(f"\n  [VAL] reward={mean_reward:+.2f}  "
              f"P={mean_metrics['precision']:.3f}  "
              f"R={mean_metrics['recall']:.3f}  "
              f"FPR={mean_metrics['fpr']:.3f}  "
              f"F1={mean_metrics['f1']:.3f}  {status}")

        return mean_reward, mean_metrics

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def _checkpoint(self, val_reward: float, epoch: int):
        """Save checkpoint if val_reward improves. Keep top-K."""
        ckpt_dir  = self.config["paths"]["checkpoints"]
        keep      = self.config["training"]["checkpoint_keep"]
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}_r{val_reward:.2f}.pt")

        self.agent.save(ckpt_path)

        # Track best checkpoints (min-heap by reward)
        heapq.heappush(self._best_reward_history, (val_reward, ckpt_path))

        # Remove worst if over limit
        while len(self._best_reward_history) > keep:
            worst_reward, worst_path = heapq.heappop(self._best_reward_history)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"  [CKPT] Removed: {os.path.basename(worst_path)}")

        print(f"  [CKPT] Saved:   {os.path.basename(ckpt_path)}")

    # -------------------------------------------------------------------------
    # Early stopping
    # -------------------------------------------------------------------------

    def _check_early_stopping(self, val_reward: float) -> bool:
        tc = self.config["training"]
        patience   = tc["early_stopping_patience"]
        min_delta  = tc["early_stopping_min_delta"]

        if len(self._val_rewards) < 2:
            return False

        best_prev = max(self._val_rewards[:-1])
        if val_reward < best_prev + min_delta:
            self._no_improve_count += 1
        else:
            self._no_improve_count = 0

        return self._no_improve_count >= patience

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _load_split(self, split: str) -> Dict:
        path = os.path.join(self.config["paths"]["data_processed"], f"{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Run: python scripts/prepare_data.py first"
            )
        data = np.load(path, allow_pickle=True)
        return {
            "states":       data["states"].astype(np.float32),
            "labels":       data["labels"].astype(np.int32),
            "attack_types": data["attack_types"],
        }

    def _build_env(self, data: Dict, stage: int) -> NexusWAFEnv:
        rc = self.config["reward"]
        return NexusWAFEnv(
            states         = data["states"],
            labels         = data["labels"],
            attack_types   = data["attack_types"],
            episode_length = self.config["environment"]["episode_length"],
            reward_config  = rc,
            curriculum_stage = stage,
            seed           = 42,
        )

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _print_progress(self, epoch: int, ep_reward: float,
                        mean_reward: float, metrics: Dict):
        loss_str = ""
        if self._recent_losses:
            loss_str = f"  loss={np.mean(list(self._recent_losses)):.4f}"

        print(f"  Epoch {epoch:3d} | "
              f"reward={ep_reward:+7.2f}  mean={mean_reward:+7.2f} | "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"FPR={metrics['fpr']:.3f} | "
              f"ε={self.agent.epsilon:.3f}{loss_str}")

    def _save_final_results(self):
        if not self._best_reward_history:
            return
        best_reward, best_path = max(self._best_reward_history)
        results = {
            "best_checkpoint": best_path,
            "best_val_reward": best_reward,
            "total_epochs":    len(self._val_rewards),
            "val_reward_history": self._val_rewards,
        }
        out = os.path.join(self.config["paths"]["results"], "training_summary.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Best checkpoint: {os.path.basename(best_path)}")
        print(f"  Best val reward: {best_reward:.4f}")
        print(f"  Results saved:   {out}")
