"""
nexus-ml/src/environment/waf_env.py
====================================
NexusWAF Reinforcement Learning Environment.

Implements the Gymnasium interface so it's compatible with any RL library
(Stable-Baselines3, CleanRL, custom DQN, etc.).

MDP Definition:
  State:   15-dimensional feature vector (see feature_extractor.py)
  Actions: 7 discrete actions (see config.yaml actions section)
  Reward:  See _compute_reward() — encodes security vs. latency tradeoff

Episode:
  One episode = one sliding window of `episode_length` consecutive requests
  drawn from the dataset. The agent processes them in order.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    # Fallback stub for environments without gymnasium installed
    HAS_GYM = False
    class _FakeGym:
        class Env: pass
        class spaces:
            class Box:
                def __init__(self, **kw): pass
            class Discrete:
                def __init__(self, n): self.n = n
    gym = _FakeGym()
    spaces = _FakeGym.spaces
    print("[WARN] gymnasium not installed. Install with: pip install gymnasium")


# =============================================================================
# Action constants — must stay in sync with config.yaml
# =============================================================================

class Action:
    ALLOW_NO_ML        = 0   # Allow, skip ML inference entirely
    INVOKE_ML          = 1   # Call ML service for semantic analysis
    BLOCK_IMMEDIATE    = 2   # Block without ML (fast block)
    LOG_AND_ALLOW      = 3   # Allow but log for GPS analysis
    RAISE_THRESHOLD    = 4   # Increase risk threshold (under load)
    LOWER_THRESHOLD    = 5   # Decrease risk threshold (under attack)
    RATE_LIMIT_IP      = 6   # Soft rate-limit this client IP

    NAMES = {
        0: "allow_no_ml",
        1: "invoke_ml",
        2: "block_immediate",
        3: "log_and_allow",
        4: "raise_threshold",
        5: "lower_threshold",
        6: "rate_limit_ip",
    }
    N = 7


# =============================================================================
# Default reward config — can be overridden from config.yaml
# =============================================================================

DEFAULT_REWARD_CONFIG = {
    "true_positive":   1.0,
    "true_negative":   0.5,
    "false_positive": -1.5,
    "false_negative": -2.0,
    "ml_waste":       -0.1,
    "ml_bonus":        0.3,
    "latency_penalty": 0.2,
    "latency_budget":  0.5,   # normalised latency above which penalty applies
}

# Feature indices (must match feature_extractor.py)
IDX_RISK_SCORE       = 0
IDX_TAG_COUNT        = 1
IDX_HAS_SQLI         = 2
IDX_HAS_XSS          = 3
IDX_HAS_PATH         = 4
IDX_HAS_CMD          = 5
IDX_BODY_LEN         = 6
IDX_URI_LEN          = 7
IDX_PARAM_COUNT      = 8
IDX_RATE_LIMITED     = 9
IDX_ATTACK_RATE      = 10
IDX_LATENCY          = 11
IDX_HAS_REFERER      = 12
IDX_CONTENT_TYPE     = 13
IDX_GRAMMAR_RISK     = 14

STATE_DIM = 15


# =============================================================================
# NexusWAF Environment
# =============================================================================

class NexusWAFEnv(gym.Env):
    """
    Gymnasium environment wrapping the NexusWAF request processing pipeline.

    Each step processes one HTTP request and returns:
      - observation: 15-dim state vector
      - reward:      scalar reward based on action correctness
      - terminated:  True when episode window is exhausted
      - truncated:   Always False (no time limit beyond episode_length)
      - info:        Dict with label, attack_type, action_name for logging
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        states:       np.ndarray,     # (N, 15) float32
        labels:       np.ndarray,     # (N,)    int32  0=benign 1=attack
        attack_types: np.ndarray,     # (N,)    str
        episode_length: int = 500,
        reward_config: Optional[Dict] = None,
        curriculum_stage: int = 3,    # 1=easy, 2=medium, 3=full
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.reward_config  = reward_config or DEFAULT_REWARD_CONFIG
        self.episode_length = episode_length
        self.curriculum_stage = curriculum_stage

        # Build curriculum-filtered indices
        self._all_states       = states
        self._all_labels       = labels
        self._all_attack_types = attack_types
        self._eligible_indices = self._build_curriculum_indices(curriculum_stage)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(Action.N)

        # Episode state
        self._rng           = np.random.default_rng(seed)
        self._episode_start = 0
        self._step_count    = 0
        self._episode_states: Optional[np.ndarray] = None
        self._episode_labels: Optional[np.ndarray] = None
        self._episode_types:  Optional[np.ndarray] = None

        # Adaptive threshold (modified by actions 4 & 5)
        self._risk_threshold = 0.7

        # Statistics (reset per episode)
        self._ep_stats = self._init_stats()

    # -------------------------------------------------------------------------
    # Gymnasium interface
    # -------------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick a random window of episode_length from eligible indices
        n_eligible = len(self._eligible_indices)
        start = self._rng.integers(0, max(1, n_eligible - self.episode_length))
        window_idx = self._eligible_indices[start : start + self.episode_length]

        self._episode_states = self._all_states[window_idx]
        self._episode_labels = self._all_labels[window_idx]
        self._episode_types  = self._all_attack_types[window_idx]

        self._step_count     = 0
        self._risk_threshold = 0.7
        self._ep_stats       = self._init_stats()

        obs = self._episode_states[0]
        return obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._episode_states is not None, "Call reset() before step()"
        assert 0 <= action < Action.N, f"Invalid action: {action}"

        state       = self._episode_states[self._step_count]
        true_label  = int(self._episode_labels[self._step_count])
        attack_type = str(self._episode_types[self._step_count])

        # Compute reward
        reward, outcome = self._compute_reward(action, state, true_label, attack_type)

        # Update adaptive threshold (actions 4 and 5)
        if action == Action.RAISE_THRESHOLD:
            self._risk_threshold = min(0.95, self._risk_threshold + 0.05)
        elif action == Action.LOWER_THRESHOLD:
            self._risk_threshold = max(0.3, self._risk_threshold - 0.05)

        # Update episode statistics
        self._update_stats(outcome, action)

        # Advance
        self._step_count += 1
        terminated = (self._step_count >= len(self._episode_states))

        # Next observation
        if not terminated:
            next_obs = self._episode_states[self._step_count].copy()
        else:
            next_obs = np.zeros(STATE_DIM, dtype=np.float32)

        info = {
            "label":       true_label,
            "attack_type": attack_type,
            "action_name": Action.NAMES[action],
            "outcome":     outcome,
            "reward":      reward,
            "threshold":   self._risk_threshold,
            **({k: v for k, v in self._ep_stats.items()} if terminated else {}),
        }

        return next_obs, reward, terminated, False, info

    def render(self):
        pass  # No visual rendering needed

    def close(self):
        pass

    # -------------------------------------------------------------------------
    # Reward function
    # -------------------------------------------------------------------------

    def _compute_reward(
        self, action: int, state: np.ndarray, true_label: int, attack_type: str
    ) -> Tuple[float, str]:
        """
        Compute reward for (action, true_label) pair.

        Returns:
            (reward_value, outcome_string)
            outcome in: "tp", "tn", "fp", "fn", "ml_waste", "neutral"
        """
        rc = self.reward_config
        risk_score  = float(state[IDX_RISK_SCORE])
        latency     = float(state[IDX_LATENCY])
        is_attack   = (true_label == 1)

        # --- Determine effective decision from action -------------------------
        # Actions 2 (block) and 6 (rate_limit) → blocking decision
        # Actions 0 (allow_no_ml), 3 (log_allow) → allowing decision
        # Action 1 (invoke_ml) → ML decides; we simulate ML as near-perfect
        #                         on clear cases, probabilistic on ambiguous
        # Actions 4, 5 (threshold adjust) → neutral for this step

        if action == Action.RAISE_THRESHOLD or action == Action.LOWER_THRESHOLD:
            # Threshold adjustment — evaluate current request with current threshold
            decision_block = (risk_score >= self._risk_threshold)
        elif action == Action.BLOCK_IMMEDIATE or action == Action.RATE_LIMIT_IP:
            decision_block = True
        elif action == Action.ALLOW_NO_ML or action == Action.LOG_AND_ALLOW:
            decision_block = False
        elif action == Action.INVOKE_ML:
            # Simulate ML result: near-perfect on extreme risk, probabilistic otherwise
            ml_confidence = self._simulate_ml(state, is_attack)
            decision_block = (ml_confidence >= 0.8)
        else:
            decision_block = False

        # --- Compute base reward from TP/TN/FP/FN ----------------------------
        if is_attack and decision_block:
            reward  = rc["true_positive"]
            outcome = "tp"
        elif not is_attack and not decision_block:
            reward  = rc["true_negative"]
            outcome = "tn"
        elif not is_attack and decision_block:
            reward  = rc["false_positive"]
            outcome = "fp"
        else:  # is_attack and not decision_block
            reward  = rc["false_negative"]
            outcome = "fn"

        # --- ML invocation modifiers -----------------------------------------
        if action == Action.INVOKE_ML:
            if outcome == "tp":
                # ML catches an attack the fast layers couldn't definitively block
                if risk_score < 0.6:  # Genuinely ambiguous
                    reward += rc["ml_bonus"]
                else:
                    # ML invoked on obvious attack — wasteful
                    reward += rc["ml_waste"]
            elif outcome in ("tp", "fn"):
                pass  # Normal ML use

        # --- Latency penalty -------------------------------------------------
        if latency > rc["latency_budget"]:
            excess = latency - rc["latency_budget"]
            reward -= rc["latency_penalty"] * excess
            # Extra penalty if we also called ML (doubles latency cost)
            if action == Action.INVOKE_ML:
                reward -= rc["latency_penalty"] * excess * 0.5

        return float(reward), outcome

    def _simulate_ml(self, state: np.ndarray, is_attack: bool) -> float:
        """
        Simulate ML microservice confidence score.

        In training we don't have a real ML service, so we simulate one:
        - Clear attacks (high risk): correctly identified with high confidence
        - Clear benign (low risk):   correctly identified with high confidence
        - Ambiguous (mid risk):      probabilistic — sometimes wrong
        
        This forces the RL agent to learn that invoking ML on obvious cases
        is wasteful (ML just confirms what the risk score already said).
        """
        risk = float(state[IDX_RISK_SCORE])
        noise = self._rng.normal(0, 0.1)

        if is_attack:
            # ML confidence that it's an attack (should be high for real attacks)
            base = 0.5 + 0.5 * risk   # Higher risk → higher ML confidence
        else:
            # ML confidence that it's benign (low attack confidence for benign)
            base = 0.5 - 0.5 * risk   # Lower risk → lower attack confidence

        return float(np.clip(base + noise, 0.0, 1.0))

    # -------------------------------------------------------------------------
    # Curriculum filtering
    # -------------------------------------------------------------------------

    def _build_curriculum_indices(self, stage: int) -> np.ndarray:
        """
        Filter eligible sample indices based on curriculum stage.

        Stage 1: Only obvious samples (risk≈0 or risk>0.7)
        Stage 2: Add ambiguous samples (risk 0.3-0.7)
        Stage 3: Full dataset (all samples)
        """
        n = len(self._all_states)

        if stage == 3:
            return np.arange(n)

        risk_scores = self._all_states[:, IDX_RISK_SCORE]

        if stage == 1:
            # Easy: very clear cases
            mask = (risk_scores < 0.2) | (risk_scores > 0.7)
        elif stage == 2:
            # Medium: include ambiguous
            mask = (risk_scores < 0.3) | (risk_scores > 0.5)
        else:
            mask = np.ones(n, dtype=bool)

        indices = np.where(mask)[0]
        if len(indices) < self.episode_length:
            print(f"[WARN] Stage {stage} filter left only {len(indices)} samples. "
                  f"Using all samples.")
            return np.arange(n)

        return indices

    def set_curriculum_stage(self, stage: int):
        """Hot-switch curriculum stage during training."""
        self.curriculum_stage = stage
        self._eligible_indices = self._build_curriculum_indices(stage)
        print(f"  Curriculum stage set to {stage} "
              f"({len(self._eligible_indices)} eligible samples)")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _init_stats(self) -> Dict:
        return {
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "ml_invocations": 0, "total_steps": 0,
            "total_reward": 0.0,
        }

    def _update_stats(self, outcome: str, action: int):
        if outcome in self._ep_stats:
            self._ep_stats[outcome] += 1
        if action == Action.INVOKE_ML:
            self._ep_stats["ml_invocations"] += 1
        self._ep_stats["total_steps"] += 1

    def episode_metrics(self) -> Dict:
        """Compute precision, recall, FPR for the current episode."""
        s = self._ep_stats
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
        total = max(s["total_steps"], 1)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        fpr       = fp / max(fp + tn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        ml_rate   = s["ml_invocations"] / total

        return {
            "precision": precision,
            "recall":    recall,
            "fpr":       fpr,
            "f1":        f1,
            "ml_rate":   ml_rate,
            "total_reward": s["total_reward"],
            "accuracy":  (tp + tn) / total,
        }
