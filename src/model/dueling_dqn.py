"""
nexus-ml/src/model/dueling_dqn.py
==================================
Dueling Deep Q-Network implementation.

Architecture:
  Input (15) → FC-128 → BN → ReLU
             → FC-128 → BN → ReLU
             → FC-64  →      ReLU
             ┌────────────────────────────┐
             │ Value stream:              │
             │   FC-32 → ReLU → FC-1     │  V(s)
             ├────────────────────────────┤
             │ Advantage stream:          │
             │   FC-32 → ReLU → FC-7     │  A(s,a)
             └────────────────────────────┘
  Q(s,a) = V(s) + A(s,a) − mean(A(s,·))

Why Dueling:
  Separating V(s) from A(s,a) helps when many actions have similar
  Q-values. The agent learns "how good is this state" independently from
  "how much better is action X vs. Y", which stabilises training on the
  WAF policy where most benign requests lead to the same outcome regardless
  of the fine-grained action chosen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DuelingDQN(nn.Module):
    """
    Dueling DQN with Batch Normalisation on shared layers.

    Args:
        input_dim:        State vector dimension (15)
        hidden_dims:      List of shared layer sizes [128, 128, 64]
        value_hidden:     Hidden size of value stream (32)
        advantage_hidden: Hidden size of advantage stream (32)
        num_actions:      Number of discrete actions (7)
    """

    def __init__(
        self,
        input_dim:        int = 15,
        hidden_dims:      List[int] = None,
        value_hidden:     int = 32,
        advantage_hidden: int = 32,
        num_actions:      int = 7,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.input_dim   = input_dim
        self.num_actions = num_actions

        # --- Shared feature extraction layers --------------------------------
        shared_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*shared_layers)

        # --- Value stream V(s) -----------------------------------------------
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )

        # --- Advantage stream A(s,a) -----------------------------------------
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, advantage_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(advantage_hidden, num_actions),
        )

        # Orthogonal initialisation (better than default for RL)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            Q-values of shape (batch_size, num_actions)
        """
        features   = self.shared(x)                          # (B, last_hidden)
        value      = self.value_stream(features)              # (B, 1)
        advantage  = self.advantage_stream(features)          # (B, num_actions)

        # Combine: Q = V + (A - mean(A))
        # Subtracting mean makes the decomposition unique
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def get_action(self, state: torch.Tensor) -> int:
        """
        Greedy action selection (no gradient computation).

        Args:
            state: Shape (state_dim,) or (1, state_dim)

        Returns:
            Integer action index
        """
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return int(q_values.argmax(dim=1).item())


# =============================================================================
# Experience Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Circular replay buffer for storing (s, a, r, s', done) transitions.

    Uses pre-allocated numpy arrays for efficiency (avoids Python list overhead
    on 50K+ transitions). Samples uniformly — sufficient for this problem size.
    """

    def __init__(self, capacity: int = 50_000, state_dim: int = 15):
        self.capacity  = capacity
        self.state_dim = state_dim
        self._ptr      = 0      # write pointer
        self._size     = 0      # current number of stored transitions

        # Pre-allocate storage
        self.states      = torch.zeros(capacity, state_dim,  dtype=torch.float32)
        self.actions     = torch.zeros(capacity,             dtype=torch.long)
        self.rewards     = torch.zeros(capacity,             dtype=torch.float32)
        self.next_states = torch.zeros(capacity, state_dim,  dtype=torch.float32)
        self.dones       = torch.zeros(capacity,             dtype=torch.float32)

    def push(
        self,
        state:      torch.Tensor,
        action:     int,
        reward:     float,
        next_state: torch.Tensor,
        done:       bool,
    ):
        """Store one transition."""
        i = self._ptr
        self.states[i]      = state.float()
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state.float()
        self.dones[i]       = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        """Uniform random sample of batch_size transitions."""
        indices = torch.randint(0, self._size, (batch_size,))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= 1000  # min_replay_size from config


# =============================================================================
# DQN Agent
# =============================================================================

class DQNAgent:
    """
    Wraps the DuelingDQN network with:
      - ε-greedy exploration
      - target network (hard update)
      - gradient clipping
      - optimiser

    This is the agent that interacts with NexusWAFEnv.
    """

    def __init__(
        self,
        state_dim:         int   = 15,
        num_actions:       int   = 7,
        hidden_dims:       List[int] = None,
        learning_rate:     float = 3e-4,
        gamma:             float = 0.99,
        epsilon_start:     float = 1.0,
        epsilon_end:       float = 0.05,
        epsilon_decay:     int   = 20_000,
        batch_size:        int   = 256,
        replay_capacity:   int   = 50_000,
        target_update_freq:int   = 500,
        gradient_clip:     float = 10.0,
        device:            str   = "cpu",
    ):
        self.state_dim          = state_dim
        self.num_actions        = num_actions
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip      = gradient_clip
        self.device             = torch.device(device)
        self._steps             = 0   # total steps taken (for epsilon decay)
        self._updates           = 0   # total gradient updates (for target net sync)

        # Networks
        net_kwargs = dict(
            input_dim=state_dim,
            hidden_dims=hidden_dims or [128, 128, 64],
            num_actions=num_actions,
        )
        self.online_net = DuelingDQN(**net_kwargs).to(self.device)
        self.target_net = DuelingDQN(**net_kwargs).to(self.device)
        self._sync_target()  # target ← online (initial hard copy)
        self.target_net.eval()

        # Optimiser
        self.optimiser = torch.optim.Adam(
            self.online_net.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.replay = ReplayBuffer(
            capacity=replay_capacity, state_dim=state_dim
        )

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.num_actions, (1,)).item()
        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.online_net.get_action(s)

    def decay_epsilon(self):
        """Linear decay of epsilon."""
        self._steps += 1
        fraction = min(1.0, self._steps / self.epsilon_decay)
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * (1.0 - fraction)

    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------

    def update(self) -> float:
        """
        One gradient update step.

        Returns:
            TD loss value (float), or 0.0 if buffer not ready
        """
        if not self.replay.is_ready or len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay.sample(self.batch_size)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        self.online_net.train()

        # Current Q-values Q(s, a; θ)
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ · max_a' Q(s', a'; θ⁻)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Huber loss (smooth L1) — more robust to outlier rewards than MSE
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimiser.step()

        self._updates += 1
        if self._updates % self.target_update_freq == 0:
            self._sync_target()

        return float(loss.item())

    def _sync_target(self):
        """Hard update: copy online → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "online_state_dict":  self.online_net.state_dict(),
            "target_state_dict":  self.target_net.state_dict(),
            "optimiser_state":    self.optimiser.state_dict(),
            "epsilon":            self.epsilon,
            "steps":              self._steps,
            "updates":            self._updates,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimiser.load_state_dict(ckpt["optimiser_state"])
        self.epsilon   = ckpt["epsilon"]
        self._steps    = ckpt["steps"]
        self._updates  = ckpt["updates"]
        print(f"Loaded checkpoint: step={self._steps}, ε={self.epsilon:.4f}")

    @property
    def total_steps(self) -> int:
        return self._steps

    @property
    def total_updates(self) -> int:
        return self._updates


# =============================================================================
# Quick architecture summary
# =============================================================================

if __name__ == "__main__":
    net = DuelingDQN(input_dim=15, hidden_dims=[128, 128, 64], num_actions=7)
    total_params = sum(p.numel() for p in net.parameters())
    trainable    = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"DuelingDQN Architecture:")
    print(net)
    print(f"\nTotal parameters:    {total_params:,}")
    print(f"Trainable parameters:{trainable:,}")

    # Test forward pass
    dummy = torch.randn(4, 15)  # batch of 4
    q = net(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {q.shape}")   # should be (4, 7)
    print(f"Q-values sample:\n{q[0].detach().numpy().round(3)}")
