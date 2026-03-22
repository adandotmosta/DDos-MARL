"""
iql.py
Independent Q-Learning (IQL) for CoopIDS.

Each agent maintains its own DQN and replay buffer.
No centralized critic — agents treat each other as part of the environment.
Comparison baseline against MAPPO (S8) for ablation S6.

Components:
  QNetwork      — local Q-function (obs_i → Q-values per action)
  ReplayBuffer  — per-agent experience replay
  IQL           — wraps N independent DQN agents
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from config import (
        STATE_DIM, ACTION_DIM, HIDDEN_DIM,
        LR_ACTOR, GAMMA, BATCH_SIZE, N_AGENTS,
    )
except ImportError:
    from coopids.config import (
        STATE_DIM, ACTION_DIM, HIDDEN_DIM,
        LR_ACTOR, GAMMA, BATCH_SIZE, N_AGENTS,
    )

# IQL-specific hyperparameters
BUFFER_SIZE      = 50_000   # replay buffer capacity per agent
TARGET_UPDATE    = 10        # episodes between target network hard updates
EPS_START        = 1.0       # initial epsilon for greedy exploration
EPS_END          = 0.05      # minimum epsilon
EPS_DECAY        = 0.995     # multiplicative decay per episode
MIN_BUFFER       = 500       # minimum transitions before learning starts


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Local Q-function for one agent.
    Input:  local obs  (STATE_DIM,)
    Output: Q-values   (ACTION_DIM,)
    Same architecture as Actor for fair comparison.
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer (per agent)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size experience replay buffer for one agent.
    Stores (obs, action, reward, next_obs, done) tuples.
    """

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.append((
            obs.copy(), action, reward, next_obs.copy(), float(done)
        ))

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        obs, actions, rewards, next_obs, dones = zip(
            *[self.buffer[i] for i in idx]
        )
        return {
            "obs":      torch.FloatTensor(np.array(obs)),
            "actions":  torch.LongTensor(np.array(actions)),
            "rewards":  torch.FloatTensor(np.array(rewards)),
            "next_obs": torch.FloatTensor(np.array(next_obs)),
            "dones":    torch.FloatTensor(np.array(dones)),
        }

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# IQL
# ─────────────────────────────────────────────────────────────────────────────

class IQL:
    """
    Independent Q-Learning — N decentralized DQN agents.

    Each agent:
      - Has its own Q-network and target network
      - Maintains its own replay buffer
      - Updates independently using (local_obs, action, reward, next_local_obs)
      - No access to other agents' observations during training or execution

    Epsilon-greedy exploration decays across episodes.
    Target network updated every TARGET_UPDATE episodes.
    """

    def __init__(
        self,
        n_agents:   int   = N_AGENTS,
        state_dim:  int   = STATE_DIM,
        action_dim: int   = ACTION_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        lr:         float = LR_ACTOR,
        gamma:      float = GAMMA,
        batch_size: int   = BATCH_SIZE,
        device:     str   = "cpu",
    ) -> None:
        self.n_agents   = n_agents
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size
        self.device     = torch.device(device)

        # One Q-network + target network per agent
        self.q_nets = [
            QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ]
        self.target_nets = [
            QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ]
        # Initialise targets identical to online nets
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())
            t.eval()

        self.optimizers = [
            optim.Adam(q.parameters(), lr=lr)
            for q in self.q_nets
        ]

        # Per-agent replay buffers
        self.buffers = [ReplayBuffer() for _ in range(n_agents)]

        # Exploration
        self.epsilon     = EPS_START
        self.update_count = 0

    # ── Act ──────────────────────────────────────────────────────────────────

    def act(
        self,
        local_states: List[np.ndarray],
        greedy: bool = False,
    ) -> List[int]:
        """
        Select actions for all agents.
        greedy=True uses argmax (evaluation); False uses epsilon-greedy (training).
        """
        actions = []
        for i, q_net in enumerate(self.q_nets):
            if not greedy and np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.action_dim))
            else:
                obs_t = torch.FloatTensor(local_states[i]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_vals = q_net(obs_t)
                actions.append(int(q_vals.argmax().item()))
        return actions

    # ── Store ────────────────────────────────────────────────────────────────

    def store(
        self,
        local_obs:  List[np.ndarray],
        actions:    List[int],
        reward:     float,
        next_obs:   List[np.ndarray],
        done:       bool,
    ) -> None:
        """Store transition in each agent's replay buffer."""
        for i in range(self.n_agents):
            self.buffers[i].store(
                local_obs[i], actions[i], reward, next_obs[i], done
            )

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Update all agents' Q-networks from their replay buffers.
        Returns dict of losses.
        """
        if any(len(b) < MIN_BUFFER for b in self.buffers):
            return {}

        total_loss = 0.0

        for i, (q_net, target_net, optimizer, buffer) in enumerate(
            zip(self.q_nets, self.target_nets, self.optimizers, self.buffers)
        ):
            batch = buffer.sample(self.batch_size)
            obs      = batch["obs"].to(self.device)
            actions  = batch["actions"].to(self.device)
            rewards  = batch["rewards"].to(self.device)
            next_obs = batch["next_obs"].to(self.device)
            dones    = batch["dones"].to(self.device)

            # Current Q-values for taken actions
            q_values = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values: r + gamma * max_a Q_target(s', a)
            with torch.no_grad():
                next_q = target_net(next_obs).max(1)[0]
                target = rewards + self.gamma * next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        self.update_count += 1
        return {"q_loss": round(total_loss / self.n_agents, 6)}

    # ── Epsilon decay ────────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    # ── Target network sync ──────────────────────────────────────────────────

    def sync_targets(self) -> None:
        """Hard update: copy online weights to target networks."""
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, q_net in enumerate(self.q_nets):
            torch.save(q_net.state_dict(), save_dir / f"q_net_r{i+1}.pt")

    def load(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        for i, q_net in enumerate(self.q_nets):
            path = save_dir / f"q_net_r{i+1}.pt"
            if path.exists():
                q_net.load_state_dict(
                    torch.load(path, map_location=self.device)
                )
                self.target_nets[i].load_state_dict(q_net.state_dict())
