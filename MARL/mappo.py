"""
mappo.py
MAPPO implementation for CoopIDS.

Components:
  Actor          — local policy (obs_i → action)
  Critic         — centralized value (joint_obs → value)
  RolloutBuffer  — stores transitions per episode
  MAPPO          — wraps actors + critic + PPO update
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    from config import (
        STATE_DIM, JOINT_DIM, ACTION_DIM, HIDDEN_DIM,
        GAMMA, GAE_LAMBDA, LR_ACTOR, LR_CRITIC,
        CLIP_EPS, ENTROPY_COEF, VALUE_COEF,
        MAX_GRAD_NORM, EPOCHS_PER_UPDATE, BATCH_SIZE,
        N_AGENTS,
    )
except ImportError:
    from coopids.config import (
        STATE_DIM, JOINT_DIM, ACTION_DIM, HIDDEN_DIM,
        GAMMA, GAE_LAMBDA, LR_ACTOR, LR_CRITIC,
        CLIP_EPS, ENTROPY_COEF, VALUE_COEF,
        MAX_GRAD_NORM, EPOCHS_PER_UPDATE, BATCH_SIZE,
        N_AGENTS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Actor
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Local policy network for one agent.
    Input:  local obs  (STATE_DIM,)
    Output: action probabilities  (ACTION_DIM,)

    Used at execution time — only sees local obs (CTDE).
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
        """Returns raw logits."""
        return self.net(obs)

    def get_dist(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Sample action from policy.
        Returns (action, log_prob) as Python scalars.
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)
        dist   = self.get_dist(obs_t)
        action = dist.sample()
        logp   = dist.log_prob(action)
        return int(action.item()), float(logp.item())

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns log_probs and entropy for given obs/actions.
        Used during PPO update.
        """
        dist    = self.get_dist(obs)
        logp    = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Critic (Centralized)
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Centralized value network.
    Input:  joint obs  (JOINT_DIM,)  — all agents' states concatenated
    Output: scalar value estimate

    Only used during training (CTDE).
    """

    def __init__(
        self,
        joint_dim:  int = JOINT_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate."""
        return self.net(joint_obs).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# RolloutBuffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one episode of transitions for all agents.

    Stores per step:
      - local_obs[i]:  obs for each agent i
      - joint_obs:     concatenated obs (for critic)
      - actions[i]:    action for each agent i
      - log_probs[i]:  log prob for each agent i
      - reward:        global scalar reward
      - done:          episode done flag
      - value:         critic value estimate
    """

    def __init__(self, n_agents: int = N_AGENTS) -> None:
        self.n_agents  = n_agents
        self.clear()

    def clear(self) -> None:
        self.local_obs  = [[] for _ in range(self.n_agents)]
        self.joint_obs  = []
        self.actions    = [[] for _ in range(self.n_agents)]
        self.log_probs  = [[] for _ in range(self.n_agents)]
        self.rewards    = []
        self.dones      = []
        self.values     = []

    def store(
        self,
        local_obs:  List[np.ndarray],
        joint_obs:  np.ndarray,
        actions:    List[int],
        log_probs:  List[float],
        reward:     float,
        done:       bool,
        value:      float,
    ) -> None:
        for i in range(self.n_agents):
            self.local_obs[i].append(local_obs[i].copy())
            self.actions[i].append(actions[i])
            self.log_probs[i].append(log_probs[i])
        self.joint_obs.append(joint_obs.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:      float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.
        """
        T      = len(self.rewards)
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards= np.array(self.rewards, dtype=np.float32)
        dones  = np.array(self.dones,   dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:T]
        return returns, advantages

    def get_tensors(
        self,
        agent_id:    int,
        returns:     np.ndarray,
        advantages:  np.ndarray,
    ) -> dict:
        """Return tensors for one agent's PPO update."""
        return {
            "obs":        torch.FloatTensor(np.array(self.local_obs[agent_id])),
            "joint_obs":  torch.FloatTensor(np.array(self.joint_obs)),
            "actions":    torch.LongTensor(np.array(self.actions[agent_id])),
            "old_logp":   torch.FloatTensor(np.array(self.log_probs[agent_id])),
            "returns":    torch.FloatTensor(returns),
            "advantages": torch.FloatTensor(advantages),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAPPO
# ─────────────────────────────────────────────────────────────────────────────

class MAPPO:
    """
    Multi-Agent PPO with Centralized Training Decentralized Execution.

    - One Actor per agent (local obs → action)
    - One shared Critic (joint obs → value)
    - PPO clipped objective
    - Shared reward → all actors receive same return signal
    """

    def __init__(
        self,
        n_agents:    int   = N_AGENTS,
        state_dim:   int   = STATE_DIM,
        joint_dim:   int   = JOINT_DIM,
        action_dim:  int   = ACTION_DIM,
        hidden_dim:  int   = HIDDEN_DIM,
        lr_actor:    float = LR_ACTOR,
        lr_critic:   float = LR_CRITIC,
        clip_eps:    float = CLIP_EPS,
        entropy_coef:float = ENTROPY_COEF,
        value_coef:  float = VALUE_COEF,
        max_grad:    float = MAX_GRAD_NORM,
        epochs:      int   = EPOCHS_PER_UPDATE,
        batch_size:  int   = BATCH_SIZE,
        gamma:       float = GAMMA,
        gae_lambda:  float = GAE_LAMBDA,
        device:      str   = "cpu",
    ) -> None:
        self.n_agents     = n_agents
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.max_grad     = max_grad
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.device       = torch.device(device)

        # One actor per agent
        self.actors = [
            Actor(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ]

        # Shared centralized critic
        self.critic = Critic(joint_dim, hidden_dim).to(self.device)

        # Separate optimizer per actor + one for critic
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Logging
        self.update_count = 0

    # ── Act ──────────────────────────────────────────────────────────────────

    def act(
        self,
        local_states: List[np.ndarray],
        joint_obs:    np.ndarray,
    ) -> Tuple[List[int], List[float], float]:
        """
        Sample actions for all agents.
        Returns:
            actions:   list of action ints
            log_probs: list of log probs
            value:     critic estimate
        """
        actions   = []
        log_probs = []

        for i, actor in enumerate(self.actors):
            action, logp = actor.act(local_states[i])
            actions.append(action)
            log_probs.append(logp)

        # Critic value from joint obs
        joint_t = torch.FloatTensor(joint_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = float(self.critic(joint_t).item())

        return actions, log_probs, value

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        """
        PPO update using collected rollout.
        Returns dict of losses for logging.
        """
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0

        T = len(buffer)

        for epoch in range(self.epochs):
            # Random mini-batch indices
            indices = np.random.permutation(T)

            for start in range(0, T, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]

                # ── Critic update (shared across agents)
                batch_joint = torch.FloatTensor(
                    np.array(buffer.joint_obs)[batch_idx]
                ).to(self.device)
                batch_ret = torch.FloatTensor(returns[batch_idx]).to(self.device)

                values_pred = self.critic(batch_joint)
                critic_loss = nn.MSELoss()(values_pred, batch_ret)

                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                self.critic_optimizer.step()

                total_critic_loss += critic_loss.item()

                # ── Actor update (one per agent)
                batch_adv = torch.FloatTensor(advantages[batch_idx]).to(self.device)

                for i, (actor, opt) in enumerate(
                    zip(self.actors, self.actor_optimizers)
                ):
                    tensors = buffer.get_tensors(i, returns, advantages)

                    batch_obs     = tensors["obs"][batch_idx].to(self.device)
                    batch_actions = tensors["actions"][batch_idx].to(self.device)
                    batch_old_lp  = tensors["old_logp"][batch_idx].to(self.device)

                    new_logp, entropy = actor.evaluate(batch_obs, batch_actions)

                    # PPO clipped ratio
                    ratio = torch.exp(new_logp - batch_old_lp)
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv

                    actor_loss = (
                        -torch.min(surr1, surr2).mean()
                        - self.entropy_coef * entropy.mean()
                    )

                    opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad)
                    opt.step()

                    total_actor_loss += actor_loss.item()
                    total_entropy    += entropy.mean().item()

        self.update_count += 1
        n_updates = self.epochs * max(1, T // self.batch_size)

        return {
            "actor_loss":  total_actor_loss  / (n_updates * self.n_agents),
            "critic_loss": total_critic_loss / n_updates,
            "entropy":     total_entropy     / (n_updates * self.n_agents),
            "update":      self.update_count,
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, save_dir: Path) -> None:
        """Save all actors and critic to save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), save_dir / f"actor_r{i+1}.pt")

        torch.save(self.critic.state_dict(), save_dir / "critic.pt")

    def load(self, save_dir: Path) -> None:
        """Load all actors and critic from save_dir."""
        save_dir = Path(save_dir)

        for i, actor in enumerate(self.actors):
            path = save_dir / f"actor_r{i+1}.pt"
            if path.exists():
                actor.load_state_dict(
                    torch.load(path, map_location=self.device)
                )

        critic_path = save_dir / "critic.pt"
        if critic_path.exists():
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
