"""
ippo.py
Independent PPO (IPPO) for CoopIDS ablation study (S7).

Each agent runs full PPO independently — local actor + local critic.
No centralized critic, no joint observation.
Comparison baseline: isolates whether MAPPO's advantage comes from
the centralized critic (CTDE) or just from using PPO over DQN.

Components:
  Actor         — reused from mappo.py (local obs → action)
  LocalCritic   — local value function (local obs → value), one per agent
  IPPORollout   — per-agent rollout buffer (no joint_obs)
  IPPO          — wraps N independent PPO agents
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from config import (
        STATE_DIM, ACTION_DIM, HIDDEN_DIM,
        GAMMA, GAE_LAMBDA, LR_ACTOR, LR_CRITIC,
        CLIP_EPS, ENTROPY_COEF, VALUE_COEF,
        MAX_GRAD_NORM, EPOCHS_PER_UPDATE, BATCH_SIZE,
        N_AGENTS,
    )
    from src.mappo import Actor
except ImportError:
    from coopids.config import (
        STATE_DIM, ACTION_DIM, HIDDEN_DIM,
        GAMMA, GAE_LAMBDA, LR_ACTOR, LR_CRITIC,
        CLIP_EPS, ENTROPY_COEF, VALUE_COEF,
        MAX_GRAD_NORM, EPOCHS_PER_UPDATE, BATCH_SIZE,
        N_AGENTS,
    )
    from coopids.src.mappo import Actor


# ─────────────────────────────────────────────────────────────────────────────
# Local Critic (per agent, local obs only)
# ─────────────────────────────────────────────────────────────────────────────

class LocalCritic(nn.Module):
    """
    Decentralized value network for one agent.
    Input:  local obs  (STATE_DIM,)   — same dim as actor input
    Output: scalar value estimate

    Unlike MAPPO's centralized Critic (JOINT_DIM=15), this only sees
    the agent's own STATE_DIM=5 features. No access to other agents.
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class IPPORollout:
    """
    Rollout buffer for one IPPO agent.
    Stores (obs, action, log_prob, reward, done, value) per step.
    No joint_obs — fully local.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []
        self.values    = []

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        log_prob: float,
        reward:   float,
        done:     bool,
        value:    float,
    ) -> None:
        self.obs.append(obs.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
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
        returns:    np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        return {
            "obs":        torch.FloatTensor(np.array(self.obs)),
            "actions":    torch.LongTensor(np.array(self.actions)),
            "old_logp":   torch.FloatTensor(np.array(self.log_probs)),
            "returns":    torch.FloatTensor(returns),
            "advantages": torch.FloatTensor(advantages),
        }


# ─────────────────────────────────────────────────────────────────────────────
# IPPO
# ─────────────────────────────────────────────────────────────────────────────

class IPPO:
    """
    Independent PPO — N fully decentralized PPO agents.

    Each agent:
      - Has its own Actor (local obs → action) — same arch as MAPPO
      - Has its own LocalCritic (local obs → value) — no joint obs
      - Maintains its own rollout buffer
      - Updates via PPO clipped objective independently

    No shared information during training or execution.
    Isolates the effect of the centralized critic vs PPO optimizer.
    """

    def __init__(
        self,
        n_agents:     int   = N_AGENTS,
        state_dim:    int   = STATE_DIM,
        action_dim:   int   = ACTION_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        lr_actor:     float = LR_ACTOR,
        lr_critic:    float = LR_CRITIC,
        clip_eps:     float = CLIP_EPS,
        entropy_coef: float = ENTROPY_COEF,
        value_coef:   float = VALUE_COEF,
        max_grad:     float = MAX_GRAD_NORM,
        epochs:       int   = EPOCHS_PER_UPDATE,
        batch_size:   int   = BATCH_SIZE,
        gamma:        float = GAMMA,
        gae_lambda:   float = GAE_LAMBDA,
        device:       str   = "cpu",
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

        # One actor per agent (same architecture as MAPPO)
        self.actors = [
            Actor(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ]

        # One LOCAL critic per agent (local obs only — no joint)
        self.critics = [
            LocalCritic(state_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ]

        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic)
            for critic in self.critics
        ]

        # Per-agent rollout buffers
        self.buffers = [IPPORollout() for _ in range(n_agents)]

        self.update_count = 0

    # ── Act ──────────────────────────────────────────────────────────────────

    def act(
        self,
        local_states: List[np.ndarray],
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Sample actions for all agents using local obs only.
        Returns:
            actions:   list of action ints
            log_probs: list of log probs
            values:    list of per-agent local critic estimates
        """
        actions   = []
        log_probs = []
        values    = []

        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            action, logp = actor.act(local_states[i])
            actions.append(action)
            log_probs.append(logp)

            obs_t = torch.FloatTensor(local_states[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = float(critic(obs_t).item())
            values.append(value)

        return actions, log_probs, values

    # ── Store ────────────────────────────────────────────────────────────────

    def store(
        self,
        local_obs:  List[np.ndarray],
        actions:    List[int],
        log_probs:  List[float],
        reward:     float,
        done:       bool,
        values:     List[float],
    ) -> None:
        """Store one transition into each agent's rollout buffer."""
        for i in range(self.n_agents):
            self.buffers[i].store(
                obs      = local_obs[i],
                action   = actions[i],
                log_prob = log_probs[i],
                reward   = reward,
                done     = done,
                value    = values[i],
            )

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, last_values: List[float]) -> dict:
        """
        PPO update for all agents independently from their rollout buffers.
        last_values: per-agent bootstrap value for GAE.
        Returns dict of averaged losses.
        """
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0

        for i, (actor, critic, actor_opt, critic_opt, buf) in enumerate(zip(
            self.actors, self.critics,
            self.actor_optimizers, self.critic_optimizers,
            self.buffers,
        )):
            returns, advantages = buf.compute_returns_and_advantages(
                last_values[i], self.gamma, self.gae_lambda
            )

            # Normalize advantages per agent
            adv_mean   = advantages.mean()
            adv_std    = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            T       = len(buf)
            tensors = buf.get_tensors(returns, advantages)

            for _ in range(self.epochs):
                indices = np.random.permutation(T)

                for start in range(0, T, self.batch_size):
                    idx = indices[start: start + self.batch_size]

                    batch_obs     = tensors["obs"][idx].to(self.device)
                    batch_actions = tensors["actions"][idx].to(self.device)
                    batch_old_lp  = tensors["old_logp"][idx].to(self.device)
                    batch_adv     = tensors["advantages"][idx].to(self.device)
                    batch_ret     = tensors["returns"][idx].to(self.device)

                    # ── Critic update
                    values_pred = critic(batch_obs)
                    critic_loss = nn.MSELoss()(values_pred, batch_ret)

                    critic_opt.zero_grad()
                    (self.value_coef * critic_loss).backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad)
                    critic_opt.step()

                    total_critic_loss += critic_loss.item()

                    # ── Actor update
                    new_logp, entropy = actor.evaluate(batch_obs, batch_actions)

                    ratio = torch.exp(new_logp - batch_old_lp)
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_eps, 1 + self.clip_eps
                    ) * batch_adv

                    actor_loss = (
                        -torch.min(surr1, surr2).mean()
                        - self.entropy_coef * entropy.mean()
                    )

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad)
                    actor_opt.step()

                    total_actor_loss += actor_loss.item()
                    total_entropy    += entropy.mean().item()

        self.update_count += 1
        n_batches = self.n_agents * self.epochs * max(1, len(self.buffers[0]) // self.batch_size)

        return {
            "actor_loss":  total_actor_loss  / n_batches,
            "critic_loss": total_critic_loss / n_batches,
            "entropy":     total_entropy     / n_batches,
            "update":      self.update_count,
        }

    def clear_buffers(self) -> None:
        for buf in self.buffers:
            buf.clear()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            torch.save(actor.state_dict(),  save_dir / f"actor_r{i+1}.pt")
            torch.save(critic.state_dict(), save_dir / f"critic_r{i+1}.pt")

    def load(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            actor_path  = save_dir / f"actor_r{i+1}.pt"
            critic_path = save_dir / f"critic_r{i+1}.pt"
            if actor_path.exists():
                actor.load_state_dict(
                    torch.load(actor_path, map_location=self.device)
                )
            if critic_path.exists():
                critic.load_state_dict(
                    torch.load(critic_path, map_location=self.device)
                )
