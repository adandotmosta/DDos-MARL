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


class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_dist(self, obs):
        return Categorical(logits=self.forward(obs))

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)
        dist   = self.get_dist(obs_t)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    def evaluate(self, obs, actions):
        dist = self.get_dist(obs)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    def __init__(self, joint_dim=JOINT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self, n_agents: int = N_AGENTS):
        self.n_agents = n_agents
        self.clear()

    def clear(self):
        self.local_obs = [[] for _ in range(self.n_agents)]
        self.joint_obs = []
        self.actions   = [[] for _ in range(self.n_agents)]
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.rewards   = []
        self.dones     = []
        self.values    = []

    def store(self, local_obs, joint_obs, actions, log_probs, reward, done, value):
        for i in range(self.n_agents):
            self.local_obs[i].append(local_obs[i].copy())
            self.actions[i].append(actions[i])
            self.log_probs[i].append(log_probs[i])
        self.joint_obs.append(joint_obs.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)

    def compute_returns_and_advantages(self, last_value, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        T      = len(self.rewards)
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards= np.array(self.rewards, dtype=np.float32)
        dones  = np.array(self.dones,   dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta        = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae          = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t]= gae

        returns = advantages + values[:T]
        return returns, advantages

    def get_tensors(self, agent_id, returns, advantages):
        return {
            "obs":        torch.FloatTensor(np.array(self.local_obs[agent_id])),
            "joint_obs":  torch.FloatTensor(np.array(self.joint_obs)),
            "actions":    torch.LongTensor(np.array(self.actions[agent_id])),
            "old_logp":   torch.FloatTensor(np.array(self.log_probs[agent_id])),
            "returns":    torch.FloatTensor(returns),
            "advantages": torch.FloatTensor(advantages),
        }


class MAPPO:
    def __init__(
        self,
        n_agents=N_AGENTS, state_dim=STATE_DIM, joint_dim=JOINT_DIM,
        action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, clip_eps=CLIP_EPS,
        entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
        max_grad=MAX_GRAD_NORM, epochs=EPOCHS_PER_UPDATE,
        batch_size=BATCH_SIZE, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        device="cpu",
    ):
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
        self.update_count = 0

        self.actors = [
            Actor(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ]
        self.critic = Critic(joint_dim, hidden_dim).to(self.device)
        self.actor_optimizers = [
            optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, local_states, joint_obs):
        actions, log_probs = [], []
        for i, actor in enumerate(self.actors):
            a, lp = actor.act(local_states[i])
            actions.append(a)
            log_probs.append(lp)
        joint_t = torch.FloatTensor(joint_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = float(self.critic(joint_t).item())
        return actions, log_probs, value

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = total_critic_loss = total_entropy = 0.0
        T = len(buffer)

        for _ in range(self.epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                idx = indices[start: start + self.batch_size]

                batch_joint = torch.FloatTensor(np.array(buffer.joint_obs)[idx]).to(self.device)
                batch_ret   = torch.FloatTensor(returns[idx]).to(self.device)
                batch_adv   = torch.FloatTensor(advantages[idx]).to(self.device)

                critic_loss = nn.MSELoss()(self.critic(batch_joint), batch_ret)
                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                self.critic_optimizer.step()
                total_critic_loss += critic_loss.item()

                for i, (actor, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                    t = buffer.get_tensors(i, returns, advantages)
                    new_logp, entropy = actor.evaluate(
                        t["obs"][idx].to(self.device),
                        t["actions"][idx].to(self.device),
                    )
                    ratio  = torch.exp(new_logp - t["old_logp"][idx].to(self.device))
                    surr1  = ratio * batch_adv
                    surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                    aloss  = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                    opt.zero_grad()
                    aloss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad)
                    opt.step()
                    total_actor_loss += aloss.item()
                    total_entropy    += entropy.mean().item()

        self.update_count += 1
        n = self.epochs * max(1, T // self.batch_size)
        return {
            "actor_loss":  total_actor_loss  / (n * self.n_agents),
            "critic_loss": total_critic_loss / n,
            "entropy":     total_entropy     / (n * self.n_agents),
            "update":      self.update_count,
        }

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), save_dir / f"actor_r{i+1}.pt")
        torch.save(self.critic.state_dict(), save_dir / "critic.pt")

    def load(self, save_dir: Path):
        save_dir = Path(save_dir)
        for i, actor in enumerate(self.actors):
            p = save_dir / f"actor_r{i+1}.pt"
            if p.exists():
                actor.load_state_dict(torch.load(p, map_location=self.device))
        cp = save_dir / "critic.pt"
        if cp.exists():
            self.critic.load_state_dict(torch.load(cp, map_location=self.device))
