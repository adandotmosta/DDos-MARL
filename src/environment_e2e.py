"""
environment_e2e.py
End-to-end environment for S39: FlowEncoder replaces LightGBM.

Identical to environment_tiered_v2.py EXCEPT:
  - No LightGBM model/bundle required
  - Risk scores computed by a trainable FlowEncoder (MLP)
  - Encoder is re-run at each episode reset (scores evolve during training)
  - State: [embed_mean_0, embed_mean_1, embed_mean_2, flow_count_norm]  →  4-dim
  - Risk: sigmoid(risk_head) per flow  →  same tiered profiles as S27

Tiered profiles are identical to S27 (p_risk calibrated thresholds).
Reward, routing, CAPTCHA mechanics are all unchanged.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from config import (
        N_AGENTS, WINDOW_SIZE, Us, VICTIM_IPS,
        FLOW_COUNT_MAX, REWARD_LEGIT_W, REWARD_ATTACK_W,
    )
except ImportError:
    from coopids.config import (
        N_AGENTS, WINDOW_SIZE, Us, VICTIM_IPS,
        FLOW_COUNT_MAX, REWARD_LEGIT_W, REWARD_ATTACK_W,
    )

from src.flow_encoder import FlowEncoder, prepare_features

# ── Tiered action profiles (identical to S27 / environment_tiered_v2) ─────────
TIERED_PROFILES = {
    0: (1.00, 1.00),   # passive:    forward all
    1: (0.90, 0.60),   # light
    2: (0.80, 0.40),   # moderate
    3: (0.70, 0.25),   # firm
    4: (0.60, 0.15),   # aggressive
    5: (0.50, 0.10),   # lockdown
}

ACTION_DIM = len(TIERED_PROFILES)   # 6
STATE_DIM  = 4                      # 3 embed dims + flow_count_norm
JOINT_DIM  = STATE_DIM * N_AGENTS   # 12

CAPTCHA_PASS = {
    "Benign":     0.95,
    "Suspicious": 0.50,
    "Attack":     0.05,
}

LOCAL_REWARD = True

ACTION_LABELS = {
    0: "passive",
    1: "light",
    2: "moderate",
    3: "firm",
    4: "aggressive",
    5: "lockdown",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def assign_agents(df: pd.DataFrame, n_agents: int = N_AGENTS) -> pd.DataFrame:
    df = df.copy()
    df["agent"] = df["src_ip"].apply(lambda ip: hash(str(ip)) % n_agents)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RouterEnvironment (encoder-based)
# ─────────────────────────────────────────────────────────────────────────────

class RouterEnvironmentE2E:
    """
    One ISP edge router using a learned FlowEncoder instead of LightGBM.

    Preprocessing is done once at init. Encoding is re-done at each episode
    reset via encode_all(), so risk scores evolve as the encoder trains.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        agent_id: int,
        X_preprocessed: np.ndarray,
        agent_mask: np.ndarray,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.agent_id = agent_id
        self.window_size = window_size

        # Filter to this agent's flows
        agent_indices = np.where(agent_mask)[0]
        self.flows = df.iloc[agent_indices].sort_values("timestamp").reset_index(drop=True)

        # Sort X_preprocessed to match flows order
        sort_order = df.iloc[agent_indices].sort_values("timestamp").index.values
        # Map from original df indices to X_preprocessed positions
        orig_positions = agent_indices[np.argsort(df.iloc[agent_indices]["timestamp"].values)]
        # Actually, simpler: agent_indices are positions in original df
        # We need them sorted by timestamp
        ts_vals = df.iloc[agent_indices]["timestamp"].values
        sorted_within = np.argsort(ts_vals)
        sorted_positions = agent_indices[sorted_within]
        self.X_all = X_preprocessed[sorted_positions].copy()  # (n_agent_flows, 42)

        # Labels for encoder supervised training
        self._labels = self.flows["label"].values if "label" in self.flows.columns else None

        # Timestamps for window lookup
        self._ts = pd.to_datetime(self.flows["timestamp"])
        _epoch = pd.Timestamp("1970-01-01")
        self._ts_sec = (self._ts - _epoch).dt.total_seconds()
        self._ts_sec_np = self._ts_sec.values.astype(np.float64)

        self.t_min = float(self._ts_sec.min()) if len(self._ts_sec) else 0.0
        self.t_max = float(self._ts_sec.max()) if len(self._ts_sec) else 0.0

        # Placeholders — filled by encode_all()
        n = len(self.flows)
        self._embed_all = np.zeros((n, 3), dtype=np.float32)
        self._risk_all  = np.zeros(n, dtype=np.float32)

    def encode_all(self, encoder: FlowEncoder) -> None:
        """Re-encode all flows through the current encoder. Called at episode reset."""
        if len(self.X_all) == 0:
            return
        encoder.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(self.X_all)
            embed, risk = encoder(X_t)
            self._embed_all = embed.numpy()
            self._risk_all = risk.numpy()

    def reset(self) -> None:
        pass

    def get_window(self, t_start: float) -> pd.DataFrame:
        t_end = t_start + self.window_size
        lo = int(np.searchsorted(self._ts_sec_np, t_start, side="left"))
        hi = int(np.searchsorted(self._ts_sec_np, t_end, side="left"))
        return self.flows.iloc[lo:hi].copy()

    def compute_state(self, window: pd.DataFrame) -> np.ndarray:
        """
        4-dim state: [embed_mean_0, embed_mean_1, embed_mean_2, flow_count_norm].
        """
        if len(window) == 0:
            return np.zeros(STATE_DIM, dtype=np.float32)

        idx = window.index.values
        return np.array([
            float(self._embed_all[idx, 0].mean()),
            float(self._embed_all[idx, 1].mean()),
            float(self._embed_all[idx, 2].mean()),
            min(len(window) / FLOW_COUNT_MAX, 1.0),
        ], dtype=np.float32)

    def apply_action(
        self,
        window: pd.DataFrame,
        action: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Apply tiered threshold profile using learned risk score.
        Identical logic to S27 but risk comes from FlowEncoder, not LightGBM.
        """
        if len(window) == 0:
            return window

        t_block, t_puzzle = TIERED_PROFILES.get(action, (1.0, 1.0))

        if t_block >= 1.0 and t_puzzle >= 1.0:
            return window  # passive

        n   = len(window)
        idx = window.index.values
        risk = self._risk_all[idx]

        block_mask  = risk >= t_block
        puzzle_mask = (~block_mask) & (risk >= t_puzzle)

        # CAPTCHA pass rates based on true label
        puzzle_pass = np.ones(n, dtype=np.float32)
        if puzzle_mask.any():
            puzzle_labels = window.loc[window.index[puzzle_mask], "label"]
            puzzle_pass[puzzle_mask] = (
                puzzle_labels.map(CAPTCHA_PASS).fillna(0.5).values.astype(np.float32)
            )

        captcha_result = rng.random(n) < puzzle_pass
        keep_mask = (~block_mask) & (~puzzle_mask | captcha_result)
        return window[keep_mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentEnv (encoder-based)
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEnvE2E:
    """
    Coordinates N RouterEnvironmentE2E instances.
    Identical interface to environment_tiered_v2.MultiAgentEnv.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        X_preprocessed: np.ndarray,
        encoder: FlowEncoder,
        window_size: int = WINDOW_SIZE,
        us: float = Us,
        n_agents: int = N_AGENTS,
        seed: int = 42,
    ) -> None:
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)
        self.encoder     = encoder

        df = assign_agents(df, n_agents)

        self.routers: List[RouterEnvironmentE2E] = []
        for i in range(n_agents):
            agent_mask = (df["agent"] == i).values
            router = RouterEnvironmentE2E(
                df, agent_id=i, X_preprocessed=X_preprocessed,
                agent_mask=agent_mask, window_size=window_size,
            )
            self.routers.append(router)

        ts = pd.to_datetime(df["timestamp"])
        _epoch = pd.Timestamp("1970-01-01")
        ts_sec = (ts - _epoch).dt.total_seconds()
        self.t_start   = float(ts_sec.min())
        self.t_end     = float(ts_sec.max())
        self.t_current = self.t_start

        self.victim_ips = VICTIM_IPS
        self._df = df
        self.last_step_info: dict = {}

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Reset and re-encode all flows through the current encoder."""
        self.t_current = self.t_start
        for router in self.routers:
            router.encode_all(self.encoder)
            router.reset()
        local_states = self._get_local_states()
        joint_obs    = np.concatenate(local_states).astype(np.float32)
        return local_states, joint_obs

    def step(
        self,
        actions: List[int],
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool]:
        assert len(actions) == self.n_agents

        t = self.t_current
        all_forwarded: List[pd.DataFrame] = []

        for router, action in zip(self.routers, actions):
            window_t1 = router.get_window(t + self.window_size)
            forwarded = router.apply_action(window_t1, action, self.rng)
            all_forwarded.append(forwarded)

        if LOCAL_REWARD:
            reward = self._compute_local_rewards(all_forwarded, t + self.window_size)
        else:
            reward = self._compute_reward(all_forwarded, t + self.window_size)

        self.t_current = t + self.window_size
        done = self.t_current >= self.t_end

        next_local_states = self._get_local_states()
        next_joint_obs    = np.concatenate(next_local_states).astype(np.float32)

        return next_local_states, next_joint_obs, reward, done

    def _get_local_states(self) -> List[np.ndarray]:
        return [r.compute_state(r.get_window(self.t_current)) for r in self.routers]

    def _compute_reward(
        self, all_forwarded: List[pd.DataFrame], t_window: float,
    ) -> float:
        legit_total  = 0
        attack_total = 0
        for router in self.routers:
            window = router.get_window(t_window)
            if "label" in window.columns:
                legit_total  += int((window["label"] == "Benign").sum())
                attack_total += int((window["label"] == "Attack").sum())

        if legit_total == 0 and attack_total == 0:
            self.last_step_info = {
                "benign_total": 0, "attack_total": 0,
                "benign_forwarded": 0, "attack_forwarded": 0,
                "attack_blocked": 0,
            }
            return 0.0

        non_empty = [f for f in all_forwarded if len(f) > 0]
        if not non_empty:
            self.last_step_info = {
                "benign_total": legit_total, "attack_total": attack_total,
                "benign_forwarded": 0, "attack_forwarded": 0,
                "attack_blocked": attack_total,
            }
            return -1.0 if legit_total > 0 else 0.0

        forwarded_all = pd.concat(non_empty, ignore_index=True)
        legit_passed  = int((forwarded_all["label"] == "Benign").sum()) if "label" in forwarded_all.columns else 0
        attack_passed = int((forwarded_all["label"] == "Attack").sum()) if "label" in forwarded_all.columns else 0

        self.last_step_info = {
            "benign_total":     legit_total,
            "attack_total":     attack_total,
            "benign_forwarded": legit_passed,
            "attack_forwarded": attack_passed,
            "attack_blocked":   attack_total - attack_passed,
        }

        legit_ratio  = legit_passed  / max(legit_total,  1)
        attack_ratio = attack_passed / max(attack_total, 1)
        return float(REWARD_LEGIT_W * legit_ratio - REWARD_ATTACK_W * attack_ratio)

    def _compute_local_rewards(
        self, all_forwarded: List[pd.DataFrame], t_window: float,
    ) -> List[float]:
        rewards = []
        per_agent_data = []
        for router, forwarded in zip(self.routers, all_forwarded):
            window = router.get_window(t_window)
            if "label" not in window.columns:
                rewards.append(0.0)
                per_agent_data.append({"benign_total": 0, "attack_total": 0,
                                       "benign_passed": 0, "attack_passed": 0})
                continue

            legit_total  = int((window["label"] == "Benign").sum())
            attack_total = int((window["label"] == "Attack").sum())

            if legit_total == 0 and attack_total == 0:
                rewards.append(0.0)
                per_agent_data.append({"benign_total": 0, "attack_total": 0,
                                       "benign_passed": 0, "attack_passed": 0})
                continue

            if len(forwarded) == 0:
                rewards.append(-1.0 if legit_total > 0 else 0.0)
                per_agent_data.append({"benign_total": legit_total, "attack_total": attack_total,
                                       "benign_passed": 0, "attack_passed": 0})
                continue

            legit_passed  = int((forwarded["label"] == "Benign").sum())
            attack_passed = int((forwarded["label"] == "Attack").sum())
            per_agent_data.append({"benign_total": legit_total, "attack_total": attack_total,
                                   "benign_passed": legit_passed, "attack_passed": attack_passed})

            legit_ratio  = legit_passed  / max(legit_total,  1)
            attack_ratio = attack_passed / max(attack_total, 1)
            rewards.append(float(REWARD_LEGIT_W * legit_ratio - REWARD_ATTACK_W * attack_ratio))

        self._compute_reward(all_forwarded, t_window)
        self.last_step_info["per_agent"] = per_agent_data
        return rewards

    @property
    def n_windows(self) -> int:
        return int((self.t_end - self.t_start) / self.window_size)
