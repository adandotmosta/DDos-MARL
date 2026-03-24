"""
environment_tiered_v2.py
Tiered action environment for CoopIDS (S27+).

Identical to environment_tiered.py EXCEPT:
  - Routing signal: p_risk = 1 - p_benign  (instead of p_attack)
  - Thresholds recalibrated for p_risk scale (p_risk is systematically
    higher than p_attack — it sums two probabilities).

Why p_risk generalises better:
  Monday-only scorer degrades on Tuesday: attack p_attack p50 collapses
  from 1.000 → 0.071, but p_risk (= p_attack + p_suspicious) p50 drops
  only 1.000 → 0.982.  The scorer can still tell traffic is NOT benign
  even when it can't decide attack vs suspicious.

State is unchanged: 4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]
so the agent can still observe the full probability breakdown.

Each action selects a (T_block, T_puzzle) pair applied to p_risk:
  flow.p_risk >= T_block   →  BLOCK   (dropped)
  flow.p_risk >= T_puzzle  →  PUZZLE  (CAPTCHA: Attack=5% pass, Benign=95% pass)
  flow.p_risk <  T_puzzle  →  FORWARD (passed)

Calibrated from analyse_prob_distributions.py:
  p_risk thresholds are higher than p_attack ones because p_risk saturates
  at 1.0 for most attacks and is stochastically larger than p_attack.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")

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

# ── Tiered action profiles (p_risk calibrated) ───────────────────────────────
# Each action = (T_block, T_puzzle) applied to p_risk = 1 - p_benign.
# Thresholds derived from analyse_prob_distributions.py Section 5:
#   firm       (0.70, 0.25) → Mon 98.6% TPR / Tue 85.4% TPR / Tue FPR 4.9%
#   aggressive (0.60, 0.15) → Mon 99.3% TPR / Tue 88.3-92% TPR / Tue FPR ~5-11%
#   lockdown   (0.50, 0.10) → Mon 99.6% TPR / Tue 94.1% TPR / Tue FPR 12.9%
# Agent learns which tier to use per window.

TIERED_PROFILES_V2 = {
    0: (1.00, 1.00),   # passive:    forward all
    1: (0.90, 0.60),   # light:      very conservative — only clearly non-benign
    2: (0.80, 0.40),   # moderate:   Mon ~98% TPR, Tue ~82% TPR, low FPR
    3: (0.70, 0.25),   # firm:       Mon ~99% TPR, Tue ~85% TPR, Tue FPR ~5%
    4: (0.60, 0.15),   # aggressive: Mon ~99% TPR, Tue ~88% TPR, Tue FPR ~5%
    5: (0.50, 0.10),   # lockdown:   Mon ~99% TPR, Tue ~94% TPR, Tue FPR ~13%
}

ACTION_DIM = len(TIERED_PROFILES_V2)   # 6
STATE_DIM  = 4                          # same as S26
JOINT_DIM  = STATE_DIM * N_AGENTS      # 12

# CAPTCHA pass rates — same as S26 (realistic, not oracle)
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
# RouterEnvironment
# ─────────────────────────────────────────────────────────────────────────────

class RouterEnvironment:
    """
    One ISP edge router with p_risk-based tiered action profiles.

    Routing uses p_risk = 1 - p_benign.
    State exposes [p_attack, p_benign, p_suspicious, flow_count_norm]
    so the agent can observe the full breakdown while routing on p_risk.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        agent_id: int,
        bundle: dict,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.agent_id    = agent_id
        self.bundle      = bundle
        self.model       = bundle["model_3class"]
        self.features    = bundle["features"]
        self.imputer     = bundle["imputer"]
        self.hs_mapping  = bundle["handshake_mapping"]
        self.window_size = window_size

        self.flows = (
            df[df["agent"] == agent_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        self._ts = pd.to_datetime(self.flows["timestamp"])
        _epoch = pd.Timestamp("1970-01-01")
        self._ts_sec = (self._ts - _epoch).dt.total_seconds()
        self._ts_sec_np = self._ts_sec.values.astype(np.float64)

        self.t_min = float(self._ts_sec.min()) if len(self._ts_sec) else 0.0
        self.t_max = float(self._ts_sec.max()) if len(self._ts_sec) else 0.0

        # Precompute all probabilities once
        X_all = self._prepare_features(self.flows)
        if X_all is not None:
            _proba = self.model.predict_proba(X_all)
            _cls   = self.bundle["classes_3class"]
            self._p_attack_all = _proba[:, _cls.index("attack")].astype(np.float32)
            self._p_benign_all = _proba[:, _cls.index("benign")].astype(np.float32)
            self._p_susp_all   = _proba[:, _cls.index("suspicious")].astype(np.float32)
            # Routing signal: p_risk = 1 - p_benign
            self._p_risk_all   = (1.0 - self._p_benign_all).astype(np.float32)
        else:
            n = len(self.flows)
            self._p_attack_all = np.zeros(n, np.float32)
            self._p_benign_all = np.ones(n,  np.float32)
            self._p_susp_all   = np.zeros(n, np.float32)
            self._p_risk_all   = np.zeros(n, np.float32)

    def reset(self) -> None:
        pass

    def _prepare_features(self, window: pd.DataFrame) -> Optional[np.ndarray]:
        if len(window) == 0:
            return None
        X = window[self.features].copy()
        if "handshake_state" in X.columns and self.hs_mapping:
            X["handshake_state"] = (
                X["handshake_state"].astype(str).map(self.hs_mapping)
                .fillna(-1).astype(np.float32)
            )
        for col in X.columns:
            if col != "handshake_state":
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
        return self.imputer.transform(X)

    def get_window(self, t_start: float) -> pd.DataFrame:
        t_end = t_start + self.window_size
        lo = int(np.searchsorted(self._ts_sec_np, t_start, side='left'))
        hi = int(np.searchsorted(self._ts_sec_np, t_end,   side='left'))
        return self.flows.iloc[lo:hi].copy()

    def compute_state(self, window: pd.DataFrame) -> np.ndarray:
        """
        4-dim state (identical to S26):
          [p_attack_mean, p_benign_mean, p_suspicious_mean, flow_count_norm]

        The agent observes the full probability breakdown.
        Routing uses p_risk internally but the state exposes the components
        so the agent can learn richer context (e.g. confident attack vs
        attack-suspicious mix).
        """
        if len(window) == 0:
            return np.zeros(STATE_DIM, dtype=np.float32)

        idx = window.index.values
        return np.array([
            float(self._p_attack_all[idx].mean()),
            float(self._p_benign_all[idx].mean()),
            float(self._p_susp_all[idx].mean()),
            min(len(window) / FLOW_COUNT_MAX, 1.0),
        ], dtype=np.float32)

    def apply_action(
        self,
        window: pd.DataFrame,
        action: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Apply tiered threshold profile to window using p_risk = 1 - p_benign.

          p_risk >= T_block   →  BLOCK  (dropped)
          p_risk >= T_puzzle  →  PUZZLE (CAPTCHA challenge)
          p_risk <  T_puzzle  →  FORWARD
        """
        if len(window) == 0:
            return window

        t_block, t_puzzle = TIERED_PROFILES_V2.get(action, (1.0, 1.0))

        if t_block >= 1.0 and t_puzzle >= 1.0:
            return window

        n      = len(window)
        idx    = window.index.values
        p_risk = self._p_risk_all[idx]   # 1 - p_benign

        block_mask  = p_risk >= t_block
        puzzle_mask = (~block_mask) & (p_risk >= t_puzzle)

        # CAPTCHA: pass rate based on true label
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
# MultiAgentEnv
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEnv:
    """
    Coordinates N RouterEnvironments with p_risk-based tiered action profiles.
    Interface identical to environment_tiered.MultiAgentEnv.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bundle: dict,
        window_size: int = WINDOW_SIZE,
        us: float = Us,
        n_agents: int = N_AGENTS,
        seed: int = 42,
    ) -> None:
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)

        df = assign_agents(df, n_agents)

        self.routers: List[RouterEnvironment] = [
            RouterEnvironment(df, agent_id=i, bundle=bundle,
                              window_size=window_size)
            for i in range(n_agents)
        ]

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
        self.t_current = self.t_start
        for router in self.routers:
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
        self,
        all_forwarded: List[pd.DataFrame],
        t_window: float,
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
        self,
        all_forwarded: List[pd.DataFrame],
        t_window: float,
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
