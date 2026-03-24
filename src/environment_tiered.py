"""
environment_tiered.py
Tiered action environment for CoopIDS (S26+).

Key differences from environment.py:
  - No IP risk table — every flow is routed individually by its p_attack score.
  - No throttle — replaced by graduated block/puzzle/forward tiers.
  - No oracle — puzzle uses realistic CAPTCHA pass rates, routing uses LightGBM p_attack.
  - Actions are threshold profiles, not single tools.
  - State is 4-dim: [p_attack, p_benign, p_suspicious, flow_count_norm].

Each action selects a (T_block, T_puzzle) pair:
  flow.p_attack >= T_block   →  BLOCK   (dropped)
  flow.p_attack >= T_puzzle  →  PUZZLE  (CAPTCHA: Attack=5% pass, Benign=95% pass)
  flow.p_attack <  T_puzzle  →  FORWARD (passed)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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

# ── Tiered action profiles ───────────────────────────────────────────────────
# Each action = (T_block, T_puzzle).
# Flows with p_attack >= T_block are dropped.
# Flows with p_attack >= T_puzzle (but < T_block) get CAPTCHA challenge.
# Flows with p_attack <  T_puzzle are forwarded.

TIERED_PROFILES = {
    0: (1.00, 1.00),   # passive:    forward all
    1: (0.85, 0.35),   # light:      block very confident, puzzle moderate
    2: (0.60, 0.20),   # moderate:   ~94% TPR, 0.4% FPR
    3: (0.35, 0.10),   # firm:       ~97% TPR, 1.8% FPR
    4: (0.20, 0.05),   # aggressive: ~99% TPR, 7% FPR
    5: (0.10, 0.02),   # lockdown:   near total, higher FPR
}

ACTION_DIM    = len(TIERED_PROFILES)  # 6
STATE_DIM     = 4
JOINT_DIM     = STATE_DIM * N_AGENTS  # 12

# CAPTCHA pass rates — realistic, not oracle routing.
# The CAPTCHA mechanism itself separates bots from humans.
CAPTCHA_PASS = {
    "Benign":     0.95,
    "Suspicious": 0.50,
    "Attack":     0.05,
}

# Per-agent local reward (same as environment.py)
LOCAL_REWARD = True

# Action labels for logging
ACTION_LABELS = {
    0: "passive",
    1: "light",
    2: "moderate",
    3: "firm",
    4: "aggressive",
    5: "lockdown",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper — assign src_ip to agent
# ─────────────────────────────────────────────────────────────────────────────

def assign_agents(df: pd.DataFrame, n_agents: int = N_AGENTS) -> pd.DataFrame:
    df = df.copy()
    df["agent"] = df["src_ip"].apply(lambda ip: hash(str(ip)) % n_agents)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RouterEnvironment (Tiered)
# ─────────────────────────────────────────────────────────────────────────────

class RouterEnvironment:
    """
    Simulates one ISP edge router with tiered action profiles.

    No IP risk table. Every flow is individually routed by p_attack.
    State: [p_attack_mean, p_benign_mean, p_suspicious_mean, flow_count_norm]
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

        # Filter flows for this agent, sorted by timestamp
        self.flows = (
            df[df["agent"] == agent_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Timestamp array for fast window retrieval
        self._ts = pd.to_datetime(self.flows["timestamp"])
        _epoch = pd.Timestamp("1970-01-01")
        self._ts_sec = (self._ts - _epoch).dt.total_seconds()
        self._ts_sec_np = self._ts_sec.values.astype(np.float64)

        # Track start/end
        self.t_min = float(self._ts_sec.min()) if len(self._ts_sec) else 0.0
        self.t_max = float(self._ts_sec.max()) if len(self._ts_sec) else 0.0

        # Precompute LightGBM probabilities for ALL flows once.
        X_all = self._prepare_features(self.flows)
        if X_all is not None:
            _proba = self.model.predict_proba(X_all)
            _cls   = self.bundle["classes_3class"]
            self._p_attack_all = _proba[:, _cls.index("attack")].astype(np.float32)
            self._p_benign_all = _proba[:, _cls.index("benign")].astype(np.float32)
            self._p_susp_all   = _proba[:, _cls.index("suspicious")].astype(np.float32)
        else:
            n = len(self.flows)
            self._p_attack_all = np.zeros(n, np.float32)
            self._p_benign_all = np.ones(n,  np.float32)
            self._p_susp_all   = np.zeros(n, np.float32)

    def reset(self) -> None:
        """Reset at start of each episode. No IP table to clear."""
        pass

    # ── Feature preparation ──────────────────────────────────────────────────

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

    # ── Window retrieval ─────────────────────────────────────────────────────

    def get_window(self, t_start: float) -> pd.DataFrame:
        t_end = t_start + self.window_size
        lo = int(np.searchsorted(self._ts_sec_np, t_start, side='left'))
        hi = int(np.searchsorted(self._ts_sec_np, t_end,   side='left'))
        return self.flows.iloc[lo:hi].copy()

    # ── State computation ────────────────────────────────────────────────────

    def compute_state(self, window: pd.DataFrame) -> np.ndarray:
        """
        4-dim state: [p_attack, p_benign, p_suspicious, flow_count_norm]
        """
        if len(window) == 0:
            return np.zeros(STATE_DIM, dtype=np.float32)

        idx          = window.index.values
        p_attack     = float(self._p_attack_all[idx].mean())
        p_benign     = float(self._p_benign_all[idx].mean())
        p_suspicious = float(self._p_susp_all[idx].mean())
        flow_count_norm = min(len(window) / FLOW_COUNT_MAX, 1.0)

        return np.array(
            [p_attack, p_benign, p_suspicious, flow_count_norm],
            dtype=np.float32,
        )

    # ── Apply tiered action ──────────────────────────────────────────────────

    def apply_action(
        self,
        window: pd.DataFrame,
        action: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Apply tiered threshold profile to window.

        Each flow is individually routed by its p_attack:
          p_attack >= T_block   →  dropped (blocked)
          p_attack >= T_puzzle  →  CAPTCHA challenge (label-based pass rate)
          p_attack <  T_puzzle  →  forwarded
        """
        if len(window) == 0:
            return window

        t_block, t_puzzle = TIERED_PROFILES.get(action, (1.0, 1.0))

        # Passive — forward all
        if t_block >= 1.0 and t_puzzle >= 1.0:
            return window

        n   = len(window)
        idx = window.index.values
        p_attack = self._p_attack_all[idx]

        # Block tier: drop entirely
        block_mask = p_attack >= t_block

        # Puzzle tier: CAPTCHA challenge
        puzzle_mask = (~block_mask) & (p_attack >= t_puzzle)

        # Forward tier: pass through
        # forward_mask = (~block_mask) & (~puzzle_mask)  # implicit

        # Apply CAPTCHA to puzzle-tier flows
        # Pass rate depends on true label (realistic CAPTCHA behaviour)
        puzzle_pass = np.ones(n, dtype=np.float32)
        if puzzle_mask.any():
            puzzle_labels = window.loc[window.index[puzzle_mask], "label"]
            puzzle_pass[puzzle_mask] = (
                puzzle_labels.map(CAPTCHA_PASS).fillna(0.5).values.astype(np.float32)
            )

        # Stochastic CAPTCHA: each puzzled flow passes with its label-based rate
        captcha_result = rng.random(n) < puzzle_pass

        # Keep flow if: not blocked AND (not puzzled OR passed captcha)
        keep_mask = (~block_mask) & (~puzzle_mask | captcha_result)

        return window[keep_mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentEnv (Tiered)
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEnv:
    """
    Coordinates N RouterEnvironments with tiered action profiles.

    Simpler than the original: no IP risk table management, no module-level
    flags. Each agent picks a threshold profile per window.
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

    # ── Episode control ──────────────────────────────────────────────────────

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        self.t_current = self.t_start
        for router in self.routers:
            router.reset()

        local_states = self._get_local_states()
        joint_obs    = np.concatenate(local_states).astype(np.float32)
        return local_states, joint_obs

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: List[int],
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool]:
        """
        Advance one time window.

        1. Each router gets window t+1
        2. Apply tiered action on window t+1
        3. Compute reward from forwarded flows
        4. Advance clock
        5. Compute next states
        """
        assert len(actions) == self.n_agents

        t = self.t_current
        all_forwarded: List[pd.DataFrame] = []

        for router, action in zip(self.routers, actions):
            window_t1 = router.get_window(t + self.window_size)
            forwarded = router.apply_action(window_t1, action, self.rng)
            all_forwarded.append(forwarded)

        # Reward
        if LOCAL_REWARD:
            reward = self._compute_local_rewards(all_forwarded, t + self.window_size)
        else:
            reward = self._compute_reward(all_forwarded, t + self.window_size)

        # Advance clock
        self.t_current = t + self.window_size
        done = self.t_current >= self.t_end

        # Next states
        next_local_states = self._get_local_states()
        next_joint_obs    = np.concatenate(next_local_states).astype(np.float32)

        return next_local_states, next_joint_obs, reward, done

    # ── State helpers ────────────────────────────────────────────────────────

    def _get_local_states(self) -> List[np.ndarray]:
        states = []
        for router in self.routers:
            window = router.get_window(self.t_current)
            state  = router.compute_state(window)
            states.append(state)
        return states

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        all_forwarded: List[pd.DataFrame],
        t_window: float,
    ) -> float:
        """
        reward = legit_ratio - attack_ratio
        Range [-1, +1].
        """
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

        legit_passed  = 0
        attack_passed = 0
        if "label" in forwarded_all.columns:
            legit_passed  = int((forwarded_all["label"] == "Benign").sum())
            attack_passed = int((forwarded_all["label"] == "Attack").sum())

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
        """Per-agent local reward."""
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
                r = -1.0 if legit_total > 0 else 0.0
                rewards.append(r)
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

        # Update global last_step_info for eval logging
        self._compute_reward(all_forwarded, t_window)
        self.last_step_info["per_agent"] = per_agent_data
        return rewards

    # ── Info ─────────────────────────────────────────────────────────────────

    @property
    def n_windows(self) -> int:
        return int((self.t_end - self.t_start) / self.window_size)
