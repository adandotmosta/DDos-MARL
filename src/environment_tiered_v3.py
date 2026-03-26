"""
environment_tiered_v3.py
Tiered action environment with shared IP blocklist (S33/S34).

Extends TieredEnvV2 with:
  - Per-router IP risk table: IPs flagged when p_risk >= FLAG_THRESHOLD
  - Shared threat signal: total flagged IPs across ALL routers / SHARED_IP_MAX
  - 5-dim state: [p_attack, p_benign, p_suspicious, flow_count_norm, shared_threat_level]

Why this creates coordination potential:
  With 4-dim state, each agent sees only its own subnet's risk signals.
  The shared_threat_level gives Agent 2 visibility into what Agent 1 is observing.
  If Agent 1's subnet is flooded (many IPs flagged), Agent 2 can pre-emptively
  escalate its own thresholds before its subnet is targeted.

  This is the first mechanism in CoopIDS where MAPPO's joint critic could
  provide genuine value over IPPO's independent critics — the shared signal
  creates cross-agent information that benefits from centralized coordination.

State: [p_attack_mean, p_benign_mean, p_suspicious_mean, flow_count_norm, shared_threat_level]
  - shared_threat_level = sum(ip_table sizes across all routers) / SHARED_IP_MAX
  - Clipped to [0, 1]

IP flagging:
  - IP added when p_risk >= FLAG_THRESHOLD (default 0.70 = firm T_block)
  - IP table persists for whole episode (no expiry — simplest design)
  - Flagging happens during apply_action (when we already have p_risk per flow)

Comparison design:
  S33 (MAPPO): joint critic sees 15-dim joint state — can coordinate responses
  S34 (IPPO):  each agent sees 5-dim local state only — independent but sees shared signal
  If S33 > S34 → CTDE exploits shared signal for genuine coordination
  If S33 ≈ S34 → shared signal helps but CTDE not needed (independent use is enough)
  If both > S29 → shared threat level improves detection regardless of algorithm
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

# Reuse same action profiles as V2
TIERED_PROFILES_V3 = {
    0: (1.00, 1.00),   # passive
    1: (0.90, 0.60),   # light
    2: (0.80, 0.40),   # moderate
    3: (0.70, 0.25),   # firm
    4: (0.60, 0.15),   # aggressive
    5: (0.50, 0.10),   # lockdown
}

ACTION_DIM    = len(TIERED_PROFILES_V3)   # 6
STATE_DIM     = 5                          # 4-dim + shared_threat_level
JOINT_DIM     = STATE_DIM * N_AGENTS      # 15

FLAG_THRESHOLD = 0.70    # p_risk threshold to add IP to blocklist (firm level)
SHARED_IP_MAX  = 500     # normalisation constant for shared threat level

CAPTCHA_PASS = {
    "Benign":     0.95,
    "Suspicious": 0.50,
    "Attack":     0.05,
}

LOCAL_REWARD = True

ACTION_LABELS = {
    0: "passive", 1: "light", 2: "moderate",
    3: "firm", 4: "aggressive", 5: "lockdown",
}


def assign_agents(df: pd.DataFrame, n_agents: int = N_AGENTS,
                  routing_seed: int | None = None) -> pd.DataFrame:
    df = df.copy()
    if routing_seed is None:
        df["agent"] = df["src_ip"].apply(lambda ip: hash(str(ip)) % n_agents)
    else:
        rng = np.random.RandomState(routing_seed)
        unique_ips = df["src_ip"].unique()
        ip_map = {ip: int(rng.randint(0, n_agents)) for ip in unique_ips}
        df["agent"] = df["src_ip"].map(ip_map)
    return df


class RouterEnvironmentV3:
    """
    ISP edge router with p_risk tiered profiles + IP blocklist tracking.
    Exposes ip_table_size so MultiAgentEnvV3 can compute shared_threat_level.
    """

    def __init__(self, df, agent_id, bundle, window_size=WINDOW_SIZE):
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
        self._ts_sec    = (self._ts - _epoch).dt.total_seconds()
        self._ts_sec_np = self._ts_sec.values.astype(np.float64)
        self.t_min = float(self._ts_sec.min()) if len(self._ts_sec) else 0.0
        self.t_max = float(self._ts_sec.max()) if len(self._ts_sec) else 0.0

        X_all = self._prepare_features(self.flows)
        if X_all is not None:
            _proba = self.model.predict_proba(X_all)
            _cls   = bundle["classes_3class"]
            self._p_attack_all = _proba[:, _cls.index("attack")].astype(np.float32)
            self._p_benign_all = _proba[:, _cls.index("benign")].astype(np.float32)
            self._p_susp_all   = _proba[:, _cls.index("suspicious")].astype(np.float32)
            self._p_risk_all   = (1.0 - self._p_benign_all).astype(np.float32)
        else:
            n = len(self.flows)
            self._p_attack_all = np.zeros(n, np.float32)
            self._p_benign_all = np.ones(n,  np.float32)
            self._p_susp_all   = np.zeros(n, np.float32)
            self._p_risk_all   = np.zeros(n, np.float32)

        # IP blocklist — persists for the episode
        self.ip_blocklist: set = set()

    def reset(self) -> None:
        self.ip_blocklist = set()

    @property
    def ip_table_size(self) -> int:
        return len(self.ip_blocklist)

    def _prepare_features(self, window):
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
        lo = int(np.searchsorted(self._ts_sec_np, t_start, side="left"))
        hi = int(np.searchsorted(self._ts_sec_np, t_end,   side="left"))
        return self.flows.iloc[lo:hi].copy()

    def compute_local_state(self, window: pd.DataFrame) -> np.ndarray:
        """
        4-dim local state (shared_threat_level appended by MultiAgentEnvV3).
        [p_attack_mean, p_benign_mean, p_suspicious_mean, flow_count_norm]
        """
        if len(window) == 0:
            return np.zeros(4, dtype=np.float32)
        idx = window.index.values
        return np.array([
            float(self._p_attack_all[idx].mean()),
            float(self._p_benign_all[idx].mean()),
            float(self._p_susp_all[idx].mean()),
            min(len(window) / FLOW_COUNT_MAX, 1.0),
        ], dtype=np.float32)

    def apply_action(self, window, action, rng):
        if len(window) == 0:
            return window

        t_block, t_puzzle = TIERED_PROFILES_V3.get(action, (1.0, 1.0))
        if t_block >= 1.0 and t_puzzle >= 1.0:
            return window

        n      = len(window)
        idx    = window.index.values
        p_risk = self._p_risk_all[idx]

        # Flag IPs with high p_risk into blocklist
        high_risk_mask = p_risk >= FLAG_THRESHOLD
        if high_risk_mask.any():
            flagged_ips = window.loc[window.index[high_risk_mask], "src_ip"].values
            self.ip_blocklist.update(flagged_ips.tolist())

        block_mask  = p_risk >= t_block
        puzzle_mask = (~block_mask) & (p_risk >= t_puzzle)

        puzzle_pass = np.ones(n, dtype=np.float32)
        if puzzle_mask.any():
            puzzle_labels = window.loc[window.index[puzzle_mask], "label"]
            puzzle_pass[puzzle_mask] = (
                puzzle_labels.map(CAPTCHA_PASS).fillna(0.5).values.astype(np.float32)
            )

        captcha_result = rng.random(n) < puzzle_pass
        keep_mask = (~block_mask) & (~puzzle_mask | captcha_result)
        return window[keep_mask].copy()


class MultiAgentEnvV3:
    """
    Coordinates N RouterEnvironmentV3 instances.
    Appends shared_threat_level to each agent's local state.

    shared_threat_level = sum(router.ip_table_size for all routers) / SHARED_IP_MAX
    Clipped to [0, 1].

    This is the coordination signal: each agent sees the aggregate IP flagging
    rate across the whole system, not just its own router.
    """

    def __init__(self, df, bundle, window_size=WINDOW_SIZE, us=Us,
                 n_agents=N_AGENTS, seed=42):
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)

        df = assign_agents(df, n_agents)

        self.routers: List[RouterEnvironmentV3] = [
            RouterEnvironmentV3(df, agent_id=i, bundle=bundle,
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
        self.last_step_info: dict = {}

    def _shared_threat_level(self) -> float:
        total = sum(r.ip_table_size for r in self.routers)
        return min(total / SHARED_IP_MAX, 1.0)

    def _get_local_states(self) -> List[np.ndarray]:
        stl = self._shared_threat_level()
        states = []
        for r in self.routers:
            window = r.get_window(self.t_current)
            local4 = r.compute_local_state(window)
            # Append shared threat level → 5-dim state
            state5 = np.append(local4, stl).astype(np.float32)
            states.append(state5)
        return states

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        self.t_current = self.t_start
        for r in self.routers:
            r.reset()
        local_states = self._get_local_states()
        joint_obs    = np.concatenate(local_states).astype(np.float32)
        return local_states, joint_obs

    def step(self, actions):
        assert len(actions) == self.n_agents

        t = self.t_current
        all_forwarded = []

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

        next_local = self._get_local_states()
        next_joint = np.concatenate(next_local).astype(np.float32)

        return next_local, next_joint, reward, done

    def _compute_reward(self, all_forwarded, t_window):
        legit_total = attack_total = 0
        for r in self.routers:
            w = r.get_window(t_window)
            if "label" in w.columns:
                legit_total  += int((w["label"] == "Benign").sum())
                attack_total += int((w["label"] == "Attack").sum())

        if legit_total == 0 and attack_total == 0:
            self.last_step_info = {
                "benign_total": 0, "attack_total": 0,
                "benign_forwarded": 0, "attack_blocked": 0,
            }
            return 0.0

        non_empty = [f for f in all_forwarded if len(f) > 0]
        if not non_empty:
            self.last_step_info = {
                "benign_total": legit_total, "attack_total": attack_total,
                "benign_forwarded": 0, "attack_blocked": attack_total,
            }
            return -1.0 if legit_total > 0 else 0.0

        fwd = pd.concat(non_empty, ignore_index=True)
        legit_passed  = int((fwd["label"] == "Benign").sum()) if "label" in fwd.columns else 0
        attack_passed = int((fwd["label"] == "Attack").sum()) if "label" in fwd.columns else 0

        self.last_step_info = {
            "benign_total": legit_total, "attack_total": attack_total,
            "benign_forwarded": legit_passed,
            "attack_blocked": attack_total - attack_passed,
        }
        legit_ratio  = legit_passed  / max(legit_total,  1)
        attack_ratio = attack_passed / max(attack_total, 1)
        return float(REWARD_LEGIT_W * legit_ratio - REWARD_ATTACK_W * attack_ratio)

    def _compute_local_rewards(self, all_forwarded, t_window):
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
                per_agent_data.append({"benign_total": legit_total,
                                       "attack_total": attack_total,
                                       "benign_passed": 0, "attack_passed": 0})
                continue

            legit_passed  = int((forwarded["label"] == "Benign").sum())
            attack_passed = int((forwarded["label"] == "Attack").sum())
            per_agent_data.append({"benign_total": legit_total,
                                   "attack_total": attack_total,
                                   "benign_passed": legit_passed,
                                   "attack_passed": attack_passed})
            legit_ratio  = legit_passed  / max(legit_total,  1)
            attack_ratio = attack_passed / max(attack_total, 1)
            rewards.append(float(REWARD_LEGIT_W * legit_ratio - REWARD_ATTACK_W * attack_ratio))

        self._compute_reward(all_forwarded, t_window)
        self.last_step_info["per_agent"] = per_agent_data
        return rewards

    @property
    def n_windows(self) -> int:
        return int((self.t_end - self.t_start) / self.window_size)
