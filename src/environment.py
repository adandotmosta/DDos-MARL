"""
environment.py
RouterEnvironment  — single router simulation
MultiAgentEnv      — 3 routers synchronized, MARL-compatible
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# LightGBM predict_proba raises this when passed a numpy array instead of
# a named DataFrame — harmless here since we control feature order.
warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")

# ── import config (works whether run from project root or src/)
try:
    from config import (
        N_AGENTS, WINDOW_SIZE, Us, IP_TIMEOUT, BASE_THRESHOLD,
        VICTIM_IPS, PUZZLE_PASS, DROP_PROBS, STATE_DIM, JOINT_DIM,
        ACTION_DIM, FLOW_COUNT_MAX, IP_TABLE_MAX, UNIQUE_IP_MAX,
        REWARD_LEGIT_W, REWARD_ATTACK_W, LOCAL_REWARD, EXTENDED_STATE, PUZZLE_COST,
    )
except ImportError:
    from coopids.config import (
        N_AGENTS, WINDOW_SIZE, Us, IP_TIMEOUT, BASE_THRESHOLD,
        VICTIM_IPS, PUZZLE_PASS, DROP_PROBS, STATE_DIM, JOINT_DIM,
        ACTION_DIM, FLOW_COUNT_MAX, IP_TABLE_MAX, UNIQUE_IP_MAX,
        REWARD_LEGIT_W, REWARD_ATTACK_W, LOCAL_REWARD, EXTENDED_STATE, PUZZLE_COST,
    )

# Patch this to False in train_s12 / evaluate.py S12 block to disable puzzle action.
PUZZLE_ENABLED = True

# Patch this to True in train_s14 to add puzzle_block_rate as 6th state feature.
PUZZLE_STATE = False

# S15: Risk-aware throttle — drop high P(attack) flows instead of random drops.
SMART_THROTTLE = False

# S16: Dynamic puzzle — attack pass rate drifts up during episode.
DYNAMIC_PUZZLE = False

# S17: LightGBM prediction entropy as 6th state feature.
ENTROPY_STATE = False

# S23: Selective puzzle — puzzle only challenges flows predicted as Suspicious.
# Attack-predicted and Benign-predicted flows pass through puzzle unchanged,
# forcing agents to use throttle/block for high-confidence attack flows.
SELECTIVE_PUZZLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper — load lgbm bundle
# ─────────────────────────────────────────────────────────────────────────────

def load_lgbm_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    required = {"model_3class", "features", "imputer", "handshake_mapping"}
    missing  = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Bundle missing keys: {missing}")
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Helper — assign src_ip to agent
# ─────────────────────────────────────────────────────────────────────────────

def assign_agents(df: pd.DataFrame, n_agents: int = N_AGENTS) -> pd.DataFrame:
    """
    Permanently assign each src_ip to one agent via hash.
    Same IP always goes through same router — realistic routing assumption.
    """
    df = df.copy()
    df["agent"] = df["src_ip"].apply(lambda ip: hash(str(ip)) % n_agents)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RouterEnvironment
# ─────────────────────────────────────────────────────────────────────────────

class RouterEnvironment:
    """
    Simulates one ISP edge router.

    Each router:
      - Owns a fixed subset of src_IPs (assigned via hash)
      - Observes 30-second windows of flows
      - Maintains an IP risk table
      - Applies actions (forward / puzzle / throttle / block)
      - Computes local state: [p_attack, p_benign, p_suspicious, p_anomaly, load]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        agent_id: int,
        bundle: dict,
        window_size: int = WINDOW_SIZE,
        us: float = Us,
        base_threshold: float = BASE_THRESHOLD,
        ip_timeout: int = IP_TIMEOUT,
    ) -> None:
        self.agent_id      = agent_id
        self.bundle        = bundle
        self.model         = bundle["model_3class"]
        self.features      = bundle["features"]
        self.imputer       = bundle["imputer"]
        self.hs_mapping    = bundle["handshake_mapping"]
        self.window_size   = window_size
        self.Us            = us
        self.base_threshold= base_threshold
        self.ip_timeout    = ip_timeout

        # Filter flows for this agent only, sorted by timestamp
        self.flows = (
            df[df["agent"] == agent_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Convert timestamp to float seconds for fast comparison
        # pandas 3.0: astype(np.int64) returns microseconds, not nanoseconds.
        # Use total_seconds() instead.
        self._ts = pd.to_datetime(self.flows["timestamp"])
        _epoch = pd.Timestamp("1970-01-01")
        self._ts_sec = (self._ts - _epoch).dt.total_seconds()

        # Sorted numpy array for O(log n) searchsorted window retrieval
        self._ts_sec_np = self._ts_sec.values.astype(np.float64)

        # IP risk table: {src_ip: (risk_score, timestamp_sec)}
        self.ip_risk_table: Dict[str, Tuple[float, float]] = {}

        # Track start/end
        self.t_min = float(self._ts_sec.min()) if len(self._ts_sec) else 0.0
        self.t_max = float(self._ts_sec.max()) if len(self._ts_sec) else 0.0

        # Rolling puzzle outcome tracking for PUZZLE_STATE feature
        # Tracks attacks blocked vs total flows passed through puzzle over last N steps
        self._puzzle_attacks_blocked = 0
        self._puzzle_total_passed    = 0

        # Precompute LightGBM probabilities for ALL flows once.
        # Monday data is fixed — probabilities are identical every episode.
        # Eliminates ~5,808 predict_proba calls/episode at runtime.
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
        """Clear IP risk table at start of each episode."""
        self.ip_risk_table = {}
        self._puzzle_attacks_blocked = 0
        self._puzzle_total_passed    = 0

    # ── Feature preparation ──────────────────────────────────────────────────

    def _prepare_features(self, window: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract and clean feature matrix from a window of flows."""
        if len(window) == 0:
            return None

        X = window[self.features].copy()

        # Encode handshake_state if present
        if "handshake_state" in X.columns and self.hs_mapping:
            X["handshake_state"] = (
                X["handshake_state"]
                .astype(str)
                .map(self.hs_mapping)
                .fillna(-1)
                .astype(np.float32)
            )

        # Numeric conversion and clean
        for col in X.columns:
            if col != "handshake_state":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
        return self.imputer.transform(X)

    # ── Window retrieval ─────────────────────────────────────────────────────

    def get_window(self, t_start: float) -> pd.DataFrame:
        """
        Return all flows in [t_start, t_start + window_size).
        Binary search on sorted timestamp array: O(log n) vs O(n) boolean mask.
        Correct because self.flows is sorted by timestamp and reset_index(drop=True),
        so iloc[lo:hi] gives the same slice as the boolean mask did.
        """
        t_end = t_start + self.window_size
        lo = int(np.searchsorted(self._ts_sec_np, t_start, side='left'))
        hi = int(np.searchsorted(self._ts_sec_np, t_end,   side='left'))
        return self.flows.iloc[lo:hi].copy()

    # ── State computation ────────────────────────────────────────────────────

    def compute_state(self, window: pd.DataFrame) -> np.ndarray:
        """
        Compute local state vector from a window of flows.

        State: [p_attack, p_benign, p_suspicious, ip_table_ratio, flow_count_norm]
        - p_attack, p_benign, p_suspicious: mean LightGBM probabilities
        - ip_table_ratio  = current flagged-IP count / IP_TABLE_MAX   (agent memory)
        - flow_count_norm = flows in this window / FLOW_COUNT_MAX      (traffic volume)

        Replaced from earlier design:
        - p_anomaly removed: it's p_attack + p_suspicious — a perfect linear
          combination of two already-present features; zero new information.
        - load_norm removed: sum(bytes_rate)/Us ≈ 0.009 always (per-flow
          bytes_rate ~570 B/s, Us=1e8); effectively constant, no signal.
        """
        if len(window) == 0:
            dim = STATE_DIM + (1 if PUZZLE_STATE else 0) + (1 if ENTROPY_STATE else 0) + (2 if EXTENDED_STATE else 0)
            return np.zeros(dim, dtype=np.float32)

        # Use precomputed probability arrays — zero inference at runtime.
        # window.index.values = [lo, lo+1, ..., hi-1] (positions in self.flows
        # after reset_index + searchsorted iloc slice).
        idx          = window.index.values
        p_attack     = float(self._p_attack_all[idx].mean())
        p_benign     = float(self._p_benign_all[idx].mean())
        p_suspicious = float(self._p_susp_all[idx].mean())

        # IP table size — tells agent how many IPs are currently flagged
        ip_table_ratio = min(len(self.ip_risk_table) / IP_TABLE_MAX, 1.0)

        # Flow count — high during attacks, low during benign periods
        flow_count_norm = min(len(window) / FLOW_COUNT_MAX, 1.0)

        base_state = [p_attack, p_benign, p_suspicious, ip_table_ratio, flow_count_norm]

        if PUZZLE_STATE:
            # puzzle_block_rate: fraction of puzzle-stopped flows that were attacks
            puzzle_block_rate = (
                self._puzzle_attacks_blocked / max(self._puzzle_total_passed, 1)
            )
            return np.array(base_state + [puzzle_block_rate], dtype=np.float32)

        if ENTROPY_STATE:
            # S17: Shannon entropy of LightGBM predictions — model confidence signal
            idx = window.index.values
            probs = np.stack([
                self._p_attack_all[idx],
                self._p_benign_all[idx],
                self._p_susp_all[idx],
            ], axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            prediction_entropy = float(entropy.mean()) / np.log(3)  # normalize [0,1]
            return np.array(base_state + [prediction_entropy], dtype=np.float32)

        if not EXTENDED_STATE:
            return np.array(base_state, dtype=np.float32)

        # E13 — Spatial features (DDoS signature detection)
        # unique_src_ips_norm: many sources → DDoS; few sources → targeted attack
        unique_src_ips_norm = min(
            window["src_ip"].nunique() / UNIQUE_IP_MAX, 1.0
        ) if "src_ip" in window.columns else 0.0

        # attack_ip_ratio: fraction of flows from already-flagged IPs
        # Tells agent whether IP table is "hot" right now
        if len(window) > 0 and "src_ip" in window.columns:
            flagged_count = window["src_ip"].astype(str).isin(self.ip_risk_table).sum()
            attack_ip_ratio = float(flagged_count) / len(window)
        else:
            attack_ip_ratio = 0.0

        return np.array(
            base_state + [unique_src_ips_norm, attack_ip_ratio],
            dtype=np.float32,
        )

    # ── IP risk table ────────────────────────────────────────────────────────

    def update_ip_table(self, window: pd.DataFrame) -> None:
        """
        Flag src_IPs with high attack probability into the risk table.
        Called on window t before acting on window t+1.
        Uses precomputed probabilities and vectorised numpy — no LightGBM call.
        """
        if len(window) == 0:
            return

        idx   = window.index.values
        risks = self._p_attack_all[idx]

        # Vectorised threshold: find positions in the window slice
        flagged_positions = np.where(risks >= self.base_threshold)[0]
        if len(flagged_positions) == 0:
            return

        flagged_rows  = window.iloc[flagged_positions]
        flagged_risks = risks[flagged_positions]
        flagged_ts    = self._ts_sec_np[flagged_rows.index.values]

        for ip, risk, ts in zip(
            flagged_rows["src_ip"].astype(str).values,
            flagged_risks,
            flagged_ts,
        ):
            self.ip_risk_table[ip] = (float(risk), float(ts))

    def decay_ip_table(self, current_time: float) -> None:
        """Remove IP table entries older than ip_timeout seconds."""
        expired = [
            ip for ip, (_, t) in self.ip_risk_table.items()
            if current_time - t > self.ip_timeout
        ]
        for ip in expired:
            del self.ip_risk_table[ip]

    # ── Apply action ─────────────────────────────────────────────────────────

    def apply_action(
        self,
        window: pd.DataFrame,
        action: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Apply agent action to window t+1 flows.

        Actions:
          0 → forward all
          1 → puzzle   (pass rate depends on true label)
          2 → throttle_light  (drop 30% of flagged IPs)
          3 → throttle_medium (drop 60% of flagged IPs)
          4 → throttle_heavy  (drop 90% of flagged IPs)
          5 → block           (drop 100% of flagged IPs)

        Returns: DataFrame of forwarded flows only.
        """
        if len(window) == 0 or action == 0:
            return window

        n = len(window)

        if action == 1 and PUZZLE_ENABLED:
            # Puzzle — applies to ALL flows; pass rate depends on true label.
            if SELECTIVE_PUZZLE:
                # S23: pass rate based on LightGBM predicted class, not ground truth.
                # Attack-predicted → pass_rate=1.0 (puzzle useless, must block/throttle)
                # Suspicious-predicted → pass_rate=0.5 (genuine uncertainty, verify)
                # Benign-predicted → pass_rate=1.0 (passes through, no need to challenge)
                idx = window.index.values
                pred_class = np.argmax(
                    np.stack([self._p_benign_all[idx],
                              self._p_susp_all[idx],
                              self._p_attack_all[idx]], axis=1), axis=1
                )  # 0=benign-predicted, 1=suspicious-predicted, 2=attack-predicted
                pass_rates = np.where(pred_class == 1, 0.50, 1.00).astype(np.float32)
            elif DYNAMIC_PUZZLE:
                # S16: attack pass rate drifts up during episode
                progress = getattr(self, '_episode_progress', 0.0)
                dynamic_pass = {
                    "Benign":     0.95 - 0.10 * progress,
                    "Suspicious": 0.50 - 0.10 * progress,
                    "Attack":     0.05 + 0.25 * progress,
                }
                pass_rates = window["label"].map(dynamic_pass).fillna(0.5).values
            else:
                pass_rates = window["label"].map(PUZZLE_PASS).fillna(0.5).values
            keep_mask  = rng.random(n) < pass_rates
            forwarded  = window[keep_mask].copy()
            # Track puzzle outcomes for PUZZLE_STATE feature
            if PUZZLE_STATE:
                blocked = int((window[~keep_mask]["label"] == "Attack").sum())
                self._puzzle_attacks_blocked += blocked
                self._puzzle_total_passed    += int((~keep_mask).sum())
            return forwarded

        if action in DROP_PROBS:
            # Throttle / block — only affects IPs in the risk table.
            flagged = window["src_ip"].astype(str).isin(self.ip_risk_table).values
            if SMART_THROTTLE:
                # S15: drop flows with highest P(attack) instead of random
                idx = window.index.values
                flow_risks = self._p_attack_all[idx]
                risk_threshold = 1.0 - DROP_PROBS[action]
                drop_mask = flagged & (flow_risks > risk_threshold)
            else:
                drop_mask = flagged & (rng.random(n) < DROP_PROBS[action])
            return window[~drop_mask].copy()

        return window


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentEnv
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEnv:
    """
    Coordinates 3 RouterEnvironments.

    - All routers advance through Monday data in lockstep (same timestamp).
    - CTDE: each agent acts on local obs; critic sees joint obs.
    - Reward is global (shared across agents).
    - One episode = one full replay of Monday data.
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

        # Assign src_ips to agents
        df = assign_agents(df, n_agents)

        # One router per agent
        self.routers: List[RouterEnvironment] = [
            RouterEnvironment(df, agent_id=i, bundle=bundle,
                              window_size=window_size, us=us)
            for i in range(n_agents)
        ]

        # Global time grid (all routers share same clock)
        # pandas 3.0: astype(np.int64) returns microseconds, not nanoseconds.
        ts = pd.to_datetime(df["timestamp"])
        _epoch = pd.Timestamp("1970-01-01")
        ts_sec = (ts - _epoch).dt.total_seconds()
        self.t_start   = float(ts_sec.min())
        self.t_end     = float(ts_sec.max())
        self.t_current = self.t_start

        # Victim IPs for reward computation
        self.victim_ips = VICTIM_IPS

        # Keep original df for reward computation
        self._df = df

        # Populated by _compute_reward() each step — read by evaluator
        self.last_step_info: dict = {}

    # ── Episode control ──────────────────────────────────────────────────────

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Reset to start of Monday.
        Returns:
            local_states: list of state vectors, one per agent
            joint_obs:    concatenation of all local states (for critic)
        """
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

        1. Each router observes window t → updates IP table → acts on window t+1
        2. Compute global reward from all forwarded flows
        3. Advance clock by window_size
        4. Compute next states

        Args:
            actions: list of action ints, one per agent

        Returns:
            next_local_states: list of state vectors
            next_joint_obs:    concatenation for critic
            reward:            global scalar reward
            done:              True if end of episode
        """
        assert len(actions) == self.n_agents

        t = self.t_current

        # S16: propagate episode progress for dynamic puzzle pass rates
        if DYNAMIC_PUZZLE:
            progress = (t - self.t_start) / max(self.t_end - self.t_start, 1)
            for router in self.routers:
                router._episode_progress = progress

        # ── For each router: observe t, update table, act on t+1
        all_forwarded: List[pd.DataFrame] = []

        for i, (router, action) in enumerate(zip(self.routers, actions)):
            window_t  = router.get_window(t)
            window_t1 = router.get_window(t + self.window_size)

            # Update IP table from current window
            router.update_ip_table(window_t)

            # Apply action on next window
            forwarded = router.apply_action(window_t1, action, self.rng)
            all_forwarded.append(forwarded)

            # Decay old IP table entries
            router.decay_ip_table(t)

        # ── Reward (global scalar or per-agent list)
        if LOCAL_REWARD:
            reward = self._compute_local_rewards(all_forwarded, t + self.window_size)
        else:
            reward = self._compute_reward(all_forwarded, t + self.window_size)

        # ── Puzzle cost — penalise each puzzle action taken this step
        # PUZZLE_COST = 0.0 by default (no-op, preserves all prior experiments).
        # Set > 0 in config or patch _env_module.PUZZLE_COST to study tradeoff.
        if PUZZLE_COST > 0.0:
            puzzle_count = sum(1 for a in actions if a == 1 and PUZZLE_ENABLED)
            if isinstance(reward, list):
                reward = [r - PUZZLE_COST * puzzle_count for r in reward]
            else:
                reward = reward - PUZZLE_COST * puzzle_count

        # ── Advance clock
        self.t_current = t + self.window_size
        done = self.t_current >= self.t_end

        # ── Next states
        next_local_states = self._get_local_states()
        next_joint_obs    = np.concatenate(next_local_states).astype(np.float32)

        return next_local_states, next_joint_obs, reward, done

    # ── State helpers ─────────────────────────────────────────────────────────

    def _get_local_states(self) -> List[np.ndarray]:
        """Compute current state for each router."""
        states = []
        for router in self.routers:
            window = router.get_window(self.t_current)
            state  = router.compute_state(window)
            states.append(state)
        return states

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        all_forwarded: List[pd.DataFrame],
        t_window: float,
    ) -> float:
        """
        Direct detection reward:
            reward = legit_ratio - attack_ratio

        where:
            legit_ratio  = Benign flows forwarded / total Benign in window  → want HIGH
            attack_ratio = Attack flows forwarded / total Attack in window  → want LOW

        Range: [-1, +1].
          +1 = perfect (all benign pass, all attacks blocked)
          -1 = worst   (all benign blocked, all attacks pass)
          ~0 = random policy (roughly equal benign/attack pass-through)

        Suspicious flows are excluded — they drive the puzzle action via
        state signal (p_suspicious), not via direct reward.

        Note: bytes_rate / Us is still used in the state (load_norm) for the
        observation — Us is preserved in config.py for that purpose.
        """
        # ── Totals in this window (ground truth, training only)
        legit_total  = 0
        attack_total = 0
        for router in self.routers:
            window = router.get_window(t_window)
            if "label" in window.columns:
                legit_total  += int((window["label"] == "Benign").sum())
                attack_total += int((window["label"] == "Attack").sum())

        # Empty window — nothing happened
        if legit_total == 0 and attack_total == 0:
            self.last_step_info = {
                "benign_total": 0, "attack_total": 0,
                "benign_forwarded": 0, "attack_forwarded": 0,
                "attack_blocked": 0,
            }
            return 0.0

        # All traffic was dropped
        non_empty = [f for f in all_forwarded if len(f) > 0]
        if not non_empty:
            self.last_step_info = {
                "benign_total":     legit_total,
                "attack_total":     attack_total,
                "benign_forwarded": 0,
                "attack_forwarded": 0,
                "attack_blocked":   attack_total,
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
        """
        Per-agent local reward variant of _compute_reward.
        Each agent is rewarded only on its own router's flows.
        Returns a list of N_AGENTS floats.
        """
        rewards = []
        per_agent_data = []
        for i, (router, forwarded) in enumerate(zip(self.routers, all_forwarded)):
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

        # Update last_step_info using global totals for logging consistency
        self._compute_reward(all_forwarded, t_window)
        # Expose per-agent breakdowns for counterfactual reward computation (S22)
        self.last_step_info["per_agent"] = per_agent_data
        return rewards

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        """Return current environment state for logging."""
        return {
            "t_current":    self.t_current,
            "t_start":      self.t_start,
            "t_end":        self.t_end,
            "progress":     (self.t_current - self.t_start) / (self.t_end - self.t_start),
            "ip_table_sizes": [len(r.ip_risk_table) for r in self.routers],
        }

    @property
    def n_windows(self) -> int:
        """Approximate number of windows per episode."""
        return int((self.t_end - self.t_start) / self.window_size)
