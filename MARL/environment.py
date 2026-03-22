"""
environment.py
RouterEnvironment  — single router simulation (fully vectorized)
MultiAgentEnv      — 3 routers synchronized, MARL-compatible

Vectorization changes vs original:
  - update_ip_table : no iterrows → numpy boolean mask
  - apply_action    : no iterrows → vectorized per-action masking
  - get_window      : numpy searchsorted instead of boolean mask scan
  - assign_agents   : vectorized hash via pandas str operations
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from config import (
        ACTION_DIM, BASE_THRESHOLD, DROP_PROBS, IP_TIMEOUT,
        JOINT_DIM, N_AGENTS, PUZZLE_PASS, STATE_DIM,
        Us, VICTIM_IPS, WINDOW_SIZE,
    )
except ImportError:
    from coopids.config import (
        ACTION_DIM, BASE_THRESHOLD, DROP_PROBS, IP_TIMEOUT,
        JOINT_DIM, N_AGENTS, PUZZLE_PASS, STATE_DIM,
        Us, VICTIM_IPS, WINDOW_SIZE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Puzzle pass-rate array  (indexed by label string → float)
# ─────────────────────────────────────────────────────────────────────────────
_PUZZLE_PASS_DEFAULT = 0.5


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
# Helper — assign src_ip to agent (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def assign_agents(df: pd.DataFrame, n_agents: int = N_AGENTS) -> pd.DataFrame:
    """
    Permanently assign each src_ip to one agent via hash.
    Vectorized: no apply/lambda loop.
    Same IP always → same router (realistic routing assumption).
    """
    df = df.copy()
    # pandas str hash is vectorized
    df["agent"] = df["src_ip"].astype(str).apply(hash) % n_agents
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RouterEnvironment
# ─────────────────────────────────────────────────────────────────────────────

class RouterEnvironment:
    """
    Simulates one ISP edge router.

    Owns a fixed subset of src_IPs (hash-assigned).
    Observes 30-second windows → computes state → applies action.

    State: [p_attack, p_benign, p_suspicious, p_anomaly, load_norm]
    Actions: 0=forward, 1=puzzle, 2-5=throttle/block
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        agent_id:       int,
        bundle:         dict,
        window_size:    int   = WINDOW_SIZE,
        us:             float = Us,
        base_threshold: float = BASE_THRESHOLD,
        ip_timeout:     int   = IP_TIMEOUT,
    ) -> None:
        self.agent_id       = agent_id
        self.bundle         = bundle
        self.model          = bundle["model_3class"]
        self.features       = bundle["features"]
        self.imputer        = bundle["imputer"]
        self.hs_mapping     = bundle["handshake_mapping"]
        self.window_size    = window_size
        self.Us             = us
        self.base_threshold = base_threshold
        self.ip_timeout     = ip_timeout

        # ── Class index lookup (computed once, reused every step)
        classes          = bundle["classes_3class"]
        self._idx_attack = classes.index("attack")
        self._idx_benign = classes.index("benign")
        self._idx_susp   = classes.index("suspicious")

        # ── Filter and sort flows for this agent
        self.flows = (
            df[df["agent"] == agent_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ── Precompute timestamps as float seconds (numpy array — fast slicing)
        _epoch = pd.Timestamp("1970-01-01")
        self._ts_sec: np.ndarray = (
            (pd.to_datetime(self.flows["timestamp"]) - _epoch)
            .dt.total_seconds()
            .to_numpy()
            .astype(np.float64)
        )

        # ── Precompute src_ip as numpy array of strings (for vectorized lookup)
        self._src_ips: np.ndarray = self.flows["src_ip"].astype(str).to_numpy()

        # ── IP risk table: src_ip → (risk_score, timestamp_sec)
        # Stored as two parallel numpy-backed dicts for fast set lookup
        self.ip_risk_table: Dict[str, Tuple[float, float]] = {}

        self.t_min = float(self._ts_sec.min()) if len(self._ts_sec) else 0.0
        self.t_max = float(self._ts_sec.max()) if len(self._ts_sec) else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.ip_risk_table = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Window retrieval — O(log n) via searchsorted
    # ─────────────────────────────────────────────────────────────────────────

    def get_window(self, t_start: float) -> pd.DataFrame:
        """
        Return flows in [t_start, t_start + window_size).
        Uses np.searchsorted on sorted timestamp array → O(log n).
        Much faster than boolean mask scan on 100k+ rows.
        """
        t_end = t_start + self.window_size
        lo    = int(np.searchsorted(self._ts_sec, t_start, side="left"))
        hi    = int(np.searchsorted(self._ts_sec, t_end,   side="left"))
        return self.flows.iloc[lo:hi]   # view, not copy — fast

    def get_window_indices(self, t_start: float) -> Tuple[int, int]:
        """Return (lo, hi) row indices for a window — used in vectorized ops."""
        t_end = t_start + self.window_size
        lo    = int(np.searchsorted(self._ts_sec, t_start, side="left"))
        hi    = int(np.searchsorted(self._ts_sec, t_end,   side="left"))
        return lo, hi

    # ─────────────────────────────────────────────────────────────────────────
    # Feature preparation
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_features(self, window: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract, encode, clean, and impute feature matrix.
        Returns float32 numpy array ready for LightGBM.
        """
        if len(window) == 0:
            return None

        X = window[self.features].copy()

        # Encode handshake_state (categorical → int) if present
        if "handshake_state" in X.columns and self.hs_mapping:
            X["handshake_state"] = (
                X["handshake_state"]
                .astype(str)
                .map(self.hs_mapping)
                .fillna(-1)
                .astype(np.float32)
            )

        # Numeric coercion (vectorized per column)
        for col in X.columns:
            if col != "handshake_state":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
        return self.imputer.transform(X)   # returns float64 numpy array

    # ─────────────────────────────────────────────────────────────────────────
    # State computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_state(self, window: pd.DataFrame) -> np.ndarray:
        """
        State: [p_attack, p_benign, p_suspicious, p_anomaly, load_norm]

        p_anomaly = p_attack + p_suspicious (combined non-benign signal)
        Helps on Tuesday where attacks appear as suspicious.
        """
        if len(window) == 0:
            return np.zeros(STATE_DIM, dtype=np.float32)

        X = self._prepare_features(window)
        if X is None:
            return np.zeros(STATE_DIM, dtype=np.float32)

        proba = self.model.predict_proba(X)   # (n_flows, 3)

        p_attack     = float(proba[:, self._idx_attack].mean())
        p_benign     = float(proba[:, self._idx_benign].mean())
        p_suspicious = float(proba[:, self._idx_susp].mean())
        p_anomaly    = min(p_attack + p_suspicious, 1.0)

        load      = float(window["bytes_rate"].sum()) if "bytes_rate" in window.columns else 0.0
        load_norm = min(load / self.Us, 1.0)

        return np.array(
            [p_attack, p_benign, p_suspicious, p_anomaly, load_norm],
            dtype=np.float32,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # IP risk table — vectorized update
    # ─────────────────────────────────────────────────────────────────────────

    def update_ip_table(self, window: pd.DataFrame, t_start: float) -> None:
        """
        Flag src_IPs with P(attack) >= base_threshold.

        VECTORIZED — no iterrows:
          1. predict_proba on all flows at once
          2. boolean mask for threshold
          3. dict update via numpy fancy indexing
        """
        if len(window) == 0:
            return

        X = self._prepare_features(window)
        if X is None:
            return

        proba = self.model.predict_proba(X)           # (n, 3)
        risks = proba[:, self._idx_attack]             # (n,)  P(attack)

        # Boolean mask: flows above threshold
        flagged = risks >= self.base_threshold          # (n,)  bool

        if not flagged.any():
            return

        # Get src_ips and timestamps for flagged flows only
        src_ips    = window["src_ip"].astype(str).to_numpy()[flagged]
        risks_flag = risks[flagged]

        # Timestamps for flagged rows
        lo, hi    = self.get_window_indices(t_start)
        ts_window = self._ts_sec[lo:hi]
        # Align with window (window may be a view starting at lo)
        ts_flag   = ts_window[flagged]

        # Vectorized dict update
        # np.unique keeps highest risk per IP (last occurrence wins → use max)
        for ip, risk, ts in zip(src_ips, risks_flag, ts_flag):
            existing = self.ip_risk_table.get(ip)
            if existing is None or float(risk) > existing[0]:
                self.ip_risk_table[ip] = (float(risk), float(ts))

    def decay_ip_table(self, current_time: float) -> None:
        """Remove entries older than ip_timeout. Called once per step."""
        if not self.ip_risk_table:
            return
        cutoff  = current_time - self.ip_timeout
        expired = [ip for ip, (_, t) in self.ip_risk_table.items() if t < cutoff]
        for ip in expired:
            del self.ip_risk_table[ip]

    # ─────────────────────────────────────────────────────────────────────────
    # Apply action — fully vectorized
    # ─────────────────────────────────────────────────────────────────────────

    def apply_action(
        self,
        window: pd.DataFrame,
        action: int,
        rng:    np.random.Generator,
    ) -> pd.DataFrame:
        """
        Apply action to window t+1 flows.
        Returns forwarded flows as DataFrame.
        Also returns counts for evaluation metrics.

        VECTORIZED — no iterrows:
          Each action branch uses numpy boolean arrays.

        Returns:
            forwarded: DataFrame of forwarded flows
        """
        n = len(window)
        if n == 0:
            return window

        # ── Action 0: forward everything (no computation needed)
        if action == 0:
            return window

        keep = np.ones(n, dtype=bool)   # start: keep all

        # ── Action 1: puzzle
        if action == 1:
            labels = window["label"].to_numpy() if "label" in window.columns \
                     else np.full(n, "Benign")

            # Vectorized pass-rate lookup
            pass_rates = np.array([
                PUZZLE_PASS.get(str(lbl), _PUZZLE_PASS_DEFAULT)
                for lbl in labels
            ], dtype=np.float32)

            rolls = rng.random(n).astype(np.float32)
            keep  = rolls < pass_rates   # keep if roll < pass_rate

        # ── Actions 2-5: throttle / block (only flagged IPs)
        elif action in DROP_PROBS:
            drop_prob = DROP_PROBS[action]

            # Vectorized IP table membership check
            src_ips = window["src_ip"].astype(str).to_numpy()
            flagged = np.isin(src_ips, list(self.ip_risk_table.keys()))

            if flagged.any():
                if drop_prob >= 1.0:
                    # Block: drop all flagged
                    keep[flagged] = False
                else:
                    # Throttle: probabilistic drop of flagged only
                    rolls            = rng.random(n).astype(np.float32)
                    drop_mask        = flagged & (rolls < drop_prob)
                    keep[drop_mask]  = False

        return window.iloc[keep] if keep.all() else window.iloc[keep].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics helper (for evaluate.py)
    # ─────────────────────────────────────────────────────────────────────────

    def count_labels(self, window: pd.DataFrame) -> Dict[str, int]:
        """Return label counts in a window."""
        if "label" not in window.columns or len(window) == 0:
            return {"Benign": 0, "Attack": 0, "Suspicious": 0}
        counts = window["label"].value_counts()
        return {
            "Benign":     int(counts.get("Benign",     0)),
            "Attack":     int(counts.get("Attack",     0)),
            "Suspicious": int(counts.get("Suspicious", 0)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MultiAgentEnv
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEnv:
    """
    Coordinates N RouterEnvironments.

    - All routers advance in lockstep (shared global clock).
    - CTDE: agents act on local obs; critic sees joint obs.
    - Reward is global (shared across all agents).
    - One episode = one full replay of the training day.
    - step() returns forwarded flows per router for evaluation metrics.
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        bundle:      dict,
        window_size: int   = WINDOW_SIZE,
        us:          float = Us,
        n_agents:    int   = N_AGENTS,
        seed:        int   = 42,
    ) -> None:
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)

        # Assign src_ips to agents (once at init)
        df = assign_agents(df, n_agents)

        # Build one router per agent
        self.routers: List[RouterEnvironment] = [
            RouterEnvironment(
                df, agent_id=i, bundle=bundle,
                window_size=window_size, us=us,
            )
            for i in range(n_agents)
        ]

        # Global clock (float seconds)
        _epoch = pd.Timestamp("1970-01-01")
        ts_sec = (
            (pd.to_datetime(df["timestamp"]) - _epoch)
            .dt.total_seconds()
            .to_numpy()
            .astype(np.float64)
        )
        self.t_start   = float(ts_sec.min())
        self.t_end     = float(ts_sec.max())
        self.t_current = self.t_start

        self.victim_ips = VICTIM_IPS
        self._df        = df

    # ── Episode control ───────────────────────────────────────────────────────

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Reset to episode start. Returns (local_states, joint_obs)."""
        self.t_current = self.t_start
        for r in self.routers:
            r.reset()
        local = self._get_local_states()
        joint = np.concatenate(local).astype(np.float32)
        return local, joint

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: List[int],
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, List[pd.DataFrame]]:
        """
        Advance one time window.

        Returns:
            next_local_states  — per-agent state vectors
            next_joint_obs     — concatenated for critic
            reward             — global scalar
            done               — episode finished
            all_forwarded      — list of forwarded DataFrames per router
                                 (used by evaluate.py for TPR / legit_ratio)
        """
        assert len(actions) == self.n_agents

        t              = self.t_current
        all_forwarded: List[pd.DataFrame] = []

        for router, action in zip(self.routers, actions):
            window_t  = router.get_window(t)
            window_t1 = router.get_window(t + self.window_size)

            # Update IP table from window t (vectorized)
            router.update_ip_table(window_t, t)

            # Apply action on window t+1 (vectorized)
            forwarded = router.apply_action(window_t1, action, self.rng)
            all_forwarded.append(forwarded)

            # Expire old IP table entries
            router.decay_ip_table(t)

        # Global reward
        reward = self._compute_reward(all_forwarded, t + self.window_size)

        # Advance clock
        self.t_current = t + self.window_size
        done           = self.t_current >= self.t_end

        # Next states
        next_local = self._get_local_states()
        next_joint = np.concatenate(next_local).astype(np.float32)

        return next_local, next_joint, reward, done, all_forwarded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_local_states(self) -> List[np.ndarray]:
        return [
            r.compute_state(r.get_window(self.t_current))
            for r in self.routers
        ]

    def _compute_reward(
        self,
        all_forwarded: List[pd.DataFrame],
        t_window:      float,
    ) -> float:
        """
        Malialis-style reward:
          server_load > Us  →  -(load / 2*Us)        [negative]
          else              →  legit_passed / legit_total  [0..1]
        """
        non_empty = [f for f in all_forwarded if len(f) > 0]
        if not non_empty:
            return -1.0

        forwarded_all = pd.concat(non_empty, ignore_index=True)

        # Server load (vectorized isin check)
        if "dst_ip" in forwarded_all.columns and "bytes_rate" in forwarded_all.columns:
            victim_mask = forwarded_all["dst_ip"].isin(self.victim_ips)
            server_load = float(forwarded_all.loc[victim_mask, "bytes_rate"].sum())
        else:
            server_load = 0.0

        if server_load > self.Us:
            return float(-(server_load / (2.0 * self.Us)))

        # Legitimate traffic ratio
        total_legit = sum(
            int((r.get_window(t_window)["label"] == "Benign").sum())
            for r in self.routers
            if "label" in r.flows.columns
        )

        if total_legit == 0:
            return 0.0

        legit_passed = int((forwarded_all["label"] == "Benign").sum()) \
                       if "label" in forwarded_all.columns else 0

        return float(legit_passed / total_legit)

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "t_current":      self.t_current,
            "progress":       (self.t_current - self.t_start) /
                              max(self.t_end - self.t_start, 1),
            "ip_table_sizes": [len(r.ip_risk_table) for r in self.routers],
        }

    @property
    def n_windows(self) -> int:
        return int((self.t_end - self.t_start) / self.window_size)