"""
environment_tiered_v6.py  —  S42 experiment

State:      7-dim  [mean(z_0..z_5), flow_count_norm]
            where z = SupCon encoder embedding (6-dim, frozen).

Enforcement: identical to v2/v5 — LightGBM p_risk vs (T_block, T_puzzle).

The SupCon encoder is pre-trained and frozen during RL training.
It maps each flow's 42 raw features to a 6-dim embedding that separates
threat classes via contrastive geometry, regardless of class frequency.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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

from scripts.pretrain_supcon_encoder import SupConEncoder

SUPCON_ENCODER_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts" / "encoder" / "v2" / "supcon_encoder.pt"
)
SUPCON_PREPROCESSOR_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts" / "encoder" / "v2" / "preprocessor.pkl"
)

TIERED_PROFILES_V2 = {
    0: (1.00, 1.00),
    1: (0.90, 0.60),
    2: (0.80, 0.40),
    3: (0.70, 0.25),
    4: (0.60, 0.15),
    5: (0.50, 0.10),
}

ACTION_DIM    = len(TIERED_PROFILES_V2)
EMBED_DIM     = 6
STATE_DIM     = EMBED_DIM + 1     # 7
JOINT_DIM     = STATE_DIM * N_AGENTS  # 21
CAPTCHA_PASS  = {"Benign": 0.95, "Suspicious": 0.50, "Attack": 0.05}
LOCAL_REWARD  = True
ACTION_LABELS = {0: "passive", 1: "light",  2: "moderate",
                 3: "firm",    4: "aggressive", 5: "lockdown"}


def load_supcon_encoder() -> Tuple[SupConEncoder, dict]:
    if not SUPCON_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"SupCon encoder not found at {SUPCON_ENCODER_PATH}.\n"
            f"Run: python scripts/pretrain_supcon_encoder.py"
        )
    with open(SUPCON_PREPROCESSOR_PATH, "rb") as f:
        prep = pickle.load(f)
    encoder = SupConEncoder.load(SUPCON_ENCODER_PATH,
                                 n_features=prep["n_features"],
                                 embed_dim=prep["embed_dim"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, prep


def assign_agents(df: pd.DataFrame, n_agents: int = N_AGENTS,
                  routing_seed: int | None = None) -> pd.DataFrame:
    df = df.copy()
    if routing_seed is None:
        df["agent"] = df["src_ip"].apply(lambda ip: hash(str(ip)) % n_agents)
    else:
        rng = np.random.RandomState(routing_seed)
        ip_map = {ip: int(rng.randint(0, n_agents)) for ip in df["src_ip"].unique()}
        df["agent"] = df["src_ip"].map(ip_map)
    return df


class RouterEnvironment:
    def __init__(self, df: pd.DataFrame, agent_id: int, bundle: dict,
                 encoder: SupConEncoder, prep: dict,
                 window_size: int = WINDOW_SIZE):
        self.agent_id    = agent_id
        self.bundle      = bundle
        self.model       = bundle["model_3class"]
        self.features    = bundle["features"]
        self.hs_mapping  = bundle["handshake_mapping"]
        self.window_size = window_size
        self.encoder     = encoder
        self.prep        = prep

        self.flows = (
            df[df["agent"] == agent_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        _epoch = pd.Timestamp("1970-01-01")
        ts = (pd.to_datetime(self.flows["timestamp"]) - _epoch).dt.total_seconds()
        self._ts_sec_np = ts.values.astype(np.float64)
        self.t_min = float(ts.min()) if len(ts) else 0.0
        self.t_max = float(ts.max()) if len(ts) else 0.0

        # LightGBM p_risk — enforcement, pre-computed for all flows
        X_lgbm = self._prepare_lgbm(self.flows)
        if X_lgbm is not None:
            proba = self.model.predict_proba(X_lgbm)
            cls   = bundle["classes_3class"]
            self._p_risk_all = (
                1.0 - proba[:, cls.index("benign")]
            ).astype(np.float32)
        else:
            self._p_risk_all = np.zeros(len(self.flows), np.float32)

        # SupCon embeddings — state, pre-computed for all flows
        self._embed_all = self._compute_embeddings(self.flows)

    def _prepare_lgbm(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if len(df) == 0:
            return None
        X = df[self.features].copy()
        if "handshake_state" in X.columns and self.hs_mapping:
            X["handshake_state"] = (
                X["handshake_state"].astype(str).map(self.hs_mapping)
                .fillna(-1).astype(np.float32)
            )
        for col in X.columns:
            if col != "handshake_state":
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
        return self.bundle["imputer"].transform(X)

    def _compute_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        prep = self.prep
        X = df[prep["features"]].copy()
        hs_map = prep["handshake_mapping"]
        if "handshake_state" in X.columns and hs_map:
            X["handshake_state"] = (
                X["handshake_state"].astype(str).map(hs_map)
                .fillna(-1).astype(np.float32)
            )
        for col in X.columns:
            if col != "handshake_state":
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).astype("float32")

        X_imp    = prep["imputer"].transform(X)
        X_scaled = prep["scaler"].transform(X_imp)

        X_t = torch.FloatTensor(X_scaled)
        embeds = []
        batch_size = 4096
        with torch.no_grad():
            for start in range(0, len(X_t), batch_size):
                embeds.append(self.encoder.embed(X_t[start: start + batch_size]).numpy())
        return np.concatenate(embeds, axis=0).astype(np.float32)

    def reset(self):
        pass

    def get_window(self, t_start: float) -> pd.DataFrame:
        t_end = t_start + self.window_size
        lo = int(np.searchsorted(self._ts_sec_np, t_start, side="left"))
        hi = int(np.searchsorted(self._ts_sec_np, t_end,   side="left"))
        return self.flows.iloc[lo:hi].copy()

    def compute_state(self, window: pd.DataFrame) -> np.ndarray:
        if len(window) == 0:
            return np.zeros(STATE_DIM, dtype=np.float32)
        idx       = window.index.values
        emb_mean  = self._embed_all[idx].mean(axis=0)     # (6,)
        flow_norm = min(len(window) / FLOW_COUNT_MAX, 1.0)
        return np.append(emb_mean, flow_norm).astype(np.float32)

    def apply_action(self, window: pd.DataFrame, action: int,
                     rng: np.random.Generator) -> pd.DataFrame:
        if len(window) == 0:
            return window
        t_block, t_puzzle = TIERED_PROFILES_V2.get(action, (1.0, 1.0))
        if t_block >= 1.0 and t_puzzle >= 1.0:
            return window

        n      = len(window)
        idx    = window.index.values
        p_risk = self._p_risk_all[idx]

        block_mask  = p_risk >= t_block
        puzzle_mask = (~block_mask) & (p_risk >= t_puzzle)

        puzzle_pass = np.ones(n, dtype=np.float32)
        if puzzle_mask.any():
            labels = window.loc[window.index[puzzle_mask], "label"]
            puzzle_pass[puzzle_mask] = (
                labels.map(CAPTCHA_PASS).fillna(0.5).values.astype(np.float32)
            )

        captcha_result = rng.random(n) < puzzle_pass
        keep_mask = (~block_mask) & (~puzzle_mask | captcha_result)
        return window[keep_mask].copy()


class MultiAgentEnv:
    def __init__(self, df: pd.DataFrame, bundle: dict,
                 window_size: int = WINDOW_SIZE, us: float = Us,
                 n_agents: int = N_AGENTS, seed: int = 42,
                 routing_seed: int | None = None):
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)

        print("  Loading SupCon encoder …")
        encoder, prep = load_supcon_encoder()

        df = assign_agents(df, n_agents, routing_seed=routing_seed)
        self.routers: List[RouterEnvironment] = [
            RouterEnvironment(df, agent_id=i, bundle=bundle,
                              encoder=encoder, prep=prep,
                              window_size=window_size)
            for i in range(n_agents)
        ]

        _epoch = pd.Timestamp("1970-01-01")
        ts_sec = (pd.to_datetime(df["timestamp"]) - _epoch).dt.total_seconds()
        self.t_start   = float(ts_sec.min())
        self.t_end     = float(ts_sec.max())
        self.t_current = self.t_start
        self.last_step_info: dict = {}

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        self.t_current = self.t_start
        for r in self.routers:
            r.reset()
        local = self._get_local_states()
        return local, np.concatenate(local).astype(np.float32)

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], np.ndarray, float, bool]:
        assert len(actions) == self.n_agents
        t = self.t_current
        all_fwd = [
            r.apply_action(r.get_window(t + self.window_size), a, self.rng)
            for r, a in zip(self.routers, actions)
        ]
        reward = (self._compute_local_rewards(all_fwd, t + self.window_size)
                  if LOCAL_REWARD else
                  self._compute_reward(all_fwd, t + self.window_size))
        self.t_current = t + self.window_size
        done = self.t_current >= self.t_end
        nxt  = self._get_local_states()
        return nxt, np.concatenate(nxt).astype(np.float32), reward, done

    def _get_local_states(self) -> List[np.ndarray]:
        return [r.compute_state(r.get_window(self.t_current)) for r in self.routers]

    def _compute_reward(self, all_fwd, t_window) -> float:
        legit_total = attack_total = 0
        for r in self.routers:
            w = r.get_window(t_window)
            if "label" in w.columns:
                legit_total  += int((w["label"] == "Benign").sum())
                attack_total += int((w["label"] == "Attack").sum())
        if legit_total == 0 and attack_total == 0:
            return 0.0
        non_empty = [f for f in all_fwd if len(f) > 0]
        if not non_empty:
            return -1.0 if legit_total > 0 else 0.0
        fwd = pd.concat(non_empty, ignore_index=True)
        lp  = int((fwd["label"] == "Benign").sum()) if "label" in fwd.columns else 0
        ap  = int((fwd["label"] == "Attack").sum()) if "label" in fwd.columns else 0
        self.last_step_info = {
            "benign_total": legit_total, "attack_total": attack_total,
            "benign_forwarded": lp, "attack_forwarded": ap,
            "attack_blocked": attack_total - ap,
        }
        return float(REWARD_LEGIT_W * lp / max(legit_total, 1)
                     - REWARD_ATTACK_W * ap / max(attack_total, 1))

    def _compute_local_rewards(self, all_fwd, t_window) -> List[float]:
        rewards, per_agent = [], []
        for r, fwd in zip(self.routers, all_fwd):
            w = r.get_window(t_window)
            if "label" not in w.columns:
                rewards.append(0.0); per_agent.append({}); continue
            lt = int((w["label"] == "Benign").sum())
            at = int((w["label"] == "Attack").sum())
            if lt == 0 and at == 0:
                rewards.append(0.0); per_agent.append({}); continue
            if len(fwd) == 0:
                rewards.append(-1.0 if lt > 0 else 0.0)
                per_agent.append({"benign_total": lt, "attack_total": at,
                                  "benign_passed": 0, "attack_passed": 0})
                continue
            lp = int((fwd["label"] == "Benign").sum())
            ap = int((fwd["label"] == "Attack").sum())
            per_agent.append({"benign_total": lt, "attack_total": at,
                               "benign_passed": lp, "attack_passed": ap})
            rewards.append(float(REWARD_LEGIT_W * lp / max(lt, 1)
                                 - REWARD_ATTACK_W * ap / max(at, 1)))
        self._compute_reward(all_fwd, t_window)
        self.last_step_info["per_agent"] = per_agent
        return rewards

    @property
    def n_windows(self) -> int:
        return int((self.t_end - self.t_start) / self.window_size)
