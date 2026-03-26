"""
environment_tiered_ecmp.py
Tiered action environment with 5-tuple ECMP routing (S38).

One change from V2: routing by hash(src_ip, src_port, dst_ip, dst_port, protocol) % N
instead of hash(src_ip) % N.

WHY ECMP:
  Real ISP edge routers use ECMP (Equal-Cost Multi-Path) with per-flow hashing
  on the full 5-tuple. This ensures:
    1. Same TCP flow always hits the same router (no packet reordering)
    2. Load is balanced across all routers
    3. DDoS attack flows (varying src_ip) distribute across all routers naturally

  Compared to src_ip-only hash (S29):
    - Same load balancing property
    - More realistic — real networks hash on 5-tuple, not just src_ip
    - DDoS with fixed src_ip but varying ports still distributes across routers

  ECMP empirically verified on Monday data:
    - 0 windows with any empty router (out of 968 windows)
    - Perfect 110/110/110 mean split across 3 routers
    - Min flows/window: 2+ flows on every router
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.environment_tiered_v2 import (
    MultiAgentEnv as MultiAgentEnvV2,
    RouterEnvironment,
    STATE_DIM, JOINT_DIM, ACTION_DIM, ACTION_LABELS,
    WINDOW_SIZE, Us,
)

try:
    from config import N_AGENTS, VICTIM_IPS
except ImportError:
    from coopids.config import N_AGENTS, VICTIM_IPS


def assign_agents_ecmp(df: pd.DataFrame, n_agents: int = N_AGENTS) -> pd.DataFrame:
    """
    Route by 5-tuple ECMP hash: hash(src_ip, src_port, dst_ip, dst_port, protocol) % N.
    Each unique flow (TCP/UDP connection) always maps to the same router.
    """
    df = df.copy()
    five_tuple = (
        df["src_ip"].astype(str) + "_" +
        df["src_port"].astype(str) + "_" +
        df["dst_ip"].astype(str) + "_" +
        df["dst_port"].astype(str) + "_" +
        df["protocol"].astype(str)
    )
    df["agent"] = five_tuple.apply(lambda t: hash(t) % n_agents)
    return df


class MultiAgentEnvECMP(MultiAgentEnvV2):
    """
    ECMP-routed environment.
    Identical to V2 except uses 5-tuple hash instead of src_ip hash.
    """

    def __init__(self, df, bundle, window_size=WINDOW_SIZE, us=Us,
                 n_agents=N_AGENTS, seed=42):
        # Apply ECMP routing before calling parent setup
        df = assign_agents_ecmp(df, n_agents)

        # Bypass parent's assign_agents call — initialise directly
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)

        self.routers = [
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
