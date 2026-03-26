"""
environment_tiered_v4.py
Tiered action environment with victim-based routing + shared IP blocklist (S35/S36).

Extends V3 with ONE change: routing by dst_ip (victim) instead of hash(src_ip).

WHY THIS MATTERS:
  Hash routing (V2/V3): every router sees ~33% attack all day → symmetric
  Victim routing (V4):  each router protects specific victim servers

  Monday attack schedule hits specific victims at specific times:
    09:00-09:20  TCP-Valid-SYN    → Windows-machine-1  (Router 0 flooded)
    09:30-09:50  TCP-BYPass-V1   → Linux-admin         (Router 0 flooded)
    10:00-10:20  TCP-Killall-v2  → Linux-webserver      (Router 1 flooded)
    10:32-10:52  TCP-IGMP        → Windows-machine-2    (Router 1 flooded)
    11:00-11:20  TCP-SYN         → Linux-admin          (Router 0 flooded)
    11:30-11:50  Killer-TCP      → Windows-machine-4    (Router 2 flooded)
    13:00-13:20  TCP-Control     → Linux-webserver      (Router 1 flooded)
    13:30-13:50  TCP-Flag-MIX    → Linux-admin          (Router 0 flooded)
    14:00-14:20  TCP-Flag-SYN    → Windows-machine-2    (Router 1 flooded)
    14:30-14:50  TCP-Flag-ACK    → Windows-machine-3    (Router 2 flooded)
    15:00-15:20  TCP-Flag-SYN-ACK→ Linux-webserver      (Router 1 flooded)
    15:30-15:50  TCP-Flag-ACK-PSH→ Windows-machine-4    (Router 2 flooded)

  Routers now face GENUINELY different threat levels simultaneously:
    At 09:00: Router 0 = 90% attack, Router 1 = benign, Router 2 = benign
    At 11:30: Router 0 = quiet,      Router 1 = quiet,   Router 2 = flooded

  With shared_threat_level, Router 1 and 2 can observe Router 0 being attacked
  and pre-emptively escalate BEFORE their own subnet is targeted.
  This is the architectural condition for CTDE to genuinely outperform IPPO.

Victim → Router assignment (2 victims per router):
  Router 0: Windows-machine-1 (3.96.128.96),  Linux-admin     (99.79.45.168)
  Router 1: Linux-webserver   (15.222.45.224), Windows-machine-2 (35.182.194.19)
  Router 2: Windows-machine-3 (3.99.186.200),  Windows-machine-4 (35.183.15.52)

Non-victim flows (background benign traffic): fallback to hash(src_ip) % n_agents.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")

# Import everything from V3, only override routing
from src.environment_tiered_v3 import (
    RouterEnvironmentV3,
    MultiAgentEnvV3,
    TIERED_PROFILES_V3,
    ACTION_DIM, STATE_DIM, JOINT_DIM,
    FLAG_THRESHOLD, SHARED_IP_MAX,
    CAPTCHA_PASS, LOCAL_REWARD, ACTION_LABELS,
)

try:
    from config import N_AGENTS, WINDOW_SIZE, Us, VICTIM_IPS
except ImportError:
    from coopids.config import N_AGENTS, WINDOW_SIZE, Us, VICTIM_IPS


# Victim → router assignment
VICTIM_TO_AGENT = {
    "3.96.128.96":   0,   # Windows-machine-1  → Router 0
    "99.79.45.168":  0,   # Linux-admin        → Router 0
    "15.222.45.224": 1,   # Linux-webserver    → Router 1
    "35.182.194.19": 1,   # Windows-machine-2  → Router 1
    "3.99.186.200":  2,   # Windows-machine-3  → Router 2
    "35.183.15.52":  2,   # Windows-machine-4  → Router 2
}


def assign_agents_victim(df: pd.DataFrame, n_agents: int = N_AGENTS) -> pd.DataFrame:
    """
    Route by destination IP (victim server).
    Attack flows targeting victim X always go to the router protecting X.
    Non-victim flows fall back to hash(src_ip) % n_agents.

    This creates asymmetric threat distributions:
      Router protecting a targeted victim gets flooded;
      others see only background benign traffic during that attack window.
    """
    df = df.copy()
    # Primary: route by dst_ip if it's a known victim
    dst_agent = df["dst_ip"].map(VICTIM_TO_AGENT)
    # Fallback: hash(src_ip) for non-victim destinations
    src_hash  = df["src_ip"].apply(lambda ip: hash(str(ip)) % n_agents)
    df["agent"] = dst_agent.fillna(src_hash).astype(int)
    return df


class MultiAgentEnvV4(MultiAgentEnvV3):
    """
    Victim-based routing environment.
    Identical to V3 except uses assign_agents_victim instead of hash routing.
    """

    def __init__(self, df, bundle, window_size=WINDOW_SIZE, us=Us,
                 n_agents=N_AGENTS, seed=42):
        # Apply victim-based routing before calling parent __init__
        df = assign_agents_victim(df, n_agents)

        # Bypass parent's assign_agents call by directly initialising routers
        import numpy as np
        self.n_agents    = n_agents
        self.window_size = window_size
        self.Us          = us
        self.rng         = np.random.default_rng(seed)

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
