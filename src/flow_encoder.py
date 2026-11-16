"""
flow_encoder.py
Learnable per-flow risk encoder for S39 (replaces LightGBM).

Architecture:
  42 raw flow features → MLP backbone → 3-dim embedding + 1 risk scalar
  State: [mean(embed_0), mean(embed_1), mean(embed_2), flow_count_norm]  →  4-dim
  Risk:  sigmoid(risk_head) per flow  →  used in tiered profiles (same as S27)

Two training signals:
  1. Supervised BCE: risk_score vs binary label (attack/benign)  — teaches what "risk" means
  2. Co-evolution: encoder updates every K episodes alongside policy  — adapts to action space

Differences from LightGBM:
  - Simpler model (3-layer MLP vs gradient-boosted trees)
  - Jointly trained with MARL policy (not pre-trained + frozen)
  - Can specialise representation for the tiered action space
  - Continuously adapts during training (not fixed after pre-training)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Feature preprocessing (replaces LightGBM bundle preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_names(features_json: Path) -> list[str]:
    """Load the 42 selected feature names."""
    with open(features_json) as f:
        return json.load(f)


def build_handshake_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Build categorical encoding for handshake_state from data."""
    if "handshake_state" not in df.columns:
        return {}
    unique_states = sorted(df["handshake_state"].dropna().astype(str).unique())
    return {s: i for i, s in enumerate(unique_states)}


def prepare_features(
    df: pd.DataFrame,
    features: list[str],
    hs_mapping: dict[str, int],
    imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """
    Preprocess raw features into a clean, standardised float32 matrix.

    Neural networks (unlike tree-based LightGBM) need normalised inputs.
    Returns (X, imputer, scaler) — fitted on first call, reused after.
    """
    X = df[features].copy()

    # Encode handshake_state
    if "handshake_state" in X.columns and hs_mapping:
        X["handshake_state"] = (
            X["handshake_state"].astype(str).map(hs_mapping)
            .fillna(-1).astype(np.float32)
        )

    # Convert to numeric
    for col in X.columns:
        if col != "handshake_state":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")

    # Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_out = imputer.fit_transform(X)
    else:
        X_out = imputer.transform(X)

    # Standardise features (zero mean, unit variance) — critical for MLP
    if scaler is None:
        scaler = StandardScaler()
        X_out = scaler.fit_transform(X_out)
    else:
        X_out = scaler.transform(X_out)

    return X_out.astype(np.float32), imputer, scaler


# ─────────────────────────────────────────────────────────────────────────────
# FlowEncoder neural network
# ─────────────────────────────────────────────────────────────────────────────

class FlowEncoder(nn.Module):
    """
    Per-flow feature encoder: (42,) → (3-dim embedding, 1 risk scalar).

    The risk scalar (sigmoid output in [0,1]) replaces LightGBM's p_risk
    for tiered action profiles. The 3-dim embedding is mean-pooled per window
    to form the first 3 dimensions of the agent's state.
    """

    def __init__(self, n_features: int = 42, embed_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim

        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
        )
        self.embed_head = nn.Linear(hidden // 2, embed_dim)
        self.risk_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (embeddings, risk_scores).
          embeddings:  (batch, embed_dim)
          risk_scores: (batch,)  in [0, 1]
        """
        h = self.backbone(x)
        embed = self.embed_head(h)
        risk = torch.sigmoid(self.risk_head(h)).squeeze(-1)
        return embed, risk

    def risk_only(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_features) → (batch,) risk scores in [0, 1]."""
        h = self.backbone(x)
        return torch.sigmoid(self.risk_head(h)).squeeze(-1)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "flow_encoder.pt")

    @classmethod
    def load(cls, path: Path, n_features: int = 42, embed_dim: int = 3,
             hidden: int = 64) -> "FlowEncoder":
        model = cls(n_features=n_features, embed_dim=embed_dim, hidden=hidden)
        model.load_state_dict(torch.load(path / "flow_encoder.pt",
                                         weights_only=True))
        model.eval()
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Encoder training utilities
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_encoder(
    encoder: FlowEncoder,
    X: np.ndarray,
    labels: np.ndarray,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
) -> list[float]:
    """
    Warm-up supervised training: teach the encoder what "risk" means.

    Target: 1.0 for attack/suspicious flows, 0.0 for benign flows.
    This gives the encoder a reasonable starting point so tiered thresholds
    produce meaningful filtering from episode 1.
    """
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    y = torch.FloatTensor((labels != "Benign").astype(np.float32))
    X_t = torch.FloatTensor(X)
    n = len(X)

    losses = []
    encoder.train()
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            risk = encoder.risk_only(X_t[batch_idx])
            loss = F.binary_cross_entropy(risk, y[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))
    return losses


def train_encoder_step(
    encoder: FlowEncoder,
    optimizer: torch.optim.Optimizer,
    X: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 512,
) -> float:
    """One step of supervised encoder training during RL."""
    n = len(X)
    idx = np.random.choice(n, min(batch_size, n), replace=False)

    X_t = torch.FloatTensor(X[idx])
    y = torch.FloatTensor((labels[idx] != "Benign").astype(np.float32))

    encoder.train()
    risk = encoder.risk_only(X_t)
    loss = F.binary_cross_entropy(risk, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
