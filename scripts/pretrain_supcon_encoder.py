"""
pretrain_supcon_encoder.py

Pre-trains a ThreatEncoder with Supervised Contrastive Loss (SupCon).

Unlike cross-entropy, SupCon operates on pairs — it pulls embeddings of
same-class flows together and pushes different-class flows apart, regardless
of class frequency. This addresses the Suspicious class imbalance at the
objective level rather than through loss weighting.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

Outputs
-------
artifacts/encoder/v2/
  supcon_encoder.pt    — encoder weights (backbone only, no classifier head)
  preprocessor.pkl     — fitted (imputer, scaler, features)
  training_loss.png    — SupCon loss curve
  tsne.png             — t-SNE of Tuesday embeddings coloured by class
  eval_report.json     — linear probe accuracy + per-class metrics

Usage
-----
  python scripts/pretrain_supcon_encoder.py
  python scripts/pretrain_supcon_encoder.py --epochs 80 --temp 0.1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "BCC-2024"
FEAT_JSON  = BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "selected_features.json"
OUT_DIR    = BASE_DIR / "artifacts" / "encoder" / "v2"

TRAIN_CSVS = [
    DATA_DIR / "Thursday_14_Dec_2023.csv",
    DATA_DIR / "Saturday_16_Dec_2023.csv",
    DATA_DIR / "Monday_18_Dec_2023.csv",
]
TEST_CSV   = DATA_DIR / "Tuesday_19_Dec_2023.csv"

LABEL_COL  = "label"
CLASSES    = ["Benign", "Suspicious", "Attack"]
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}


# ─── preprocessing ───────────────────────────────────────────────────────────

def load_features(feat_json: Path) -> list[str]:
    with open(feat_json) as f:
        return json.load(f)


def build_handshake_mapping(paths: list[Path]) -> dict[str, int]:
    seen: set[str] = set()
    for p in paths:
        df = pd.read_csv(p, usecols=["handshake_state"], low_memory=False, nrows=50000)
        if "handshake_state" in df.columns:
            seen.update(df["handshake_state"].dropna().astype(str).unique())
    return {s: i for i, s in enumerate(sorted(seen))}


def preprocess(
    df: pd.DataFrame,
    features: list[str],
    hs_map: dict[str, int],
    imputer: SimpleImputer | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    X = df[features].copy()
    if "handshake_state" in X.columns and hs_map:
        X["handshake_state"] = (
            X["handshake_state"].astype(str).map(hs_map)
            .fillna(-1).astype(np.float32)
        )
    for col in X.columns:
        if col != "handshake_state":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        Xn = imputer.fit_transform(X)
    else:
        Xn = imputer.transform(X)

    if scaler is None:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(Xn)
    else:
        Xn = scaler.transform(Xn)

    return Xn.astype(np.float32), imputer, scaler


def load_dataset(
    paths: list[Path],
    features: list[str],
    hs_map: dict[str, int],
    imputer=None,
    scaler=None,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    dfs = []
    for p in paths:
        print(f"  loading {p.name} …")
        df = pd.read_csv(p, usecols=features + [LABEL_COL], low_memory=False)
        df = df[df[LABEL_COL].isin(CLASSES)]
        dfs.append(df)
    full   = pd.concat(dfs, ignore_index=True)
    labels = full[LABEL_COL].values
    X, imputer, scaler = preprocess(full, features, hs_map, imputer, scaler)
    y = np.array([CLASS2IDX[l] for l in labels], dtype=np.int64)
    return X, y, imputer, scaler


# ─── model ───────────────────────────────────────────────────────────────────

class SupConEncoder(nn.Module):
    """
    42 raw features → 6-dim L2-normalised embedding (for SupCon training).

    After pre-training, unnormalised embeddings from backbone are used
    as RL state (scale matters for actor linear layers).

    Architecture: 42 → 256 → 128 → 64 → 6
    """

    def __init__(self, n_features: int = 42, embed_dim: int = 6, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, embed_dim),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised embedding — used as RL state."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised embedding — used for SupCon loss."""
        return F.normalize(self.backbone(x), dim=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, n_features: int = 42, embed_dim: int = 6) -> "SupConEncoder":
        m = cls(n_features=n_features, embed_dim=embed_dim)
        m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        m.eval()
        return m


# ─── SupCon loss ─────────────────────────────────────────────────────────────

def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    features: (N, D) L2-normalised embeddings
    labels:   (N,)   integer class labels
    """
    device = features.device
    N = features.shape[0]

    # Similarity matrix
    sim = torch.matmul(features, features.T) / temperature  # (N, N)

    # Numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Positive mask: same class, excluding self
    labels_col = labels.unsqueeze(1)                          # (N, 1)
    pos_mask   = (labels_col == labels_col.T).float()         # (N, N)
    self_mask  = torch.eye(N, device=device)
    pos_mask   = pos_mask - self_mask                         # exclude self

    # Denominator: all pairs except self
    exp_sim     = torch.exp(sim) * (1 - self_mask)
    log_prob    = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-prob over positives
    n_positives = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(pos_mask * log_prob).sum(dim=1) / n_positives
    return loss.mean()


# ─── training ────────────────────────────────────────────────────────────────

def train(
    encoder: SupConEncoder,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    device: torch.device,
) -> list[float]:
    encoder.to(device).train()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    n   = len(X)
    losses: list[float] = []

    # Binary labels: Benign=0, (Suspicious+Attack)=1
    # Matches the RL enforcement objective and avoids Suspicious distribution shift
    y_binary = (y > 0).astype(np.int64)
    y_bin_t  = torch.LongTensor(y_binary).to(device)
    n_ben = int((y_binary == 0).sum())
    n_thr = int((y_binary == 1).sum())
    print(f"  binary labels: Benign={n_ben} ({100*n_ben/n:.1f}%)  "
          f"Threat={n_thr} ({100*n_thr/n:.1f}%)")

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            b        = idx[start: start + batch_size]
            z        = encoder(X_t[b])          # L2-normalised
            loss     = supcon_loss(z, y_bin_t[b], temperature)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  supcon_loss={avg:.4f}")

    return losses


# ─── evaluation ──────────────────────────────────────────────────────────────

def extract_embeddings(
    encoder: SupConEncoder, X: np.ndarray, device: torch.device,
    batch_size: int = 4096
) -> np.ndarray:
    encoder.eval()
    X_t    = torch.FloatTensor(X).to(device)
    embeds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            embeds.append(encoder.embed(X_t[start: start + batch_size]).cpu().numpy())
    return np.concatenate(embeds, axis=0)


def linear_probe(Z_train, y_train, Z_test, y_test) -> dict:
    clf  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(Z_train, y_train)
    pred = clf.predict(Z_test)
    acc  = accuracy_score(y_test, pred)
    rep  = classification_report(y_test, pred, target_names=CLASSES, output_dict=True)
    print(f"\n  Linear probe accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=CLASSES))
    return {"accuracy": acc, "per_class": rep}


def plot_loss(losses, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1.5, color="#6366F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SupCon loss")
    ax.set_title("SupCon encoder pre-training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def plot_tsne(Z, y, out: Path, n_sample: int = 6000) -> None:
    rng = np.random.default_rng(42)
    if len(Z) > n_sample:
        idx = rng.choice(len(Z), n_sample, replace=False)
        Z, y = Z[idx], y[idx]
    print("  running t-SNE …")
    Z2 = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1).fit_transform(Z)
    colours = ["#3B82F6", "#F59E0B", "#EF4444"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(CLASSES):
        mask = y == i
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=6, alpha=0.5, label=cls, color=colours[i])
    ax.legend(markerscale=3, fontsize=10)
    ax.set_title("t-SNE of 6-dim SupCon embeddings (Tuesday test set)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=80)
    p.add_argument("--embed-dim", type=int,   default=6)
    p.add_argument("--batch",     type=int,   default=512)
    p.add_argument("--lr",        type=float, default=3e-3)
    p.add_argument("--temp",      type=float, default=0.5,
                   help="SupCon temperature τ (lower = tighter clusters)")
    p.add_argument("--dropout",   type=float, default=0.2)
    p.add_argument("--device",    type=str,   default="cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== SupCon ThreatEncoder pre-training ===")
    features = load_features(FEAT_JSON)
    print(f"  features: {len(features)}  |  embed_dim: {args.embed_dim}  "
          f"|  τ={args.temp}  |  epochs={args.epochs}")

    print("\n[1/5] Building handshake mapping …")
    hs_map = build_handshake_mapping(TRAIN_CSVS + [TEST_CSV])
    print(f"  handshake states: {len(hs_map)}")

    print("\n[2/5] Loading training data (Thu + Sat + Mon) …")
    X_train, y_train, imputer, scaler = load_dataset(TRAIN_CSVS, features, hs_map)
    print(f"  train: {X_train.shape}  classes: {np.bincount(y_train)}")

    print("\n[3/5] Loading test data (Tuesday) …")
    X_test, y_test, _, _ = load_dataset([TEST_CSV], features, hs_map, imputer, scaler)
    print(f"  test:  {X_test.shape}  classes: {np.bincount(y_test)}")

    print(f"\n[4/5] Training SupCon encoder …")
    encoder = SupConEncoder(n_features=len(features), embed_dim=args.embed_dim,
                            dropout=args.dropout)
    print(f"  params: {sum(p.numel() for p in encoder.parameters()):,}")
    losses = train(encoder, X_train, y_train, args.epochs, args.batch,
                   args.lr, args.temp, device)

    print("\n[5/5] Evaluating …")
    Z_train = extract_embeddings(encoder, X_train, device)
    Z_test  = extract_embeddings(encoder, X_test,  device)
    probe   = linear_probe(Z_train, y_train, Z_test, y_test)

    # ── save ──────────────────────────────────────────────────────────────
    encoder.save(OUT_DIR / "supcon_encoder.pt")
    print(f"  saved encoder → {OUT_DIR / 'supcon_encoder.pt'}")

    preprocessor = {
        "features":          features,
        "handshake_mapping": hs_map,
        "imputer":           imputer,
        "scaler":            scaler,
        "embed_dim":         args.embed_dim,
        "n_features":        len(features),
        "temperature":       args.temp,
    }
    with open(OUT_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    with open(OUT_DIR / "eval_report.json", "w") as f:
        json.dump({"linear_probe": probe, "embed_dim": args.embed_dim,
                   "epochs": args.epochs, "temperature": args.temp,
                   "final_loss": losses[-1]}, f, indent=2)

    plot_loss(losses, OUT_DIR / "training_loss.png")
    plot_tsne(Z_test, y_test, OUT_DIR / "tsne.png")

    print(f"\nDone. Artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
