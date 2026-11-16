"""
pretrain_threat_encoder.py

Pre-trains a ThreatEncoder (42 raw flow features → 6-dim threat embedding)
on CICIDS training days (Thursday, Saturday, Monday) using 3-class
cross-entropy supervision (Benign / Suspicious / Attack).

Outputs
-------
artifacts/encoder/v1/
  threat_encoder.pt        — encoder weights
  preprocessor.pkl         — fitted (imputer, scaler, handshake_mapping, features)
  training_loss.png        — loss curve
  tsne.png                 — t-SNE of test embeddings coloured by class
  eval_report.json         — linear probe accuracy + per-class metrics

Usage
-----
  python scripts/pretrain_threat_encoder.py
  python scripts/pretrain_threat_encoder.py --epochs 60 --embed-dim 6 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import pickle
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "BCC-2024"
FEAT_JSON  = BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "selected_features.json"
OUT_DIR    = BASE_DIR / "artifacts" / "encoder" / "v1"

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


def build_handshake_mapping(dfs: list[pd.DataFrame]) -> dict[str, int]:
    seen: set[str] = set()
    for df in dfs:
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
    imputer: SimpleImputer | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    dfs = []
    for p in paths:
        print(f"  loading {p.name} …")
        df = pd.read_csv(p, usecols=features + [LABEL_COL], low_memory=False)
        df = df[df[LABEL_COL].isin(CLASSES)]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    labels = full[LABEL_COL].values
    X, imputer, scaler = preprocess(full, features, hs_map, imputer, scaler)
    y = np.array([CLASS2IDX[l] for l in labels], dtype=np.int64)
    return X, y, imputer, scaler


# ─── model ───────────────────────────────────────────────────────────────────

class ThreatEncoder(nn.Module):
    """
    42 raw flow features  →  6-dim threat embedding.

    Architecture: 42 → 256 → 128 → 64 → 6  (BatchNorm + Dropout)
    Classification head: 6 → 3  (used only during pre-training)
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
        self.classifier = nn.Linear(embed_dim, len(CLASSES))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, n_features: int = 42, embed_dim: int = 6) -> "ThreatEncoder":
        m = cls(n_features=n_features, embed_dim=embed_dim)
        m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        m.eval()
        return m


# ─── training ────────────────────────────────────────────────────────────────

def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    weights = counts.sum() / (len(CLASSES) * counts)
    print(f"  class weights: " + " ".join(f"{CLASSES[i]}={weights[i]:.2f}" for i in range(len(CLASSES))))
    return torch.FloatTensor(weights).to(device)


def train(
    encoder: ThreatEncoder,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    encoder.to(device).train()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    weights = class_weights(y, device)

    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    n   = len(X)
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, n, batch_size):
            b = idx[start: start + batch_size]
            logits = encoder(X_t[b])
            loss   = F.cross_entropy(logits, y_t[b], weight=weights)
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
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    return losses


# ─── evaluation ──────────────────────────────────────────────────────────────

def extract_embeddings(
    encoder: ThreatEncoder, X: np.ndarray, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    encoder.eval()
    X_t = torch.FloatTensor(X).to(device)
    embeds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            embeds.append(encoder.embed(X_t[start: start + batch_size]).cpu().numpy())
    return np.concatenate(embeds, axis=0)


def linear_probe(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray,  y_test: np.ndarray,
) -> dict:
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(Z_train, y_train)
    preds = clf.predict(Z_test)
    acc   = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds,
        target_names=CLASSES,
        output_dict=True,
    )
    print(f"\n  Linear probe accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=CLASSES))
    return {"accuracy": acc, "per_class": report}


def plot_loss(losses: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("ThreatEncoder pre-training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_tsne(
    Z: np.ndarray, y: np.ndarray, out: Path, n_sample: int = 6000
) -> None:
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
        ax.scatter(Z2[mask, 0], Z2[mask, 1],
                   s=6, alpha=0.5, label=cls, color=colours[i])
    ax.legend(markerscale=3, fontsize=10)
    ax.set_title("t-SNE of 6-dim ThreatEncoder embeddings (Tuesday test set)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=80)
    p.add_argument("--embed-dim", type=int,   default=6)
    p.add_argument("--batch",     type=int,   default=2048)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--dropout",   type=float, default=0.2)
    p.add_argument("--device",    type=str,   default="cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ThreatEncoder pre-training ===")
    features = load_features(FEAT_JSON)
    print(f"  features: {len(features)}")

    # ── build handshake mapping from all CSVs ──────────────────────────────
    print("\n[1/5] Building handshake mapping …")
    sample_dfs = []
    for p in TRAIN_CSVS + [TEST_CSV]:
        df = pd.read_csv(p, usecols=["handshake_state"], low_memory=False, nrows=50000)
        sample_dfs.append(df)
    hs_map = build_handshake_mapping(sample_dfs)
    print(f"  handshake states: {len(hs_map)}")

    # ── load training data ─────────────────────────────────────────────────
    print("\n[2/5] Loading training data (Thu + Sat + Mon) …")
    X_train, y_train, imputer, scaler = load_dataset(
        TRAIN_CSVS, features, hs_map
    )
    print(f"  train: {X_train.shape}  classes: {np.bincount(y_train)}")

    # ── load test data ─────────────────────────────────────────────────────
    print("\n[3/5] Loading test data (Tuesday) …")
    X_test, y_test, _, _ = load_dataset(
        [TEST_CSV], features, hs_map, imputer, scaler
    )
    print(f"  test:  {X_test.shape}  classes: {np.bincount(y_test)}")

    # ── train encoder ──────────────────────────────────────────────────────
    print(f"\n[4/5] Training ThreatEncoder ({args.embed_dim}-dim, {args.epochs} epochs) …")
    encoder = ThreatEncoder(
        n_features=len(features),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    )
    print(f"  params: {sum(p.numel() for p in encoder.parameters()):,}")
    losses = train(encoder, X_train, y_train, args.epochs, args.batch, args.lr, device)

    # ── evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating …")
    Z_train = extract_embeddings(encoder, X_train, device)
    Z_test  = extract_embeddings(encoder, X_test,  device)
    probe   = linear_probe(Z_train, y_train, Z_test, y_test)

    # ── save ───────────────────────────────────────────────────────────────
    encoder.save(OUT_DIR / "threat_encoder.pt")
    print(f"  saved encoder → {OUT_DIR / 'threat_encoder.pt'}")

    preprocessor = {
        "features":           features,
        "handshake_mapping":  hs_map,
        "imputer":            imputer,
        "scaler":             scaler,
        "embed_dim":          args.embed_dim,
        "n_features":         len(features),
    }
    with open(OUT_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"  saved preprocessor → {OUT_DIR / 'preprocessor.pkl'}")

    eval_result = {
        "linear_probe":   probe,
        "embed_dim":      args.embed_dim,
        "epochs":         args.epochs,
        "train_shape":    list(X_train.shape),
        "test_shape":     list(X_test.shape),
        "final_loss":     losses[-1],
    }
    with open(OUT_DIR / "eval_report.json", "w") as f:
        json.dump(eval_result, f, indent=2)

    plot_loss(losses, OUT_DIR / "training_loss.png")
    plot_tsne(Z_test, y_test, OUT_DIR / "tsne.png")

    print(f"\nDone. All artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
