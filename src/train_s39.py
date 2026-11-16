"""
train_s39.py
End-to-End Flow Encoder + MAPPO (S39).

Replaces the pre-trained LightGBM risk model with a jointly-trained MLP encoder.
The FlowEncoder learns per-flow risk scores alongside the MARL policy.

Architecture:
  FlowEncoder:  42 raw features → (3-dim embedding, 1 risk scalar)
  State:        mean-pool embeddings + flow_count_norm → 4-dim (same as S27)
  Policy:       MAPPO with same actor/critic architecture as S27
  Actions:      Same 6 tiered profiles as S27 (p_risk thresholds)

Training:
  Phase 1 (warm-up):  20 epochs of supervised BCE on Monday labels
  Phase 2 (joint):    MAPPO + 1 encoder update per UPDATE_EVERY episodes

The key ablation question:
  S27 uses a frozen pre-trained LightGBM (gradient boosting, ~50 features → 3 classes).
  S39 uses a jointly-trained MLP (42 features → risk score).
  Does co-training the risk model with the policy improve generalisation?

Saves to: artifacts/marl/v1/ablation/S39/

Usage:
    python main.py --stage s39
    python src/train_s39.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from config import (
        CSV_MONDAY, FEATURES_JSON,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
except ImportError:
    from coopids.config import (
        CSV_MONDAY, FEATURES_JSON,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )

from src.flow_encoder import (
    FlowEncoder, load_feature_names, build_handshake_mapping,
    prepare_features, pretrain_encoder, train_encoder_step,
)
from src.environment_e2e import MultiAgentEnvE2E, STATE_DIM, ACTION_DIM, JOINT_DIM
from src.mappo import MAPPO, RolloutBuffer

_S39_DIR = MARL_DIR / "ablation" / "S39"

WARMUP_EPOCHS = 20      # supervised pre-training epochs
ENCODER_LR    = 1e-3    # encoder learning rate during joint training


def load_day(csv_path: Path, features: list[str]) -> pd.DataFrame:
    """Load a day's CSV with required columns."""
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train() -> None:
    _S39_DIR.mkdir(parents=True, exist_ok=True)

    print("S39 config: END-TO-END FLOW ENCODER + MAPPO")
    print("  Encoder:    MLP (42→64→32→3+1) replaces LightGBM")
    print("  Algorithm:  MAPPO (same as S27)")
    print("  Training:   Encoder warm-up (20 epochs BCE) + joint training")
    print(f"  State dim:  {STATE_DIM}, Action dim: {ACTION_DIM}, Joint dim: {JOINT_DIM}")

    # ── Load features and data ─────────────────────────────────────────────
    features = load_feature_names(FEATURES_JSON)
    print(f"\nLoading Monday data...")
    df_monday = load_day(CSV_MONDAY, features)
    print(f"  Monday: {len(df_monday):,} flows, {len(features)} features")

    # ── Preprocess features ────────────────────────────────────────────────
    print("Preprocessing features...")
    hs_mapping = build_handshake_mapping(df_monday)
    X_monday, imputer, scaler = prepare_features(df_monday, features, hs_mapping)
    labels_monday = df_monday["label"].values
    print(f"  Feature matrix: {X_monday.shape}")
    print(f"  Feature stats after scaling: mean={X_monday.mean():.4f}, std={X_monday.std():.4f}")
    print(f"  Label distribution: {pd.Series(labels_monday).value_counts().to_dict()}")

    # Save preprocessing artifacts (needed for evaluation)
    preproc = {
        "features": features,
        "hs_mapping": hs_mapping,
        "imputer": imputer,
        "scaler": scaler,
    }
    with open(_S39_DIR / "preprocessing.pkl", "wb") as f:
        pickle.dump(preproc, f)

    # ── Create encoder and warm up ─────────────────────────────────────────
    encoder = FlowEncoder(n_features=len(features), embed_dim=3, hidden=64)

    print(f"\nPhase 1: Encoder warm-up ({WARMUP_EPOCHS} epochs supervised BCE)...")
    warmup_losses = pretrain_encoder(
        encoder, X_monday, labels_monday,
        epochs=WARMUP_EPOCHS, batch_size=1024, lr=ENCODER_LR,
    )
    print(f"  Warm-up loss: {warmup_losses[0]:.4f} → {warmup_losses[-1]:.4f}")

    # Verify encoder risk scores after warm-up
    encoder.eval()
    with torch.no_grad():
        _, risk_scores = encoder(torch.FloatTensor(X_monday))
        risk_np = risk_scores.numpy()
        benign_mask = labels_monday == "Benign"
        attack_mask = labels_monday == "Attack"
        print(f"  Risk scores after warm-up:")
        print(f"    Benign mean:  {risk_np[benign_mask].mean():.4f}")
        if attack_mask.any():
            print(f"    Attack mean:  {risk_np[attack_mask].mean():.4f}")

    # ── Create environment and MAPPO ───────────────────────────────────────
    env = MultiAgentEnvE2E(
        df=df_monday, X_preprocessed=X_monday, encoder=encoder,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE,
    )
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM,
                   action_dim=ACTION_DIM, n_agents=N_AGENTS)
    buffer = RolloutBuffer(n_agents=N_AGENTS)

    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=ENCODER_LR * 0.1)

    log         = []
    best_reward = -np.inf

    print(f"\nPhase 2: Joint training for {N_EPISODES} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Enc L':>8} {'Time':>7}")
    print("-" * 75)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()

        local_states, joint_obs = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            actions, log_probs, value = mappo.act(local_states, joint_obs)
            next_local, next_joint, rewards, done = env.step(actions)
            mean_reward = float(np.mean(rewards))

            buffer.store(
                local_obs=local_states,
                joint_obs=joint_obs,
                actions=actions,
                log_probs=log_probs,
                reward=mean_reward,
                done=done,
                value=value,
            )

            local_states = next_local
            joint_obs    = next_joint
            ep_reward   += mean_reward
            ep_steps    += 1

        # ── PPO + encoder update ───────────────────────────────────────────
        update_info = {}
        enc_loss = 0.0
        if episode % UPDATE_EVERY == 0 and len(buffer) > 0:
            _, _, last_val = mappo.act(local_states, joint_obs)
            update_info = mappo.update(buffer, last_value=last_val)
            buffer.clear()

            # Encoder supervised update (joint training)
            enc_loss = train_encoder_step(
                encoder, encoder_opt, X_monday, labels_monday,
                batch_size=1024,
            )

        elapsed = time.time() - t0
        entry = {
            "episode":      episode,
            "reward":       round(ep_reward, 4),
            "steps":        ep_steps,
            "actor_loss":   round(update_info.get("actor_loss",  0.0), 6),
            "critic_loss":  round(update_info.get("critic_loss", 0.0), 6),
            "entropy":      round(update_info.get("entropy",     0.0), 6),
            "encoder_loss": round(enc_loss, 6),
            "time_s":       round(elapsed, 2),
        }
        log.append(entry)

        if episode % 10 == 0 or episode == 1:
            print(f"{episode:>8d} {ep_reward:>10.4f} {ep_steps:>7d} "
                  f"{entry['actor_loss']:>9.6f} {entry['critic_loss']:>9.6f} "
                  f"{entry['entropy']:>9.6f} {enc_loss:>8.5f} {elapsed:>6.1f}s")

        if episode % CHECKPOINT_EVERY == 0:
            mappo.save(_S39_DIR)
            encoder.save(_S39_DIR)
            with open(_S39_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S39_DIR / "best")
            encoder.save(_S39_DIR / "best")

    # ── Final save ─────────────────────────────────────────────────────────
    mappo.save(_S39_DIR)
    encoder.save(_S39_DIR)
    with open(_S39_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Save final encoder risk distribution
    encoder.eval()
    with torch.no_grad():
        _, risk_final = encoder(torch.FloatTensor(X_monday))
        risk_np = risk_final.numpy()
        benign_mask = labels_monday == "Benign"
        attack_mask = labels_monday == "Attack"
        print(f"\nFinal encoder risk scores:")
        print(f"  Benign mean:  {risk_np[benign_mask].mean():.4f}")
        if attack_mask.any():
            print(f"  Attack mean:  {risk_np[attack_mask].mean():.4f}")

    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S39_DIR}")


if __name__ == "__main__":
    train()
