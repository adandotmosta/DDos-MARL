"""
generate_plots.py
Generate all paper figures from existing experiment data.

Saves PNG images (300 dpi) to: artifacts/plots/

Usage:
    python generate_plots.py            # all plots
    python generate_plots.py --only 1   # only figure 1
    python generate_plots.py --only 1,3,5  # specific figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent
EVAL    = BASE / "artifacts" / "evaluation" / "v1" / "s12"
TRAIN   = BASE / "artifacts" / "marl" / "v1"
OUT_DIR = BASE / "artifacts" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "monday": "#2196F3",
    "tuesday": "#FF5722",
    "thursday": "#4CAF50",
    "saturday": "#9C27B0",
    "tpr": "#2196F3",
    "fpr": "#FF5722",
    "mappo": "#1565C0",
    "ippo": "#E65100",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_eval(model: str, day: str) -> dict | None:
    """Load evaluation result for a model on a given day."""
    return load_json(EVAL / f"{model}_{day}_results.json")


def load_training_log(stage: str) -> list | None:
    """Load training_log.json for a given ablation stage."""
    # Try ablation dir first, then root
    path = TRAIN / "ablation" / stage / "training_log.json"
    if not path.exists():
        path = TRAIN / "training_log.json"
    return load_json(path)


def savefig(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.relative_to(BASE)}")


# ── Model registry ──────────────────────────────────────────────────────────
# Maps display name → eval file prefix
# Grouped by experiment family for selective plotting

EVAL_FILE_MAP = {
    # Early ablations (throttle-based, S6-S12)
    "S6 (IQL)":        "iql",
    "S7 (IPPO)":       "ippo",
    "S8 (MAPPO)":      "",       # monday_results.json (no prefix)
    "S9 (Local Rew)":  "mapl",
    "S10 (Ext State)":  "e13",
    "S11 (10-Act)":    "s11",
    "S12 (No Puzzle)": "s12",
    # Tiered ablations (S26+)
    "S26 (Tiered)":    "s26",
    "S26b (Mon LGB)":  "s26b",
    "S27 (p_risk)":    "s27",
    "S28 (Mixed Train)":"s28",
    "S29 (IPPO)":      "s29",
    "S30 (Comb LGB)":  "s30" if (EVAL / "s30_monday_results.json").exists() else None,
    "S31 (IPPO 6ag)":  "s31",
    "S32 (1-dim)":     "s32",
    "S33 (Blocklist)": "s33",
    "S34 (IPPO BL)":   "s34",
    "S35 (Victim R)":  "s35",
    "S36 (IPPO VR)":   "s36",
    "S37 (MAPPO 6ag)": "s37",
    "S38 (ECMP)":      "s38",
    "S39 (FlowEnc)":   "s39",
    "S40 (Global R)":  "s40",
}
# Filter out None entries
EVAL_FILE_MAP = {k: v for k, v in EVAL_FILE_MAP.items() if v is not None}


def _eval_path(prefix: str, day: str) -> Path:
    if prefix == "":
        return EVAL / f"{day}_results.json"
    return EVAL / f"{prefix}_{day}_results.json"


def load_all_evals(day: str) -> dict[str, dict]:
    """Load all eval results for a given day, returns {display_name: data}."""
    results = {}
    for name, prefix in EVAL_FILE_MAP.items():
        data = load_json(_eval_path(prefix, day))
        if data is not None:
            results[name] = data
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: TPR & FPR bar chart — all experiments, Monday vs Tuesday
# ─────────────────────────────────────────────────────────────────────────────
def fig1_tpr_fpr_bars():
    print("Figure 1: TPR & FPR bar chart (all experiments)")

    monday = load_all_evals("monday")
    tuesday = load_all_evals("tuesday")

    # Use only models that have both days
    names = [n for n in EVAL_FILE_MAP if n in monday and n in tuesday]
    n = len(names)

    tpr_mon = [monday[n]["tpr"] for n in names]
    fpr_mon = [monday[n]["fpr"] for n in names]
    tpr_tue = [tuesday[n]["tpr"] for n in names]
    fpr_tue = [tuesday[n]["fpr"] for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(n)
    w = 0.35

    # TPR
    ax1.bar(x - w/2, tpr_mon, w, label="Monday (train day)", color=COLORS["monday"], alpha=0.85)
    ax1.bar(x + w/2, tpr_tue, w, label="Tuesday (unseen day)", color=COLORS["tuesday"], alpha=0.85)
    ax1.set_ylabel("TPR (Attack Detection Rate)")
    ax1.set_title("Attack Detection Rate (TPR) — Higher is Better")
    ax1.legend(loc="lower left")
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(y=0.377, color="gray", ls="--", lw=1, alpha=0.6)
    ax1.annotate("LightGBM-only baseline (37.7%)", xy=(0.02, 0.377),
                 xycoords=("axes fraction", "data"), fontsize=8, color="gray")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # FPR
    ax2.bar(x - w/2, fpr_mon, w, label="Monday", color=COLORS["monday"], alpha=0.85)
    ax2.bar(x + w/2, fpr_tue, w, label="Tuesday", color=COLORS["tuesday"], alpha=0.85)
    ax2.set_ylabel("FPR (False Positive Rate)")
    ax2.set_title("False Positive Rate (FPR) — Lower is Better")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, max(max(fpr_mon), max(fpr_tue)) * 1.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=55, ha="right", fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig1_tpr_fpr_all_experiments")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Key experiments comparison (selected subset for paper clarity)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_key_comparison():
    print("Figure 2: Key experiments comparison (selected)")

    key_models = [
        "S6 (IQL)", "S7 (IPPO)", "S8 (MAPPO)", "S9 (Local Rew)",
        "S12 (No Puzzle)", "S27 (p_risk)", "S29 (IPPO)",
        "S39 (FlowEnc)", "S40 (Global R)",
    ]

    monday = load_all_evals("monday")
    tuesday = load_all_evals("tuesday")

    names = [n for n in key_models if n in monday and n in tuesday]
    n = len(names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(n)
    w = 0.35

    # TPR
    ax = axes[0]
    ax.bar(x - w/2, [monday[n]["tpr"] for n in names], w,
           label="Monday", color=COLORS["monday"], alpha=0.85)
    ax.bar(x + w/2, [tuesday[n]["tpr"] for n in names], w,
           label="Tuesday", color=COLORS["tuesday"], alpha=0.85)
    ax.set_ylabel("TPR")
    ax.set_title("Attack Detection Rate (TPR)")
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    # FPR
    ax = axes[1]
    ax.bar(x - w/2, [monday[n]["fpr"] for n in names], w,
           label="Monday", color=COLORS["monday"], alpha=0.85)
    ax.bar(x + w/2, [tuesday[n]["fpr"] for n in names], w,
           label="Tuesday", color=COLORS["tuesday"], alpha=0.85)
    ax.set_ylabel("FPR")
    ax.set_title("False Positive Rate (FPR)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Key Ablation Comparison — Monday (train) vs Tuesday (unseen)", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig2_key_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Generalisation gap — TPR drop from Monday to Tuesday
# ─────────────────────────────────────────────────────────────────────────────
def fig3_generalisation_gap():
    print("Figure 3: Generalisation gap (Monday→Tuesday TPR drop)")

    monday = load_all_evals("monday")
    tuesday = load_all_evals("tuesday")

    names = [n for n in EVAL_FILE_MAP if n in monday and n in tuesday]
    gaps = [monday[n]["tpr"] - tuesday[n]["tpr"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if g <= 0.05 else "#FF9800" if g <= 0.10 else "#F44336" for g in gaps]
    bars = ax.bar(range(len(names)), [g * 100 for g in gaps], color=colors, alpha=0.85)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("TPR Drop (percentage points)")
    ax.set_title("Generalisation Gap: Monday TPR − Tuesday TPR")
    ax.axhline(y=5, color="green", ls="--", lw=1, alpha=0.5, label="<5pp (good)")
    ax.axhline(y=10, color="orange", ls="--", lw=1, alpha=0.5, label="<10pp (ok)")
    ax.legend(fontsize=8)

    # Annotate values
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{gap*100:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    savefig(fig, "fig3_generalisation_gap")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Training reward curves (selected experiments)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_training_curves():
    print("Figure 4: Training reward curves")

    experiments = {
        "S8 (MAPPO)":      ("S8",  None),       # root training_log.json
        "S9 (Local Rew)":  ("S9",  None),
        "S27 (p_risk)":    ("S27", None),
        "S29 (IPPO)":      ("S29", None),
        "S39 (FlowEnc)":   ("S39", None),
        "S40 (Global R)":  ("S40", None),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10

    for idx, (label, (stage, _)) in enumerate(experiments.items()):
        log = load_training_log(stage)
        if log is None:
            continue
        episodes = [e["episode"] for e in log]
        rewards  = [e["reward"]  for e in log]

        # Smooth with moving average (window=20)
        window = min(20, len(rewards) // 5) if len(rewards) > 20 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            eps_smooth = episodes[window-1:]
        else:
            smoothed = rewards
            eps_smooth = episodes

        ax.plot(eps_smooth, smoothed, label=label, color=cmap(idx), lw=1.5, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Curves (smoothed, window=20)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig4_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Training curves — actor loss, critic loss, entropy
# ─────────────────────────────────────────────────────────────────────────────
def fig5_training_losses():
    print("Figure 5: Training losses & entropy")

    stages = ["S27", "S29", "S39", "S40"]
    labels = ["S27 (MAPPO p_risk)", "S29 (IPPO)", "S39 (FlowEnc)", "S40 (Global R)"]
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["actor_loss", "critic_loss", "entropy"]
    titles  = ["Actor Loss", "Critic Loss", "Entropy"]

    for stage, label, idx in zip(stages, labels, range(len(stages))):
        log = load_training_log(stage)
        if log is None:
            continue
        episodes = [e["episode"] for e in log]

        for col, metric in enumerate(metrics):
            vals = [e[metric] for e in log]
            # Only plot non-zero segments (updates happen every UPDATE_EVERY)
            nonzero = [(ep, v) for ep, v in zip(episodes, vals) if v != 0.0]
            if not nonzero:
                continue
            eps_nz, vals_nz = zip(*nonzero)

            # Smooth
            window = min(10, len(vals_nz) // 5) if len(vals_nz) > 10 else 1
            if window > 1:
                smoothed = np.convolve(vals_nz, np.ones(window)/window, mode="valid")
                eps_sm = eps_nz[window-1:]
            else:
                smoothed = vals_nz
                eps_sm = eps_nz

            axes[col].plot(eps_sm, smoothed, label=label, color=cmap(idx), lw=1.2, alpha=0.8)

    for col, title in enumerate(titles):
        axes[col].set_xlabel("Episode")
        axes[col].set_ylabel(title)
        axes[col].set_title(title)
        axes[col].legend(fontsize=7)
        axes[col].grid(True, alpha=0.3)

    fig.suptitle("Training Dynamics", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig5_training_losses")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: MAPPO vs IPPO comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig6_mappo_vs_ippo():
    print("Figure 6: MAPPO vs IPPO comparison")

    pairs = [
        ("S27 (MAPPO)", "s27", "S29 (IPPO)", "s29", "3 agents, p_risk tiered"),
        ("S33 (MAPPO BL)", "s33", "S34 (IPPO BL)", "s34", "3 agents, blocklist"),
        ("S35 (MAPPO VR)", "s35", "S36 (IPPO VR)", "s36", "3 agents, victim routing"),
        ("S37 (MAPPO 6ag)", "s37", "S31 (IPPO 6ag)", "s31", "6 agents, p_risk tiered"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(pairs))
    w = 0.35

    tpr_mappo_mon, tpr_ippo_mon = [], []
    tpr_mappo_tue, tpr_ippo_tue = [], []
    xlabels = []

    for mappo_name, mappo_pf, ippo_name, ippo_pf, desc in pairs:
        mm = load_json(_eval_path(mappo_pf, "monday"))
        mt = load_json(_eval_path(mappo_pf, "tuesday"))
        im = load_json(_eval_path(ippo_pf, "monday"))
        it_ = load_json(_eval_path(ippo_pf, "tuesday"))

        tpr_mappo_mon.append(mm["tpr"] if mm else 0)
        tpr_mappo_tue.append(mt["tpr"] if mt else 0)
        tpr_ippo_mon.append(im["tpr"] if im else 0)
        tpr_ippo_tue.append(it_["tpr"] if it_ else 0)
        xlabels.append(desc)

    # Monday
    ax = axes[0]
    ax.bar(x - w/2, tpr_mappo_mon, w, label="MAPPO", color=COLORS["mappo"], alpha=0.85)
    ax.bar(x + w/2, tpr_ippo_mon, w, label="IPPO", color=COLORS["ippo"], alpha=0.85)
    ax.set_ylabel("TPR")
    ax.set_title("Monday (Train Day)")
    ax.set_ylim(0.8, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=8)
    ax.legend()

    # Tuesday
    ax = axes[1]
    ax.bar(x - w/2, tpr_mappo_tue, w, label="MAPPO", color=COLORS["mappo"], alpha=0.85)
    ax.bar(x + w/2, tpr_ippo_tue, w, label="IPPO", color=COLORS["ippo"], alpha=0.85)
    ax.set_ylabel("TPR")
    ax.set_title("Tuesday (Unseen Day)")
    ax.set_ylim(0.6, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=8)
    ax.legend()

    fig.suptitle("MAPPO (CTDE) vs IPPO (Independent) — TPR Comparison", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig6_mappo_vs_ippo")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Scalability — 3 agents vs 6 agents
# ─────────────────────────────────────────────────────────────────────────────
def fig7_scalability():
    print("Figure 7: Scalability (3 vs 6 agents)")

    configs = {
        "IPPO 3ag (S29)":  "s29",
        "IPPO 6ag (S31)":  "s31",
        "MAPPO 3ag (S27)": "s27",
        "MAPPO 6ag (S37)": "s37",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for day_idx, day in enumerate(["monday", "tuesday"]):
        ax = axes[day_idx]
        names, tprs, fprs = [], [], []
        for name, prefix in configs.items():
            data = load_json(_eval_path(prefix, day))
            if data:
                names.append(name)
                tprs.append(data["tpr"])
                fprs.append(data["fpr"])

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, tprs, w, label="TPR", color=COLORS["tpr"], alpha=0.85)
        ax.bar(x + w/2, fprs, w, label="FPR", color=COLORS["fpr"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{day.capitalize()}")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("Scalability: 3 Agents vs 6 Agents", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig7_scalability")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Action distribution heatmaps (tiered experiments)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_action_heatmaps():
    print("Figure 8: Action distribution heatmaps")

    TIERED_ACTIONS = ["passive", "light", "moderate", "firm", "aggressive", "lockdown"]

    models = {
        "S27 Monday":  ("s27", "monday"),
        "S27 Tuesday": ("s27", "tuesday"),
        "S29 Monday":  ("s29", "monday"),
        "S29 Tuesday": ("s29", "tuesday"),
        "S40 Monday":  ("s40", "monday"),
        "S40 Tuesday": ("s40", "tuesday"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for idx, (title, (prefix, day)) in enumerate(models.items()):
        data = load_json(_eval_path(prefix, day))
        if data is None:
            axes_flat[idx].set_visible(False)
            continue

        ad = data["action_distribution"]
        n_agents = len(ad)

        # Build matrix: agents x actions (normalised per agent)
        matrix = np.zeros((n_agents, len(TIERED_ACTIONS)))
        for i, agent_dict in enumerate(ad):
            total = sum(agent_dict.values())
            for j, act in enumerate(TIERED_ACTIONS):
                matrix[i, j] = agent_dict.get(act, 0) / max(total, 1)

        ax = axes_flat[idx]
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(TIERED_ACTIONS)))
        ax.set_xticklabels(TIERED_ACTIONS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels([f"Router {i+1}" for i in range(n_agents)], fontsize=9)
        ax.set_title(title, fontsize=10)

        # Annotate cells
        for i in range(n_agents):
            for j in range(len(TIERED_ACTIONS)):
                val = matrix[i, j]
                if val > 0.005:
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Action Proportion")
    fig.suptitle("Per-Router Action Distribution (Tiered Experiments)", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig8_action_heatmaps")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Action distribution heatmaps (throttle-based experiments)
# ─────────────────────────────────────────────────────────────────────────────
def fig9_action_heatmaps_throttle():
    print("Figure 9: Action distribution heatmaps (throttle-based)")

    THROTTLE_ACTIONS = ["forward", "puzzle", "throttle_10", "throttle_20",
                        "throttle_30", "throttle_50", "throttle_70",
                        "throttle_90", "throttle_95", "block"]

    models = {
        "S8 Monday":  ("", "monday"),
        "S8 Tuesday": ("", "tuesday"),
        "S9 Monday":  ("mapl", "monday"),
        "S9 Tuesday": ("mapl", "tuesday"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, (title, (prefix, day)) in enumerate(models.items()):
        data = load_json(_eval_path(prefix, day))
        if data is None:
            axes_flat[idx].set_visible(False)
            continue

        ad = data["action_distribution"]
        n_agents = len(ad)

        matrix = np.zeros((n_agents, len(THROTTLE_ACTIONS)))
        for i, agent_dict in enumerate(ad):
            total = sum(agent_dict.values())
            for j, act in enumerate(THROTTLE_ACTIONS):
                matrix[i, j] = agent_dict.get(act, 0) / max(total, 1)

        ax = axes_flat[idx]
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(THROTTLE_ACTIONS)))
        ax.set_xticklabels(THROTTLE_ACTIONS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels([f"Router {i+1}" for i in range(n_agents)], fontsize=9)
        ax.set_title(title, fontsize=10)

        for i in range(n_agents):
            for j in range(len(THROTTLE_ACTIONS)):
                val = matrix[i, j]
                if val > 0.005:
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Action Proportion")
    fig.suptitle("Per-Router Action Distribution (Throttle-Based Experiments)", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig9_action_heatmaps_throttle")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10: Architecture evolution — from throttle to tiered
# ─────────────────────────────────────────────────────────────────────────────
def fig10_architecture_evolution():
    print("Figure 10: Architecture evolution (throttle → tiered)")

    # Chronological progression of key design decisions
    evolution = [
        ("S6\nIQL",         "s6",   "iql"),
        ("S7\nIPPO",        "s7",   "ippo"),
        ("S8\nMAPPO",       "s8",   ""),
        ("S9\nLocal R",     "s9",   "mapl"),
        ("S11\n10-Act",     "s11",  "s11"),
        ("S12\nNo Puzzle",  "s12",  "s12"),
        ("S27\np_risk",     "s27",  "s27"),
        ("S29\nIPPO tier",  "s29",  "s29"),
        ("S39\nFlowEnc",    "s39",  "s39"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    names, tpr_mon, tpr_tue = [], [], []
    for label, stage, prefix in evolution:
        mm = load_json(_eval_path(prefix, "monday"))
        mt = load_json(_eval_path(prefix, "tuesday"))
        if mm and mt:
            names.append(label)
            tpr_mon.append(mm["tpr"])
            tpr_tue.append(mt["tpr"])

    x = np.arange(len(names))

    ax.plot(x, tpr_mon, "o-", color=COLORS["monday"], lw=2, ms=8,
            label="Monday TPR", zorder=3)
    ax.plot(x, tpr_tue, "s--", color=COLORS["tuesday"], lw=2, ms=8,
            label="Tuesday TPR", zorder=3)

    # Shade the architecture transition
    ax.axvspan(5.5, len(names) - 0.5, alpha=0.08, color="green", label="Tiered architecture")
    ax.axvspan(-0.5, 5.5, alpha=0.08, color="blue", label="Throttle architecture")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("TPR")
    ax.set_title("Architecture Evolution: Key Experiments")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.7, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    savefig(fig, "fig10_architecture_evolution")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11: Cross-day evaluation (Thursday/Saturday for S27, S28)
# ─────────────────────────────────────────────────────────────────────────────
def fig11_cross_day():
    print("Figure 11: Cross-day generalisation (Mon/Tue/Thu/Sat)")

    models = ["s27", "s26b", "s28"]
    labels = ["S27 (p_risk)", "S26b (tiered)", "S28 (mixed train)"]
    days   = ["monday", "tuesday", "thursday", "saturday"]
    day_colors = [COLORS["monday"], COLORS["tuesday"], COLORS["thursday"], COLORS["saturday"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(models)
    n_days = len(days)
    w = 0.18
    x = np.arange(n_models)

    for d_idx, day in enumerate(days):
        tprs = []
        for prefix in models:
            data = load_json(_eval_path(prefix, day))
            tprs.append(data["tpr"] if data else 0)
        offset = (d_idx - n_days / 2 + 0.5) * w
        ax.bar(x + offset, tprs, w, label=day.capitalize(),
               color=day_colors[d_idx], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("TPR")
    ax.set_title("Cross-Day Generalisation (4 evaluation days)")
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    savefig(fig, "fig11_cross_day_generalisation")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12: Local vs Global reward (S27 vs S40)
# ─────────────────────────────────────────────────────────────────────────────
def fig12_local_vs_global():
    print("Figure 12: Local vs Global reward (S27 vs S40)")

    TIERED_ACTIONS = ["passive", "light", "moderate", "firm", "aggressive", "lockdown"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    configs = [
        ("S27 Local — Monday",  "s27", "monday"),
        ("S40 Global — Monday", "s40", "monday"),
        ("S27 Local — Tuesday",  "s27", "tuesday"),
        ("S40 Global — Tuesday", "s40", "tuesday"),
    ]

    for idx, (title, prefix, day) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        data = load_json(_eval_path(prefix, day))
        if data is None:
            ax.set_visible(False)
            continue

        ad = data["action_distribution"]
        n_agents = len(ad)

        matrix = np.zeros((n_agents, len(TIERED_ACTIONS)))
        for i, agent_dict in enumerate(ad):
            total = sum(agent_dict.values())
            for j, act in enumerate(TIERED_ACTIONS):
                matrix[i, j] = agent_dict.get(act, 0) / max(total, 1)

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(TIERED_ACTIONS)))
        ax.set_xticklabels(TIERED_ACTIONS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels([f"R{i+1}" for i in range(n_agents)], fontsize=9)

        tpr, fpr = data["tpr"], data["fpr"]
        ax.set_title(f"{title}\nTPR={tpr:.1%}  FPR={fpr:.1%}", fontsize=10)

        for i in range(n_agents):
            for j in range(len(TIERED_ACTIONS)):
                val = matrix[i, j]
                if val > 0.005:
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.5, label="Action Proportion")
    fig.suptitle("Local vs Global Reward — Same TPR/FPR, Different Action Distributions", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig12_local_vs_global_reward")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13: Coordination mechanisms comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig13_coordination():
    print("Figure 13: Coordination mechanisms comparison")

    # Group by coordination mechanism
    groups = {
        "Baseline\n(p_risk tiered)":     [("S27 MAPPO", "s27"), ("S29 IPPO", "s29")],
        "Shared IP\nBlocklist":          [("S33 MAPPO", "s33"), ("S34 IPPO", "s34")],
        "Victim\nRouting":               [("S35 MAPPO", "s35"), ("S36 IPPO", "s36")],
        "ECMP\nRouting":                 [("S38 IPPO", "s38")],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for day_idx, day in enumerate(["monday", "tuesday"]):
        ax = axes[day_idx]
        x_pos = 0
        ticks, tick_labels = [], []

        for group_name, models in groups.items():
            for model_name, prefix in models:
                data = load_json(_eval_path(prefix, day))
                if data is None:
                    x_pos += 1
                    continue
                is_mappo = "MAPPO" in model_name
                color = COLORS["mappo"] if is_mappo else COLORS["ippo"]
                ax.bar(x_pos, data["tpr"], 0.7, color=color, alpha=0.85,
                       edgecolor="white", lw=0.5)
                ax.text(x_pos, data["tpr"] + 0.005, f"{data['tpr']:.1%}",
                        ha="center", va="bottom", fontsize=7)
                ticks.append(x_pos)
                tick_labels.append(model_name.split()[0])  # S27, S29, etc.
                x_pos += 1
            x_pos += 0.5  # gap between groups

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("TPR")
        ax.set_title(f"{day.capitalize()}")
        ax.set_ylim(0.7, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # Add group labels at the bottom
        from matplotlib.patches import FancyBboxPatch
        ax.legend(handles=[
            plt.Rectangle((0,0), 1, 1, fc=COLORS["mappo"], alpha=0.85, label="MAPPO"),
            plt.Rectangle((0,0), 1, 1, fc=COLORS["ippo"], alpha=0.85, label="IPPO"),
        ], labels=["MAPPO", "IPPO"], fontsize=8)

    fig.suptitle("Coordination Mechanisms — TPR by Architecture", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig13_coordination_mechanisms")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 14: Summary scatter — TPR vs FPR trade-off (all experiments)
# ─────────────────────────────────────────────────────────────────────────────
def fig14_tradeoff_scatter():
    print("Figure 14: TPR vs FPR trade-off scatter")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for day_idx, day in enumerate(["monday", "tuesday"]):
        ax = axes[day_idx]
        evals = load_all_evals(day)

        for name, data in evals.items():
            tpr, fpr = data["tpr"], data["fpr"]
            # Color by family
            if any(t in name for t in ["S27", "S28", "S30", "S33", "S35", "S37", "S39", "S40"]):
                c = COLORS["mappo"]
                marker = "o"
            elif any(t in name for t in ["S29", "S31", "S32", "S34", "S36", "S38"]):
                c = COLORS["ippo"]
                marker = "s"
            else:
                c = "gray"
                marker = "^"

            ax.scatter(fpr, tpr, c=c, marker=marker, s=60, alpha=0.8, zorder=3)
            ax.annotate(name.split("(")[0].strip(), (fpr, tpr),
                       fontsize=6, xytext=(4, 4), textcoords="offset points")

        ax.set_xlabel("FPR (False Positive Rate)")
        ax.set_ylabel("TPR (Attack Detection Rate)")
        ax.set_title(f"{day.capitalize()}")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # Ideal corner
        ax.annotate("Ideal\n(high TPR,\nlow FPR)", xy=(0, 1), fontsize=7,
                    color="green", alpha=0.5, ha="left", va="top")
        ax.grid(True, alpha=0.2)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["mappo"],
               markersize=8, label="MAPPO variants"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["ippo"],
               markersize=8, label="IPPO variants"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=8, label="Early ablations"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle("TPR vs FPR Trade-off — All Experiments", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig14_tpr_fpr_tradeoff")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 15: State dimensionality comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig15_state_dim():
    print("Figure 15: State dimensionality comparison")

    models = {
        "S32 (1-dim)":    "s32",
        "S29 (5-dim)":    "s29",
        "S10 (7-dim)":    "e13",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    w = 0.35

    names = list(models.keys())
    tpr_mon = []
    tpr_tue = []
    for name, prefix in models.items():
        mm = load_json(_eval_path(prefix, "monday"))
        mt = load_json(_eval_path(prefix, "tuesday"))
        tpr_mon.append(mm["tpr"] if mm else 0)
        tpr_tue.append(mt["tpr"] if mt else 0)

    ax.bar(x - w/2, tpr_mon, w, label="Monday", color=COLORS["monday"], alpha=0.85)
    ax.bar(x + w/2, tpr_tue, w, label="Tuesday", color=COLORS["tuesday"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("TPR")
    ax.set_title("Effect of State Dimensionality on Detection Rate")
    ax.legend()
    ax.set_ylim(0.7, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    savefig(fig, "fig15_state_dimensionality")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 16: Summary table as image
# ─────────────────────────────────────────────────────────────────────────────
def fig16_summary_table():
    print("Figure 16: Summary results table")

    monday = load_all_evals("monday")
    tuesday = load_all_evals("tuesday")

    names = [n for n in EVAL_FILE_MAP if n in monday and n in tuesday]

    rows = []
    for name in names:
        m, t = monday[name], tuesday[name]
        rows.append([
            name,
            f"{m['tpr']:.1%}", f"{m['fpr']:.1%}",
            f"{t['tpr']:.1%}", f"{t['fpr']:.1%}",
            f"{(m['tpr'] - t['tpr'])*100:+.1f}pp",
        ])

    # Add static baselines
    rows.append(["LightGBM only", "—", "—", "37.7%", "—", "—"])
    rows.append(["Static Aggressive", "—", "—", "91.8%", "18.2%", "—"])

    fig, ax = plt.subplots(figsize=(14, max(6, len(rows) * 0.35 + 1.5)))
    ax.axis("off")

    col_labels = ["Experiment", "Mon TPR", "Mon FPR", "Tue TPR", "Tue FPR", "TPR Gap"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title("Full Ablation Study Results Summary (S6–S40)", fontsize=14, pad=20)

    fig.tight_layout()
    savefig(fig, "fig16_summary_table")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
ALL_FIGURES = {
    1:  fig1_tpr_fpr_bars,
    2:  fig2_key_comparison,
    3:  fig3_generalisation_gap,
    4:  fig4_training_curves,
    5:  fig5_training_losses,
    6:  fig6_mappo_vs_ippo,
    7:  fig7_scalability,
    8:  fig8_action_heatmaps,
    9:  fig9_action_heatmaps_throttle,
    10: fig10_architecture_evolution,
    11: fig11_cross_day,
    12: fig12_local_vs_global,
    13: fig13_coordination,
    14: fig14_tradeoff_scatter,
    15: fig15_state_dim,
    16: fig16_summary_table,
}

def main():
    parser = argparse.ArgumentParser(description="Generate paper plots")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated figure numbers to generate (e.g. 1,3,5)")
    args = parser.parse_args()

    if args.only:
        figs = [int(x.strip()) for x in args.only.split(",")]
    else:
        figs = sorted(ALL_FIGURES.keys())

    print(f"Generating {len(figs)} figures → {OUT_DIR.relative_to(BASE)}/\n")

    for num in figs:
        if num in ALL_FIGURES:
            try:
                ALL_FIGURES[num]()
            except Exception as e:
                print(f"  ERROR in figure {num}: {e}")
        else:
            print(f"  Unknown figure number: {num}")

    print(f"\nDone! All images saved to: {OUT_DIR.relative_to(BASE)}/")


if __name__ == "__main__":
    main()
