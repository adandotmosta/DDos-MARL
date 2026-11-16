#!/usr/bin/env python3
"""Generate publication-quality figures for the CoopIDS research paper."""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = pathlib.Path(__file__).parent
ARTIFACT_DIR = pathlib.Path(__file__).parent.parent / "artifacts"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Colours
C_ENV   = "#4A90D9"   # blue  – environment
C_ACTOR = "#5CB85C"   # green – actor
C_CRITIC = "#F0AD4E"  # orange – critic
C_TIER  = "#9B59B6"   # purple – tiered profiles
C_ARROW = "#333333"
C_BG    = "#F8F8F8"


# ===================================================================
# Figure 1: Architecture Diagram
# ===================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, color, label, fontsize=6.5, alpha=0.85, text_color="white"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor="black", linewidth=0.6, alpha=alpha)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", wrap=True)
        return b

    def arrow(x0, y0, x1, y1, color=C_ARROW, style="->", lw=0.8, ls="-"):
        a = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle=style, color=color,
                            linewidth=lw, linestyle=ls,
                            mutation_scale=10,
                            connectionstyle="arc3,rad=0")
        ax.add_patch(a)

    # --- Title ---
    ax.text(5, 6.75, "CoopIDS: CTDE Architecture",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(5, 6.4, "Centralized Training, Decentralized Execution",
            ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#555555")

    # --- Three router pipelines ---
    router_xs = [0.6, 3.6, 6.6]
    labels = ["Router 1", "Router 2", "Router 3"]
    flow_labels = ["Flows A", "Flows B", "Flows C"]

    for i, (rx, rl, fl) in enumerate(zip(router_xs, labels, flow_labels)):
        w = 2.2
        # Flows (top)
        box(rx, 5.7, w, 0.45, C_ENV, f"{rl}: {fl}", fontsize=6)
        # LightGBM risk scorer
        box(rx, 5.0, w, 0.45, C_ENV, "LightGBM Risk Scorer", fontsize=6)
        arrow(rx + w/2, 5.7, rx + w/2, 5.48)
        # State aggregation
        box(rx, 4.25, w, 0.5, C_ENV, f"State Aggregation\n$s_{i+1}$ (4-dim)", fontsize=6)
        arrow(rx + w/2, 5.0, rx + w/2, 4.78)
        # MAPPO Actor
        box(rx, 3.35, w, 0.55, C_ACTOR, f"MAPPO Actor $\\pi_{i+1}$", fontsize=6.5)
        arrow(rx + w/2, 4.25, rx + w/2, 3.93)
        # Tiered action
        box(rx, 2.55, w, 0.5, C_TIER, "Tiered Action\n(allow/flag/block)", fontsize=5.5)
        arrow(rx + w/2, 3.35, rx + w/2, 3.08)

    # --- Centralised Critic ---
    cx, cy, cw, ch = 1.5, 1.2, 7, 0.7
    box(cx, cy, cw, ch, C_CRITIC, "Centralised Critic  $V(s_1 \\| s_2 \\| s_3)$   "
        "— Joint State (12-dim)", fontsize=7, text_color="black")

    # Arrows from each local state to critic (dashed – training only)
    for rx in router_xs:
        mid = rx + 1.1
        arrow(mid, 2.55, mid, 1.93, color=C_CRITIC, style="->", lw=0.7, ls="--")

    # Label the dashed arrows
    ax.text(9.3, 2.0, "training\nonly", fontsize=5.5, color=C_CRITIC,
            ha="center", va="center", fontstyle="italic")

    # Arrow from critic back (value estimate, dashed)
    arrow(cx + cw/2, 1.2, cx + cw/2, 0.65, color=C_CRITIC, style="->", lw=0.7, ls="--")
    ax.text(cx + cw/2, 0.45, "Value estimate (training only)",
            ha="center", fontsize=5.5, color="#666666", fontstyle="italic")

    # Legend
    legend_items = [
        (C_ENV, "Environment / Feature extraction"),
        (C_ACTOR, "Decentralized Actor"),
        (C_CRITIC, "Centralized Critic"),
        (C_TIER, "Tiered Action Profiles"),
    ]
    for idx, (c, lab) in enumerate(legend_items):
        bx = 0.2 + idx * 2.5
        p = FancyBboxPatch((bx, 0.02), 0.25, 0.2, boxstyle="round,pad=0.02",
                           facecolor=c, edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.add_patch(p)
        ax.text(bx + 0.35, 0.12, lab, fontsize=5.5, va="center")

    fig.savefig(OUT_DIR / "fig_architecture.pdf")
    plt.close(fig)
    print("Saved fig_architecture.pdf")


# ===================================================================
# Figure 2: Training Curve
# ===================================================================
def fig_training_curve():
    log_path = ARTIFACT_DIR / "marl/v1/ablation/S27/training_log.json"
    with open(log_path) as f:
        data = json.load(f)

    episodes = np.array([d["episode"] for d in data])
    rewards  = np.array([d["reward"]  for d in data])
    n = len(episodes)

    # Rolling average
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_x = episodes[window - 1:]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))

    # Raw data (light)
    ax.plot(episodes, rewards, color="#4A90D9", alpha=0.25, linewidth=0.5, label="Raw reward")
    # Smoothed
    ax.plot(smooth_x, smoothed, color="#4A90D9", linewidth=1.3, label=f"Rolling avg (w={window})")

    # First PPO update
    ax.axvline(x=10, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(13, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
            "1st PPO\nupdate", fontsize=5.5, color="#E74C3C", va="bottom")

    # Convergence region – shade from episode 200 onward (or 60% mark if fewer episodes)
    conv_start = min(200, int(n * 0.6))
    ylims = ax.get_ylim()
    ax.axvspan(conv_start, episodes[-1], color="#2ECC71", alpha=0.10)
    ax.text(conv_start + (episodes[-1] - conv_start) / 2, ylims[1] * 0.995,
            "Convergence\nregion", fontsize=5.5, color="#27AE60", ha="center", va="top")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("S27 Training Reward Convergence", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, episodes[-1])

    fig.savefig(OUT_DIR / "fig_training_curve.pdf")
    plt.close(fig)
    print("Saved fig_training_curve.pdf")


# ===================================================================
# Figure 3: Monday vs Tuesday Bar Chart
# ===================================================================
def fig_results_comparison():
    models = [
        "LightGBM\nonly",
        "S9\nMAPPO\nbinary",
        "S27\nMAPPO\ntiered",
        "S29\nIPPO\ntiered",
        "S39\nFlowEnc",
        "S40\nGlobal\nreward",
        "S30\nRetrained\nsensor",
    ]
    mon = [82.01, 95.02, 99.52, 99.35, 98.28, 99.15, 99.10]
    tue = [37.73, 87.93, 85.78, 88.38, 83.96, 87.88, 94.68]
    drops = [m - t for m, t in zip(mon, tue)]

    x = np.arange(len(models))
    bw = 0.34

    fig, ax = plt.subplots(figsize=(7, 3.0))

    bars_mon = ax.bar(x - bw / 2, mon, bw, color="#5CB85C", edgecolor="white",
                      linewidth=0.4, label="Monday (seen attacks)", zorder=3)
    bars_tue = ax.bar(x + bw / 2, tue, bw, color="#F0AD4E", edgecolor="white",
                      linewidth=0.4, label="Tuesday (unseen attacks)", zorder=3)

    # Drop annotations
    for i, (bar, drop) in enumerate(zip(bars_tue, drops)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"\u2013{drop:.1f}", ha="center", va="bottom", fontsize=5.5,
                color="#E74C3C", fontweight="bold")

    ax.set_ylim(30, 104)
    ax.set_ylabel("TPR (%)")
    ax.set_title("TPR Comparison: Monday (Seen) vs Tuesday (Unseen Attacks)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6)
    ax.legend(loc="lower right", framealpha=0.9)

    # Gridlines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.7)
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    for yl in [80, 90, 100]:
        ax.axhline(y=yl, color="#AAAAAA", linewidth=0.6, linestyle="-", zorder=1)

    plt.xticks(rotation=0)

    fig.savefig(OUT_DIR / "fig_results_comparison.pdf")
    plt.close(fig)
    print("Saved fig_results_comparison.pdf")


# ===================================================================
if __name__ == "__main__":
    fig_architecture()
    fig_training_curve()
    fig_results_comparison()
    print("All figures generated.")
