"""Generate a dark-themed training curve plot for S27, suitable for PowerPoint."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Paths ──────────────────────────────────────────────────────────────────
LOG_PATH = (
    "/Users/abdenourbelarbi/Documents/MultiAgents/MARL_Cybersecurity/"
    "Research_project/artifacts/marl/v1/ablation/S27/training_log.json"
)
OUT_PATH = (
    "/Users/abdenourbelarbi/Documents/MultiAgents/MARL_Cybersecurity/"
    "Research_project/report/training_curve_dark.png"
)

# ── Load data ──────────────────────────────────────────────────────────────
with open(LOG_PATH, "r") as f:
    log = json.load(f)

episodes = np.array([entry["episode"] for entry in log])
rewards = np.array([entry["reward"] for entry in log])

# Rolling average (window=20)
window = 20
rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
rolling_episodes = episodes[window - 1 :]

# ── Theme constants ────────────────────────────────────────────────────────
BG_COLOR = "#1B1B2F"
CYAN = "#00D2FF"
RAW_GREY = "#AAAAAA"
YELLOW = "#FFD700"
DARK_GREEN = "#0D4D2B"
WHITE = "#FFFFFF"

# ── Plot ───────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
})

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Raw rewards – faint light grey
ax.plot(episodes, rewards, color=RAW_GREY, alpha=0.25, linewidth=0.7, label="Raw reward")

# Rolling average – cyan
ax.plot(rolling_episodes, rolling_avg, color=CYAN, linewidth=2, label=f"Rolling avg (w={window})")

# First PPO update at episode 10
ax.axvline(x=10, color=YELLOW, linestyle="--", linewidth=1.2, label="First PPO update")

# Convergence shaded region starting at ~episode 200
ax.axvspan(200, episodes[-1], color=DARK_GREEN, alpha=0.25, label="Convergence region")

# Labels and title
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Reward", fontsize=12)
ax.set_title("S27 Reward Convergence", fontsize=14, color=WHITE, fontweight="bold")

# Legend
legend = ax.legend(loc="lower right", fontsize=9, facecolor="#2A2A4A", edgecolor="#444466")
for text in legend.get_texts():
    text.set_color(WHITE)

# Spine styling
for spine in ax.spines.values():
    spine.set_color("#444466")

ax.grid(True, color="#333355", linewidth=0.4, alpha=0.5)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, transparent=False, facecolor=BG_COLOR)
plt.close(fig)

print(f"Saved plot to {OUT_PATH}")
