"""Generate training/validation figures for ECANet_ImageNet100_analysis.md.

Parses ECANet_output.txt and writes PNGs to figures/.
Usage: python plot_training.py [log_path] [out_dir]
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "ECANet_output.txt")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
OUT.mkdir(exist_ok=True)

ITERS_PER_EPOCH = 660
LR_STEP = 30

# Palette (validated reference categorical slots 1/2) and neutral inks.
TRAIN = "#2a78d6"
VAL = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SHADE = "#f1f0ec"

train_re = re.compile(
    r"Epoch: \[(\d+)\]\[(\d+)/\d+\].*?Loss ([\d.]+) \(([\d.]+)\)\s+"
    r"Prec@1 ([\d.]+) \(([\d.]+)\)\s+Prec@5 ([\d.]+) \(([\d.]+)\)"
)
val_re = re.compile(r"\* Prec@1 ([\d.]+) Prec@5 ([\d.]+)")

iters = []  # (global_iter, instantaneous loss)
epoch_last = {}  # epoch -> (running loss, running top1, running top5) at last log line
val1, val5 = [], []
for line in LOG.read_text().splitlines():
    if m := train_re.search(line):
        ep, it = int(m[1]), int(m[2])
        iters.append((ep + it / ITERS_PER_EPOCH, float(m[3])))
        epoch_last[ep] = (float(m[4]), float(m[6]), float(m[8]))
    elif m := val_re.search(line):
        val1.append(float(m[1]))
        val5.append(float(m[2]))

epochs = sorted(epoch_last)[: len(val1)]
tr_loss = [epoch_last[e][0] for e in epochs]
tr1 = [epoch_last[e][1] for e in epochs]
tr5 = [epoch_last[e][2] for e in epochs]
assert len(epochs) == len(val1) == 50, (len(epochs), len(val1))

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

PHASES = [(-0.5, 9.5, "A"), (9.5, 29.5, "B"), (29.5, 31.5, "C"), (31.5, 49.5, "D")]


def decorate(ax, phases=True):
    if phases:
        for i, (a, b, name) in enumerate(PHASES):
            if i % 2 == 0:
                ax.axvspan(a, b, color=SHADE, zorder=0, lw=0)
            ax.text((a + b) / 2, 1.01, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2, fontsize=9)
    ax.axvline(LR_STEP - 0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.5, 49.5)
    ax.set_xlabel("Epoch")


def line(ax, x, y, color, label, **kw):
    ax.plot(x, y, color=color, lw=2, label=label, zorder=3, **kw)


# Figure 1: top-1 and top-5 accuracy, train vs. val
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, tr, va, name in [(axes[0], tr1, val1, "Top-1"), (axes[1], tr5, val5, "Top-5")]:
    decorate(ax)
    line(ax, epochs, tr, TRAIN, "Train (running avg.)")
    line(ax, epochs, va, VAL, "Validation")
    best = max(range(len(va)), key=va.__getitem__)
    ax.plot(best, va[best], "o", ms=8, color=VAL, mec="white", mew=2, zorder=4)
    ax.annotate(f"best {va[best]:.2f}% @ ep {best}", (best, va[best]),
                xytext=(0, -22), textcoords="offset points", ha="center",
                color=INK, fontsize=9)
    ax.text(LR_STEP - 0.2, 0.04, "LR ×0.1", transform=ax.get_xaxis_transform(),
            color=INK2, fontsize=9, ha="left")
    ax.set_ylabel(f"{name} accuracy (%)")
    ax.set_title(f"{name} accuracy", loc="left", color=INK, pad=16)
axes[0].legend(loc="lower right", bbox_to_anchor=(1, 0.08))
fig.tight_layout()
fig.savefig(OUT / "fig1_accuracy.png")
plt.close(fig)

# Figure 2: training loss (logged every 100 iterations + epoch running average)
fig, ax = plt.subplots(figsize=(8, 4))
decorate(ax)
xs, ls = zip(*iters)
ax.plot([x - 0.5 for x in xs], ls, color=TRAIN, lw=1, alpha=0.35,
        label="Per-batch loss (every 100 iters)", zorder=2)
line(ax, epochs, tr_loss, TRAIN, "Epoch running average")
ax.text(LR_STEP - 0.2, 0.92, "LR ×0.1", transform=ax.get_xaxis_transform(),
        color=INK2, fontsize=9, ha="left")
ax.set_ylabel("Cross-entropy loss")
ax.set_ylim(0, None)
ax.set_title("Training loss", loc="left", color=INK, pad=16)
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
fig.tight_layout()
fig.savefig(OUT / "fig2_train_loss.png")
plt.close(fig)

# Figure 4: train − validation top-1 gap
gap = [t - v for t, v in zip(tr1, val1)]
fig, ax = plt.subplots(figsize=(8, 3.6))
decorate(ax)
ax.axhline(0, color=INK2, lw=1, zorder=1)
ax.bar(epochs, gap, width=0.75, color=[VAL if g >= 0 else TRAIN for g in gap], zorder=3)
ax.annotate(f"{gap[0]:+.2f}", (0, 0), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8.5, color=INK)
ax.annotate(f"{gap[-1]:+.2f}", (49, gap[-1]), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8.5, color=INK)
ax.set_ylabel("Train − val top-1 (pts)")
ax.set_title("Generalization gap (blue: val > train; orange: train > val)",
             loc="left", color=INK, pad=16)
fig.tight_layout()
fig.savefig(OUT / "fig4_gap.png")
plt.close(fig)

# Figure 3: validation top-1 per-epoch change (makes the LR-step jump explicit)
delta = [val1[0]] + [b - a for a, b in zip(val1, val1[1:])]
fig, ax = plt.subplots(figsize=(8, 3.6))
decorate(ax)
ax.axhline(0, color=INK2, lw=1, zorder=1)
ax.bar(epochs[1:], delta[1:], width=0.75, color=VAL, zorder=3)
ax.annotate(f"{delta[30]:+.2f}", (30, delta[30]), xytext=(4, -2),
            textcoords="offset points", ha="left", fontsize=8.5, color=INK)
ax.set_ylabel("Δ val top-1 vs. prev. epoch (pts)")
ax.set_title("Epoch-over-epoch validation top-1 change", loc="left", color=INK, pad=16)
fig.tight_layout()
fig.savefig(OUT / "fig3_val_delta.png")
plt.close(fig)

print(f"epochs={len(epochs)} best top-1={max(val1):.2f} best top-5={max(val5):.2f}")
print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
