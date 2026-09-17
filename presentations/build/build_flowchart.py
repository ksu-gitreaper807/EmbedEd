#!/usr/bin/env python3
"""
Generate the research-process flowchart as a standalone asset.

Corporate / computer-science diagram conventions:
  - muted corporate palette (navy, slate, steel blue, amber, green)
  - rounded process boxes, straight orthogonal connectors, downward flow
  - sans-serif type, clear hierarchy, generous whitespace
  - numbered steps down the left edge
  - vector SVG output with text kept as editable <text> elements

Outputs:
  presentations/diagrams/research-process-flowchart.svg
  presentations/diagrams/research-process-flowchart.png

Run:  /tmp/pptxenv/bin/python presentations/build/build_flowchart.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(ROOT, "presentations", "diagrams")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- palette
NAVY = "#12263A"
INK = "#16202B"
BODY = "#444E59"
MUTED = "#5A6672"
ACC = "#1F6FB2"       # steel blue  - process
ACC2 = "#E8734A"      # amber       - the manipulated variable
GOOD = "#2E8B6F"      # green       - outputs
GREY = "#8A94A0"      # neutral branch
RULE = "#D9DFE6"
ARROW = "#9AA5B1"
PANEL = "#F7FAFD"

plt.rcParams["font.family"] = "DejaVu Sans"
# keep text as editable <text> elements rather than outline paths
plt.rcParams["svg.fonttype"] = "none"

FIG_W, FIG_H = 11.0, 8.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

M = 0.7                      # side margin
CW = FIG_W - 2 * M           # 9.6


# ---------------------------------------------------------------- helpers
def box(x, y, w, h, title, sub=None, fill="white", edge=ACC, tcolor=INK,
        scolor=BODY, tsize=11.5, ssize=9.5, spine=None, lw=1.4, tweight="bold"):
    """Rounded process box with optional left colour spine."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.07",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3))
    if spine:
        ax.add_patch(Rectangle((x + 0.02, y + 0.06), 0.055, h - 0.12,
                               facecolor=spine, edgecolor="none", zorder=4))
    tx = x + (0.20 if spine else 0.16)
    if sub:
        ax.text(tx, y + h * 0.60, title, ha="left", va="center",
                fontsize=tsize, fontweight=tweight, color=tcolor, zorder=5)
        ax.text(tx, y + h * 0.27, sub, ha="left", va="center",
                fontsize=ssize, color=scolor, zorder=5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, fontweight=tweight, color=tcolor, zorder=5)


def arrow(x1, y1, x2, y2, color=ARROW, style="-", lw=1.5, dash=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, linestyle=style, zorder=2,
        shrinkA=0, shrinkB=0))


def step(n, y, h):
    """Numbered chip on the left margin."""
    cx, r = M - 0.42, 0.155
    ax.add_patch(plt.Circle((cx, y + h / 2), r, facecolor=PANEL,
                            edgecolor=RULE, linewidth=1.2, zorder=3))
    ax.text(cx, y + h / 2, str(n), ha="center", va="center",
            fontsize=10, fontweight="bold", color=MUTED, zorder=5)


# ---------------------------------------------------------------- title bar
ax.add_patch(Rectangle((0, FIG_H - 0.95), FIG_W, 0.95,
                       facecolor=NAVY, edgecolor="none", zorder=2))
ax.add_patch(Rectangle((0, FIG_H - 0.95), 0.16, 0.95,
                       facecolor=ACC2, edgecolor="none", zorder=3))
ax.text(M, FIG_H - 0.35, "Research Process — Negative-Sampling Strategy Comparison",
        ha="left", va="center", fontsize=15.5, fontweight="bold", color="white", zorder=5)
ax.text(M, FIG_H - 0.68, "Domain-Specific Contrastive Embeddings for Code Similarity Detection"
        "   ·   GraphCodeBERT   ·   BigCloneBench (CodeXGLUE)",
        ha="left", va="center", fontsize=10, color="#B9C6D4", zorder=5)


# ---------------------------------------------------------------- rows
# (top, height)
R1 = (7.10, 0.72)
R2 = (6.02, 0.62)
R3 = (5.04, 0.78)
R4 = (3.92, 0.62)
R5 = (2.94, 0.58)
R6 = (2.00, 0.68)
R7 = (0.96, 0.56)
GAP = 0.36

# three-column geometry
BW = (CW - 2 * 0.375) / 3.0          # 2.95
CX = [M, M + BW + 0.375, M + 2 * (BW + 0.375)]

# --- 1. dataset -------------------------------------------------------
y, h = R1
step(1, y, h)
box(M, y, CW, h, "Domain Dataset — BigCloneBench (CodeXGLUE)",
    "9,134 Java fragments   ·   901,028 / 415,416 / 415,416 train / validation / test pairs",
    fill=NAVY, edge=NAVY, tcolor="white", scolor="#B9C6D4")
arrow(FIG_W / 2, y, FIG_W / 2, y - GAP)

# --- 2. positives -----------------------------------------------------
y, h = R2
step(2, y, h)
box(M, y, CW, h, "Positive Pairs",
    "true clone pairs — fixed, and shared unchanged by every condition",
    fill="white", edge=ACC, spine=ACC)
arrow(FIG_W / 2, y, FIG_W / 2, y - GAP)

# --- 3. negative strategies -------------------------------------------
y, h = R3
step(3, y, h)
lbl = M
ax.text(lbl, y + h + 0.14, "THE INDEPENDENT VARIABLE — only this changes between conditions",
        ha="left", va="bottom", fontsize=9, fontweight="bold", color=ACC2, zorder=5)
neg = [
    ("Random Negatives", "any non-clone, sampled at random", GREY),
    ("BM25 Negatives", "top keyword matches, via rank_bm25", ACC),
    ("Semantic Negatives", "base-model cosine nearest neighbours", ACC2),
]
for i, (t, s, c) in enumerate(neg):
    box(CX[i], y, BW, h, t, s, fill="white", edge=c, spine=c, lw=1.5)
ax.text(M, y - 0.10, "known clones of the anchor are excluded in all three — verified before use",
        ha="left", va="top", fontsize=8.5, color=MUTED, zorder=5)
for i in range(3):
    arrow(CX[i] + BW / 2, y, CX[i] + BW / 2, y - GAP)

# --- 4. fine-tune -----------------------------------------------------
y, h = R4
step(4, y, h)
for i, (t, c) in enumerate([("Fine-tune  C1", GREY), ("Fine-tune  C2", ACC),
                            ("Fine-tune  C3", ACC2)]):
    box(CX[i], y, BW, h, t, "contrastive, identical settings", fill=PANEL,
        edge=c, spine=c, scolor=MUTED, ssize=8.8)
for i in range(3):
    arrow(CX[i] + BW / 2, y, CX[i] + BW / 2, y - GAP)

# --- 5. baseline ------------------------------------------------------
y, h = R5
step(5, y, h)
box(M, y, CW, h, "+  Baseline  C0 — off-the-shelf model, no fine-tuning  (the control)",
    fill="#EEF2F7", edge=MUTED, tcolor=INK, tsize=11.5)
for i in range(3):
    arrow(CX[i] + BW / 2, y, CX[i] + BW / 2, y - GAP)

# --- 6. evaluation ----------------------------------------------------
y, h = R6
step(6, y, h)
outs = [
    ("Standard Benchmark Evaluation", "F1 · precision · recall   +   MAP@R", ACC),
    ("Generalisation Evaluation", "held-out unseen functionalities   ·   Δ", ACC2),
    ("Embedding Visualisation", "t-SNE / UMAP, fixed seed", GOOD),
]
for i, (t, s, c) in enumerate(outs):
    box(CX[i], y, BW, h, t, s, fill="white", edge=c, spine=c, ssize=8.8)
for i in range(3):
    cx = CX[i] + BW / 2
    arrow(cx, y, cx, y - GAP * 0.45)
    arrow(cx, y - GAP * 0.45, FIG_W / 2, y - GAP * 0.45, lw=1.2)
arrow(FIG_W / 2, y - GAP * 0.45, FIG_W / 2, y - GAP)

# --- 7. results -------------------------------------------------------
y, h = R7
step(7, y, h)
box(M, y, CW, h, "Results Comparison Across the Four Conditions",
    "differences reported, not absolute levels — the benchmark's labels are themselves imperfect",
    fill=NAVY, edge=NAVY, tcolor="white", scolor="#B9C6D4")

# ---------------------------------------------------------------- footer
ax.text(M, 0.52, "Only the negative-selection method differs between C1, C2 and C3 — "
                 "model, loss, positives, subset, epochs and learning rate are held constant.",
        ha="left", va="center", fontsize=8.8, color=MUTED, zorder=5)
ax.plot([M, FIG_W - M], [0.36, 0.36], color=RULE, linewidth=1.0, zorder=2)

fig.savefig(os.path.join(OUTDIR, "research-process-flowchart.svg"),
            bbox_inches="tight", facecolor="white", format="svg")
plt.savefig(os.path.join(OUTDIR, "research-process-flowchart.png"),
            bbox_inches="tight", facecolor="white", dpi=200)
plt.close(fig)
print("wrote flowchart svg + png to", OUTDIR)
