"""Stylized confusion-matrix board for every Phase 2 model found on disk.

Rows = actual label, columns = model prediction at the run's validation-chosen
threshold (`max_f1_on_valid`, applied to the shared test split unchanged).
Cells show row-% (large) and raw pair counts (small). Panels are
auto-discovered: every `artifacts/runs/C{0,1,2,3}_<seed>/` directory that has a
`predictions.npz` gets a panel (so extension campaigns — e.g. seeds 16/17/18 —
appear automatically after a pull; smoke and sprime dirs are ignored). Reads
ONLY `predictions.npz` — no model, no GPU — so it runs on any VM or local pull
that has the run dirs:

    python -m scripts.plot_confusion_matrices [--runs-dir DIR] [--out PNG] [--dpi N]

Output: report/figures/phase2_confusion_matrices.png (+ a stdout summary table
whose P/R/F1 must match each run's eval_metrics.json — an implicit consistency
check, same arithmetic as embeded.evaluate.precision_recall_f1).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from embeded import settings as S

CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
COND_COLOR = {"C0": "#64748b", "C1": "#0e7490", "C2": "#b45309", "C3": "#6d28d9", "C4": "#15803d"}
CELL_TAG = (("TP", "FN"), ("FP", "TN"))     # [actual][predicted]; positive = clone
INK = "#0f172a"


def discover_models(runs_dir: str | Path) -> list[tuple[str, int]]:
    """Every (condition, seed) with a predictions.npz, C0 first, seeds ascending.
    Only dirs named C{0..4}_<seed> qualify — smoke_/sprime_ dirs never match."""
    found: set[tuple[str, int]] = set()
    for d in Path(runs_dir).glob("C[0-4]_*"):
        cond, _, seed = d.name.rpartition("_")
        if cond in CONDITIONS:
            if (d / "predictions.npz").exists():
                found.add((cond, int(seed)))
    return sorted(found, key=lambda t: (CONDITIONS.index(t[0]), t[1]))


def confusion(scores: np.ndarray, labels: np.ndarray, thr: float) -> np.ndarray:
    """[[TP, FN], [FP, TN]] — identical predicate to embeded.evaluate (score >= thr)."""
    pred = scores >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fn = int(np.sum(~pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    tn = int(np.sum(~pred & (labels == 0)))
    return np.array([[tp, fn], [fp, tn]], dtype=float)


def draw_panel(ax, cond: str, seed: int, m: np.ndarray | None, thr: float | None) -> tuple:
    """One stylized matrix; returns (precision, recall, f1) for the stdout table."""
    ax.set_xticks([0, 1], labels=["pred clone", "pred not-clone"], fontsize=8)
    ax.set_yticks([0, 1], labels=["actual clone", "actual not-clone"], fontsize=8)
    if m is None:
        ax.set_facecolor("#f1f5f9")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#e2e8f0")
        ax.text(0.5, 0.55, "run directory missing", ha="center", va="center",
                fontsize=10, color="#94a3b8", transform=ax.transAxes)
        ax.set_title(f"{cond} · seed {seed}", color=COND_COLOR[cond],
                     fontsize=12, fontweight="bold")
        return (float("nan"),) * 3

    rown = m / m.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list(f"cm_{cond}", ["#ffffff", COND_COLOR[cond]])
    ax.imshow(rown, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(2):
        for j in range(2):
            pct, n, tag = rown[i, j], int(m[i, j]), CELL_TAG[i][j]
            ax.text(j, i - 0.13, tag, ha="center", va="center", fontsize=8,
                    color="#ffffff" if pct > 0.6 else "#64748b")
            ax.text(j, i + 0.05, f"{pct * 100:.1f}%", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="#ffffff" if pct > 0.6 else INK)
            ax.text(j, i + 0.24, f"{n:,}", ha="center", va="center", fontsize=8,
                    color="#e2e8f0" if pct > 0.6 else "#64748b")
    tp, fn, fp, tn = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ax.set_title(f"{cond} · seed {seed}\nF1 {f1:.4f} · threshold {thr:.4f}",
                 color=COND_COLOR[cond], fontsize=12, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return prec, rec, f1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None,
                    help="default: $EMBEDED_ARTIFACTS/runs")
    ap.add_argument("--out", default=None,
                    help="default: report/figures/phase2_confusion_matrices.png")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)

    runs = a.runs_dir or str(S.ARTIFACTS / S.RUNS_SUBDIR)
    out = a.out or str(S.REPORT_MD.parent / "figures" / "phase2_confusion_matrices.png")

    models = discover_models(runs)
    if not models:
        print(f"no C0..C3 run dirs with predictions.npz under {runs} — pull artifacts first")
        return 1
    n_rows = max(1, math.ceil(len(models) / 5))
    fig, axes = plt.subplots(n_rows, 5, figsize=(24, 4.6 * n_rows + 0.6), facecolor="white",
                             squeeze=False)
    print(f"== confusion matrices from {runs} ({len(models)} runs) ==")
    print(f"{'run':>9} | {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7} | "
          f"{'P':>6} {'R':>6} {'F1':>6} {'thr':>6} {'n_pred_pos':>10}")
    k = 0
    for ax, (cond, seed) in zip(axes.flat, models):
        m = thr = None
        npz = f"{runs}/{cond}_{seed}/predictions.npz"
        try:
            z = np.load(npz)
            m = confusion(z["test_scores"], z["test_labels"], float(z["threshold"]))
            thr = float(z["threshold"])
            tp, fn, fp, tn = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            print(f"{cond}_{seed:>5} | {int(tp):>7,} {int(fp):>7,} {int(fn):>7,} "
                  f"{int(tn):>7,} | {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} "
                  f"{thr:>6.4f} {int(tp + fp):>10,}")
        except (FileNotFoundError, KeyError) as e:
            print(f"{cond}_{seed:>5} | MISSING ({e.__class__.__name__})")
        draw_panel(ax, cond, seed, m, thr)
        k += bool(m is not None)

    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Phase 2 — test-split confusion matrices, {len(models)} models",
                 fontsize=17, fontweight="bold", color=INK, y=0.99)
    fig.text(0.012, 0.005,
             f"n = {S.EXPECTED_SPLIT_ROWS['test']:,} shared test pairs per panel · "
             "threshold chosen on validation (max_f1_on_valid), applied unchanged · "
             "cell colour = row share · C0 = untuned baseline",
             fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, 0.015, 1, 0.96), h_pad=3.2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print(f"wrote {out} ({k}/{len(models)} runs found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
