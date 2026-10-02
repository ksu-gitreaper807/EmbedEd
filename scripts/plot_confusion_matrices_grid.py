"""Grid-layout confusion-matrix board — companion to plot_confusion_matrices.py.

Same panels, same arithmetic, same stdout consistency table as the original
script (they are imported, not copied) — only the LAYOUT differs:

    row 0        : C0 (the untuned baseline), centred ("top middle")
    rows 1..6    : every other run in a 6-column grid, sorted C1..C6 then seed
                   ascending — with the full 6-seed campaign this puts ONE
                   CONDITION PER ROW and seeds 13..18 across the columns.

Panels are auto-discovered exactly like the original (`C{0..6}_<seed>/` with a
`predictions.npz`; smoke_/sprime_ never match). Reads ONLY predictions.npz —
no model, no GPU — so it runs on any VM or local pull:

    python -m scripts.plot_confusion_matrices_grid [--runs-dir DIR] [--out PNG] [--dpi N]

Output default: report/figures/phase2_confusion_matrices_grid.png.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embeded import settings as S
# one implementation of the panel + discovery + arithmetic: imported, not forked
from scripts.plot_confusion_matrices import (COND_COLOR, INK, CONDITIONS,
                                             confusion, discover_models,
                                             draw_panel)

GRID_COLS = 6


def split_models(models: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """C0 runs go to the banner; everything else keeps the canonical order."""
    banner = [m for m in models if m[0] == "C0"]
    rest = [m for m in models if m[0] != "C0"]
    return banner, rest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None,
                    help="default: $EMBEDED_ARTIFACTS/runs")
    ap.add_argument("--out", default=None,
                    help="default: report/figures/phase2_confusion_matrices_grid.png")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)

    runs = a.runs_dir or str(S.ARTIFACTS / S.RUNS_SUBDIR)
    out = a.out or str(S.REPORT_MD.parent / "figures" / "phase2_confusion_matrices_grid.png")

    models = discover_models(runs)
    if not models:
        print(f"no C0..C6 run dirs with predictions.npz under {runs} — pull artifacts first")
        return 1
    banner, rest = split_models(models)

    grid_rows = max(1, math.ceil(len(rest) / GRID_COLS))
    fig = plt.figure(figsize=(4.8 * GRID_COLS, 4.6 * (grid_rows + 1) + 0.8), facecolor="white")
    gs = fig.add_gridspec(grid_rows + 1, GRID_COLS)

    print(f"== confusion-matrix GRID from {runs} ({len(models)} runs: "
          f"C0 banner + {len(rest)} in a {grid_rows}x{GRID_COLS} grid) ==")
    print(f"{'run':>9} | {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7} | "
          f"{'P':>6} {'R':>6} {'F1':>6} {'thr':>6} {'n_pred_pos':>10}")

    def render(ax, cond: str, seed: int) -> bool:
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
        return m is not None

    k = 0
    # --- row 0: C0 centred across the two middle columns ("top middle") -----
    banner_ax = fig.add_subplot(gs[0, GRID_COLS // 2 - 1:GRID_COLS // 2 + 1])
    if banner:
        for cond, seed in banner:                  # one C0 today; more would overlap
            k += render(banner_ax, cond, seed)
    else:
        banner_ax.axis("off")
        print("(no C0 run found — banner slot left empty)")

    # --- rows 1..: the rest, 6 per row, sorted C1..C6 then seeds ascending ---
    for i, (cond, seed) in enumerate(rest):
        r, c = divmod(i, GRID_COLS)
        k += render(fig.add_subplot(gs[r + 1, c]), cond, seed)
    for j in range(len(rest), grid_rows * GRID_COLS):
        r, c = divmod(j, GRID_COLS)
        fig.add_subplot(gs[r + 1, c]).axis("off")

    fig.suptitle(f"Phase 2 — test-split confusion matrices, {len(models)} models "
                 f"(C0 centred; one row per condition, seeds ascending)",
                 fontsize=17, fontweight="bold", color=INK, y=0.995)
    fig.text(0.012, 0.004,
             f"n = {S.EXPECTED_SPLIT_ROWS['test']:,} shared test pairs per panel · "
             "threshold chosen on validation (max_f1_on_valid), applied unchanged · "
             "cell colour = row share · C0 = untuned baseline (banner)",
             fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, 0.012, 1, 0.97), h_pad=3.2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print(f"wrote {out} ({k}/{len(models)} runs found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
