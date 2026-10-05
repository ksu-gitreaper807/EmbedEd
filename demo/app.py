"""The §5 demo (PHASE3_PLAN §5, SCOPE P2-21; decision D6): one file, Gradio,
no serving infrastructure — a local single-file app, not a deployment.

- Default view: REAL held-out test pairs with their gold labels (not synthetic
  examples, not a blank text box). Scores come from the recorded
  predictions.npz of the two runs — the exact arrays the tables were built
  from, row-aligned to pairs_test.tsv (the same order evaluate() scored).
- Per pair: gold / C0 cosine vs its threshold / C_best cosine vs its threshold,
  each threshold drawn as a visible marker on the score bar, so the reader can
  see WHY the decision came out as it did.
- All four outcome modes (D6 = the 2x2 of (C0 decision) x (C_best decision))
  must have at least one real pair in the picker; --check asserts it headless.
- The free-text paste stays, labelled "illustrative only - not part of the
  evaluation", in those words.
- It never trains and never re-encodes the corpus at import; checkpoints live
  outside Git, so this app's first action on a fresh machine is:

      python -m scripts.hf_artifacts pull --if-configured

  (sections 9/12 pull 'artifacts/runs/C[0-6]_1[345]/*', mining brings pairs_*).

CLI:  python -m demo.app --check            headless contract check
      python -m demo.app                    serve the Gradio UI
      python -m demo.app --best C5 --seed 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embeded import settings as S
from embeded.data.prepare_data import load_fragments, load_pairs

ILLUSTRATIVE_LABEL = "illustrative only — not part of the evaluation"
DEFAULT_BEST = "C1"   # Phase-2's best trained condition (0.7464 ± 0.0176, n=6);
                      # C5 is close but its audit gate is still open, so it cannot headline.
CELL_NAMES = {
    (True, True): "both say clone",
    (False, True): "C0 wrong / C_best right",
    (True, False): "C0 right / C_best wrong",
    (False, False): "both say non-clone",
}


def _npz(condition: str, seed: int, runs_root: Path) -> dict:
    import numpy as np

    p = runs_root / f"{condition}_{seed}" / "predictions.npz"
    if not p.exists():
        raise SystemExit(f"missing {p} — the demo reads recorded predictions, it never "
                         "trains. First action on a fresh machine: "
                         "`python -m scripts.hf_artifacts pull --if-configured`")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def collect(runs_root: Path | None = None, *, c0: str = "C0", best: str = DEFAULT_BEST,
            seed0: int | None = None, seedb: int | None = None,
            pairs=None, frags: dict[int, str] | None = None):
    """Align the two runs' test scores with the real test pairs and group them
    into D6's four cells. Returns (rows, meta); rows carry everything the UI
    needs. Injectable `pairs`/`frags` for tests. The seed policy (defaults):
    among locally available seeds, pick the (seed0, seedb) pair that maximises
    the worst-cell coverage — documented, deterministic, no label cherry-picking
    (the labels are whatever the recorded npz scored)."""
    import numpy as np

    root = Path(runs_root) if runs_root else S.ARTIFACTS / S.RUNS_SUBDIR
    frags = load_fragments() if frags is None else frags
    pairs = load_pairs("test") if pairs is None else pairs
    seeds = list(S.SEEDS)

    def load(condition, seed):
        z = _npz(condition, seed, root)
        thr = float(np.asarray(z["threshold"]).ravel()[0])
        return (np.asarray(z["test_scores"], dtype=float),
                thr, z)

    def cell_counts(s0, sb):
        s_a, thr_a, _ = load(c0, s0)
        s_b, thr_b, _ = load(best, sb)
        if len(s_a) != len(pairs) or len(s_b) != len(pairs):
            raise SystemExit(f"predictions.npz rows ({len(s_a)}/{len(s_b)}) != pairs_test.tsv "
                             f"rows ({len(pairs)}) — artifacts from different data eras; "
                             "re-pull matching runs")
        counts = {(True, True): 0, (False, True): 0, (True, False): 0, (False, False): 0}
        for k, (_i, _j, lab) in enumerate(pairs):
            counts[(bool(s_a[k] >= thr_a), bool(s_b[k] >= thr_b))] += 1
        return min(counts.values()), counts

    if seed0 is None or seedb is None:
        best_pick, best_key = None, None
        for a in seeds:
            for b in seeds:
                try:
                    worst, counts = cell_counts(a, b)
                except SystemExit:
                    continue          # that seed pair simply is not pulled locally
                key = (worst, -abs(a - b), -a, -b)
                if best_key is None or key > best_key:
                    best_pick, best_key = (a, b), key
        if best_pick is None:
            raise SystemExit("no pulled (C0, C_best) seed pair found in runs/ — the demo "
                             "reads recorded predictions only; first action: "
                             "`python -m scripts.hf_artifacts pull --if-configured`")
        seed0, seedb = best_pick
        print(f"[demo] seed policy (max worst-cell coverage over pulled seeds): "
              f"C0 seed {seed0}, {best} seed {seedb}")
    worst, counts = cell_counts(seed0, seedb)

    s_a, thr_a, _ = load(c0, seed0)
    s_b, thr_b, _ = load(best, seedb)
    empty = [CELL_NAMES[c] for c, n in counts.items() if n == 0]
    rows = []
    for k, (i, j, lab) in enumerate(pairs):
        d0, db = bool(s_a[k] >= thr_a), bool(s_b[k] >= thr_b)
        rows.append({"k": k, "frag_i": int(i), "frag_j": int(j), "gold": int(lab),
                     "s0": float(s_a[k]), "thr0": thr_a,
                     "sb": float(s_b[k]), "thrb": thr_b,
                     "cell": (d0, db),
                     "text_a": frags.get(int(i)), "text_b": frags.get(int(j))})
    meta = {"c0": c0, "best": best, "seed0": seed0, "seedb": seedb,
            "counts": {f"{CELL_NAMES[c]}": n for c, n in sorted(counts.items())},
            "n_test_pairs": len(pairs), "worst_cell": worst, "empty": empty}
    return rows, meta


def check(runs_root: Path | None = None, **kw) -> int:
    """Headless --check: recorded metadata + npz only; no training, no corpus
    re-encode; asserts all four D6 cells have a real pair in the picker."""
    rows, meta = collect(runs_root, **kw)
    if meta["empty"]:
        raise SystemExit(f"D6 outcome mode(s) with no real pair: {meta['empty']} — try "
                         "--seed/--best overrides; if no seed pair covers a cell, that is "
                         "a finding about the thresholds, not something to fake with "
                         "synthetic pairs")
    print(f"[demo] {meta['n_test_pairs']} real held-out test pairs; "
          f"C0_{meta['seed0']} vs {meta['best']}_{meta['seedb']}; cells: {meta['counts']}")
    assert ILLUSTRATIVE_LABEL in _source(), \
        "the free-text box lost its 'illustrative only' label"
    print(f"free-text box labelled {ILLUSTRATIVE_LABEL!r}")
    return 0


def _source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------- UI

def _bar_figure(score: float, thr: float, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.barh([0], [1.0], color="#e8e8e8")
    ax.axvline(thr, color="crimson", lw=2)
    ax.plot([score], [0], marker="o", color="navy", markersize=11)
    ax.text(score, 0.32, f"cosine {score:.4f}", ha="center", fontsize=9)
    ax.text(thr, -0.42, f"thr {thr:.4f}", color="crimson", ha="center", fontsize=8)
    ax.set_xlim(0, 1), ax.set_yticks([]), ax.set_ylim(-0.7, 0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("clone" if False else "", fontsize=1)
    return fig


def serve(runs_root: Path | None = None, *, c0: str = "C0",
          best: str = DEFAULT_BEST, seed0: int | None = None,
          seedb: int | None = None, pairs=None, frags=None, max_in_picker=24,
          share: bool = False):
    import gradio as gr

    rows, meta = collect(runs_root, c0=c0, best=best, seed0=seed0, seedb=seedb,
                         pairs=pairs, frags=frags)
    if meta["empty"]:
        print(f"[warn] cells with no real pair: {meta['empty']} — the picker covers "
              f"the rest (--check would refuse to ship this)")
    # picker: round-robin across the four cells so every mode is reachable first-page
    by_cell: dict[tuple, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    picker, order = [], []
    rings = [iter(v) for v in by_cell.values()]
    while len(picker) < max_in_picker:
        progressed = False
        for it in rings:
            for r in it:
                picker.append(r); order.append(
                    f"[{CELL_NAMES[r['cell']]}] gold={'clone' if r['gold'] else 'non-clone'}"
                    f" · frag {r['frag_i']} ↔ {r['frag_j']}")
                progressed = True
                break
        if not progressed:
            break
    by_label = dict(zip(order, picker))

    def render(label):
        r = by_label[label]
        d0, db = "clone" if r["s0"] >= r["thr0"] else "non-clone", \
                 "clone" if r["sb"] >= r["thrb"] else "non-clone"
        return (r["text_a"] or f"(fragment {r['frag_i']})", r["text_b"] or f"(fragment {r['frag_j']})",
                f"**gold: {'clone' if r['gold'] else 'non-clone'}** · C0 says {d0} · "
                f"{best} says {db}",
                _bar_figure(r["s0"], r["thr0"], f"{c0} (seed {meta['seed0']})"),
                _bar_figure(r["sb"], r["thrb"], f"{best} (seed {meta['seedb']})"))

    with gr.Blocks(title="EmbedEd demo") as demo:
        gr.Markdown(
            f"## EmbedEd — real held-out test pairs\n"
            f"C0 = untuned base model (seed {meta['seed0']}), C_best = **{best}** "
            f"(seed {meta['seedb']}); thresholds are each run's recorded validation "
            f"threshold (red line), the dot is the pair's cosine. D6 cells present: "
            f"{meta['counts']}")
        pick = gr.Dropdown(choices=order, value=order[0], label="pair")
        ta = gr.Code(label="fragment A (test pair)", language="markdown")
        tb = gr.Code(label="fragment B (test pair)", language="markdown")
        gold = gr.Markdown()
        with gr.Row():
            g0 = gr.Plot(label=f"{c0} score vs threshold")
            gb = gr.Plot(label=f"{best} score vs threshold")
        pick.change(render, inputs=pick, outputs=[ta, tb, gold, g0, gb])
        gr.Markdown(f"### free-text paste — **{ILLUSTRATIVE_LABEL}**")

        def score_free(text_a: str, text_b: str):
            if not text_a.strip() or not text_b.strip():
                return "paste two snippets", None, None
            import numpy as np

            from embeded.evaluate import default_embed_fn
            root = Path(runs_root) if runs_root else S.ARTIFACTS / S.RUNS_SUBDIR
            out, figs = [], []
            for cond, seed in ((c0, meta["seed0"]), (best, meta["seedb"])):
                fn, _dev, _note = default_embed_fn(cond, seed)
                v = fn([text_a, text_b])
                cos = float(np.dot(v[0], v[1]) /
                            ((np.linalg.norm(v[0]) * np.linalg.norm(v[1])) + 1e-12))
                thr = float(np.asarray(_npz(cond, seed, root)["threshold"]).ravel()[0])
                out.append(f"{cond} seed {seed}: cosine {cos:.4f} vs threshold "
                           f"{thr:.4f} → **{'clone' if cos >= thr else 'non-clone'}**")
                figs.append(_bar_figure(cos, thr, f"{cond} seed {seed} (free text)"))
            return "\n\n".join(out), figs[0], figs[1]

        with gr.Row():
            fa = gr.Textbox(lines=6, label="snippet A")
            fb = gr.Textbox(lines=6, label="snippet B")
        btn = gr.Button("score pasted pair")
        o_md, o_g0, o_gb = gr.Markdown(), gr.Plot(), gr.Plot()
        btn.click(score_free, inputs=[fa, fb], outputs=[o_md, o_g0, o_gb])
    # --share: Colab prints a public gradio.live URL and this call BLOCKS —
    # interrupt the notebook cell to stop the server (local app, not a deployment).
    demo.launch(server_name="0.0.0.0", inbrowser=False, show_error=True, share=share)


def build_parser() -> argparse.ArgumentParser:
    """The CLI contract the notebook's §12 cells call (--check / --share); kept
    test-addressable so the authored contract cannot drift from the module."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="headless contract check")
    ap.add_argument("--best", default=DEFAULT_BEST)
    ap.add_argument("--seed", type=int, default=None, help="pin C0's seed")
    ap.add_argument("--seed-best", type=int, default=None, help="pin C_best's seed")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--share", action="store_true",
                    help="gradio public URL (Colab convenience); the call blocks until "
                         "interrupted — the demo is a local app, not a deployment")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    kw = {"best": a.best, "runs_root": Path(a.runs_root) if a.runs_root else None,
          "share": a.share}
    if a.seed:
        kw["seed0"] = a.seed
    if a.seed_best:
        kw["seedb"] = a.seed_best
    if a.check:
        return check(**kw)
    serve(**kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
