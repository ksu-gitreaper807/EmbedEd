"""The §4 figure: seven UMAP panels, C0-C6 (PHASE3_PLAN §4, SCOPE P2-18).

A figure, not evidence: it ships AFTER the Δ table and no claim in the write-up
may rest on it. The rules it implements, in one place:

- fixed seed and fixed n_neighbors / min_dist, all recorded in settings.py
  BEFORE the figure exists (P2-18);
- the SAME sample of fragments in every panel (different samples per panel make
  the panels incomparable) — deterministic under settings.VERSION + the sample
  size, and recorded fragment-id-exact in umap_params.json;
- colour by functionality, including the held-out ones (star markers), so the
  unseen-functionality story is visible next to the Δ table;
- C0 shows the untuned base encoder; every other condition shows the s′
  checkpoint of --model-seed (default settings.UMAP_MODEL_SEED), loaded with the
  era-tolerant scoring load, its fingerprint recorded in umap_params.json.

The fragment→functionality map is derived once from the s′ frame and written to
`sprime_fragment_functions.json` (synced in the `sprime` pull section), so later
VMs never need the raw corpus just to draw the figure.

CLI:  python -m embeded.visualize --panels C0,C1,C2,C3,C4,C5,C6
                                    [--model-seed 13] [--n-per-functionality 18]
Writes report/figures/umap_panels.png + umap_params.json. Generated files,
never hand-edited.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import settings as S
from .generalize import (build_fragments, load_split, load_sprime_fragments,
                         load_sprime_frame, sprime_run_dir)
from .train import CHECKPOINT_NAME, load_trained_checkpoint

FRAG_FUNCS_NAME = "sprime_fragment_functions.json"
KNOWN_CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
LEGEND_NOTE = ("star marker = held-out functionality ({holdout}); same fragment sample "
               "in every panel; a figure, not evidence - ships after the Δ table "
               "(PHASE3_PLAN §4)")


# ------------------------------------------------------------- fragment map + sample

def fragment_functions_from_frame(frame) -> dict[int, str]:
    """fragment idx -> functionality id, via the canonical build_fragments mapping
    (first appearance wins; frame order is frozen by the split's source md5)."""
    _, texts_map = build_fragments(frame)
    frag_func: dict[int, str] = {}
    for c1, c2, fid in zip(frame["code1"], frame["code2"], frame["functionality_id"]):
        f = str(fid)
        for code in (c1, c2):
            frag_func.setdefault(texts_map[str(code)], f)
    if len(frag_func) != len(texts_map):
        raise SystemExit("fragment→functionality derivation missed fragments — the "
                         "frame changed under the canonical fragment table; re-run --mine")
    return frag_func


def fragment_functions() -> dict[int, str]:
    """Portable artifact first; otherwise derive from the s′ frame once and WRITE
    it, so later VMs pull the map instead of the 481 MB raw corpus."""
    p = S.artifact(FRAG_FUNCS_NAME)
    if p.exists():
        return {int(k): str(v) for k, v in json.loads(p.read_text()).items()}
    frame, _prov = load_sprime_frame()
    frag_func = fragment_functions_from_frame(frame)
    p.write_text(json.dumps({str(k): v for k, v in sorted(frag_func.items())},
                            indent=0))
    print(f"[figure] wrote {p.name} ({len(frag_func)} fragments) — synced with the "
          "'sprime' pull section from now on")
    return frag_func


def sample_fragment_ids(frag_func: dict[int, str], n_per_func: int,
                        *, version: str | None = None) -> list[int]:
    """The ONE fragment sample every panel shares: n_per_func fragments per
    functionality, seeded shuffle, sorted output. Deterministic under
    (version, n_per_func) — re-running the figure re-samples identically."""
    by_func: dict[str, list[int]] = defaultdict(list)
    for idx in sorted(frag_func):
        by_func[frag_func[idx]].append(idx)
    rng = random.Random(f"umap-panels#{version or S.VERSION}#{n_per_func}")
    out: list[int] = []
    for fid in sorted(by_func):
        idxs = list(by_func[fid])
        rng.shuffle(idxs)
        out.extend(idxs[:n_per_func])
    return sorted(out)


# ------------------------------------------------------------------ model loading

def default_embed_fn_for(condition: str, model_seed: int):
    """(embed_fn, fingerprint_or_'base') — C0 the untouched base encoder, any
    other condition the s′ checkpoint of model_seed via the era-tolerant
    SCORING load (its fingerprint is recorded in umap_params.json)."""
    from .encoder import encode_texts, load_encoder

    fp = "base"
    if condition != "C0":
        ck = sprime_run_dir(condition, model_seed) / CHECKPOINT_NAME
        if not ck.exists():
            raise SystemExit(f"missing {ck} — the {condition} panel shows the s′ model of "
                             f"seed {model_seed}: finish section 8 on a trainer VM and pull "
                             f"it here first: "
                             f"hf_main(['pull', '--if-configured', '--include', "
                             f"'artifacts/runs/sprime_C[1-6]_{model_seed}/*'])")
    tok, model, dev = load_encoder(S.MODEL_ID)   # AFTER the gate: fail before the download
    if condition != "C0":
        loaded = load_trained_checkpoint(sprime_run_dir(condition, model_seed) / CHECKPOINT_NAME,
                                         model)
        fp = str(loaded.get("fingerprint"))
    model.eval()

    def embed_fn(texts):
        return encode_texts(model, tok, list(texts), max_len=S.MAX_LEN, device=dev,
                            batch_size=S.EVAL_BATCH, amp_dtype="float32")

    return embed_fn, fp


# ----------------------------------------------------------------------- panels

def panels(conditions, *, model_seed: int = S.UMAP_MODEL_SEED,
           n_per_func: int = S.UMAP_N_PER_FUNCTIONALITY,
           embed_fn_for=None, reducer_factory=None,
           texts: dict[int, str] | None = None, frag_func: dict[int, str] | None = None,
           holdout: list[str] | None = None, out_dir: Path | None = None) -> dict:
    """Build the figure + params. injectable for tests: embed_fn_for(condition)
    -> (fn, fingerprint), reducer_factory() -> fitted-transformer (else umap)."""
    unknown = [c for c in conditions if c not in KNOWN_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown} — one of {list(KNOWN_CONDITIONS)}")
    out_dir = Path(out_dir) if out_dir else Path(S.REPORT_MD).parent / "figures"

    texts = load_sprime_fragments() if texts is None else texts
    frag_func = fragment_functions() if frag_func is None else frag_func
    holdout = load_split()["holdout"] if holdout is None else holdout

    ids = sample_fragment_ids(frag_func, n_per_func)
    missing = [i for i in ids if i not in texts]
    if missing:
        raise SystemExit(f"{len(missing)} sampled fragments are not in sprime_fragments.jsonl "
                         f"(e.g. {missing[:3]}) — stale artifacts, re-run --mine")
    sample_texts = [texts[i] for i in ids]
    holdout_set = {str(f) for f in holdout}

    if reducer_factory is None:
        def reducer_factory():
            import umap
            return umap.UMAP(n_neighbors=S.UMAP_N_NEIGHBORS, min_dist=S.UMAP_MIN_DIST,
                             metric=S.UMAP_METRIC, random_state=S.UMAP_SEED)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = [*plt.get_cmap("tab20").colors, *plt.get_cmap("tab20b").colors]
    funcs = sorted({frag_func[i] for i in ids})
    color_of = {f: palette[k % len(palette)] for k, f in enumerate(funcs)}

    # resolve every model BEFORE any rendering: a missing checkpoint must stop the
    # figure before the first download, not after three models are loaded
    resolved = []
    for cond in conditions:
        fn_fp = embed_fn_for(cond, model_seed) if embed_fn_for \
            else default_embed_fn_for(cond, model_seed)
        resolved.append((cond, fn_fp[0], fn_fp[1]))
    checkpoints: dict[str, str] = {}
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.ravel()
    for k, (cond, embed_fn, fp) in enumerate(resolved):
        X = embed_fn(sample_texts)
        pos = reducer_factory().fit_transform(X)
        ax = axes[k]
        for fid in funcs:
            pts = [j for j, i in enumerate(ids) if frag_func[i] == fid]
            held = fid in holdout_set
            ax.scatter(pos[pts, 0], pos[pts, 1], s=26 if not held else 60,
                       c=[color_of[fid]], marker="*" if held else "o",
                       edgecolors="black" if held else "none", linewidths=0.4,
                       label=fid, zorder=3 if not held else 4)
        checkpoints[cond] = fp
        title = f"{cond} — {'untuned base' if cond == 'C0' else f's′ seed {model_seed}'}"
        ax.set_title(f"{title}\n(ckpt {fp})" if cond != "C0" else title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])

    legend_ax = axes[len(conditions)]
    handles = [Line2D([], [], linestyle="none", marker="*" if f in holdout_set else "o",
                      color=color_of[f], markeredgecolor="black",
                      markersize=9 if f in holdout_set else 6,
                      label=f + (" (held out)" if f in holdout_set else ""))
               for f in funcs]
    legend_ax.legend(handles=handles, loc="center left", fontsize=8, frameon=False)
    legend_ax.axis("off")
    for ax in axes[len(conditions) + 1:]:
        ax.axis("off")
    fig.suptitle("s′ fragment manifold per negative-selection strategy — same sample, "
                 "fixed UMAP params (SCOPE P2-18)", fontsize=13)
    fig.text(0.5, 0.015, LEGEND_NOTE.format(holdout=", ".join(sorted(holdout_set))),
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    png, params_path = out_dir / "umap_panels.png", out_dir / "umap_params.json"
    fig.savefig(png, dpi=150)
    params = {
        "version": S.VERSION,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": list(conditions),
        "model_seed": int(model_seed),
        "seed": int(S.UMAP_SEED),
        "n_neighbors": int(S.UMAP_N_NEIGHBORS),
        "min_dist": float(S.UMAP_MIN_DIST),
        "metric": str(S.UMAP_METRIC),
        "n_per_functionality": int(n_per_func),
        "fragment_ids": ids,
        "holdout": sorted(holdout_set),
        "checkpoints": checkpoints,
        "note": LEGEND_NOTE.format(holdout=", ".join(sorted(holdout_set))),
    }
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"figure written: {png} ({len(ids)} fragments x {len(conditions)} panels); "
          f"params: {params_path}")
    print("it is a figure, not evidence — presented after the Δ table, no claim rests on it")
    return params


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", required=True,
                    help=f"comma-separated conditions, e.g. {','.join(KNOWN_CONDITIONS)}")
    ap.add_argument("--model-seed", type=int, default=S.UMAP_MODEL_SEED,
                    help="which s′ seed's checkpoint each non-C0 panel shows")
    ap.add_argument("--n-per-functionality", type=int,
                    default=S.UMAP_N_PER_FUNCTIONALITY,
                    help="fragments per functionality in the shared sample")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    conditions = [c.strip() for c in a.panels.split(",") if c.strip()]
    out = Path(a.out_dir) if a.out_dir else None
    panels(conditions, model_seed=a.model_seed, n_per_func=a.n_per_functionality,
           out_dir=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
