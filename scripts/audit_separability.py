"""Is the audited contamination separable? (decision D2 input)

    python -m scripts.audit_separability
    python -m scripts.audit_separability --sweep 0.6 0.5 0.4   # override all three metric sweeps

The blind audit found that a large share of mined "negatives" are functionally
equivalent to their anchor (C2 21/50, C3 15/50 — see report/measurements.md).
D2's pre-registered remedy is to extend the exclusion to *near-verbatim*
candidates, which only helps if those contaminated pairs are textually close to
their anchor. This measures that directly, on the same 100 pairs that were
labelled, and reports the trade-off: at each similarity threshold, how much of
the condition's negatives you would throw away to remove how much contamination.

Prints aggregate statistics only — never a fragment's text, never the key's rows.
Needs `audit_key.csv`, which lives only on the machine that scored the audit.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from statistics import median

from embeded import settings as S

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in WORD.findall(text)}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _ratio(a: str, b: str, cap: int = 2000) -> float:
    """difflib similarity on a prefix — quadratic, and 100 pairs is small."""
    return difflib.SequenceMatcher(None, a[:cap], b[:cap]).ratio()


def pair_features(anchor: str, negative: str, cosine: float | None) -> dict:
    return {"cosine": cosine, "jaccard": _jaccard(anchor, negative),
            "ratio": _ratio(anchor, negative),
            "len_ratio": min(len(anchor), len(negative)) / max(1, max(len(anchor), len(negative)))}


def load() -> list[dict]:
    """One row per audited pair: condition, label, and its similarity features."""
    from embeded.data.prepare_data import load_fragments

    audit = S.ARTIFACTS / "audit"
    for name in ("audit_key.csv", "audit_labels.csv"):
        if not (audit / name).exists():
            raise SystemExit(f"missing {audit / name} — this runs where the audit was scored")
    with open(audit / "audit_key.csv", encoding="utf-8") as fh:
        key = {r["id"]: r for r in csv.DictReader(fh)}
    with open(audit / "audit_labels.csv", encoding="utf-8") as fh:
        labels = {r["id"]: (r.get("label") or "").strip().lower() for r in csv.DictReader(fh)}

    frags = load_fragments()
    emb = pos = None
    if (S.ARTIFACTS / "corpus_emb.npy").exists():
        import numpy as np
        emb = np.load(S.ARTIFACTS / "corpus_emb.npy")
        pos = {int(c): i for i, c in enumerate(sorted(frags))}
        if emb.shape[0] != len(frags):
            raise SystemExit(f"corpus_emb.npy has {emb.shape[0]} rows but there are {len(frags)} "
                             "fragments — embeddings and fragments are from different runs")

    out = []
    for pid, k in key.items():
        a, n = int(k["anchor"]), int(k["negative"])
        cos = float(emb[pos[a]] @ emb[pos[n]]) if emb is not None else None
        out.append({"id": pid, "condition": k["condition"], "label": labels.get(pid, ""),
                    **pair_features(frags[a], frags[n], cos)})
    return out


def _quartiles(values: list[float]) -> str:
    v = sorted(values)
    if not v:
        return "n=0"
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]          # noqa: E731
    return f"n={len(v)} p25={q(0.25):.3f} med={median(v):.3f} p75={q(0.75):.3f} max={v[-1]:.3f}"


# Per metric, because the scales are not comparable: cosine of two GraphCodeBERT
# embeddings lives in 0.96-1.00, while token Jaccard between a clone pair sits
# around 0.35-0.45. One shared list of thresholds measures nothing.
DEFAULT_SWEEP = {"cosine": [0.995, 0.99, 0.985, 0.98],
                 "ratio": [0.6, 0.5, 0.4, 0.3],
                 "jaccard": [0.6, 0.5, 0.4, 0.35, 0.3]}


def report(rows: list[dict], sweep: list[float] | None = None) -> None:
    conds = sorted({r["condition"] for r in rows})
    print(f"audited pairs: {len(rows)}  (version {S.VERSION})\n")
    print("== how similar is a 'negative' to its anchor? ==")
    for cond in conds:
        for label in ("clone", "not_clone", "unsure"):
            sub = [r for r in rows if r["condition"] == cond and r["label"] == label]
            if not sub:
                continue
            print(f"  {cond} {label:<10} cosine  {_quartiles([r['cosine'] for r in sub if r['cosine'] is not None])}")
            print(f"  {'':<3} {'':<10} jaccard {_quartiles([r['jaccard'] for r in sub])}")
            print(f"  {'':<3} {'':<10} ratio   {_quartiles([r['ratio'] for r in sub])}")

    print("\n== cleanest cut per metric: the threshold that drops no clean pair ==")
    print("  (in-sample: the maximum of ~30 not_clone values has wide sampling error, so this")
    print("   identifies a candidate threshold, it does not validate one)\n")
    for metric in ("cosine", "ratio", "jaccard"):
        if metric == "cosine" and rows[0]["cosine"] is None:
            continue
        for cond in conds:
            sub = [r for r in rows if r["condition"] == cond]
            clean = [r[metric] for r in sub if r["label"] == "not_clone" and r[metric] is not None]
            dirty = [r[metric] for r in sub if r["label"] == "clone" and r[metric] is not None]
            if not clean or not dirty:
                continue
            top = max(clean)
            caught = sum(1 for v in dirty if v > top)
            print(f"  {cond} {metric:<8} every not_clone <= {top:.3f}; a cut just above it removes "
                  f"{caught}/{len(dirty)} = {caught/len(dirty):.0%} of the contamination and no "
                  f"clean pair (median clone = {median(dirty):.3f})")

    print("\n== what would an exclusion buy? ==")
    print("  drop every pair at or above the threshold; 'removed' counts the labelled clones "
          "it catches, 'residual' is the contamination left behind.\n")
    for metric in ("cosine", "ratio", "jaccard"):
        if metric == "cosine" and rows[0]["cosine"] is None:
            continue
        print(f"  --- filter on {metric} ---")
        print("  | threshold | pairs dropped | of the condition | contamination removed | residual FN rate |")
        print("  |---|---|---|---|---|")
        for t in (sweep or DEFAULT_SWEEP[metric]):
            for cond in conds:
                sub = [r for r in rows if r["condition"] == cond]
                drop = [r for r in sub if (r[metric] or 0) >= t]
                keep = [r for r in sub if (r[metric] or 0) < t]
                clones_drop = sum(1 for r in drop if r["label"] == "clone")
                clones_all = sum(1 for r in sub if r["label"] == "clone")
                resid = (sum(1 for r in keep if r["label"] == "clone") / len(keep)) if keep else 0.0
                print(f"  | {metric} \u2265 {t:.2f} ({cond}) | {len(drop)}/{len(sub)} "
                      f"= {len(drop)/len(sub):.0%} | {clones_drop}/{clones_all} "
                      f"= {(clones_drop/clones_all if clones_all else 0):.0%} | {resid:.0%} |")
        print()
    print("Read it as: a threshold that drops few pairs while removing most contamination means "
          "D2's near-verbatim exclusion is cheap; one that only removes contamination by dropping "
          "most of the condition means the noise is not textually separable, and the honest move "
          "is to keep the mining and report the rate as a label-noise floor.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=float, nargs="+", default=None,
                    help="override the per-metric threshold lists")
    ap.add_argument("--json", default=None, help="also dump the per-pair features (no text) here")
    a = ap.parse_args(argv)
    rows = load()
    report(rows, a.sweep)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"[written] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
