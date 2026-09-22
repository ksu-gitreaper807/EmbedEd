"""Gate G1b — the hardness check (FINAL_SPEC §6.1, correction 4). Run BEFORE
training; if the manipulation did not take effect, the experiment produces no
evidence. Require mean cos(anchor, negative):  C1 < C2 <= C3, C2−C1 >= margin.

    python -m code.hardcheck          # real data (needs code/artifacts/corpus_emb.npy)

evaluate(means, margin) is pure so the gate logic itself is unit-tested.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

from . import settings as S


def evaluate(means: dict[str, float], margin: float) -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    if not all(k in means for k in ("C1", "C2", "C3")):
        return False, [f"missing conditions: have {sorted(means)}, need C1/C2/C3"]
    if means["C2"] - means["C1"] < margin:
        ok = False
        msgs.append(f"C1→C2 gap {means['C2'] - means['C1']:+.4f} < margin {margin} "
                    "(mining produced equally easy negatives)")
    if means["C3"] < means["C2"] - 1e-9:
        ok = False
        msgs.append(f"C3 ({means['C3']:.4f}) harder than C2 ({means['C2']:.4f}) "
                    "violates C2 <= C3 — check semantic exclusion order")
    if not msgs:
        msgs.append(f"order C1 < C2 <= C3 holds with visible first gap "
                    f"({means['C1']:.4f} < {means['C2']:.4f} <= {means['C3']:.4f})")
    return ok, msgs


def measure(triples_files: dict[str, str], emb: np.ndarray, row_of: dict[int, int],
            n_anchors: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for cond, path in triples_files.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        anchors = sorted({r["anchor"] for r in rows})
        take = anchors if len(anchors) <= n_anchors else list(
            rng.choice(anchors, size=n_anchors, replace=False))
        take = set(int(t) for t in take)
        sims = [float(emb[row_of[r["anchor"]]] @ emb[row_of[r["negative"]]])
                for r in rows if r["anchor"] in take and r["negative"] in row_of]
        if not sims:
            raise SystemExit(f"{cond}: no (anchor,negative) pair found in corpus index — stale artifacts?")
        out[cond] = {"mean_cos": round(float(np.mean(sims)), 5),
                     "std": round(float(np.std(sims)), 5),
                     "n_samples": len(sims)}
    return out


def append_measurements(per_cond, ok, msgs) -> None:
    block = ["", "## Gate G1 — hardness check", ""]
    for c in ("C1", "C2", "C3"):
        m = per_cond[c]
        block.append(f"- {c}: mean cos(anchor, negative) = {m['mean_cos']:.5f} "
                     f"(sd {m['std']:.3f}, n = {m['n_samples']})")
    block.append(f"- **verdict: {'PASS' if ok else 'FAIL'}** — " + "; ".join(msgs))
    block.append("")
    if S.REPORT_MD.exists():
        t = S.REPORT_MD.read_text()
        t = re.sub(r"\n## Gate G1 — hardness check.*?(?=\n## |\Z)", "\n" + "\n".join(block), t, flags=re.S)
        S.REPORT_MD.write_text(t if "## Gate G1" in t else t + "\n".join(block))
    else:
        S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
        S.REPORT_MD.write_text("# Measurements\n" + "\n".join(block))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=S.HARDNESS_MARGIN)
    ap.add_argument("--anchors", type=int, default=S.HARDNESS_CHECK_ANCHORS)
    a = ap.parse_args(argv)

    from .data.prepare_data import load_fragments
    from .mining.semantic_index import load_corpus_emb

    emb, meta = load_corpus_emb()
    row_of = {c: i for i, c in enumerate(sorted(load_fragments()))}
    files = {c: str(S.artifact(f"triples_{c}.jsonl")) for c in ("C1", "C2", "C3")}
    for p in files.values():
        try:
            open(p).close()
        except FileNotFoundError:
            raise SystemExit(f"missing {p} — run `python -m code.negatives` first")
    per = measure(files, emb, row_of, a.anchors, S.SEED)
    means = {c: per[c]["mean_cos"] for c in per}
    ok, msgs = evaluate(means, a.margin)
    print(json.dumps(per, indent=2))
    print("HARDNESS GATE:", "PASS" if ok else "FAIL")
    for m in msgs:
        print(" -", m)
    append_measurements(per, ok, msgs)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
