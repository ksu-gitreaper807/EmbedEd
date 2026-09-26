"""Gate G1 diagnostics — what the hardness numbers MEAN in this embedding space.

    python -m scripts.hardness_diagnostics [--examples 2]

Computes, it does not judge: the verdict stays with `embeded.hardcheck` and
`settings.HARDNESS_MARGIN`. Run it when the gate fails (or passes narrowly)
before touching either the mining or the margin.

Why it exists (2026-09-26, T4 run): mean cos(anchor, negative) came out
C1 0.961 / C2 0.975 / C3 0.988 — the whole ruler is 0.027 wide because the
untuned encoder is anisotropic (a random corpus fragment already scores 0.96).
An absolute margin cannot be read without that scale, so this script reports:

- the scale: mean cos of random corpus pairs, and cos(anchor, positive);
- per condition: mean/sd, standardised gap d vs C1, position in the C1->C3
  range, the PERCENTILE of each negative in its anchor's similarity
  distribution over the corpus (50% = random, 99.7% = top-20 of 6,451), the
  share of negatives inside the anchor's top-20 / top-100, and the share of
  triples whose negative is closer than the labelled positive;
- overlap between the conditions' negative sets per anchor;
- a few anchor / positive / negative texts to eyeball.

Writes ARTIFACTS/hardness_diagnostics.json and a '## Gate G1 — diagnostics'
section in report/measurements.md (data only, no verdict).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

from embeded import settings as S

CONDS = ("C1", "C2", "C3")


def _load_triples(path) -> dict[int, dict]:
    """anchor -> {"positive": p, "negatives": [n, ...]} preserving file order."""
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            a = out.setdefault(r["anchor"], {"positive": r["positive"], "negatives": []})
            a["negatives"].append(r["negative"])
    return out


def _cos(emb, row_of, i, j) -> float:
    return float(emb[row_of[i]] @ emb[row_of[j]])


def analyse(emb: np.ndarray, row_of: dict[int, int], corpus: list[int],
            triples: dict[str, dict[int, dict]], seed: int = S.SEED,
            top_ranks=(20, 100)) -> dict:
    rng = np.random.default_rng(seed)
    corpus_rows = np.array([row_of[c] for c in corpus])
    row_to_corpus_pos = {int(r): k for k, r in enumerate(corpus_rows)}

    # --- scale of the space -------------------------------------------------
    n_pairs = min(20_000, len(corpus) * (len(corpus) - 1) // 2)
    i = rng.integers(0, len(corpus), size=n_pairs)
    j = rng.integers(0, len(corpus), size=n_pairs)
    keep = i != j
    floor = emb[corpus_rows[i[keep]]] * emb[corpus_rows[j[keep]]]
    random_pair_cos = float(floor.sum(1).mean())

    anchors = list(triples["C1"])
    pos_cos = np.array([_cos(emb, row_of, a, triples["C1"][a]["positive"]) for a in anchors])

    # --- per-anchor similarity distribution over the corpus ------------------
    per = {c: defaultdict(list) for c in CONDS}
    for idx, a in enumerate(anchors):
        sims = emb[corpus_rows] @ emb[row_of[a]]
        if row_of[a] in row_to_corpus_pos:            # never rank the anchor against itself
            sims = np.delete(sims, row_to_corpus_pos[row_of[a]])
        srt = np.sort(sims)
        n = len(srt)
        ap = float(pos_cos[idx])
        for c in CONDS:
            negs = triples[c][a]["negatives"]
            nc = np.array([float(emb[row_of[a]] @ emb[row_of[x]]) for x in negs])
            below = np.searchsorted(srt, nc, side="left")             # corpus frags strictly less similar
            per[c]["cos"].extend(nc.tolist())
            per[c]["pct"].extend((below / max(n, 1)).tolist())
            rank = n - below                                          # 1 = the nearest corpus fragment
            for t in top_ranks:
                per[c][f"top{t}"].extend((rank <= t).tolist())
            per[c]["closer_than_pos"].extend((nc > ap).tolist())

    out = {"n_anchors": len(anchors), "k": len(triples["C1"][anchors[0]]["negatives"]),
           "corpus_size": len(corpus),
           "scale": {"random_pair_cos": round(random_pair_cos, 5),
                     "anchor_positive_cos_mean": round(float(pos_cos.mean()), 5),
                     "anchor_positive_cos_sd": round(float(pos_cos.std()), 5)},
           "conditions": {}}
    means = {c: float(np.mean(per[c]["cos"])) for c in CONDS}
    sds = {c: float(np.std(per[c]["cos"])) for c in CONDS}
    span = means["C3"] - means["C1"]
    for c in CONDS:
        pooled = float(np.sqrt((sds[c] ** 2 + sds["C1"] ** 2) / 2)) or 1e-9
        d = {"mean_cos": round(means[c], 5), "sd": round(sds[c], 5),
             "gap_vs_C1": round(means[c] - means["C1"], 5),
             "d_vs_C1": round((means[c] - means["C1"]) / pooled, 3),
             "position_in_C1_C3_range": round((means[c] - means["C1"]) / span, 3) if span > 0 else None,
             "percentile_mean": round(100 * float(np.mean(per[c]["pct"])), 2),
             "percentile_median": round(100 * float(np.median(per[c]["pct"])), 2),
             "closer_than_positive_frac": round(float(np.mean(per[c]["closer_than_pos"])), 4),
             "n": len(per[c]["cos"])}
        for t in top_ranks:
            d[f"in_anchor_top{t}_frac"] = round(float(np.mean(per[c][f"top{t}"])), 4)
        out["conditions"][c] = d

    # --- overlap of negative sets per anchor ----------------------------------
    ov = {}
    for x, y in (("C1", "C2"), ("C1", "C3"), ("C2", "C3")):
        shares = [len(set(triples[x][a]["negatives"]) & set(triples[y][a]["negatives"]))
                  / max(len(triples[x][a]["negatives"]), 1) for a in anchors]
        ov[f"{x}&{y}"] = round(float(np.mean(shares)), 4)
    out["negative_set_overlap_frac"] = ov
    return out


def _snippet(text: str, n: int = 160) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def examples(emb, row_of, frags, triples, n: int) -> list[str]:
    lines = []
    for a in list(triples["C1"])[:n]:
        p = triples["C1"][a]["positive"]
        lines.append(f"anchor {a}: {_snippet(frags[a])}")
        lines.append(f"  positive {p} (cos {_cos(emb, row_of, a, p):.3f}): {_snippet(frags[p])}")
        for c in CONDS:
            x = triples[c][a]["negatives"][0]
            lines.append(f"  {c} neg#1 {x} (cos {_cos(emb, row_of, a, x):.3f}): {_snippet(frags[x])}")
    return lines


def write_section(res: dict) -> None:
    sc, cd = res["scale"], res["conditions"]
    lines = ["## Gate G1 — diagnostics (scale of the space; no verdict)", "",
             f"`python -m scripts.hardness_diagnostics` — version `{S.VERSION}`, {res['n_anchors']} anchors × k={res['k']}, "
             f"corpus {res['corpus_size']}. Percentile = share of corpus fragments less similar to the anchor than the "
             "negative (50 = random, 100 = nearest).", "",
             f"Scale: random corpus pair cos = **{sc['random_pair_cos']:.4f}**; cos(anchor, positive) = "
             f"**{sc['anchor_positive_cos_mean']:.4f}** (sd {sc['anchor_positive_cos_sd']:.3f}).", "",
             "| cond | mean cos | sd | gap vs C1 | d vs C1 | position C1→C3 | pct mean | pct median | in top-20 | in top-100 | closer than positive |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONDS:
        m = cd[c]
        pos = "—" if m["position_in_C1_C3_range"] is None else f"{m['position_in_C1_C3_range']:.2f}"
        lines.append(f"| {c} | {m['mean_cos']:.4f} | {m['sd']:.3f} | {m['gap_vs_C1']:+.4f} | {m['d_vs_C1']:.2f} | {pos} | "
                     f"{m['percentile_mean']:.1f} | {m['percentile_median']:.1f} | {m['in_anchor_top20_frac']:.1%} | "
                     f"{m['in_anchor_top100_frac']:.1%} | {m['closer_than_positive_frac']:.1%} |")
    ov = res["negative_set_overlap_frac"]
    lines += ["", "Negative-set overlap per anchor: " + ", ".join(f"{k} = {v:.1%}" for k, v in ov.items()), ""]
    block = "\n".join(lines)
    S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    if S.REPORT_MD.exists():
        old = S.REPORT_MD.read_text()
        if "## Gate G1 — diagnostics" in old:
            old = re.sub(r"## Gate G1 — diagnostics.*?(?=\n## |\Z)", lambda _m: block + "\n", old, flags=re.S)
        else:
            old = old.rstrip("\n") + "\n\n" + block + "\n"
        S.REPORT_MD.write_text(old)
    else:
        S.REPORT_MD.write_text("# Measurements\n\n" + block + "\n")
    print(f"wrote {S.REPORT_MD}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=int, default=2, help="anchors to print with texts (0 = none)")
    a = ap.parse_args(argv)

    from embeded.data.prepare_data import corpus_ids, load_fragments
    from embeded.mining.semantic_index import load_corpus_emb

    emb, meta = load_corpus_emb()
    frags = load_fragments()
    row_of = {c: i for i, c in enumerate(sorted(frags))}
    triples = {}
    for c in CONDS:
        p = S.artifact(f"triples_{c}.jsonl")
        if not p.exists():
            raise SystemExit(f"missing {p} — run `python -m embeded.negatives` first")
        triples[c] = _load_triples(p)
    if not (set(triples["C1"]) == set(triples["C2"]) == set(triples["C3"])):
        raise SystemExit("anchor sets differ between conditions — re-mine all three strategies in one run")

    print(f"embeddings: {meta.get('model')} max_len={meta.get('max_len')} n={meta.get('n')}")
    res = analyse(emb, row_of, corpus_ids(), triples)
    print(json.dumps(res, indent=2))
    if a.examples:
        print("\nEXAMPLES (first negative per condition):")
        print("\n".join(examples(emb, row_of, frags, triples, a.examples)))
    S.artifact("hardness_diagnostics.json").write_text(json.dumps(res, indent=2))
    write_section(res)


if __name__ == "__main__":
    main()
