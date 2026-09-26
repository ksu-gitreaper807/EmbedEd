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


def jaccard(a_tokens: set, b_tokens: set) -> float:
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens) if (a_tokens or b_tokens) else 0.0


def label_noise(triples: dict[str, dict[int, dict]], frags: dict[int, str],
                clones_train: dict[int, set], clones_valid: dict[int, set],
                thresholds=(0.5, 0.75)) -> dict:
    """Two FLOORS on 'negatives that are really clones' (correction 5 — the
    exclusion only removes clones LABELLED in train, and harder mining finds
    more of the unlabelled ones):

    - `valid_labelled_clone_frac`: negatives that the VALID split labels as a
      clone of the anchor (exact, but only a subset of pairs is labelled at all);
    - `near_dup_frac_ge_<t>`: negatives whose BM25 token set has Jaccard >= t
      with the anchor's — near-verbatim Type-1/2 code under a different name.
    `train_labelled_clone_frac` must be 0 (the exclusion working) — reported as
    a self-check. The 50+50 manual audit remains the real measurement.
    """
    from embeded.mining.bm25_index import tokenize_code
    tok_cache: dict[int, set] = {}

    def toks(i):
        if i not in tok_cache:
            tok_cache[i] = set(tokenize_code(frags[i]))
        return tok_cache[i]

    out = {}
    for c, per_anchor in triples.items():
        n = tr = va = 0
        hits = {t: 0 for t in thresholds}
        jacc = []
        for a, rec in per_anchor.items():
            ta = toks(a)
            for x in rec["negatives"]:
                n += 1
                tr += x in clones_train.get(a, ())
                va += x in clones_valid.get(a, ())
                j = jaccard(ta, toks(x))
                jacc.append(j)
                for t in thresholds:
                    hits[t] += j >= t
        d = {"n": n, "train_labelled_clone_frac": round(tr / n, 4),
             "valid_labelled_clone_frac": round(va / n, 4),
             "jaccard_mean": round(float(np.mean(jacc)), 4),
             "jaccard_median": round(float(np.median(jacc)), 4)}
        for t in thresholds:
            d[f"near_dup_frac_ge_{t}"] = round(hits[t] / n, 4)
        out[c] = d
    # the same yardstick on the labelled positives, for reference
    pj = [jaccard(toks(a), toks(rec["positive"])) for a, rec in triples["C1"].items()]
    out["positives"] = {"jaccard_mean": round(float(np.mean(pj)), 4),
                        "jaccard_median": round(float(np.median(pj)), 4),
                        **{f"near_dup_frac_ge_{t}": round(float(np.mean([j >= t for j in pj])), 4)
                           for t in thresholds}}
    return out


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
    from embeded.mining.bm25_index import tokenize_code
    lines = []
    for a in list(triples["C1"])[:n]:
        ta = set(tokenize_code(frags[a]))
        p = triples["C1"][a]["positive"]
        lines.append(f"anchor {a}: {_snippet(frags[a])}")
        lines.append(f"  positive {p} (cos {_cos(emb, row_of, a, p):.3f}, jaccard "
                     f"{jaccard(ta, set(tokenize_code(frags[p]))):.2f}): {_snippet(frags[p])}")
        for c in CONDS:
            x = triples[c][a]["negatives"][0]
            lines.append(f"  {c} neg#1 {x} (cos {_cos(emb, row_of, a, x):.3f}, jaccard "
                         f"{jaccard(ta, set(tokenize_code(frags[x]))):.2f}): {_snippet(frags[x])}")
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
    ln = res.get("label_noise")
    if ln:
        lines += ["Label-noise floors (correction 5; proxies, not the audit): share of negatives that the valid split "
                  "labels as a clone of the anchor, and share whose BM25 token set is near-identical to the anchor's "
                  "(Jaccard). Positives shown with the same yardstick.", "",
                  "| set | train-labelled clone (must be 0) | valid-labelled clone | Jaccard median | ≥ 0.5 | ≥ 0.75 |",
                  "|---|---|---|---|---|---|"]
        for c in CONDS:
            m = ln[c]
            lines.append(f"| {c} negatives | {m['train_labelled_clone_frac']:.2%} | {m['valid_labelled_clone_frac']:.2%} | "
                         f"{m['jaccard_median']:.2f} | {m['near_dup_frac_ge_0.5']:.1%} | {m['near_dup_frac_ge_0.75']:.1%} |")
        pm = ln["positives"]
        lines.append(f"| labelled positives | — | — | {pm['jaccard_median']:.2f} | {pm['near_dup_frac_ge_0.5']:.1%} | "
                     f"{pm['near_dup_frac_ge_0.75']:.1%} |")
        lines.append("")
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
    from embeded.data.prepare_data import labeled_clones, load_pairs
    clones_valid = defaultdict(set)
    for i, j, lab in load_pairs("valid"):
        if lab == 1:
            clones_valid[i].add(j)
            clones_valid[j].add(i)
    res["label_noise"] = label_noise(triples, frags, labeled_clones(), clones_valid)
    print(json.dumps(res, indent=2))
    if a.examples:
        print("\nEXAMPLES (first negative per condition):")
        print("\n".join(examples(emb, row_of, frags, triples, a.examples)))
    S.artifact("hardness_diagnostics.json").write_text(json.dumps(res, indent=2))
    write_section(res)


if __name__ == "__main__":
    main()
