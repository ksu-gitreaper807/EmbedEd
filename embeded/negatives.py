"""One interface for all three negative strategies (FINAL_SPEC §7.1).

    python -m embeded.negatives --strategies random,bm25,semantic [--k 20]

Same k, same exclusion pass, same self-exclusion, same anchor/positive stream
for every condition — the ONLY thing that differs is how candidates are ranked
(SCOPE P1-1). The ranked indices (BM25/semantic) cover ALL fragments so that
anchors need not be in the mining corpus; candidates are then filtered to the
corpus (train minus test — §4.1 Rule 1). If a ranked strategy's overfetch
still cannot supply k clean negatives, the remainder is topped up from the
corpus tail (pad count reported: a heavily-padded C3 is C1 in disguise, and
the hardness check would catch it too).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from . import settings as S
from .data.prepare_data import corpus_ids, labeled_clones, load_fragments, positives
from .mining.exclude import exclude_labeled_clones

OVERFETCH = 4  # ranked candidates fetched before filtering = OVERFETCH * k
COND = {"random": "C1", "bm25": "C2", "semantic": "C3"}


def clean_candidates(anchor, ranked, clone_sets, corpus_set, corpus_list,
                     k, stats):
    """Order-preserving: labelled-clone + self exclusion (§7.2), then corpus
    filter (Rule 1), then exclusion-clean deterministic top-up to k."""
    ok = [c for c in exclude_labeled_clones(anchor, ranked, clone_sets) if c in corpus_set]
    if len(ok) < k:
        banned = clone_sets.get(anchor, frozenset()) | {anchor} | set(ok)
        for c in corpus_list:
            if len(ok) >= k:
                break
            if c not in banned:
                ok.append(c)
                stats["padded"] += 1
    return ok[:k]


def anchor_stream(k, cap=None):
    """Shared (anchor, positive) stream: first positive per anchor in the seeded
    shuffled order — identical for every condition (SCOPE P1-3)."""
    clones = labeled_clones()
    seen, stream = set(), []
    for anchor, p in positives():
        if anchor in clones and anchor not in seen:
            seen.add(anchor)
            stream.append((anchor, p))
    limit = cap or max(S.TRAIN_PAIRS_CAP // k, 1)
    return stream[:limit]


def mine(strategy, *, stream, corpus_list, clone_sets, k, all_ids,
         bm25=None, semantic=None, seed=None):
    rng = random.Random(seed if seed is not None else S.SEED)
    corpus_set = set(corpus_list)
    depth = k * OVERFETCH
    stats = defaultdict(int)
    triples = []
    for anchor, positive in stream:
        if strategy == "random":
            ranked = rng.sample(all_ids, min(depth + 10, len(all_ids)))
        elif strategy == "bm25":
            ranked = [i for i, _ in bm25.topk(anchor, depth)]
        elif strategy == "semantic":
            ranked = [i for i, _ in semantic.topk_for_fragment(anchor, depth)]
        else:
            raise ValueError(strategy)
        negs = clean_candidates(anchor, ranked, clone_sets, corpus_set,
                                corpus_list, k, stats)
        stats["anchors_short"] += int(len(negs) < k)
        triples.extend((anchor, positive, n) for n in negs)
    return triples, dict(stats)


def cached_summary(strategies, *, k: int, n_anchors: int) -> dict | None:
    """mining_summary.json iff it was produced by this VERSION with the same k,
    anchor count and (at least) these strategies, and the triples files exist.
    Artifact gating for resume; the whole run is otherwise deterministic."""
    p = S.ARTIFACTS / "mining_summary.json"
    if not p.exists():
        return None
    try:
        summary = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    run = summary.get("_run") or {}
    if not (run.get("version") == S.VERSION and run.get("k") == k and run.get("n_anchors") == n_anchors
            and set(strategies) <= set(run.get("strategies", []))):
        return None
    if not all((S.ARTIFACTS / f"triples_{COND[s]}.jsonl").exists() for s in strategies):
        return None
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="random,bm25,semantic")
    ap.add_argument("--k", type=int, default=S.K_NEGATIVES)
    ap.add_argument("--cap-anchors", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-mine even if matching triples for this VERSION are already present")
    a = ap.parse_args(argv)
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]

    frags = load_fragments()
    all_ids = sorted(frags)
    clone_sets = labeled_clones()
    corpus_list = corpus_ids()
    stream = anchor_stream(a.k, a.cap_anchors)
    print(f"frags={len(all_ids)} corpus={len(corpus_list)} anchors={len(stream)} "
          f"k={a.k} -> <= {len(stream) * a.k} triples per condition")
    if not a.force:
        cached = cached_summary(strategies, k=a.k, n_anchors=len(stream))
        if cached is not None:
            for st in strategies:
                print(COND[st], cached[COND[st]])
            print(f"[cache] triples for version {S.VERSION} (k={a.k}, anchors={len(stream)}) already mined "
                  "— skipping (--force to redo)")
            return

    bm25 = semantic = None
    if "bm25" in strategies:
        from .mining.bm25_index import BM25Index
        bm25 = BM25Index([frags[i] for i in all_ids])
    if "semantic" in strategies:
        from .mining.semantic_index import SemanticIndex, load_corpus_emb
        emb, meta = load_corpus_emb()
        semantic = SemanticIndex(emb, all_ids)
        print("semantic index:", meta["model"], f"max_len={meta['max_len']}",
              f"n={meta['n']}")

    summary = {}
    for st in strategies:
        triples, stats = mine(st, stream=stream, corpus_list=corpus_list,
                              clone_sets=clone_sets, k=a.k, all_ids=all_ids,
                              bm25=bm25, semantic=semantic)
        out = S.artifact(f"triples_{COND[st]}.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for an, po, ne in triples:
                fh.write(json.dumps({"anchor": an, "positive": po,
                                     "negative": ne}) + "\n")
        path = str(out.relative_to(S.ROOT)) if str(out).startswith(str(S.ROOT)) else str(out)
        summary[COND[st]] = {"strategy": st, "triples": len(triples),
                             "stats": stats, "file": path}
        print(COND[st], summary[COND[st]])
    summary["_run"] = {"version": S.VERSION, "k": a.k, "n_anchors": len(stream),
                       "train_pairs_cap": S.TRAIN_PAIRS_CAP, "seed": S.SEED,
                       "strategies": strategies}
    (S.ARTIFACTS / "mining_summary.json").write_text(json.dumps(summary, indent=2))
    print("next: pytest -q embeded/tests   then   python -m embeded.hardcheck (needs emb cache)")


if __name__ == "__main__":
    main()
