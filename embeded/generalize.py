"""Phase 3 — generalisation on unseen functionality (RQ3), the s′ runner
(PHASE3_PLAN §3, §6 item 1).

    python -m embeded.generalize --census   --holdout-k 3
    python -m embeded.generalize --mine     --holdout-k 3 --k 20 --cap <settings.SPRIME_TRAIN_CAP>
    python -m embeded.generalize --hardness --holdout-k 3          # gate G1s; exit 1 = FAIL
    python -m embeded.generalize --condition C1 --seed 13 --holdout-k 3 [--smoke] [--force]
    python -m embeded.generalize --condition C1 --seed 13 --holdout-k 3 --transfer
    python -m embeded.generalize --condition C0 --seed 13 --holdout-k 3 --transfer   # baseline
    python -m embeded.generalize --results-table

Corpus: Kitsios et al.'s s′ (`EMBEDED_ARTIFACTS/sprime/.../data_bcb_v2_sampled_bf.pickle`,
4,600 pairs over 23 functionality ids, acquired by `scripts.fetch_sprime`). Decision
D3 = (B): fine-tune on s′ with negatives mined from s′ itself, so *negative-selection
strategy* stays the independent variable on functionality-labelled data. Route A
(FINAL_SPEC §9.3): hold out k = SPRIME_HELDOUT_K whole functionalities; the holdout is
the k highest pair counts, ties broken by lowest functionality id (decision D4,
`RULE_NAME` here) — measured by `--census` and frozen before any F1 exists.

Protocol frozen (PHASE3_PLAN §3.2):
* ONE mining and ONE seen/unseen partition shared by every condition and every seed —
  like Phase 2's fixed valid/test splits, the seed never changes the split, only the
  training order/init (the `embeded.train` resume contract);
* negatives are mined inside the held-in functionalities only, so no unseen code
  enters training in any role; the same anchors/positives/k feed C1/C2/C3
  (`embeded.negatives.mine` — one implementation, the SAME one the main table used);
* the threshold comes from the seen-validation slice only (max_f1_on_valid) and is
  applied unchanged to the unseen pairs; F1_seen is the seen-validation F1 (never the
  training pairs), F1_unseen the held-out functionalities, Δ = F1_seen − F1_unseen;
* `--transfer` scores the Phase 2 checkpoints (C0 = base model) on the same partition
  without training — a secondary reading that must never enter the Δ table.

This corpus is NOT the main table's corpus: absolute F1 is not comparable across the
two tables (say it in both captions). Gate G1s re-measures the C1 < C2 <= C3 hardness
on s′ with `embeded.hardcheck.evaluate_gap` — one definition, never assumed to transfer
from CodeXGLUE (PHASE3_PLAN §2 item 4).

Runs land in `artifacts/runs/sprime_<condition>_<seed>/` (transfer readings in
`sprime_transfer_<condition>_<seed>/`, smoke harness-checks in their own
`sprime_smoke_<condition>_<seed>/` so re-running `--smoke` can never clobber a real
run) so they can never collide with Phase 2. torch
is imported lazily, and every encoder entry point is injectable, so the offline test
suite runs without a GPU (`embeded/tests/test_generalize.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import settings as S
from .evaluate import (best_f1_threshold, default_embed_fn, map_at_r,
                       precision_recall_f1, subsample)
from .hardcheck import (bootstrap_d, evaluate_gap, measure as hardness_measure,
                        missing_artifact_message, standardized_gap)
from .train import (CHECKPOINT_NAME, CONFIG_NAME, EVAL_CONFIG_NAME, EVAL_METRICS_NAME,
                    LOG_NAME, METRICS_NAME, PREDICTIONS_NAME, batches, fingerprint,
                    load_checkpoint, run_dir as phase2_run_dir, save_checkpoint, set_seed,
                    triplet_loss, verify_checkpoint)

CONDS = ("C1", "C2", "C3")                 # trained generalisation conditions
ALL_CONDITIONS = ("C0", "C1", "C2", "C3")  # + C0 as the untuned transfer-only baseline
PICKLE_GLOB = "*bcb_v2_sampled_bf*.pickle"
RULE_NAME = "max_pair_count_then_lowest_id"   # decision D4 — the notebook records this string
CENSUS_NAME = "sprime_census.json"
FRAGS_NAME = "sprime_fragments.jsonl"
SPLIT_NAME = "sprime_split.json"
SUMMARY_NAME = "sprime_mining_summary.json"
TRIPLES_NAME = "sprime_triples_{cond}.jsonl"
EMB_NAME = "sprime_emb.npy"
EMB_META_NAME = "sprime_emb.meta.json"
DECISIONS_NAME = "phase3_decisions.json"
GATE_SECTION = "## Gate G1s — hardness on s′"
CENSUS_SECTION = "## s′ census — measured pair counts and the holdout (D4)"
TABLE_SECTION = "## Phase 3 — generalisation (F1_seen / F1_unseen / Δ)"
GATE_LOG_NAME = "sprime_gate_runs.json"
# The seen-validation slice: 10% of the held-in pairs, per class, split with the one
# frozen seed — the same split for every condition and every seed (see module docstring).
VALID_FRAC = 0.10
_CORPUS_NOTE = ("corpus: s′ (Kitsios et al., bcb_v2_sampled_bf, doi:10.5281/zenodo.17238379) — "
                "NOT the Phase 2 corpus; absolute F1 is not comparable across the two tables (D3=B)")


# --------------------------------------------------------------------- plumbing

def sprime_run_dir(condition: str, seed: int, *, smoke: bool = False) -> Path:
    """Prefixed, so a generalisation run can never collide with a Phase 2 main run.
    Smoke runs land in their OWN `sprime_smoke_*` directory: §7 is the one cell a
    resumed VM re-runs after pulling finished runs from the HF checkpoint, and when
    the smoke harness-check shared the real run's directory it clobbered the pulled
    run's metrics → fingerprint mismatch → a spurious 20-minute retrain (incident
    2026-09-28). A smoke run must never be able to touch a real run's files."""
    name = f"sprime_smoke_{condition}_{seed}" if smoke else f"sprime_{condition}_{seed}"
    return S.ARTIFACTS / S.RUNS_SUBDIR / name


def sprime_transfer_dir(condition: str, seed: int) -> Path:
    return S.ARTIFACTS / S.RUNS_SUBDIR / f"sprime_transfer_{condition}_{seed}"


def _require_holdout_k(holdout_k: int) -> None:
    """The flag is a cross-check, not a dial: settings.SPRIME_HELDOUT_K stays the one
    source of truth (a deliberate change is a settings change + VERSION bump)."""
    if holdout_k != S.SPRIME_HELDOUT_K:
        raise SystemExit(
            f"--holdout-k {holdout_k} != settings.SPRIME_HELDOUT_K = {S.SPRIME_HELDOUT_K}. "
            "The holdout size is frozen in settings.py (PHASE3_PLAN §3.2); change it there "
            "deliberately (and bump VERSION) instead of passing a different number here.")


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _replace_or_append_section(report_md: Path, section: str, block: str) -> None:
    """Rebuild THIS section — header plus body — and leave every other one alone.
    `block` is the body WITHOUT the `## …` header; the helper owns it, so a first
    append and a later rebuild produce byte-identical section starts."""
    block = block.strip("\n")
    body = section + "\n\n" + block + "\n"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    if report_md.exists():
        text = report_md.read_text()
        pattern = re.escape(section) + r".*?(?=\n## |\Z)"
        if re.search(pattern, text, flags=re.S):
            text = re.sub(pattern, lambda _m: body, text, flags=re.S)
        else:
            text = text.rstrip("\n") + "\n\n" + body
        report_md.write_text(text)
    else:
        report_md.write_text("# Measurements\n\n" + body)


# --------------------------------------------------------------------- s′ input

def load_sprime_frame() -> tuple["pd.DataFrame", dict]:  # noqa: F821 - pandas lazy
    """The gate-passed pair DataFrame + provenance (path, md5, rows). Never committed;
    `python -m scripts.fetch_sprime` puts it under EMBEDED_ARTIFACTS/sprime/."""
    import pickle
    import warnings

    base = S.artifact("sprime")
    candidates = sorted(base.rglob(PICKLE_GLOB))
    if not candidates:
        raise SystemExit(missing_artifact_message(
            base / PICKLE_GLOB,
            hint="fetch s′ first (needs zenodo.org; the Arena sandbox blocks it):\n"
                 "    python -m scripts.fetch_sprime"))
    path = candidates[0]
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            with open(path, "rb") as fh:
                frame = pickle.load(fh)   # unpickling needs pandas/numpy: fail loudly here
        except Exception as exc:
            raise SystemExit(f"{path}: could not load the s′ pickle ({exc.__class__.__name__}: "
                             f"{exc}) — re-fetch with `python -m scripts.fetch_sprime`") from exc
    cols = {str(c) for c in getattr(frame, "columns", [])}
    if not {"code1", "code2", "label", "functionality_id"} <= cols:
        raise SystemExit(f"{path}: expected columns code1/code2/label/functionality_id, "
                         f"found {sorted(cols)} — the corpus changed; do not continue on it")
    return frame, {"path": path, "md5": h.hexdigest(), "rows": int(len(frame))}


def _label_series(frame) -> np.ndarray:
    return np.asarray(frame["label"].astype(int))


def pair_frag_rows_of(frame, texts: dict[str, int] | None = None) -> list[tuple[int, int, int]]:
    """(frag_a, frag_b, label) per s′ row, ids from the canonical fragment table — the
    ONE text→id mapping, so mining, runs and transfer can never disagree on ids."""
    if texts is None:
        _, texts = build_fragments(frame)
    return [(texts[str(c1)], texts[str(c2)], int(lab))
            for c1, c2, lab in zip(frame["code1"], frame["code2"], _label_series(frame))]


# --------------------------------------------------------------------- census (D4)

def build_census(frame, source: dict, holdout_k: int) -> dict:
    """Per-functionality pair counts + the D4 holdout: k highest counts, ties by lowest
    functionality id. Measured, committed, BEFORE any F1 exists (PHASE3_PLAN §3.2)."""
    counts: dict[str, int] = defaultdict(int)
    clone: dict[str, int] = defaultdict(int)
    for fid, lab in zip(frame["functionality_id"], _label_series(frame)):
        fid = str(fid)
        counts[fid] += 1
        clone[fid] += int(lab == 1)
    ranked = sorted(counts, key=lambda f: (-counts[f], f))
    return {
        "rule": RULE_NAME,
        "k": holdout_k,
        "counts": dict(sorted(counts.items())),
        "clone_counts": dict(sorted(clone.items())),
        "holdout": ranked[:holdout_k],
        "n_pairs": int(len(frame)),
        "label_counts": {str(v): int(c) for v, c in
                         zip(*np.unique(_label_series(frame), return_counts=True))},
        "n_functionalities": len(counts),
        "version": S.VERSION,
        "source_md5": source["md5"],
        "source_rows": source["rows"],
        "generated_utc": _stamp(),
    }


def census_main(holdout_k: int) -> int:
    _require_holdout_k(holdout_k)
    frame, source = load_sprime_frame()
    census = build_census(frame, source, holdout_k)
    S.artifact(CENSUS_NAME).write_text(json.dumps(census, indent=2, ensure_ascii=False))

    lines = [f"Measured from `{source['path'].name}` (md5 `{source['md5'][:12]}…`, "
             f"{source['rows']} rows) by `python -m embeded.generalize --census "
             f"--holdout-k {holdout_k}` — version `{S.VERSION}`, "
             f"recorded before any F1 exists (PHASE3_PLAN §3.2, decision D4).", "",
             "| functionality | pairs | clone | non-clone |", "|---|---|---|---|"]
    for fid in sorted(census["counts"], key=lambda f: (-census["counts"][f], f)):
        c, cl = census["counts"][fid], census["clone_counts"].get(fid, 0)
        lines.append(f"| {fid} | {c} | {cl} | {c - cl} |")
    hold = ", ".join(f"`{f}`" for f in census["holdout"])
    lines += ["", f"**Holdout (k = {holdout_k}, rule `{RULE_NAME}`): {hold}** — the same three "
               "functionalities are held out for every condition and every seed.", ""]
    _replace_or_append_section(S.REPORT_MD, CENSUS_SECTION, "\n".join(lines))

    print(f"s′: {census['n_pairs']} pairs, {census['n_functionalities']} functionalities, "
          f"labels {census['label_counts']}")
    print("per-functionality pair counts (top 8):")
    for fid in list(sorted(census["counts"], key=lambda f: (-census["counts"][f], f)))[:8]:
        print(f"  {fid:>8}  {census['counts'][fid]}")
    print(f"holdout = {census['holdout']}  (rule {RULE_NAME!r}, ties by functionality id)")
    print(f"wrote {S.artifact(CENSUS_NAME).name} and '{CENSUS_SECTION}' in {S.REPORT_MD}")
    return 0


# --------------------------------------------------------------------- fragments + split

def build_fragments(frame) -> tuple[dict[int, str], dict[str, int]]:
    """Canonical s′ fragment table: unique texts of code1/code2, keyed 0..N-1 by first
    appearance, deduplicated by text — the same rule prepare_data applies to CodeXGLUE."""
    texts: dict[str, int] = {}
    frags: dict[int, str] = {}
    for col in ("code1", "code2"):
        for t in frame[col]:
            t = str(t)
            if t not in texts:
                texts[t] = len(frags)
                frags[len(frags)] = t
    return frags, texts


def write_fragments(frags: dict[int, str]) -> None:
    from .data.prepare_data import sha1

    with open(S.artifact(FRAGS_NAME), "w", encoding="utf-8") as fh:
        for i in sorted(frags):
            fh.write(json.dumps({"idx": i, "sha1": sha1(frags[i]), "text": frags[i]},
                                ensure_ascii=False) + "\n")


def load_sprime_fragments() -> dict[int, str]:
    p = S.artifact(FRAGS_NAME)
    if not p.exists():
        raise SystemExit(missing_artifact_message(
            p, hint="mine first: python -m embeded.generalize --census --holdout-k "
                    f"{S.SPRIME_HELDOUT_K}   then   --mine ..."))
    out = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[int(r["idx"])] = r["text"]
    return out


def build_split(frame, texts: dict[str, int], holdout: list[str]) -> dict:
    """The ONE seen/unseen partition (see module docstring): unseen = every pair whose
    functionality is held out; the held-in pairs split per class into a seen-validation
    slice and the training pairs, with the frozen seed. Row indices are positions in the
    pickle (its md5 is recorded alongside), so the split is reproducible from the file."""
    labels = _label_series(frame)
    fids = [str(f) for f in frame["functionality_id"]]
    holdout_set = {str(f) for f in holdout}
    unseen, held_in = [], []
    for i, f in enumerate(fids):
        (unseen if f in holdout_set else held_in).append(i)
    rng = random.Random(f"sprime-split#{S.SEED}")
    valid, train = [], []
    for lab in (0, 1):
        rows = [i for i in held_in if labels[i] == lab]
        rng.shuffle(rows)
        n_valid = min(len(rows), int(np.ceil(VALID_FRAC * len(rows))))
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    return {
        "version": S.VERSION,
        "seed": S.SEED,
        "valid_frac": VALID_FRAC,
        "holdout": sorted(holdout_set),
        "rows": {"train": sorted(train), "valid": sorted(valid), "unseen": sorted(unseen)},
        "counts": {"train": len(train), "valid": len(valid), "unseen": len(unseen)},
        "per_class": {
            "valid": {str(l): int(np.sum(labels[valid] == l)) for l in (0, 1)},
            "unseen": {str(l): int(np.sum(labels[unseen] == l)) for l in (0, 1)},
        },
        "source_md5": None,   # filled by the caller (mining) from the provenance dict
    }


def load_split() -> dict:
    p = S.artifact(SPLIT_NAME)
    if not p.exists():
        raise SystemExit(missing_artifact_message(
            p, hint="mine first (it writes the seen/unseen partition): python -m "
                    f"embeded.generalize --mine --holdout-k {S.SPRIME_HELDOUT_K} --k "
                    f"{S.K_NEGATIVES} --cap <settings.SPRIME_TRAIN_CAP>"))
    return json.loads(p.read_text())


# --------------------------------------------------------------------- mining (D3=B)

def _sprime_clone_sets(rows: list[int], pair_frag_rows) -> dict[int, set[int]]:
    """anchor fragment -> every fragment labelled a true clone of it (ALL labelled pairs,
    not only the train slice: a validation clone pair must never become a negative)."""
    sets: dict[int, set[int]] = defaultdict(set)
    for i in rows:
        a, b, lab = pair_frag_rows[i]
        if lab == 1:
            sets[a].add(b)
            sets[b].add(a)
    return sets


def _encode_sprime(frags: dict[int, str], *, embed_fn=None, device=None) -> tuple[np.ndarray, dict]:
    """Embed every s′ fragment with the untouched base model — the single encoder path
    (`embeded.encoder` via `mining.semantic_index.encode_corpus`), cached and meta-gated
    exactly like the Phase 1 corpus_emb. `embed_fn` is the test injection seam."""
    ids = sorted(frags)
    cached_meta = None
    p_emb, p_meta = S.artifact(EMB_NAME), S.artifact(EMB_META_NAME)
    if p_emb.exists() and p_meta.exists():
        try:
            cached_meta = json.loads(p_meta.read_text())
        except json.JSONDecodeError:
            cached_meta = None
        if cached_meta and (cached_meta.get("version") == S.VERSION
                            and cached_meta.get("n") == len(ids)):
            return np.load(p_emb), cached_meta
    if embed_fn is not None:
        emb = np.asarray(embed_fn([frags[i] for i in ids]), dtype=np.float32)
        meta = {"model": "injected (test)", "max_len": None, "n": len(ids),
                "version": S.VERSION}
    else:
        from .mining.semantic_index import encode_corpus
        emb = encode_corpus([frags[i] for i in ids], model_id=S.MODEL_ID,
                            max_len=S.MAX_LEN, device=device)
        meta = {"model": S.MODEL_ID, "max_len": S.MAX_LEN, "n": len(ids),
                "pooling": S.POOLING, "version": S.VERSION}
    np.save(p_emb, emb)
    p_meta.write_text(json.dumps(meta, indent=2))
    return emb, meta


def mine(holdout_k: int, k: int, cap: int, *, embed_fn=None, force: bool = False,
         device=None) -> dict:
    """Mine C1/C2/C3 negatives from s′ itself, inside the held-in functionalities
    (`embeded.negatives.mine` — the same implementation, so the manipulation cannot
    drift from the main experiment). Also writes the fragment table and the ONE
    seen/unseen partition the runs will score."""
    _require_holdout_k(holdout_k)
    if cap >= S.TRAIN_PAIRS_CAP:
        raise SystemExit(f"cap {cap} is not smaller than TRAIN_PAIRS_CAP={S.TRAIN_PAIRS_CAP} — "
                         "the generalisation runs stay deliberately smaller (CLOUD.md §5.6, "
                         "PHASE3_PLAN §3.4)")
    census_p = S.artifact(CENSUS_NAME)
    if not census_p.exists():
        raise SystemExit(missing_artifact_message(
            census_p, hint="record the measured holdout BEFORE mining: python -m "
                           f"embeded.generalize --census --holdout-k {holdout_k}"))
    census = json.loads(census_p.read_text())
    if census.get("rule") != RULE_NAME or census.get("holdout_k", census.get("k")) != holdout_k:
        raise SystemExit(f"{CENSUS_NAME} disagrees with this run (rule {census.get('rule')!r}, "
                         f"k {census.get('holdout_k', census.get('k'))!r}) — re-run --census")

    if not force:
        cached = _cached_mining(k, cap, holdout_k)
        if cached is not None:
            for c in CONDS:
                print(c, {x: cached[c][x] for x in ("strategy", "n_anchors", "k", "n_triples")})
            print(f"[cache] s′ triples for version {S.VERSION} (k={k}, cap={cap}) already "
                  "mined — skipping (--force to redo)")
            return cached

    frame, source = load_sprime_frame()
    frags, texts = build_fragments(frame)
    write_fragments(frags)
    pair_frag_rows = pair_frag_rows_of(frame, texts)

    split = build_split(frame, texts, census["holdout"])
    split["source_md5"] = source["md5"]
    split["source_rows"] = source["rows"]
    S.artifact(SPLIT_NAME).write_text(json.dumps(split, indent=2))

    train_rows = split["rows"]["train"]
    clone_sets = _sprime_clone_sets(sorted(train_rows) + split["rows"]["valid"],
                                    pair_frag_rows)
    pool = sorted({x for i in train_rows for x in pair_frag_rows[i][:2]})
    all_ids = sorted(frags)
    # shared (anchor, positive) stream: first positive per anchor over the train-slice
    # clone pairs, seeded order — identical for every condition, capped at cap // k
    pos_of: dict[int, int] = {}
    for i in train_rows:
        a, b, lab = pair_frag_rows[i]
        if lab == 1:
            pos_of.setdefault(a, b)
    anchors = sorted(pos_of)
    random.Random(f"sprime-anchors#{S.SEED}").shuffle(anchors)
    anchors = anchors[:max(cap // k, 1)]
    stream = [(a, pos_of[a]) for a in anchors]
    print(f"s′ frags={len(all_ids)} held-in pool={len(pool)} clone anchors={len(stream)} "
          f"k={k} cap={cap} -> <= {len(stream) * k} triples per condition")

    bm25 = semantic = None
    if len(stream):
        from .mining.bm25_index import BM25Index
        bm25 = BM25Index([frags[i] for i in all_ids])
    if len(stream):
        emb, meta = _encode_sprime(frags, embed_fn=embed_fn, device=device)
        if len(emb) != len(all_ids):
            raise SystemExit(f"{EMB_NAME}: {len(emb)} rows for {len(all_ids)} fragments — "
                             "stale cache, delete it and re-run --mine")
        from .mining.semantic_index import SemanticIndex
        semantic = SemanticIndex(emb, all_ids)
        print("semantic index:", meta.get("model"), f"max_len={meta.get('max_len')}",
              f"n={meta['n']}")

    from . import negatives as NG

    summary: dict = {}
    corpus_list = pool
    for strategy, cond in NG.COND.items():
        triples, stats = NG.mine(strategy, stream=stream, corpus_list=corpus_list,
                                 clone_sets=clone_sets, k=k, all_ids=all_ids,
                                 bm25=bm25, semantic=semantic, seed=S.SEED)
        out = S.artifact(TRIPLES_NAME.format(cond=cond))
        with open(out, "w", encoding="utf-8") as fh:
            for an, po, ne in triples:
                fh.write(json.dumps({"anchor": an, "positive": po, "negative": ne}) + "\n")
        summary[cond] = {"strategy": strategy, "n_anchors": len(stream), "k": k,
                         "n_triples": len(triples), "stats": stats,
                         "file": str(out.relative_to(S.ROOT))
                         if str(out).startswith(str(S.ROOT)) else str(out)}
        print(cond, {x: summary[cond][x] for x in ("n_anchors", "k", "n_triples")},
              "stats", stats)
    summary["_run"] = {"version": S.VERSION, "k": k, "n_anchors": len(stream), "cap": cap,
                       "holdout": census["holdout"], "holdout_k": holdout_k,
                       "rule": RULE_NAME, "seed": S.SEED, "source_md5": source["md5"],
                       "sprime_pairs": int(len(frame)), "sprime_frag_pool": len(pool),
                       "split_counts": split["counts"]}
    S.artifact(SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    print(f"wrote {SPLIT_NAME}, {SUMMARY_NAME} and triples for {CONDS}")
    return summary


def _cached_mining(k: int, cap: int, holdout_k: int) -> dict | None:
    p = S.artifact(SUMMARY_NAME)
    if not p.exists():
        return None
    try:
        summary = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    run = summary.get("_run") or {}
    if not (run.get("version") == S.VERSION and run.get("k") == k and run.get("cap") == cap
            and run.get("holdout_k") == holdout_k and run.get("rule") == RULE_NAME):
        return None
    if not all(S.artifact(TRIPLES_NAME.format(cond=c)).exists() for c in CONDS):
        return None
    return summary


# --------------------------------------------------------------------- gate G1s

def _gate_extra(meta, ci) -> list[str]:
    extra = [f"embeddings: {meta.get('model')} max_len={meta.get('max_len')} n={meta.get('n')}",
             "corpus: s′ (bcb_v2_sampled_bf) — the CodeXGLUE verdict must NOT be assumed to "
             "transfer; this gate re-measures it (PHASE3_PLAN §2 item 4)"]
    if ci is not None:
        extra.append(f"95% CI for d(C1→C2) = {ci[0]:.3f}–{ci[1]:.3f} "
                     f"({S.HARDNESS_BOOTSTRAP} resamples over anchors, not negatives)")
    return extra


def hardness_main(holdout_k: int, *, embed_fn=None, anchors: int = S.HARDNESS_CHECK_ANCHORS,
                  no_bootstrap: bool = False, device=None, stamp: str | None = None) -> int:
    """Gate G1s: the D1 verdict (`hardcheck.evaluate_gap`) re-measured on the s′ triples.
    History is appended under its own section; exit 1 = FAIL ⇒ stop (notebook §6)."""
    _require_holdout_k(holdout_k)
    files = {}
    for c in CONDS:
        p = S.artifact(TRIPLES_NAME.format(cond=c))
        if not p.exists():
            raise SystemExit(missing_artifact_message(
                p, hint="mine the s′ triples first: python -m embeded.generalize --mine "
                        f"--holdout-k {holdout_k} --k {S.K_NEGATIVES} --cap <cap>"))
        files[c] = str(p)
    frags = load_sprime_fragments()
    try:
        emb, meta = _encode_sprime(frags, embed_fn=embed_fn, device=device)
    except ImportError as exc:
        raise SystemExit(f"{EMB_NAME} is missing and torch/transformers are unavailable "
                         f"({exc}) — encode on the GPU machine via --mine first") from exc
    row_of = {i: r for r, i in enumerate(sorted(frags))}
    per = hardness_measure(files, emb, row_of, anchors, S.SEED)
    ci = None if no_bootstrap else bootstrap_d(
        {c: per[c].get("per_anchor", {}) for c in CONDS})
    means = {c: per[c]["mean_cos"] for c in CONDS}
    sds = {c: per[c]["std"] for c in CONDS}
    d = standardized_gap(means["C1"], sds["C1"], means["C2"], sds["C2"])
    verdict = evaluate_gap(means, sds, ci=ci)
    ok = verdict["ok"]

    run = ["", f"### run {stamp or _stamp()} — version `{S.VERSION}`, rule: {S.HARDNESS_RULE}", ""]
    for c in CONDS:
        m = per[c]
        run.append(f"- {c}: mean cos(anchor, negative) = {m['mean_cos']:.5f} "
                   f"(sd {m['std']:.3f}, n = {m['n_samples']}, anchors = {m['n_anchors']})")
    run.extend(f"- {line}" for line in _gate_extra(meta, ci))
    run.append(f"- **verdict: {'PASS' if ok else 'FAIL'}** — " + "; ".join(verdict["msgs"]))
    run.append("")
    _replace_or_append_section(S.REPORT_MD, GATE_SECTION, "\n".join(run))

    payload = {c: {k: v for k, v in per[c].items() if k != "per_anchor"} for c in CONDS}
    log_p = S.artifact(GATE_LOG_NAME)
    runs = []
    if log_p.exists():
        try:
            runs = json.loads(log_p.read_text()).get("runs", [])
        except json.JSONDecodeError:
            runs = []
    runs.append({"stamp": stamp or _stamp(), "version": S.VERSION, "rule": S.HARDNESS_RULE,
                 "ok": ok, "d_c1_c2": round(d, 4), "ci": ci, "per_condition": payload})
    log_p.write_text(json.dumps({"runs": runs}, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"d(C1→C2) = {d:.3f}   gap = {means['C2'] - means['C1']:+.5f}")
    print("GATE G1s:", "PASS" if ok else "FAIL", f"[{S.HARDNESS_RULE}]")
    for m in verdict["msgs"]:
        print(" -", m)
    return 0 if ok else 1


# --------------------------------------------------------------------- runs

def sprime_run_config(condition: str, seed: int, holdout: list[str], *, cap: int,
                      smoke: bool = False) -> dict:
    """Every frozen choice of a generalisation run, `embeded.train`-shaped so the same
    fingerprint/checkpoint contract applies. `holdout_functionalities` is checked by the
    notebook after every run (a per-condition holdout would confound strategy with
    difficulty, PHASE3_PLAN §3.2)."""
    artifacts = {}
    for name in (FRAGS_NAME, SPLIT_NAME, SUMMARY_NAME, TRIPLES_NAME.format(cond=condition)):
        p = S.artifact(name)
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            artifacts[name] = h.hexdigest()[:16]
    cfg = {
        "version": S.VERSION, "condition": condition, "seed": int(seed),
        "smoke": bool(smoke), "kind": "generalisation", "corpus": "sprime (bcb_v2_sampled_bf)",
        "doi": S.SPRIME_DOI, "holdout_k": S.SPRIME_HELDOUT_K,
        "holdout_functionalities": sorted(str(f) for f in holdout),
        "threshold_policy": S.THRESHOLD_POLICY, "valid_frac": VALID_FRAC,
        "model_id": S.MODEL_ID, "pooling": S.POOLING, "option": "A (token-only)",
        "loss": S.LOSS, "triplet_margin": S.TRIPLET_MARGIN, "optimizer": "adamw",
        "lr": S.LR, "weight_decay": S.WEIGHT_DECAY, "warmup_steps": S.WARMUP_STEPS,
        "grad_clip": S.GRAD_CLIP, "batch": S.BATCH, "epochs": S.EPOCHS,
        "max_len": S.MAX_LEN, "amp_dtype": S.AMP_DTYPE,
        "grad_checkpoint": S.GRAD_CHECKPOINT, "k_negatives": S.K_NEGATIVES,
        "sprime_train_cap": cap, "seed_list": list(S.SEEDS), "artifacts": artifacts,
    }
    if smoke:
        cfg.update({"batch": min(4, S.BATCH), "epochs": S.SMOKE_EPOCHS,
                    "max_len": S.SMOKE_MAX_LEN, "warmup_steps": 0,
                    "amp_dtype": "float32", "max_triples": S.SMOKE_TRIPLES})
    return cfg


def load_sprime_triples(condition: str, cap: int | None = None) -> list[tuple[int, int, int]]:
    p = S.artifact(TRIPLES_NAME.format(cond=condition))
    if not p.exists():
        raise SystemExit(missing_artifact_message(
            p, hint="mine the s′ triples first: python -m embeded.generalize --mine "
                    f"--holdout-k {S.SPRIME_HELDOUT_K} --k {S.K_NEGATIVES} --cap <cap>"))
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append((int(r["anchor"]), int(r["positive"]), int(r["negative"])))
    if cap is not None and cap < len(out):
        out = out[:cap]
    return out


def _split_pairs(split: dict, pair_frag_rows) -> dict[str, list[tuple[int, int, int]]]:
    return {name: [pair_frag_rows[i] for i in split["rows"][name]]
            for name in ("valid", "unseen")}


def train_sprime(condition: str, seed: int, *, holdout: list[str], cap: int,
                 smoke: bool = False, force: bool = False, encoder=None,
                 device: str | None = None, verbose: bool = True) -> dict:
    """One generalisation run — the `embeded.train` loop on s′ triples: same objective,
    same recipe, same checkpoint/resume/fingerprint contract, only the corpus changes."""
    import torch

    from .encoder import autocast_ctx, forward_embed

    if condition not in CONDS:
        raise SystemExit(f"{condition} is not trainable on s′ — C0 is the untuned baseline: "
                         "`python -m embeded.generalize --condition C0 --seed 13 "
                         "--holdout-k 3 --transfer`")
    summary = json.loads(S.artifact(SUMMARY_NAME).read_text()) \
        if S.artifact(SUMMARY_NAME).exists() else {}
    if "cap" not in (summary.get("_run") or {}):
        raise SystemExit(missing_artifact_message(
            S.artifact(SUMMARY_NAME),
            hint="mine first: python -m embeded.generalize --mine --holdout-k "
                 f"{S.SPRIME_HELDOUT_K} --k {S.K_NEGATIVES} --cap {cap}"))
    cap = int(summary["_run"]["cap"])
    out = sprime_run_dir(condition, seed, smoke=smoke)
    out.mkdir(parents=True, exist_ok=True)
    cfg = sprime_run_config(condition, seed, holdout, cap=cap, smoke=smoke)
    cfg["fingerprint"] = fingerprint(cfg)
    cfg_path, metrics_path, ckpt_path = (out / CONFIG_NAME, out / METRICS_NAME,
                                         out / CHECKPOINT_NAME)
    if metrics_path.exists() and not force:
        prev = json.loads(metrics_path.read_text())
        if prev.get("fingerprint") == cfg["fingerprint"]:
            print(f"[cache] sprime {condition} seed={seed} already trained for version "
                  f"{S.VERSION} — skipping (--force to redo)")
            return prev
        print(f"[warn] {metrics_path} is from config {prev.get('fingerprint')}, current is "
              f"{cfg['fingerprint']} — retraining (--force silences this)")

    triples = load_sprime_triples(condition, cap=cfg.get("max_triples"))
    frags = load_sprime_fragments()

    if encoder is None:
        from .encoder import load_encoder
        tok, model, dev = load_encoder(S.MODEL_ID, device)
    else:
        tok, model, dev = encoder
    device = dev
    model.train()
    if cfg["grad_checkpoint"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ck = None if force else load_checkpoint(ckpt_path, model, cfg)
    start_epoch = int(ck["epoch"]) if ck else 0
    step = int(ck["step"]) if ck else 0
    if ck:
        print(f"[resume] {ckpt_path.name}: continuing at epoch {start_epoch}, step {step}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = max(1, cfg["epochs"] * max(1, -(-len(triples) // cfg["batch"])))
    warmup = int(cfg["warmup_steps"])

    def lr_at(s: int) -> float:
        if warmup and s < warmup:
            return (s + 1) / warmup
        left = max(0, total_steps - s)
        return left / max(1, total_steps - warmup)

    amp = cfg["amp_dtype"]
    use_scaler = amp == "float16" and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    set_seed(seed)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log = open(out / LOG_NAME, "a", encoding="utf-8")
    t0, losses, n_steps = time.time(), [], 0
    for epoch in range(start_epoch, cfg["epochs"]):
        for batch in batches(triples, cfg["batch"], seed, epoch):
            a = [frags[x] for x, _, _ in batch]
            p = [frags[y] for _, y, _ in batch]
            n = [frags[z] for _, _, z in batch]
            with autocast_ctx(device, amp):
                va = forward_embed(model, tok, a, max_len=cfg["max_len"], device=device,
                                   amp_dtype=amp)
                vp = forward_embed(model, tok, p, max_len=cfg["max_len"], device=device,
                                   amp_dtype=amp)
                vn = forward_embed(model, tok, n, max_len=cfg["max_len"], device=device,
                                   amp_dtype=amp)
                loss = triplet_loss(va, vp, vn, cfg["triplet_margin"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * lr_at(step)
            step += 1
            n_steps += 1
            losses.append(float(loss.detach()))
            if verbose and step % 25 == 0:
                el = time.time() - t0
                print(f"  sprime {condition} seed={seed} epoch {epoch} step {step}/{total_steps} "
                      f"loss {losses[-1]:.4f} mean {np.mean(losses[-25:]):.4f} "
                      f"{n_steps / el:.1f} triples/s", flush=True)
            log.write(json.dumps({"epoch": epoch, "step": step,
                                  "loss": float(loss.detach()),
                                  "triples_per_s": n_steps / max(time.time() - t0, 1e-9)}) + "\n")
        save_checkpoint(ckpt_path, model, epoch=epoch + 1, step=step, cfg=cfg,
                        loss=float(np.mean(losses)) if losses else None)
    log.close()

    metrics = {
        "condition": condition, "seed": int(seed), "version": S.VERSION,
        "fingerprint": cfg["fingerprint"], "smoke": bool(smoke), "kind": "generalisation",
        "holdout_functionalities": cfg["holdout_functionalities"],
        "loss_final_mean": round(float(np.mean(losses)), 6) if losses else None,
        "steps": step, "triples": len(triples), "epochs": cfg["epochs"],
        "train_seconds": round(time.time() - t0, 1),
        "triples_per_s": round(n_steps / max(time.time() - t0, 1e-9), 2),
        "checkpoint": str(ckpt_path), "device": str(device),
        "checkpoint_reload": verify_checkpoint(ckpt_path, model, cfg),
        "finished_utc": _stamp(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(json.dumps(metrics, indent=2))
        print(f"next: python -m embeded.generalize --condition {condition} --seed {seed} "
              f"--holdout-k {S.SPRIME_HELDOUT_K}")
    return metrics


# --------------------------------------------------------------------- evaluation

def eval_sprime(condition: str, seed: int, *, holdout: list[str], split: dict,
                pair_frag_rows: list[tuple[int, int, int]], frags: dict[int, str],
                out_dir: Path, embed_fn=None, kind: str = "generalisation",
                smoke: bool = False, force: bool = False, device: str | None = None,
                verbose: bool = True) -> dict:
    """Score one model on the seen/unseen partition (PHASE3_PLAN §3.2): threshold from
    the seen-validation slice only (`max_f1_on_valid`), applied unchanged to the unseen
    pairs. Used by the trained runs AND by --transfer — one scoring path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, pred_path = out_dir / EVAL_METRICS_NAME, out_dir / PREDICTIONS_NAME
    cfg = {"version": S.VERSION, "condition": condition, "seed": int(seed), "kind": kind,
           "smoke": bool(smoke), "corpus": "sprime (bcb_v2_sampled_bf)", "doi": S.SPRIME_DOI,
           "holdout_functionalities": sorted(str(f) for f in holdout),
           "model_id": S.MODEL_ID, "pooling": S.POOLING,
           "max_len": S.SMOKE_MAX_LEN if smoke else S.MAX_LEN,
           "eval_batch": S.EVAL_BATCH, "threshold_policy": S.THRESHOLD_POLICY,
           "valid_frac": VALID_FRAC}
    cfg["fingerprint"] = fingerprint(cfg)
    if metrics_path.exists() and not force:
        prev = json.loads(metrics_path.read_text())
        if prev.get("fingerprint") == cfg["fingerprint"]:
            print(f"[cache] {out_dir.name} already evaluated for version {S.VERSION} — "
                  "skipping (--force to redo)")
            return prev

    set_seed(seed)
    pairs = {"valid": [pair_frag_rows[i] for i in split["rows"]["valid"]],
             "test": [pair_frag_rows[i] for i in split["rows"]["unseen"]]}
    if smoke:
        pairs = {s: subsample(v, S.SMOKE_PAIRS, seed) for s, v in pairs.items()}
    ids = sorted({x for v in pairs.values() for p in v for x in p[:2]})
    missing = [i for i in ids if i not in frags]
    if missing:
        raise SystemExit(f"{len(missing)} s′ pair fragments are not in {FRAGS_NAME} "
                         f"(e.g. {missing[:3]}) — stale artifacts, re-run --mine")
    if embed_fn is None:
        fn, device, note = _sprime_default_embed_fn(condition, seed, out_dir, device,
                                                    smoke=smoke)
    else:
        fn, note = embed_fn, "injected embed_fn (test)"
    t0 = time.time()
    vecs = np.asarray(fn([frags[i] for i in ids]), dtype=np.float32)
    vec_of = {i: vecs[r] for r, i in enumerate(ids)}
    scores = {s: np.array([float(vec_of[p[0]] @ vec_of[p[1]]) for p in pairs[s]],
                          dtype=np.float32) for s in pairs}
    labels = {s: np.array([p[2] for p in pairs[s]], dtype=int) for s in pairs}

    sel = best_f1_threshold(scores["valid"], labels["valid"])   # SEEN slice only
    valid = precision_recall_f1(scores["valid"], labels["valid"], sel["threshold"])
    test = precision_recall_f1(scores["test"], labels["test"], sel["threshold"])
    metrics = {
        "condition": condition, "seed": int(seed), "version": S.VERSION,
        "fingerprint": cfg["fingerprint"], "smoke": bool(smoke), "kind": kind,
        "model": note, "corpus": "sprime (bcb_v2_sampled_bf)", "doi": S.SPRIME_DOI,
        "threshold_policy": S.THRESHOLD_POLICY,
        "holdout_functionalities": sorted(str(f) for f in holdout),
        "threshold": valid["threshold"],
        "f1_seen": valid["f1"], "f1_unseen": test["f1"],
        "delta": round(valid["f1"] - test["f1"], 4),
        "valid": {k: valid[k] for k in ("precision", "recall", "f1", "n")},
        "valid_map_at_r": round(map_at_r(scores["valid"], labels["valid"]), 4),
        "test_f1": test["f1"], "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_map_at_r": round(map_at_r(scores["test"], labels["test"]), 4),
        "test": {k: test[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n")},
        "n_pairs": {s: len(pairs[s]) for s in pairs},
        "encode_seconds": round(time.time() - t0, 1),
        "evaluated_utc": _stamp(),
    }
    (out_dir / EVAL_CONFIG_NAME).write_text(json.dumps(cfg, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    # the same npz keys as Phase 2 (valid = seen slice, test = unseen holdout), so the
    # demo and any table code can treat the two experiments uniformly
    np.savez_compressed(pred_path,
                        threshold=np.float32(valid["threshold"]),
                        valid_scores=scores["valid"].astype(np.float32),
                        valid_labels=labels["valid"],
                        test_scores=scores["test"].astype(np.float32),
                        test_labels=labels["test"])
    if verbose:
        print(json.dumps(metrics, indent=2))
    return metrics


def _sprime_default_embed_fn(condition: str, seed: int, out_dir: Path, device,
                             smoke: bool = False):
    """Production encoder for a trained s′ run: base model + THIS run's checkpoint
    (fingerprint-checked). C0 / transfer readings load the Phase 2 path instead
    (`evaluate.default_embed_fn`)."""
    from .encoder import encode_texts, load_encoder

    if condition == "C0":
        return default_embed_fn("C0", seed, smoke=smoke, device=device)
    ck_path = out_dir / CHECKPOINT_NAME
    if not ck_path.exists():
        raise SystemExit(f"missing {ck_path} — train it first: "
                         "`python -m embeded.generalize --condition "
                         f"{condition} --seed {seed} --holdout-k {S.SPRIME_HELDOUT_K}`")
    holdout = json.loads(S.artifact(SPLIT_NAME).read_text())["holdout"]
    summary = json.loads(S.artifact(SUMMARY_NAME).read_text())
    cfg = sprime_run_config(condition, seed, holdout,
                            cap=int(summary["_run"]["cap"]), smoke=smoke)
    tok, model, dev = load_encoder(S.MODEL_ID, device)
    ck = load_checkpoint(ck_path, model, cfg)
    if ck is None:
        raise SystemExit(f"{ck_path} was written by a different config — retrain "
                         "(`python -m embeded.generalize --condition "
                         f"{condition} --seed {seed} --holdout-k {S.SPRIME_HELDOUT_K} --force`); "
                         "evaluating it would mix two experiments")
    model.eval()

    max_len = S.SMOKE_MAX_LEN if smoke else S.MAX_LEN

    def embed_fn(texts):
        return encode_texts(model, tok, texts, max_len=max_len, device=dev,
                            batch_size=S.EVAL_BATCH,
                            amp_dtype=S.AMP_DTYPE if str(dev).startswith("cuda") else "float32")

    return embed_fn, dev, f"{S.MODEL_ID} + sprime {condition} seed {seed} (step {ck['step']})"


# --------------------------------------------------------------------- transfer (D3 A)

def transfer_main(condition: str, seed: int, holdout_k: int, *, embed_fn=None,
                  force: bool = False, device=None, verbose: bool = True) -> int:
    """D3 option (A) as the free secondary reading: score a Phase 2 checkpoint (C0 =
    the untouched base model) on the s′ seen/unseen partition. NO training, never part
    of the Δ table (PHASE3_PLAN §3.3)."""
    _require_holdout_k(holdout_k)
    if condition not in ALL_CONDITIONS:
        raise SystemExit(f"unknown condition {condition!r} — one of {list(ALL_CONDITIONS)}")
    if condition in CONDS:
        ck = phase2_run_dir(condition, seed) / CHECKPOINT_NAME
        if not ck.exists():
            raise SystemExit(f"missing {ck} — the transfer reading scores a Phase 2 "
                             "checkpoint; train it first: `python -m embeded.train "
                             f"--condition {condition} --seed {seed}`")
    split = load_split()
    if len(split["holdout"]) != holdout_k:
        raise SystemExit(f"{SPLIT_NAME} holds out {len(split['holdout'])} functionalities, "
                         f"expected {holdout_k} — re-run --mine")
    frame, _ = load_sprime_frame()
    frags = load_sprime_fragments()
    pair_frag_rows = pair_frag_rows_of(frame)
    out_dir = sprime_transfer_dir(condition, seed)
    if embed_fn is None:
        embed_fn, _dev, _note = default_embed_fn(condition, seed, device=device)
    eval_sprime(condition, seed, holdout=split["holdout"], split=split,
                pair_frag_rows=pair_frag_rows, frags=frags, out_dir=out_dir,
                embed_fn=embed_fn, kind="transfer", force=force, device=device,
                verbose=verbose)
    print(f"transfer reading written to {out_dir.name}/ — secondary observation only, "
          "never part of the Δ table (it never trains on functionality-labelled data)")
    return 0


# --------------------------------------------------------------------- results table

_RUN_DIR_RE = re.compile(r"^sprime_(C[123])_(\d+)$")
_TRANSFER_RE = re.compile(r"^sprime_transfer_(C[0123])_(\d+)$")


def write_results_table(runs_root: Path | None = None, *, report_md: Path | None = None) -> dict:
    """Rebuild the generalisation section: per-seed F1_seen / F1_unseen / Δ, then mean
    with the spread visible; smoke runs excluded; holdout identity asserted; the
    transfer readings kept in their own clearly-secondary block."""
    root = runs_root or (S.ARTIFACTS / S.RUNS_SUBDIR)
    rows, smoke, transfer = [], [], []
    if root.exists():
        for p in sorted(root.glob("*/" + EVAL_METRICS_NAME)):
            m = json.loads(p.read_text())
            name = p.parent.name
            if _RUN_DIR_RE.match(name):
                (smoke if m.get("smoke") else rows).append(m)
            elif _TRANSFER_RE.match(name) and not m.get("smoke"):
                transfer.append(m)
    holdouts = {tuple(m.get("holdout_functionalities") or []) for m in rows}
    if len(holdouts) > 1:
        raise SystemExit(f"the s′ runs do not share one holdout: {holdouts} — strategy would "
                         "be confounded with holdout difficulty (PHASE3_PLAN §3.2); fix the "
                         "runner and re-run before reporting Δ")

    lines = ["Generated from `artifacts/runs/sprime_<condition>_<seed>/eval_metrics.json` by "
             f"`python -m embeded.generalize --results-table` — version `{S.VERSION}`. "
             f"Threshold policy: *{S.THRESHOLD_POLICY}* — chosen on the seen slice only, "
             "applied unchanged to the unseen functionalities (the threshold column shows it "
             "was not tuned on the unseen pairs).", "",
             f"**Different corpus from the Phase 2 main table** (decision D3=B: fine-tuned on "
             f"s′, negatives mined from s′): absolute F1 is NOT comparable across the two "
             f"tables. The unseen-functionality drop itself is Kitsios et al. (ASE 2025) — "
             "the question here is only whether the size of the drop differs across "
             "negative-selection strategies.", "",
             "| condition | seed | F1_seen | F1_unseen | Δ | threshold | n seen | n unseen |",
             "|---|---|---|---|---|---|---|---|"]
    agg: dict[str, dict] = {}
    for c in ALL_CONDITIONS:
        ms = sorted([m for m in rows if m["condition"] == c], key=lambda r: r["seed"])
        for m in ms:
            delta = m["f1_seen"] - m["f1_unseen"]
            lines.append(f"| {c} | {m['seed']} | {m['f1_seen']:.4f} | {m['f1_unseen']:.4f} | "
                         f"{delta:+.4f} | {m['threshold']:.4f} | {m['valid']['n']} | "
                         f"{m['test']['n']} |")
        if ms:
            f1s = np.array([m["f1_seen"] for m in ms])
            f1u = np.array([m["f1_unseen"] for m in ms])
            dl = f1s - f1u
            sd = float(dl.std(ddof=1)) if len(ms) > 1 else 0.0
            agg[c] = {"n": len(ms), "mean_f1_seen": round(float(f1s.mean()), 4),
                      "mean_f1_unseen": round(float(f1u.mean()), 4),
                      "mean_delta": round(float(dl.mean()), 4), "sd_delta": round(sd, 4),
                      "per_seed_delta": [round(float(x), 4) for x in dl]}
    lines += ["", "| condition | runs | mean F1_seen | mean F1_unseen | mean Δ | sd Δ | "
              "per-seed Δ (spread visible) |", "|---|---|---|---|---|---|---|"]
    for c, a in agg.items():
        lines.append(f"| {c} | {a['n']} | {a['mean_f1_seen']:.4f} | {a['mean_f1_unseen']:.4f} | "
                     f"{a['mean_delta']:+.4f} | {a['sd_delta']:.4f} | "
                     f"{', '.join(f'{x:+.3f}' for x in a['per_seed_delta'])} |")
    notes = []
    if "C0" in agg:
        notes.append(f"C0 is the untuned baseline on the same partition (one evaluation, "
                     f"not a trained condition) at mean Δ {agg['C0']['mean_delta']:+.4f}.")
    trained = [c for c in CONDS if c in agg]
    if len(trained) == len(CONDS):
        best = max(trained, key=lambda c: agg[c]["mean_f1_unseen"])
        notes.append(f"On the unseen functionalities the highest mean F1 is {best} "
                     f"({agg[best]['mean_f1_unseen']:.4f}); per-seed values above — never the "
                     "best seed alone. If Δ is indistinguishable across C1/C2/C3, that is the "
                     "finding (PHASE3_PLAN §3.1): no fourth strategy (P6-1), no LR rescue (P6-2).")
    elif trained:
        notes.append(f"Partial: {', '.join(trained)} only — a partial table may only be "
                     "reported if the P6-8 seed ladder was taken deliberately "
                     "(seeds cut before strategies).")
    if smoke:
        notes.append(f"{len(smoke)} smoke-run result(s) are excluded from this table "
                     "(harness checks, not measurements).")
    if not rows:
        notes.append("No completed generalisation runs yet — this table is a placeholder.")
    lines += [""] + [f"- {n}" for n in notes] + [""]

    if transfer:
        lines += ["### Transfer readings — D3 option (A), secondary, NOT the Δ table", "",
                  "Phase 2 checkpoints (C0 = base model) scored on the same s′ partition "
                  "without training; the independent variable is gone, so these never enter "
                  "the table above (PHASE3_PLAN §3.3).", "",
                  "| condition | seed | F1_seen | F1_unseen | Δ | threshold | n unseen |",
                  "|---|---|---|---|---|---|---|"]
        for m in sorted(transfer, key=lambda r: (r["condition"], r["seed"])):
            lines.append(f"| {m['condition']} | {m['seed']} | {m['f1_seen']:.4f} | "
                         f"{m['f1_unseen']:.4f} | {m['f1_seen'] - m['f1_unseen']:+.4f} | "
                         f"{m['threshold']:.4f} | {m['test']['n']} |")
        lines.append("")

    block = "\n".join(lines)
    rep = report_md or S.REPORT_MD
    _replace_or_append_section(rep, TABLE_SECTION, block)
    print(f"wrote {rep} ({len(rows)} generalisation runs, {len(transfer)} transfer readings, "
          f"{len(smoke)} smoke excluded)")
    return {"runs": len(rows), "aggregate": agg}


# --------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--census", action="store_true",
                       help="measure per-functionality pair counts, fix the D4 holdout")
    modes.add_argument("--mine", action="store_true",
                       help="mine C1/C2/C3 negatives from s′ + write the seen/unseen split")
    modes.add_argument("--hardness", action="store_true",
                       help="gate G1s: the D1 hardness verdict re-measured on s′ (exit 1 = FAIL)")
    modes.add_argument("--results-table", action="store_true",
                       help="rebuild the generalisation section from saved run metrics")
    ap.add_argument("--condition", choices=ALL_CONDITIONS)
    ap.add_argument("--seed", type=int, default=S.SEED)
    ap.add_argument("--holdout-k", type=int, default=S.SPRIME_HELDOUT_K,
                    help=f"must equal settings.SPRIME_HELDOUT_K = {S.SPRIME_HELDOUT_K}")
    ap.add_argument("--k", type=int, default=S.K_NEGATIVES, help="negatives per anchor (--mine)")
    ap.add_argument("--cap", type=int, default=getattr(S, "SPRIME_TRAIN_CAP", None),
                    help="triples per condition (--mine); must be < TRAIN_PAIRS_CAP")
    ap.add_argument("--transfer", action="store_true",
                    help="score the Phase 2 checkpoint (C0: base model) on the s′ split; no training")
    ap.add_argument("--smoke", action="store_true", help="harness check only; never a result")
    ap.add_argument("--force", action="store_true", help="redo a completed/resumable run")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.census:
        return census_main(a.holdout_k)
    if a.mine:
        if a.cap is None:
            raise SystemExit("--cap is required (settings.SPRIME_TRAIN_CAP is not set; "
                             "PHASE3_PLAN §3.4: pick it from the census counts, keep it "
                             "smaller than TRAIN_PAIRS_CAP, record it in settings.py)")
        if a.k != S.K_NEGATIVES:
            print(f"[warn] --k {a.k} != settings.K_NEGATIVES = {S.K_NEGATIVES}: the main "
                  "experiment mined with the settings value — a different k here changes the "
                  "manipulation between corpora")
        mine(a.holdout_k, a.k, a.cap, force=a.force, device=a.device)
        return 0
    if a.hardness:
        return hardness_main(a.holdout_k, device=a.device)
    if a.results_table:
        write_results_table()
        return 0

    if not a.condition:
        ap.error("--condition is required (or use --census/--mine/--hardness/--results-table)")
    if a.condition == "C0" and not a.transfer:
        raise SystemExit("C0 is the untuned baseline and is never trained on s′: "
                         "`python -m embeded.generalize --condition C0 --seed 13 "
                         "--holdout-k 3 --transfer`")
    if a.condition in CONDS and not a.transfer:
        split = load_split()
        train_sprime(a.condition, a.seed, holdout=split["holdout"],
                     cap=getattr(S, "SPRIME_TRAIN_CAP", 0) or 0, smoke=a.smoke,
                     force=a.force, device=a.device, verbose=not a.quiet)
        frame, _ = load_sprime_frame()
        frags = load_sprime_fragments()
        pair_frag_rows = pair_frag_rows_of(frame)
        eval_sprime(a.condition, a.seed, holdout=split["holdout"], split=split,
                    pair_frag_rows=pair_frag_rows, frags=frags,
                    out_dir=sprime_run_dir(a.condition, a.seed, smoke=a.smoke),
                    kind="generalisation", smoke=a.smoke, force=a.force,
                    device=a.device, verbose=not a.quiet)
        if not a.smoke:
            write_results_table()
        return 0
    return transfer_main(a.condition, a.seed, a.holdout_k, force=a.force,
                         device=a.device, verbose=not a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
