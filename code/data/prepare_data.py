"""Phase 0 + Phase 1 data pipeline (IMPLEMENTATION_PLAN §4, Phase 0.2/.3, Phase 1.1).

- loads the CodeXGLUE BCB file in EITHER form:
    --hf          datasets.load_dataset(...)   (pair rows with func1/func2)
    --dir PATH    original layout: data.jsonl {"idx","func"} + train/valid/test.txt "i j label"
- normalises to: artifacts/fragments.jsonl ({idx, sha1, text}) + artifacts/pairs_{split}.tsv
- verifies counts against the spec, measures train/test fragment overlap, token lengths
- writes the numbers into report/measurements.md (Phase 0 gate G0 artefact)

Run:  python -m code.data.prepare_data --hf --verify-spec
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .. import settings as S

SPLITS = ("train", "valid", "test")


def sha1(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- load

# HF exposes the middle split as "validation"; the pipeline names it "valid".
HF_SPLIT_NAMES = {"valid": "validation"}


def _rows_from_hf():
    from datasets import load_dataset  # lazy: only needed with --hf

    ds = load_dataset(S.HF_DATASET)
    for split in SPLITS:
        key = HF_SPLIT_NAMES.get(split, split)
        if key not in ds:
            raise KeyError(f"split {key!r} (for {split!r}) not in dataset "
                           f"(have: {sorted(ds.keys())})")
        for r in ds[key]:
            yield split, r["func1"], r["func2"], int(bool(r["label"]))


def _load_dir(root: Path):
    """Original layout: data.jsonl carries the authoritative idx; pair files
    reference it. Fragments with no pair STILL count toward the corpus file."""
    frags: dict[int, str] = {}
    with open(root / "data.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            frags[int(d["idx"])] = d["func"]
    pairs: dict[str, list] = {s: [] for s in SPLITS}
    for split in SPLITS:
        with open(root / f"{split}.txt", encoding="utf-8") as fh:
            for line in fh:
                i, j, lab = line.split()
                pairs[split].append((int(i), int(j), int(lab)))
    return frags, pairs


# --------------------------------------------------------------------------- build

def build(rows):
    """HF format (pairs carry text) -> dedup fragments by text sha; ids assigned
    by first appearance. (In this dataset every fragment has >=1 pair by
    construction, so nothing is lost by deriving from pairs.)"""
    idx_of: dict[str, int] = {}
    frags: list[str] = []
    pairs: dict[str, list] = {s: [] for s in SPLITS}
    for split, f1, f2, lab in rows:
        ids = []
        for f in (f1, f2):
            h = sha1(f)
            if h not in idx_of:
                idx_of[h] = len(frags)
                frags.append(f)
            ids.append(idx_of[h])
        pairs[split].append((ids[0], ids[1], lab))
    return {i: x for i, x in enumerate(frags)}, pairs


def write_artifacts(frags, pairs) -> dict:
    A = S.ARTIFACTS
    A.mkdir(parents=True, exist_ok=True)
    with open(A / "fragments.jsonl", "w", encoding="utf-8") as fh:
        for i in sorted(frags):
            fh.write(json.dumps({"idx": i, "sha1": sha1(frags[i]), "text": frags[i]}) + "\n")
    for split, lst in pairs.items():
        with open(A / f"pairs_{split}.tsv", "w", encoding="utf-8") as fh:
            for i, j, lab in lst:
                fh.write(f"{i}\t{j}\t{lab}\n")
    manifest = {"version": S.VERSION, "n_fragments": len(frags),
                "n_pairs": {s: len(l) for s, l in pairs.items()}}
    (A / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- loaders for later stages

def load_fragments() -> dict[int, str]:
    out = {}
    with open(S.ARTIFACTS / "fragments.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["idx"]] = d["text"]
    return out


def load_pairs(split: str) -> list[tuple[int, int, int]]:
    out = []
    with open(S.ARTIFACTS / f"pairs_{split}.tsv", encoding="utf-8") as fh:
        for line in fh:
            i, j, lab = line.split()
            out.append((int(i), int(j), int(lab)))
    return out


def labeled_clones() -> dict[int, set[int]]:
    """anchor -> labelled true clones (TRAIN split only; mining never sees valid/test labels)."""
    cs: dict[int, set[int]] = defaultdict(set)
    for i, j, lab in load_pairs("train"):
        if lab == 1:
            cs[i].add(j)
            cs[j].add(i)
    return dict(cs)


def corpus_ids() -> list[int]:
    """§4.1 rules 1+3: fragments that appear in TRAIN pairs and NEVER in TEST pairs.
    (Valid-split fragments may be mined; valid only tunes the threshold.)"""
    tr, te = set(), set()
    for i, j, _ in load_pairs("train"):
        tr |= {i, j}
    for i, j, _ in load_pairs("test"):
        te |= {i, j}
    return sorted(tr - te)


def positives() -> list[tuple[int, int]]:
    """All train label==1 pairs (anchor, positive), deterministically shuffled by S.SEED.
    Sampled once, shared by every condition (SCOPE P1-3)."""
    import random
    pos = [(i, j) for i, j, lab in load_pairs("train") if lab == 1]
    random.Random(S.SEED).shuffle(pos)
    return pos


# --------------------------------------------------------------------------- measurements

def overlap() -> dict:
    tr, te = set(), set()
    for i, j, _ in load_pairs("train"):
        tr |= {i, j}
    for i, j, _ in load_pairs("test"):
        te |= {i, j}
    inter = tr & te
    frac = len(inter) / len(te) if te else 0.0
    d = {"train_frag": len(tr), "test_frag": len(te), "overlap": len(inter),
         "overlap_frac_of_test": round(frac, 4)}
    (S.ARTIFACTS / "overlap.json").write_text(json.dumps(d, indent=2))
    return d


def token_lengths() -> dict:
    """FINAL_SPEC §4: measure, then choose max_len. Uses the real tokenizer if
    transformers is importable, else a word-count fallback (flagged as approximate)."""
    texts = list(load_fragments().values())
    tok_kind = "wordsplit-approx"
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(S.MODEL_ID)
        enc = tk(texts, add_special_tokens=True, truncation=False)
        lens = [len(e) for e in enc["input_ids"]]
        tok_kind = S.MODEL_ID
    except Exception as e:  # offline / not installed
        lens = [len(re.findall(r"\w+", t)) + 2 for t in texts]
        print(f"[warn] tokenizer unavailable ({e.__class__.__name__}); word-count fallback — rerun on Colab before freezing MAX_LEN")
    lens.sort()
    q = lambda p: lens[min(int(p * len(lens)), len(lens) - 1)]
    d = {"tokenizer": tok_kind, "n": len(lens), "p50": q(.50), "p90": q(.90),
         "p95": q(.95), "p99": q(.99), "max": lens[-1],
         "recommended_max_len": 256 if q(.99) <= 250 else 512}
    (S.ARTIFACTS / "token_lengths.json").write_text(json.dumps(d, indent=2))
    return d


def record_measurements(manifest, ov, tl) -> None:
    S.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Phase 0 — measured numbers",
        "",
        f"generated by `python -m code.data.prepare_data` — version `{S.VERSION}`",
        "",
        "| quantity | spec value | measured | ok |",
        "|---|---|---|---|",
        f"| fragments | {S.EXPECTED_FRAGMENTS} | {manifest['n_fragments']} | {'✅' if manifest['n_fragments'] == S.EXPECTED_FRAGMENTS else '❌'} |",
    ]
    for s in SPLITS:
        lines.append(f"| {s} pairs | {S.EXPECTED_SPLIT_ROWS[s]} | {manifest['n_pairs'][s]} | "
                     f"{'✅' if manifest['n_pairs'][s] == S.EXPECTED_SPLIT_ROWS[s] else '❌'} |")
    lines += [
        "",
        f"**Train/test fragment overlap** (§4.2): {ov['overlap']} of {ov['test_frag']} test fragments "
        f"({ov['overlap_frac_of_test']:.1%}) also appear in a train pair. Standard-benchmark F1 is an "
        "optimistic upper bound to this extent; the generalisation test is there to expose it.",
        "",
        f"**Token lengths** ({tl['tokenizer']}): p50={tl['p50']} p90={tl['p90']} p95={tl['p95']} "
        f"p99={tl['p99']} max={tl['max']} → **max_len = {tl['recommended_max_len']}**.",
        "",
    ]
    block = "\n".join(lines)
    if S.REPORT_MD.exists():
        old = S.REPORT_MD.read_text()
        old = re.sub(r"## Phase 0 — measured numbers.*?(?=\n## |\Z)", block + "\n", old, flags=re.S)
        S.REPORT_MD.write_text(old if "## Phase 0" in old else old + "\n" + block)
    else:
        S.REPORT_MD.write_text("# Measurements\n\n" + block)
    print(f"wrote {S.REPORT_MD}")


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf", action="store_true", help="load via datasets.load_dataset")
    g.add_argument("--dir", type=Path, help="local CodeXGLUE-format directory (data.jsonl + splits)")
    ap.add_argument("--verify-spec", action="store_true")
    a = ap.parse_args(argv)

    if a.hf:
        frags, pairs = build(_rows_from_hf())
    else:
        frags, pairs = _load_dir(a.dir)
    manifest = write_artifacts(frags, pairs)
    print(json.dumps(manifest, indent=2))
    if a.verify_spec:
        bad = []
        if manifest["n_fragments"] != S.EXPECTED_FRAGMENTS:
            bad.append(f"fragments {manifest['n_fragments']} != {S.EXPECTED_FRAGMENTS}")
        for s, n in S.EXPECTED_SPLIT_ROWS.items():
            if manifest["n_pairs"][s] != n:
                bad.append(f"{s} {manifest['n_pairs'][s]} != {n}")
        print("SPEC CHECK:", "FAIL: " + "; ".join(bad) if bad else "all counts match ✅")
    ov = overlap()
    tl = token_lengths()
    print("overlap:", ov)
    print("tokens :", tl)
    record_measurements(manifest, ov, tl)


if __name__ == "__main__":
    main()
