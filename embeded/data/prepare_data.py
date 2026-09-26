"""Phase 0 + Phase 1 data pipeline (IMPLEMENTATION_PLAN §4, Phase 0.2/.3, Phase 1.1).

Canonical fragment list = CodeXGLUE's data.jsonl (9,126 lines, FINAL_SPEC
correction 9): 1,063 fragments are byte-identical to another fragment, so the
corpus is 8,063 unique fragment texts. Fragments are keyed 0..N-1 by first
appearance in data.jsonl and deduplicated by text sha1 — both input modes
below produce byte-identical artifacts, so Colab (--hf) and local dev (--dir)
never diverge.

- pair rows in EITHER form:
    --hf          datasets.load_dataset(...)   (func1/func2 text rows)
    --dir PATH    original layout: data.jsonl + train/valid/test.txt "i j label"
- normalises to: artifacts/fragments.jsonl ({idx, sha1, text}) + artifacts/pairs_{split}.tsv
- verifies counts against the spec, measures train/test fragment overlap, token lengths
- writes the numbers into report/measurements.md (Phase 0 gate G0 artefact)

Run:  python -m embeded.data.prepare_data --hf --verify-spec
      python -m embeded.data.prepare_data --dir CODEXGLUE_DATASET_DIR --verify-spec
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
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


def _stream(url: str, part: Path, headers: dict | None = None) -> None:
    merged = {"User-Agent": "embeded/phase0"}
    merged.update(headers or {})
    req = urllib.request.Request(url, headers=merged)
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0, last = 0, time.time(), 0.0
        with open(part, "wb") as fh:
            while chunk := r.read(1 << 16):
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if total and now - last >= 0.5:
                    last = now
                    print(f"  data.jsonl: {done / 1e6:.1f}/{total / 1e6:.1f} MB "
                          f"({done / total:.0%})", end="\r", flush=True)
    print()
    print(f"  data.jsonl: {part.stat().st_size / 1e6:.1f} MB done")
    if part.stat().st_size != S.CODEXGLUE_DATA_JSONL_SIZE:
        part.unlink(missing_ok=True)
        raise IOError(f"data.jsonl size mismatch (got {part.stat().st_size}, "
                      f"want {S.CODEXGLUE_DATA_JSONL_SIZE}) — pinned file changed?")


def fetch_data_jsonl(cache: Path) -> Path:
    """CodeXGLUE's data.jsonl — the canonical fragment list (FINAL_SPEC §4:
    'confirm with wc -l dataset/data.jsonl'). Pinned commit, size-verified,
    cached under ARTIFACTS. Set EMBEDED_DATA_JSONL=/path/to/data.jsonl to use
    a local copy (offline / Colab with a pre-fetched file)."""
    env = os.environ.get("EMBEDED_DATA_JSONL")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"EMBEDED_DATA_JSONL={p} does not exist")
        return p
    dest = cache / "codexglue_data.jsonl"
    if dest.exists() and dest.stat().st_size == S.CODEXGLUE_DATA_JSONL_SIZE:
        return dest
    if dest.exists():
        dest.unlink()
    cache.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    attempts = ((S.CODEXGLUE_DATA_JSONL_URL, None),
                (S.CODEXGLUE_DATA_JSONL_API_URL,
                 {"Accept": "application/vnd.github.raw"}))
    for url, headers in attempts:
        try:
            _stream(url, part, headers)
            part.replace(dest)
            return dest
        except Exception as e:
            part.unlink(missing_ok=True)
            print(f"[warn] data.jsonl fetch from {url.split('/')[2]} failed "
                  f"({e.__class__.__name__}: {str(e)[:120]}) — trying next source")
    raise RuntimeError("could not fetch CodeXGLUE data.jsonl from any source; "
                       "download it manually and set EMBEDED_DATA_JSONL")


# --------------------------------------------------------------------------- build

def _fragments_from_datajsonl(path: Path):
    """Dense fragment ids by first appearance in data.jsonl, dedup by text sha1.
    Returns (frags {idx: text}, src2id {source idx: idx}, sha2id, n_lines)."""
    frags: dict[int, str] = {}
    sha2id: dict[str, int] = {}
    src2id: dict[int, int] = {}
    n_lines = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            n_lines += 1
            i_src = int(d["idx"])
            if i_src in src2id:
                raise ValueError(f"data.jsonl: duplicate source idx {i_src}")
            h = sha1(d["func"])
            if h not in sha2id:
                sha2id[h] = len(frags)
                frags[len(frags)] = d["func"]
            src2id[i_src] = sha2id[h]
    return frags, src2id, sha2id, n_lines


def build_dir(root: Path):
    """Original layout: data.jsonl carries the authoritative fragment list;
    pair files reference it by source idx. Fragments with no pair STILL count
    toward the corpus file (verified: there are none in BCB)."""
    frags, src2id, _, n_lines = _fragments_from_datajsonl(root / "data.jsonl")
    pairs: dict[str, list] = {s: [] for s in SPLITS}
    for split in SPLITS:
        with open(root / f"{split}.txt", encoding="utf-8") as fh:
            for line in fh:
                i, j, lab = line.split()
                if int(i) not in src2id or int(j) not in src2id:
                    raise ValueError(f"{split}.txt references idx {i}/{j} "
                                     f"not present in data.jsonl")
                pairs[split].append((src2id[int(i)], src2id[int(j)], int(lab)))
    return frags, pairs, n_lines


def build_hf(rows, datajsonl_path: Path):
    """HF format (pairs carry text) joined onto the canonical data.jsonl
    fragment list by text sha1. If HF's texts ever drift from the released
    data.jsonl, this fails loudly instead of silently building a different
    corpus."""
    frags, _, sha2id, n_lines = _fragments_from_datajsonl(datajsonl_path)
    pairs: dict[str, list] = {s: [] for s in SPLITS}
    seen = 0
    for split, f1, f2, lab in rows:
        seen += 1
        h1, h2 = sha1(f1), sha1(f2)
        if h1 not in sha2id or h2 not in sha2id:
            raise ValueError(
                f"row {seen} ({split}): pair text not found in data.jsonl "
                f"(hf sha1s {h1[:8]}/{h2[:8]} not in the 9,126-line canonical "
                f"list) — the HF parquet has drifted from the CodeXGLUE file; "
                f"re-resolve S.CODEXGLUE_DATA_JSONL_URL")
        pairs[split].append((sha2id[h1], sha2id[h2], lab))
    return frags, pairs, n_lines


def write_artifacts(frags, pairs, n_data_lines: int) -> dict:
    A = S.ARTIFACTS
    A.mkdir(parents=True, exist_ok=True)
    with open(A / "fragments.jsonl", "w", encoding="utf-8") as fh:
        for i in sorted(frags):
            fh.write(json.dumps({"idx": i, "sha1": sha1(frags[i]), "text": frags[i]}) + "\n")
    for split, lst in pairs.items():
        with open(A / f"pairs_{split}.tsv", "w", encoding="utf-8") as fh:
            for i, j, lab in lst:
                fh.write(f"{i}\t{j}\t{lab}\n")
    manifest = {"version": S.VERSION, "n_data_lines": n_data_lines,
                "n_fragments": len(frags),
                "n_pairs": {s: len(l) for s, l in pairs.items()}}
    (A / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def cached_manifest() -> dict | None:
    """The Phase 0.2 artifacts, if a complete set for THIS settings.VERSION is on
    disk (e.g. restored by `scripts.hf_artifacts pull`). Resume protocol: every
    stage is artifact-gated so a Colab restart costs a pull, not a rerun."""
    mf = S.ARTIFACTS / "manifest.json"
    if not mf.exists():
        return None
    try:
        manifest = json.loads(mf.read_text())
    except json.JSONDecodeError:
        return None
    if manifest.get("version") != S.VERSION:
        return None
    needed = [S.ARTIFACTS / "fragments.jsonl"] + [S.ARTIFACTS / f"pairs_{s}.tsv" for s in SPLITS]
    if not all(p.exists() and p.stat().st_size > 0 for p in needed):
        return None
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
    (Valid-split fragments may be mined; valid only tunes the threshold.)
    Measured in TEXT space: identical text on both sides is leakage even when
    the source idx differs."""
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
        try:
            # instant hit when the tokenizer is already cached (Colab/Drive
            # reruns); no network at all in that case
            tk = AutoTokenizer.from_pretrained(S.MODEL_ID, local_files_only=True)
        except Exception:
            # cap per-request timeout so a dead/slow HF link degrades to the
            # fallback instead of stalling the pipeline for minutes of retries
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
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
    def ok(measured, expected):
        return "✅" if measured == expected else "❌"
    lines = [
        "## Phase 0 — measured numbers",
        "",
        f"generated by `python -m embeded.data.prepare_data` — version `{S.VERSION}`",
        "",
        "| quantity | spec value | measured | ok |",
        "|---|---|---|---|",
        f"| data.jsonl lines | {S.EXPECTED_DATA_LINES} | {manifest['n_data_lines']} | {ok(manifest['n_data_lines'], S.EXPECTED_DATA_LINES)} |",
        f"| unique fragments | {S.EXPECTED_FRAGMENTS} | {manifest['n_fragments']} | {ok(manifest['n_fragments'], S.EXPECTED_FRAGMENTS)} |",
    ]
    for s in SPLITS:
        lines.append(f"| {s} pairs | {S.EXPECTED_SPLIT_ROWS[s]} | {manifest['n_pairs'][s]} | "
                     f"{ok(manifest['n_pairs'][s], S.EXPECTED_SPLIT_ROWS[s])} |")
    lines += [
        "",
        f"**Train/test fragment overlap** (§4.2, text space): {ov['overlap']} of {ov['test_frag']} test fragments "
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
        # lambda replacement: block is data, never a regex template
        old = re.sub(r"## Phase 0 — measured numbers.*?(?=\n## |\Z)",
                     lambda _m: block + "\n", old, flags=re.S)
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
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if artifacts for this VERSION are already present")
    a = ap.parse_args(argv)

    manifest = None if a.force else cached_manifest()
    if manifest is not None:
        print(f"[cache] artifacts for version {manifest['version']} already in {S.ARTIFACTS} — "
              "skipping download/rebuild (--force to redo)")
    else:
        if a.hf:
            data_path = fetch_data_jsonl(S.ARTIFACTS)
            frags, pairs, n_lines = build_hf(_rows_from_hf(), data_path)
        else:
            frags, pairs, n_lines = build_dir(a.dir)
        manifest = write_artifacts(frags, pairs, n_lines)
    print(json.dumps(manifest, indent=2))
    if a.verify_spec:
        bad = []
        if manifest["n_data_lines"] != S.EXPECTED_DATA_LINES:
            bad.append(f"data lines {manifest['n_data_lines']} != {S.EXPECTED_DATA_LINES}")
        if manifest["n_fragments"] != S.EXPECTED_FRAGMENTS:
            bad.append(f"fragments {manifest['n_fragments']} != {S.EXPECTED_FRAGMENTS}")
        for s, n in S.EXPECTED_SPLIT_ROWS.items():
            if manifest["n_pairs"][s] != n:
                bad.append(f"{s} {manifest['n_pairs'][s]} != {n}")
        print("SPEC CHECK:", "FAIL: " + "; ".join(bad) if bad else "all counts match ✅")
    ov = overlap()
    tl = token_lengths()
    print("overlap:", ov)
    print("tokens : ", tl)
    record_measurements(manifest, ov, tl)


if __name__ == "__main__":
    main()
