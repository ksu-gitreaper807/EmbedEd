"""Deterministic mini CodeXGLUE-format fixture (data.jsonl + train/valid/test.txt).

Small enough that the whole pipeline + gates run offline in <1 s, and built so
tests can assert exact behaviour. It deliberately contains:

  * a BM25 trap: anchor 16's raw top match is labelled clone 17 (shares the
    rare token xtra99); rival 20 shares xtra99 too but lives in test -> it is
    ranked ahead of 18 yet must be dropped by the corpus filter (Rule 1,
    §4.1); after both exclusions, 18 must be promoted to first negative;
  * train/test fragment overlap (2, 7, 11, 20, 22, 23) -> none may be mined;
  * valid-only and test-only fragments -> test-only never enter the corpus,
    valid-only may (valid only tunes the threshold).

    python -m code.fixtures.make_fixture OUTDIR   # writes files + fixture_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

FAMILIES = {
    "copy": ["fileinputstream", "fileoutputstream", "bytes", "bufsize47", "transfer"],
    "fib": ["recursive", "memo", "sequence", "golden13", "basecase"],
    "sort": ["comparator", "swap", "partition", "pivot71", "quicksort"],
}
N_PER = 8  # 0-7 copy, 8-15 fib, 16-23 sort
RARE_TRAP = {16, 17, 20}  # these texts get the rare token xtra99

TRAIN = [
    (0, 1, 1), (8, 9, 1), (16, 17, 1), (2, 3, 1), (21, 22, 1),          # clones
    (16, 18, 0), (2, 6, 0), (0, 8, 0), (17, 23, 0), (20, 21, 0),         # same-fam
    (3, 11, 0), (7, 20, 0), (1, 16, 0),                                   # cross-fam
]
VALID = [(16, 23, 0), (0, 4, 0), (8, 12, 0)]
TEST = [(2, 7, 0), (10, 11, 1), (20, 22, 0), (13, 23, 0), (19, 13, 0)]


def frag_text(idx: int) -> str:
    """Instance tokens are keyed by GLOBAL idx, not idx %% N_PER: a mod-wrap
    would give families shared high-IDF tokens and scramble BM25 ranks."""
    toks = list(FAMILIES.values())[idx // N_PER]
    body = " ".join(f"{t}{i}" for i, t in enumerate(toks))
    trap = " xtra99" if idx in RARE_TRAP else ""
    return f"public void m{idx}(int n) {{ {body} value{idx} = n ;{trap} handler{idx} }}"


def main(outdir: Path) -> None:
    texts = [frag_text(i) for i in range(N_PER * 3)]
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "data.jsonl", "w") as fh:
        for i, t in enumerate(texts):
            fh.write(json.dumps({"idx": i, "func": t}) + "\n")
    for name, lst in [("train", TRAIN), ("valid", VALID), ("test", TEST)]:
        with open(outdir / f"{name}.txt", "w") as fh:
            for a, b, l in lst:
                fh.write(f"{a} {b} {l}\n")

    tr = {x for a, b, _ in TRAIN for x in (a, b)}
    te = {x for a, b, _ in TEST for x in (a, b)}
    meta = {
        "n_fragments": len(texts),
        "n_pairs": {"train": len(TRAIN), "valid": len(VALID), "test": len(TEST)},
        "overlap_ids": sorted(tr & te),
        "corpus_ids": sorted(tr - te),          # prepare_data.corpus_ids() must equal this
        "adversarial": {"anchor": 16, "clone": 17, "excluded_rival": 20,
                        "promoted": 18,
                        "note": "raw BM25 rank for 16: 17 (clone, banned), 20 "
                                "(test-overlap, not in corpus), then 18 -> must be promoted first"},
    }
    (outdir / "fixture_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fixture"))
