"""C3 plumbing with synthetic (fake) embeddings + the hardness-gate logic.
Real GraphCodeBERT embeddings are a Colab step; the LOGIC is all testable."""
import json

import numpy as np

from code import negatives as NG
from code import settings as S
from code.hardcheck import evaluate
from code.mining.semantic_index import SemanticIndex


def _fake_emb(n=24, d=8, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, d)).astype("float32")
    # give 16,17,20 a shared spike so they rank near each other
    e[16] += 3 * e[17]
    e[20] += 1.5 * e[17]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


def test_semantic_mine_matches_reference(fx):
    emb = _fake_emb()
    all_ids = sorted(fx.frags)
    sem = SemanticIndex(emb, all_ids)
    stream = [(16, 17), (0, 1), (8, 9)]
    triples, stats = NG.mine("semantic", stream=stream, corpus_list=fx.corpus,
                             clone_sets=fx.clones, k=3, all_ids=all_ids,
                             semantic=sem)
    # reference: same procedure, independent code path
    sims = emb @ emb[16]
    order = sorted(all_ids, key=lambda i: -sims[i])
    expect_first = [c for c in order if c != 16 and c not in fx.clones[16]
                    and c in set(fx.corpus)]
    got = [neg for an, pos, neg in triples if an == 16]
    assert got == expect_first[:3], f"semantic miner must match reference, got {got}"
    for an, pos, neg in triples:
        assert neg in set(fx.corpus)
        assert neg != an
        assert neg not in fx.clones.get(an, set())


def test_semantic_cli_with_cached_emb(fx, monkeypatch):
    emb = _fake_emb()
    all_ids = sorted(fx.frags)
    np.save(S.ARTIFACTS / "corpus_emb.npy", emb)
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps(
        {"model": "fake", "pooling": "mean", "max_len": 256, "n": len(emb),
         "corpus_first": all_ids[:5], "version": S.VERSION}))
    NG.main(["--strategies", "semantic", "--k", "3"])
    ts = [json.loads(l) for l in open(S.ARTIFACTS / "triples_C3.jsonl")]
    assert ts and all(t["negative"] in set(fx.corpus) for t in ts)


def test_hardness_gate_logic():
    ok, _ = evaluate({"C1": 0.20, "C2": 0.30, "C3": 0.31}, margin=0.02)
    assert ok
    ok, msgs = evaluate({"C1": 0.29, "C2": 0.30, "C3": 0.31}, margin=0.02)
    assert not ok and "gap" in msgs[0]                       # manipulation invisible
    ok, msgs = evaluate({"C1": 0.10, "C2": 0.30, "C3": 0.25}, margin=0.02)
    assert not ok and "C2 <= C3" in msgs[0]                  # ordering violated
    ok, msgs = evaluate({"C1": 0.10}, margin=0.02)
    assert not ok and "missing" in msgs[0]


def test_hardness_measure_on_fixtures(fx):
    NG.main(["--strategies", "random,bm25", "--k", "3"])
    emb = _fake_emb()
    row_of = {c: i for i, c in enumerate(sorted(fx.frags))}
    from code.hardcheck import measure
    files = {c: str(S.ARTIFACTS / f"triples_{c}.jsonl") for c in ("C1", "C2")}
    out = measure(files, emb, row_of, n_anchors=1000, seed=S.SEED)
    assert {"mean_cos", "n_samples"} <= set(out["C1"])
    assert set(out) == {"C1", "C2"}
