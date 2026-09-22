"""Phase 0/1 pipeline behaviour on the fixture (fully offline; counts are
fixture counts, not spec counts — --verify-spec runs against real data)."""
import json

from embeded import negatives as NG


def _triples(cond):
    from embeded import settings as S
    return [json.loads(l) for l in open(S.ARTIFACTS / f"triples_{cond}.jsonl")]


def test_manifest_and_artifacts(fx):
    man = json.loads((fx.tmp / "artifacts" / "manifest.json").read_text())
    assert man["n_fragments"] == fx.meta["n_fragments"]
    assert man["n_pairs"] == fx.meta["n_pairs"]
    assert json.loads((fx.tmp / "artifacts" / "overlap.json").read_text())["overlap"] \
        == len(fx.meta["overlap_ids"])


def test_corpus_rule1(fx):
    assert fx.corpus == fx.meta["corpus_ids"]
    assert not (set(fx.corpus) & set(fx.meta["overlap_ids"]))


def test_positives_shared_and_deterministic(fx):
    a, b = fx.P.positives(), fx.P.positives()
    assert a == b
    train1 = {(i, j) for i, j, l in fx.P.load_pairs("train") if l == 1}
    assert all(p in train1 for p in a)
    assert {(0, 1), (8, 9), (16, 17), (2, 3), (21, 22)} <= set(a)


def test_mining_invariants(fx):
    NG.main(["--strategies", "random,bm25", "--k", "3"])
    t1, t2 = _triples("C1"), _triples("C2")
    corpus = set(fx.corpus)
    for ts in (t1, t2):
        assert ts
        for r in ts:
            assert r["negative"] != r["anchor"]
            assert r["negative"] in corpus
            assert r["negative"] not in fx.clones.get(r["anchor"], set())
    # same anchor/positive stream in same order across conditions
    s1 = [(r["anchor"], r["positive"]) for r in t1]
    s2 = [(r["anchor"], r["positive"]) for r in t2]
    assert s1[::3] == s2[::3] and len(t1) == len(t2)
    # C2's first negative for anchor 16 is the promoted rank-2
    assert next(r["negative"] for r in t2 if r["anchor"] == 16) == 18


def test_hf_dir_parity(fx):
    """--hf and --dir must build identical artifacts (FINAL_SPEC correction 9).
    Simulates the HF row stream (text, not idx) from the fixture's own files."""
    import json as _json
    from embeded.data import prepare_data as P

    texts = {}
    for line in open(fx.fdir / "data.jsonl", encoding="utf-8"):
        d = _json.loads(line)
        texts[int(d["idx"])] = d["func"]
    rows = []
    for split in P.SPLITS:
        for line in open(fx.fdir / f"{split}.txt", encoding="utf-8"):
            i, j, lab = line.split()
            rows.append((split, texts[int(i)], texts[int(j)], int(lab)))

    frags_hf, pairs_hf, n_lines = P.build_hf(iter(rows), fx.fdir / "data.jsonl")
    assert n_lines == fx.meta["n_fragments"]
    assert frags_hf == fx.frags, "hf and dir must key fragments identically (data.jsonl order)"
    for s in P.SPLITS:
        assert pairs_hf[s] == fx.P.load_pairs(s)


def test_hf_drift_fails_loudly(fx):
    """If the HF parquet ever drifts from the released data.jsonl, build_hf must
    raise instead of silently building a different corpus."""
    from embeded.data import prepare_data as P

    rows = [("train", "public void m0(int n) { fileinputstream0 fileoutputstream1 }",
             "definitely not in data.jsonl", 0)]
    try:
        P.build_hf(iter(rows), fx.fdir / "data.jsonl")
        raise AssertionError("expected ValueError on unknown pair text")
    except ValueError as e:
        assert "not found in data.jsonl" in str(e)


def test_measurements_written(fx):
    from embeded import settings as S
    txt = S.REPORT_MD.read_text()
    assert "Phase 0 — measured numbers" in txt and "recommended" not in txt.splitlines()[0]
    tl = json.loads((fx.tmp / "artifacts" / "token_lengths.json").read_text())
    assert tl["recommended_max_len"] in (256, 512) and tl["p99"] >= tl["p50"]
