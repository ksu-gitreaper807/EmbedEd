"""scripts/hardness_diagnostics on the offline fixture with fake embeddings:
it must run end-to-end, produce the JSON + report section, and get the
sanity relations right (semantic negatives = the ruler's own top ranks)."""
import json

import numpy as np

from embeded import negatives as NG
from embeded import settings as S
from embeded.tests.test_semantic_hardcheck import _fake_emb
from scripts import hardness_diagnostics as D


def test_diagnostics_end_to_end(fx, capsys):
    emb = _fake_emb()
    all_ids = sorted(fx.frags)
    np.save(S.ARTIFACTS / "corpus_emb.npy", emb)
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps(
        {"model": "fake", "pooling": "mean", "max_len": 8, "n": len(emb),
         "corpus_first": all_ids[:5], "version": S.VERSION}))
    NG.main(["--strategies", "random,bm25,semantic", "--k", "3"])

    D.main(["--examples", "1"])
    out = capsys.readouterr().out
    res = json.loads((S.ARTIFACTS / "hardness_diagnostics.json").read_text())

    assert res["k"] == 3 and res["n_anchors"] > 0
    assert set(res["conditions"]) == {"C1", "C2", "C3"}
    c3 = res["conditions"]["C3"]
    # semantic negatives are chosen by this very embedding: they sit at the top
    # of every anchor's similarity distribution and C3 is the top of the range
    assert c3["in_anchor_top20_frac"] == 1.0
    assert c3["percentile_median"] >= res["conditions"]["C1"]["percentile_median"]
    assert c3["position_in_C1_C3_range"] == 1.0 and res["conditions"]["C1"]["position_in_C1_C3_range"] == 0.0
    for c in ("C1", "C2", "C3"):
        m = res["conditions"][c]
        assert 0 <= m["percentile_mean"] <= 100 and 0 <= m["closer_than_positive_frac"] <= 1
    assert set(res["negative_set_overlap_frac"]) == {"C1&C2", "C1&C3", "C2&C3"}
    assert -1.5 < res["scale"]["random_pair_cos"] < 1.0

    text = S.REPORT_MD.read_text()
    assert text.count("## Gate G1 — diagnostics") == 1
    assert "EXAMPLES" in out and "C3 neg#1" in out

    D.main(["--examples", "0"])                      # rerun replaces the section
    assert S.REPORT_MD.read_text().count("## Gate G1 — diagnostics") == 1


def test_label_noise_floors_on_fixture(fx):
    from collections import defaultdict
    from embeded.mining.bm25_index import tokenize_code
    from embeded import negatives as NG
    import numpy as np, json
    emb = _fake_emb()
    np.save(S.ARTIFACTS / "corpus_emb.npy", emb)
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps({"model": "fake", "max_len": 8, "n": len(emb)}))
    NG.main(["--strategies", "random,bm25,semantic", "--k", "3"])
    triples = {c: D._load_triples(S.ARTIFACTS / f"triples_{c}.jsonl") for c in ("C1", "C2", "C3")}
    clones_valid = defaultdict(set)
    for i, j, lab in fx.P.load_pairs("valid"):
        if lab == 1:
            clones_valid[i].add(j); clones_valid[j].add(i)
    ln = D.label_noise(triples, fx.frags, fx.P.labeled_clones(), clones_valid)
    for c in ("C1", "C2", "C3"):
        assert ln[c]["train_labelled_clone_frac"] == 0.0        # the exclusion did its job
        assert 0 <= ln[c]["valid_labelled_clone_frac"] <= 1
        assert 0 <= ln[c]["near_dup_frac_ge_0.75"] <= ln[c]["near_dup_frac_ge_0.5"] <= 1
    assert "jaccard_median" in ln["positives"]
    # the fixture's adversarial anchor 16 has a near-verbatim rival (18) that BM25 promotes
    assert D.jaccard(set(tokenize_code(fx.frags[16])), set(tokenize_code(fx.frags[18]))) > 0.5
