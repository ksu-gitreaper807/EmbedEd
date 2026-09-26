"""Blind 50+50 false-negative audit tooling on the offline fixture."""
import csv
import json

import numpy as np

from embeded import negatives as NG
from embeded import settings as S
from embeded.tests.test_semantic_hardcheck import _fake_emb
from scripts import audit_sample as A


def _mine(fx):
    np.save(S.ARTIFACTS / "corpus_emb.npy", _fake_emb())
    (S.ARTIFACTS / "corpus_emb.meta.json").write_text(json.dumps({"model": "fake", "max_len": 8, "n": 24}))
    NG.main(["--strategies", "random,bm25,semantic", "--k", "3"])


def test_make_is_blind_and_seeded(fx, capsys):
    _mine(fx)
    A.main(["make", "--n", "2", "--seed", "7"])
    d = S.ARTIFACTS / "audit"
    pairs_md = (d / "audit_pairs.md").read_text()
    labels = list(csv.DictReader(open(d / "audit_labels.csv")))
    key = list(csv.DictReader(open(d / "audit_key.csv")))
    assert len(labels) == len(key) == 4 and all(r["label"] == "" for r in labels)
    assert pairs_md.count("## P0") == 4 and "```java" in pairs_md
    assert "C2" not in pairs_md and "C3" not in pairs_md            # blind sheet
    assert sorted(r["condition"] for r in key) == ["C2", "C2", "C3", "C3"]
    for cond in ("C2", "C3"):                                       # distinct anchors per condition
        anchors = [r["anchor"] for r in key if r["condition"] == cond]
        assert len(set(anchors)) == len(anchors)
    again = A.sample_pairs(2, 7)
    assert [(p["condition"], p["anchor"], p["negative"]) for p in again] == \
           [(r["condition"], int(r["anchor"]), int(r["negative"])) for r in key]


def test_score_rates_and_report(fx, capsys):
    _mine(fx)
    A.make(2, 7)
    d = S.ARTIFACTS / "audit"
    key = list(csv.DictReader(open(d / "audit_key.csv")))
    lab = {}
    for r in key:                       # C2: one clone one not; C3: one clone one unsure
        ids = [k["id"] for k in key if k["condition"] == r["condition"]]
        lab[ids[0]] = "clone"
        lab[ids[1]] = "not_clone" if r["condition"] == "C2" else "unsure"
    with open(d / "audit_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "note"]); w.writeheader()
        for i, v in lab.items():
            w.writerow({"id": i, "label": v, "note": ""})
    res = A.score()
    assert res["C2"]["fn_rate"] == 0.5 and res["C2"]["unsure"] == 0
    assert res["C3"]["fn_rate"] == 0.5 and res["C3"]["fn_rate_upper_incl_unsure"] == 1.0
    lo, hi = res["C3"]["fn_rate_ci95"]
    assert 0 <= lo <= 0.5 <= hi <= 1
    assert S.REPORT_MD.read_text().count("## False-negative audit") == 1
    A.score()                                                       # rerun replaces the section
    assert S.REPORT_MD.read_text().count("## False-negative audit") == 1


def test_score_refuses_unlabelled(fx):
    _mine(fx)
    A.make(2, 7)
    import pytest
    with pytest.raises(SystemExit, match="unlabelled"):
        A.score()


def test_wilson_bounds():
    assert A.wilson(0, 50)[0] == 0.0 and A.wilson(50, 50)[1] == 1.0
    lo, hi = A.wilson(25, 50)
    assert 0.36 < lo < 0.37 and 0.63 < hi < 0.64
