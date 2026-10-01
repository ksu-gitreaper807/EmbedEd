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


def test_make_refuses_to_clobber_a_filled_sheet(fx, capsys):
    """Two hours of blind labelling must not die on a stray `make`."""
    import csv as _csv
    _mine(fx)
    A.main(["make", "--n", "2", "--seed", "7"])
    sheet = S.ARTIFACTS / "audit" / "audit_labels.csv"
    rows = list(_csv.DictReader(open(sheet)))
    rows[0]["label"] = "clone"
    with open(sheet, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["id", "label", "note"])
        w.writeheader(); w.writerows(rows)

    assert A.filled_labels(sheet) == [rows[0]["id"]]
    try:
        A.main(["make", "--n", "2", "--seed", "7"])
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "refusing to overwrite" in str(e) and "score" in str(e)
    assert A.filled_labels(sheet) == [rows[0]["id"]]          # untouched
    A.main(["make", "--n", "2", "--seed", "7", "--force"])     # explicit override works
    assert A.filled_labels(S.ARTIFACTS / "audit" / "audit_labels.csv") == []


def test_score_refuses_a_key_that_no_longer_matches_the_triples(fx, capsys):
    """The sheet outlives the artifacts: if C3 is re-mined, the labels describe
    pairs nobody trains on and must not be scored."""
    import csv as _csv
    _mine(fx)
    A.make(2, 7)
    key_path = S.ARTIFACTS / "audit" / "audit_key.csv"
    key = list(_csv.DictReader(open(key_path)))
    assert all(m["not_in_current_triples"] == 0
               for m in A.check_key_matches_triples(key).values())

    # re-mine C3 with a different k => different negatives for the same anchors
    NG.main(["--strategies", "semantic", "--k", "2", "--force"])
    drift = A.check_key_matches_triples(key)
    assert drift["C3"]["not_in_current_triples"] > 0
    try:
        A.score()
        raise AssertionError("expected SystemExit on a stale key")
    except SystemExit as e:
        assert "does not match the triples on disk" in str(e) and "C3" in str(e)


def test_c4_sheet_is_separate_and_coexists_in_report(fx, capsys, monkeypatch):
    """A C4-only audit lives in its own directory (EMBEDED_AUDIT_DIR), never
    touches the scored Phase-2 sheet, and its report section is tagged so it
    APPENDS beside the C2/C3 section instead of overwriting it."""
    _mine(fx)
    NG.main(["--strategies", "filtered", "--k", "3"])          # mine C4 on the fixture
    c4dir = S.ARTIFACTS / "audit_c4"
    monkeypatch.setenv("EMBEDED_AUDIT_DIR", str(c4dir))
    A.make(2, 7, conds=("C4",))
    key = list(csv.DictReader(open(c4dir / "audit_key.csv")))
    assert {r["condition"] for r in key} == {"C4"}
    assert not (S.ARTIFACTS / "audit" / "audit_key.csv").exists()   # Phase-2 sheet untouched
    with open(c4dir / "audit_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "note"]); w.writeheader()
        for r in key:
            w.writerow({"id": r["id"], "label": "clone", "note": ""})
    res = A.score()
    assert {k for k in res if not k.startswith("_")} == {"C4"}      # the sheet decides
    assert res["C4"]["fn_rate"] == 1.0
    report = S.REPORT_MD.read_text()
    assert "## False-negative audit — C4" in report
    # a later C2/C3 score adds its own tagged section; both survive
    monkeypatch.delenv("EMBEDED_AUDIT_DIR")
    A.make(2, 7)
    d = S.ARTIFACTS / "audit"
    key = list(csv.DictReader(open(d / "audit_key.csv")))
    with open(d / "audit_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "note"]); w.writeheader()
        for r in key:
            w.writerow({"id": r["id"], "label": "not_clone", "note": ""})
    A.score()
    report = S.REPORT_MD.read_text()
    assert "## False-negative audit — C4" in report
    assert "## False-negative audit — C2/C3" in report
    A.score()                                                    # rerun replaces only its own tag
    assert S.REPORT_MD.read_text().count("## False-negative audit") == 2
